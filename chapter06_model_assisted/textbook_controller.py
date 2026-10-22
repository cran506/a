"""第六章代码清单 6.1 的六步二自由度控制律。

该模块保留课件给出的逆动力学、理论建压泄漏、孔口压差逆及死区逆结构。
位置采用 m，力 N，绝对压力 Pa，流量 m³/s，阀指令 V。`K_flow` 的
单位是 m³/(s·V·sqrt(Pa))，已包含阀芯位移与输入电压之间的比例。
反馈复用第五章 DigitalPID：它在内部将位置误差由 m 换为 cm。
"""

from __future__ import annotations

import math
from typing import Sequence

from controllers import DigitalPID


class ModelAssistedController:
    """控制电压 = 物理机理前馈 + 数字 PID 残差反馈。

    `enable_ff=False` 为纯 PID；`enable_gain_sched=False` 使用标称供油
    压差；两者均为 True 时采用当前实测主动进油腔的压力。
    """

    def __init__(self, pid_params: dict, phys_params: dict,
                 enable_ff: bool = True, enable_gain_sched: bool = True):
        self.pid = DigitalPID(**pid_params)
        self.p = phys_params.copy()
        self.enable_ff = bool(enable_ff)
        self.enable_gain_sched = bool(enable_gain_sched)
        needed = ("A1", "A2", "m", "Ps", "P1_0", "P2_0", "Fc", "Fs",
                  "vs", "sigma", "K_flow", "deadzone", "Ct")
        missing = set(needed) - self.p.keys()
        if missing:
            raise ValueError(f"缺少控制器参数：{sorted(missing)}")
        if not all(math.isfinite(self.p[key]) for key in needed):
            raise ValueError("物理参数必须为有限数值")
        if (min(self.p[key] for key in ("A1", "A2", "m", "vs", "K_flow")) <= 0
                or self.p["Ps"] <= max(self.p["P1_0"], self.p["P2_0"])
                or min(self.p["P1_0"], self.p["P2_0"]) <= 0
                or self.p["Fs"] < self.p["Fc"] or self.p["Fc"] < 0
                or self.p["Ct"] < 0 or self.p["deadzone"] < 0):
            raise ValueError("面积、质量、压力、增益、摩擦或死区参数无效")
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.pid.reset()
        self.last = {}

    def _calc_stribeck(self, velocity: float) -> float:
        """式 6.1 中的纯 Stribeck 项；粘性项在合力式中另外计算。"""
        if abs(velocity) <= 1e-6:
            return 0.0
        p = self.p
        magnitude = p["Fc"] + (p["Fs"] - p["Fc"]) * math.exp(
            -(velocity / p["vs"]) ** 2)
        return math.copysign(magnitude, velocity)

    @staticmethod
    def _motion_direction(velocity: float, acceleration: float) -> int:
        if velocity > 1e-6:
            return 1
        if velocity < -1e-6:
            return -1
        if acceleration > 1e-4:
            return 1
        if acceleration < -1e-4:
            return -1
        return 0

    def update(self, r_state: Sequence[float], y_meas: Sequence[float],
               P_meas: Sequence[float]) -> tuple[float, float, float]:
        """返回 `(u_total, u_ff, u_fb)`，各量单位 V。

        `r_state=(x_d,v_d,a_d)`、`y_meas=(x,v)`、`P_meas=(P1,P2)`。
        当前采样只使用当前参考与测量，位置 PID 负责剩余偏差。
        """
        if len(r_state) != 3 or len(y_meas) != 2 or len(P_meas) != 2:
            raise ValueError("参考、运动测量与双腔压力的维度分别必须为 3、2、2")
        values = (*r_state, *y_meas, *P_meas)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("参考与测量必须为有限数值")
        if min(P_meas) <= 0:
            raise ValueError("双腔绝对压力必须为正")
        xd, vd, ad = r_state
        p = self.p
        u_ff = 0.0
        direction = self._motion_direction(vd, ad) if self.enable_ff else 0
        force = theoretical_pressure = leak = total_flow = 0.0
        linear_v = delta_p = 0.0
        if direction:
            # 步骤 1：依据伸出/缩回切换主动腔及返回腔。
            active_area = p["A1"] if direction > 0 else p["A2"]
            return_pressure = p["P2_0"] if direction > 0 else p["P1_0"]
            active_nominal_pressure = p["P1_0"] if direction > 0 else p["P2_0"]
            active_measured_pressure = P_meas[0] if direction > 0 else P_meas[1]
            # 步骤 2：式 6.1。低速起动以加速度方向取静摩擦单侧极限。
            effective_velocity = (vd if abs(vd) > 1e-6 else
                                  1e-5 * math.copysign(1.0, ad))
            force = p["m"] * ad + p["sigma"] * vd + self._calc_stribeck(effective_velocity)
            # 步骤 3：从非对称缸的力平衡反求理论建压，基准静压已扣除。
            if direction > 0:
                theoretical_pressure = ((force + return_pressure * p["A2"])
                                        / active_area - p["P1_0"])
            else:
                theoretical_pressure = ((return_pressure * p["A1"] - force)
                                        / active_area - p["P2_0"])
            # 步骤 4：仅完整策略从当前测量取主动进油腔压力；保护根号分母。
            drive_pressure = (active_measured_pressure if self.enable_gain_sched
                              else active_nominal_pressure)
            delta_p = max(p["Ps"] - drive_pressure, 5.0e5)
            nominal_delta_p = p["Ps"] - active_nominal_pressure
            leak = (p["Ct"] * max(theoretical_pressure, 0.0)
                    * math.sqrt(delta_p / nominal_delta_p))
            total_flow = active_area * abs(vd) + leak
            linear_v = direction * total_flow / (p["K_flow"] * math.sqrt(delta_p))
            # 步骤 5：式 6.3。u_lin 的符号决定死区逆补偿的方向。
            if abs(linear_v) > 1e-6:
                u_ff = linear_v + math.copysign(p["deadzone"], linear_v)
        # 步骤 6：与参考代码相同的 PID 残差反馈及总限幅。
        u_fb = float(self.pid.update(target=xd, feedback=y_meas[0]))
        u_total = min(10.0, max(-10.0, u_ff + u_fb))
        self.last = {
            "direction": float(direction), "F_drive_n": force,
            "P_req_est_pa": theoretical_pressure, "Q_leak_m3_s": leak,
            "Q_total_m3_s": total_flow, "delta_P_pa": delta_p,
            "u_lin_v": linear_v, "u_ff_v": u_ff, "u_fb_v": u_fb,
            "u_total_v": u_total,
        }
        return u_total, u_ff, u_fb
