"""旧版Wiener顺序对象，仅用于回归对照；主实验用input_deadzone_model.py。

这不是双腔压力数字孪生。物理模型为
    tau_valve * dz/dt = u - z
    tau_velocity * dv/dt = gain * deadzone(z) - v
u、z 的单位为 V，v 为 m/s。控制器每个采样区间零阶保持 u。
阀位单调趋近该区间指令，至多穿越两次死区边界；本模块按精确穿越
时间分段并解析积分速度，不用 Euler、输出裁剪或添加振荡信号。
"""
import math
import numpy as np


def deadzone(value, threshold=.5):
    """对标量或数组计算 sign(value)*max(abs(value)-threshold,0)。"""
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('死区阈值必须非负且有限')
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError('阀信号必须有限')
    result = np.sign(a)*np.maximum(np.abs(a)-threshold, 0.)
    return float(result) if result.ndim == 0 else result


class DeadzoneVelocityPlant:
    """±deadzone_v 硬死区的一阶阀与一阶等效机械速度对象。

    tau_valve_s=0 表示理想瞬时阀；tau_velocity_s 必须严格为正。
    step 不替控制器施加额外电压限幅，真实死区始终作用于阀位。
    """
    def __init__(self, gain_m_s_v=.01, tau_valve_s=.0008,
                 tau_velocity_s=.012, deadzone_v=.5):
        values = (gain_m_s_v, tau_valve_s, tau_velocity_s, deadzone_v)
        if (not all(np.isfinite(value) for value in values)
                or gain_m_s_v <= 0 or tau_valve_s < 0
                or tau_velocity_s <= 0 or deadzone_v < 0):
            raise ValueError('速度增益及机械时间常数必须正，阀时间常数与死区须非负')
        self.gain_m_s_v = float(gain_m_s_v)
        self.tau_valve_s = float(tau_valve_s)
        self.tau_velocity_s = float(tau_velocity_s)
        self.deadzone_v = float(deadzone_v)
        self.reset()

    def reset(self, valve_voltage_v=0., velocity_m_s=0.):
        if not all(np.isfinite(value) for value in (valve_voltage_v, velocity_m_s)):
            raise ValueError('初始阀位和速度必须有限')
        self.valve_voltage_v = float(valve_voltage_v)
        self.velocity_m_s = float(velocity_m_s)
        self.time_s = 0.
        return self.velocity_m_s

    def _active_segment(self, voltage, duration, direction):
        """在固定死区分支上精确积分；direction=0 时速度自由衰减。"""
        tv, tm = self.tau_valve_s, self.tau_velocity_s
        valve0, velocity0 = self.valve_voltage_v, self.velocity_m_s
        em, ev = math.exp(-duration/tm), math.exp(-duration/tv)
        if direction:
            alpha = 1/tm-1/tv
            ah = alpha*duration
            if alpha == 0.:
                convolution = duration/tm*em
            elif abs(ah) < 1e-5:
                convolution = duration/tm*em*math.expm1(ah)/ah
            else:
                convolution = (ev-em)/(tm*alpha)
            target = voltage-direction*self.deadzone_v
            self.velocity_m_s = (velocity0*em+self.gain_m_s_v*(
                target*(-math.expm1(-duration/tm))+(valve0-voltage)*convolution))
        else:
            self.velocity_m_s = velocity0*em
        self.valve_voltage_v = valve0+(voltage-valve0)*(-math.expm1(-duration/tv))

    def step(self, voltage_v, dt):
        """ZOH 指令下精确积分 dt 秒，返回真实速度 m/s。"""
        if not np.isfinite(voltage_v) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('电压必须有限，采样间隔必须正且有限')
        voltage, dt = float(voltage_v), float(dt)
        if self.tau_valve_s == 0.:
            self.valve_voltage_v = voltage
            decay = math.exp(-dt/self.tau_velocity_s)
            self.velocity_m_s = (self.velocity_m_s*decay+self.gain_m_s_v*
                                 deadzone(voltage, self.deadzone_v)*(1-decay))
            self.time_s += dt
            return self.velocity_m_s
        valve0 = self.valve_voltage_v
        cuts = [0., dt]
        if valve0 != voltage:
            for boundary in (-self.deadzone_v, self.deadzone_v):
                ratio = (boundary-voltage)/(valve0-voltage)
                if 0. < ratio < 1.:
                    crossing = -self.tau_valve_s*math.log(ratio)
                    if 0. < crossing < dt:
                        cuts.append(crossing)
        cuts = sorted(set(cuts))
        for start, end in zip(cuts[:-1], cuts[1:]):
            duration = end-start
            midpoint = voltage+(self.valve_voltage_v-voltage)*math.exp(
                -.5*duration/self.tau_valve_s)
            direction = (1 if midpoint > self.deadzone_v else
                         -1 if midpoint < -self.deadzone_v else 0)
            self._active_segment(voltage, duration, direction)
        self.time_s += dt
        return self.velocity_m_s
