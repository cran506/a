"""第八章公开紧凑缸灰盒四状态模型；全部机理计算采用 SI 单位。

本配置缩小缸面积以使 5000 N 成为重载故障，不是此前 .01 m² 缸的原参数。
采用连续 tanh-Stribeck 滑动近似；它不声称包含静摩擦的集合值粘住机制。
积分状态从不裁剪。C++ 只加速同一 RHS，Python RHS 供独立 ODE 重放。
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np


ROOT = Path(__file__).resolve().parent
_PARAMETERS = ("A1", "A2", "L", "V0_1", "V0_2", "m", "Ps", "Pr",
               "ambient_pressure", "beta_e", "F_c", "F_s", "v_s", "sigma",
               "gamma", "deadzone_v", "K_flow", "C_ip", "air_fraction_at_pr",
               "orifice_regularization_pa", "makeup_conductance_m3_s_pa",
               "relief_pressure_pa", "relief_conductance_m3_s_pa",
               "integration_stiffness_factor")
_LIBRARY = None


def _native_library():
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    source = ROOT / "native_twin.cpp"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / ("hydraulic_chapter8_" + digest)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    library = directory / "native_twin.so"
    if not library.exists():
        compiler = shutil.which("g++")
        if compiler is None:
            raise RuntimeError("需要 g++ 编译真实液压积分器；未提供伪造或代数替代模型")
        temporary = directory / f"native_twin.{os.getpid()}.so"
        result = subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                                 str(source), "-o", str(temporary)],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError("液压积分器编译失败：" + result.stderr[-3000:])
        temporary.replace(library)
    lib = ctypes.CDLL(str(library))
    lib.hydraulic_abi_version.restype = ctypes.c_int
    if lib.hydraulic_abi_version() != 1:
        raise RuntimeError("液压积分器 ABI 不兼容")
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    lib.hydraulic_step.argtypes = [pointer, pointer, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, pointer]
    lib.hydraulic_step.restype = ctypes.c_int
    _LIBRARY = lib
    return lib


def default_config():
    return json.loads((ROOT / "physical_config.json").read_text(encoding="utf-8"))


class HydraulicTwinCore:
    """真实液压状态 [x(m),v(m/s),P1(Pa),P2(Pa)]，瞬时阀的四阶模型。

    step 的压力/位置极值来自 RK 接受子步节点；它们不是连续时间证明。
    native 积分不修改输入状态，成功后才原子更新 self.state 和时钟。
    """
    state_units = ("m", "m/s", "Pa", "Pa")

    def __init__(self, m=None, beta_e=None, F_c=None, config=None):
        """Parameter precedence: public JSON < config mapping < explicit args.

        None means inherit the declared configuration; an omitted argument
        must not silently replace a caller's configured physical parameter.
        """
        self.p = default_config()
        if config is not None:
            self.p.update(dict(config))
        for name, value in (("m", m), ("beta_e", beta_e), ("F_c", F_c)):
            if value is not None:
                self.p[name] = float(value)
        self.p["F_s"] = self.p["static_friction_ratio"] * self.p["F_c"]
        self._validate_parameters()
        self._refresh_native_parameters()
        self._native = _native_library()
        self.t = 0.
        self.reset()

    @property
    def parameters(self):
        return self.p.copy()

    def _validate_parameters(self):
        for name in _PARAMETERS:
            if not math.isfinite(float(self.p[name])):
                raise ValueError(f"非有限物理参数：{name}")
        if min(self.p[k] for k in ("A1", "A2", "L", "V0_1", "V0_2", "m", "Ps", "Pr",
                                   "beta_e", "v_s", "K_flow", "orifice_regularization_pa")) <= 0:
            raise ValueError("几何、质量、压力和体积模量必须为正")
        if not 0 <= self.p["air_fraction_at_pr"] < 1 or self.p["Ps"] <= self.p["Pr"]:
            raise ValueError("混气比例或供回油压力无效")
        if min(self.p[k] for k in ("F_c", "F_s", "sigma", "gamma", "deadzone_v", "C_ip",
                                   "makeup_conductance_m3_s_pa", "relief_conductance_m3_s_pa")) < 0:
            raise ValueError("摩擦、死区或流导不得为负")
        if not 0 < self.p["integration_stiffness_factor"] <= 1:
            raise ValueError("刚性步长系数应在 (0,1] 内")

    def _refresh_native_parameters(self):
        self._native_parameters = np.array([self.p[name] for name in _PARAMETERS], dtype=np.float64)

    def update_physics(self, values):
        allowed = {"m", "mass", "beta_e", "F_c", "Fc"}
        if set(values) - allowed:
            raise ValueError("update_physics 仅接受质量、体积模量和库伦摩擦参数")
        old = self.p.copy()
        for key, value in values.items():
            self.p[{"mass": "m", "Fc": "F_c"}.get(key, key)] = float(value)
        self.p["F_s"] = self.p["static_friction_ratio"] * self.p["F_c"]
        try:
            self._validate_parameters()
        except Exception:
            self.p = old
            raise
        self._refresh_native_parameters()

    def friction(self, velocity):
        v = float(velocity)
        p = self.p
        return ((p["F_c"] + (p["F_s"]-p["F_c"])*math.exp(-(v/p["v_s"])**2)) *
                math.tanh(p["gamma"]*v) + p["sigma"]*v)

    def effective_bulk_modulus(self, pressure, beta_multiplier=1.):
        p = self.p
        pe = max(float(pressure), 1.)
        return 1. / ((1.-p["air_fraction_at_pr"])/(p["beta_e"]*beta_multiplier)
                     + p["air_fraction_at_pr"]*p["Pr"]/(pe*pe))

    def _phi(self, pressure_difference):
        d = float(pressure_difference)
        eps = self.p["orifice_regularization_pa"]
        return d / (d*d+eps*eps)**.25

    def rhs(self, state, u, load_n=0., beta_multiplier=1.):
        y = np.asarray(state, dtype=float)
        if y.shape != (4,) or not np.all(np.isfinite(y)):
            raise ValueError("state 必须是四维有限 SI 状态")
        u, load_n, bm = float(u), float(load_n), float(beta_multiplier)
        if not all(map(math.isfinite, (u, load_n, bm))) or bm <= 0:
            raise ValueError("输入、负载或体积模量倍率无效")
        x, v, p1, p2 = y
        p = self.p
        vol1, vol2 = p["V0_1"]+p["A1"]*x, p["V0_2"]+p["A2"]*(p["L"]-x)
        if min(vol1, vol2) <= 0:
            raise ValueError("物理液压腔容积非正")
        ue = math.copysign(max(abs(u)-p["deadzone_v"], 0.), u)
        q1 = q2 = 0.
        if ue > 0:
            q1 = p["K_flow"]*ue*self._phi(p["Ps"]-p1)
            q2 = -p["K_flow"]*ue*self._phi(p2-p["Pr"])
        elif ue < 0:
            q1 = p["K_flow"]*ue*self._phi(p1-p["Pr"])
            q2 = -p["K_flow"]*ue*self._phi(p["Ps"]-p2)
        q1 += (p["makeup_conductance_m3_s_pa"]*max(p["Pr"]-p1, 0.) -
               p["relief_conductance_m3_s_pa"]*max(p1-p["relief_pressure_pa"], 0.))
        q2 += (p["makeup_conductance_m3_s_pa"]*max(p["Pr"]-p2, 0.) -
               p["relief_conductance_m3_s_pa"]*max(p2-p["relief_pressure_pa"], 0.))
        leak = p["C_ip"]*(p1-p2)
        a = (p["A1"]*p1-p["A2"]*p2-p["ambient_pressure"]*(p["A1"]-p["A2"])
             -self.friction(v)-load_n)/p["m"]
        return np.array([v, a, self.effective_bulk_modulus(p1, bm)/vol1*(q1-p["A1"]*v-leak),
                         self.effective_bulk_modulus(p2, bm)/vol2*(q2+p["A2"]*v+leak)])

    def equilibrium_for_velocity(self, velocity=.05, load_n=0., position=None):
        """同一孔口/泄漏/摩擦方程的数值稳态，不用教材目标电压回填。"""
        from scipy.optimize import brentq
        p = self.p
        v = float(velocity)
        x = p["initial_position_m"] if position is None else float(position)
        force = self.friction(v)+float(load_n)+p["ambient_pressure"]*(p["A1"]-p["A2"])
        if abs(v) < 1e-14 and abs(float(load_n)) < 1e-14:
            # With no external load, both chambers at ambient pressure are
            # force-balanced and have no inter-chamber leakage.
            return np.array([x, 0., p["ambient_pressure"], p["ambient_pressure"]]), 0.
        # At zero speed a nonzero load still needs leakage-replenishing flow.
        # Select the port branch by the load direction, rather than pretending
        # that an unequal-pressure, internally leaking cylinder is static at u=0.
        extension_ports = v > 0.0 or (abs(v) < 1e-14 and float(load_n) > 0.0)

        def pressure1(p2):
            return (force+p["A2"]*p2)/p["A1"]

        def voltage_pair(p2):
            p1 = pressure1(p2)
            leak = p["C_ip"]*(p1-p2)
            if extension_ports:
                return ((p["A1"]*v+leak)/(p["K_flow"]*self._phi(p["Ps"]-p1)),
                        (p["A2"]*v+leak)/(p["K_flow"]*self._phi(p2-p["Pr"])))
            return ((p["A1"]*v+leak)/(p["K_flow"]*self._phi(p1-p["Pr"])),
                    (p["A2"]*v+leak)/(p["K_flow"]*self._phi(p["Ps"]-p2)))

        lo = max(p["Pr"]+1., (p["A1"]*(p["Pr"]+1.)-force)/p["A2"])
        hi = min(p["Ps"]-1., (p["A1"]*(p["Ps"]-1.)-force)/p["A2"])
        if lo >= hi:
            raise ValueError("指定负载/速度不存在供回油区间内的稳态")
        def balance(p2):
            first, second = voltage_pair(p2)
            return first-second
        p2 = brentq(balance, lo, hi, xtol=1e-6, rtol=1e-13)
        ueff = sum(voltage_pair(p2))*.5
        u = ueff+math.copysign(p["deadzone_v"], ueff)
        if abs(u) > 10:
            raise ValueError("指定稳态要求超出 ±10 V 执行器能力")
        return np.array([x, v, pressure1(p2), p2]), float(u)

    def reset(self, x0=None, velocity=None, position=None):
        velocity = self.p["initial_velocity_m_s"] if velocity is None else float(velocity)
        position = self.p["initial_position_m"] if position is None else float(position)
        y = (self.equilibrium_for_velocity(velocity, position=position)[0]
             if x0 is None else np.asarray(x0, dtype=float).copy())
        if y.shape != (4,) or not np.all(np.isfinite(y)) or min(y[2:]) <= 0:
            raise ValueError("初态必须为四维有限 SI 向量且绝对压力为正")
        if min(self.p["V0_1"]+self.p["A1"]*y[0], self.p["V0_2"]+self.p["A2"]*(self.p["L"]-y[0])) <= 0:
            raise ValueError("初态腔室容积无效")
        self.state = np.ascontiguousarray(y, dtype=np.float64)
        self.t = 0.
        return self.state.copy()

    def step(self, u, dt=.001, load_n=0., beta_multiplier=1., substeps=8):
        u, dt, load_n, bm = map(float, (u, dt, load_n, beta_multiplier))
        if not all(map(math.isfinite, (u, dt, load_n, bm))) or dt <= 0 or bm <= 0:
            raise ValueError("非有限输入或无效周期/体积模量倍率")
        if abs(u) > 10.+1e-12:
            raise ValueError("物理驱动电压必须在 ±10 V 内，限幅由控制器/环境显式执行")
        count = max(int(substeps), int(self.p["integration_min_substeps"]))
        trial = self.state.copy()
        diagnostic = np.empty(7, dtype=np.float64)
        status = self._native.hydraulic_step(trial, self._native_parameters,
            len(self._native_parameters), u, load_n, bm, dt, count,
            int(self.p["integration_max_internal_steps"]), diagnostic)
        if status:
            raise RuntimeError(f"真实液压积分失败 status={status}, t={self.t:.9f}s；未夹紧状态")
        self.state = trial
        self.t += dt
        info = dict(acceleration_m_s2=float(diagnostic[0]),
                    interval_pmin_pa=float(diagnostic[1]), interval_pmax_pa=float(diagnostic[2]),
                    interval_xmin_m=float(diagnostic[3]), interval_xmax_m=float(diagnostic[4]),
                    integration_substeps=int(diagnostic[5]), integration_rejections=int(diagnostic[6]),
                    extrema_scope="accepted_RK_substep_nodes", load_n=load_n,
                    beta_multiplier=bm, time_s=self.t)
        return self.state.copy(), info
