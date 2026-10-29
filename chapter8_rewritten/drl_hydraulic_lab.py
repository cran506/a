#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《智能流体传动与控制》第八章课堂编程实践（重写版）。

运行 python drl_hydraulic_lab.py 可复现实验；--mode train 从零训练。
8.1 五维观测/奖励；8.2 SAC；8.3 最小干预QP；8.4 域随机化；
8.5 独立故障对比；8.6 幅值剪枝、仿射量化及整数运行时。
主图(a)依用户选择复现PPT教学示意并明确标注，真实日志另存。
已附的策略来自此前选定的百万步机理引导训练，来源与离线精修均明确记录。
"""
from __future__ import annotations


"""第八章重写版：自包含四状态液压数字孪生，内部全部采用 SI 单位。

保留已验证模型的受力/连续性/孔口方程与公开参数，以便比较同一控制
实验。原来分散的配置文件和 C++ RHS 已随单文件内嵌；g++ 是可选加速，
没有编译器时自动使用 SciPy Radau，而不是换成另一组教学差分方程。

本配置缩小缸面积以使 5000 N 成为重载故障，不是此前 .01 m² 缸的原参数。
采用连续 tanh-Stribeck 滑动近似；它不声称包含静摩擦的集合值粘住机制。
积分状态从不裁剪。C++ 只加速同一 RHS，Python RHS 供独立 ODE 重放。
"""

import ctypes
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np


DEFAULT_PHYSICS = {'model_name': '第八章四状态紧凑缸教学对象：两腔死容积分别0.1/0.5mL',
 'state_units': ['m', 'm/s', 'Pa', 'Pa'],
 'A1': 0.00038,
 'A2': 0.00021,
 'L': 0.25,
 'V0_1': 1e-07,
 'V0_2': 5e-07,
 'm': 250.0,
 'Ps': 21000000.0,
 'Pr': 100000.0,
 'ambient_pressure': 100000.0,
 'beta_e': 1000000000.0,
 'F_c': 1200.0,
 'static_friction_ratio': 1.5,
 'v_s': 0.005,
 'sigma': 3000.0,
 'gamma': 500.0,
 'deadzone_v': 0.5,
 'K_flow': 1.64e-09,
 'C_ip': 1e-14,
 'air_fraction_at_pr': 0.001,
 'orifice_regularization_pa': 1000.0,
 'makeup_conductance_m3_s_pa': 5e-10,
 'relief_pressure_pa': 25000000.0,
 'relief_conductance_m3_s_pa': 1e-11,
 'initial_position_m': 0.0001,
 'initial_velocity_m_s': 0.05,
 'integration_min_substeps': 8,
 'integration_stiffness_factor': 0.8,
 'integration_max_internal_steps': 200000,
 'friction_model': '连续tanh-Stribeck滑动近似；未包含事件驱动静摩擦粘住分支',
 'flow_model': '有向压差孔口，1kPa可导正则化；低压补油、25MPa物理泄压，无积分状态裁剪',
 'bulk_modulus_model': '1/beta_eff=(1-alpha)/beta_oil+alpha*Pr/P^2；alpha为Pr处微量游离气体体积分数'}

_NATIVE_SOURCE = r'''
// Chapter 8 compact hydraulic classroom twin. SI state [x,v,P1,P2].
// RK4 with at least the requested substeps and local stiffness subdivision.
// State values are never clipped; failed trials are rejected and retried.
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {
constexpr int NP = 24;
// A1,A2,L,V01,V02,m,Ps,Pr,Patm,beta,Fc,Fs,vs,sigma,gamma,dz,Kq,Cip,
// alpha,dp_eps,makeup,relief_pressure,relief_conductance,stiffness_factor
inline double phi(double d, double eps) {
    return d / std::pow(d*d+eps*eps, .25);
}
inline double dphi(double d, double eps) {
    return (.5*d*d+eps*eps) / std::pow(d*d+eps*eps, 1.25);
}
inline double bulk(double pressure, const double* p, double bm) {
    // This floor only regularizes evaluation outside the physical domain in
    // rejected RK stages. Accepted pressure states must be strictly positive.
    double pe = std::max(pressure, 1.0);
    return 1.0 / ((1.0-p[18])/(p[9]*bm) + p[18]*p[7]/(pe*pe));
}
inline bool valid(const double* y, const double* p) {
    for(int j=0;j<4;++j) if(!std::isfinite(y[j])) return false;
    return y[2] > 0.0 && y[3] > 0.0 &&
           p[3]+p[0]*y[0] > 0.0 && p[4]+p[1]*(p[2]-y[0]) > 0.0;
}
inline double friction(double v, const double* p) {
    return (p[10]+(p[11]-p[10])*std::exp(-std::pow(v/p[12],2))) *
           std::tanh(p[14]*v) + p[13]*v;
}
inline double friction_bound(double v, const double* p) {
    double z = v/p[12], tv=std::tanh(p[14]*v);
    double mag = p[10]+(p[11]-p[10])*std::exp(-z*z);
    double dm = -2.0*(p[11]-p[10])*v/(p[12]*p[12])*std::exp(-z*z);
    return std::abs(dm*tv + mag*p[14]*(1.0-tv*tv)+p[13])/p[5];
}
inline void derivatives(const double* y, double u, double load, double bm,
                        const double* p, double* out) {
    double x=y[0],v=y[1],p1=y[2],p2=y[3];
    double ue=std::copysign(std::max(std::abs(u)-p[15],0.0),u);
    double q1=0.0,q2=0.0;
    if(ue>0.0) {
        q1=p[16]*ue*phi(p[6]-p1,p[19]);
        q2=-p[16]*ue*phi(p2-p[7],p[19]);
    } else if(ue<0.0) {
        q1=p[16]*ue*phi(p1-p[7],p[19]);
        q2=-p[16]*ue*phi(p[6]-p2,p[19]);
    }
    q1+=p[20]*std::max(p[7]-p1,0.0)-p[22]*std::max(p1-p[21],0.0);
    q2+=p[20]*std::max(p[7]-p2,0.0)-p[22]*std::max(p2-p[21],0.0);
    double leak=p[17]*(p1-p2);
    out[0]=v;
    out[1]=(p[0]*p1-p[1]*p2-p[8]*(p[0]-p[1])-friction(v,p)-load)/p[5];
    out[2]=bulk(p1,p,bm)/(p[3]+p[0]*x)*(q1-p[0]*v-leak);
    out[3]=bulk(p2,p,bm)/(p[4]+p[1]*(p[2]-x))*(q2+p[1]*v+leak);
}
inline double stable_h(const double* y,double u,double bm,const double* p) {
    double p1=y[2],p2=y[3], ue=std::max(std::abs(u)-p[15],0.0);
    double d1=0.0,d2=0.0;
    if(u>p[15]) {
        d1=p[16]*ue*dphi(p[6]-p1,p[19]);
        d2=p[16]*ue*dphi(p2-p[7],p[19]);
    } else if(u < -p[15]) {
        d1=p[16]*ue*dphi(p1-p[7],p[19]);
        d2=p[16]*ue*dphi(p[6]-p2,p[19]);
    }
    if(p1<p[7]) d1+=p[20]; if(p2<p[7]) d2+=p[20];
    if(p1>p[21]) d1+=p[22]; if(p2>p[21]) d2+=p[22];
    double vol1=p[3]+p[0]*y[0], vol2=p[4]+p[1]*(p[2]-y[0]);
    double b1=bulk(p1,p,bm),b2=bulk(p2,p,bm);
    double rate=std::max(b1/vol1*(d1+2*p[17]),b2/vol2*(d2+2*p[17]));
    rate=std::max(rate,friction_bound(y[1],p));
    rate=std::max(rate,std::sqrt((b1*p[0]*p[0]/vol1+b2*p[1]*p[1]/vol2)/p[5]));
    return p[23]/std::max(rate,1.0);
}
}
extern "C" int hydraulic_abi_version() { return 1; }
extern "C" int hydraulic_step(double* y,const double* p,int np,double u,
  double load,double bm,double dt,int substeps,int max_steps,double* diagnostics) {
    if(np!=NP || !valid(y,p) || !std::isfinite(u) || !std::isfinite(load) ||
       !std::isfinite(bm) || bm<=0 || dt<=0 || substeps<1) return -1;
    double t=0.0,maxh=dt/substeps;
    double minp=std::min(y[2],y[3]),maxp=std::max(y[2],y[3]);
    double minx=y[0],maxx=y[0];
    int steps=0,rejections=0;
    while(t < dt) {
        if(++steps>max_steps) return -2;
        double h=std::min({maxh,dt-t,stable_h(y,u,bm,p)});
        bool accepted=false;
        double k1[4],k2[4],k3[4],k4[4],a[4],b[4],c[4],next[4];
        for(int attempt=0;attempt<30;++attempt) {
            derivatives(y,u,load,bm,p,k1);
            for(int j=0;j<4;++j) a[j]=y[j]+.5*h*k1[j];
            if(valid(a,p)) {
                derivatives(a,u,load,bm,p,k2);
                for(int j=0;j<4;++j) b[j]=y[j]+.5*h*k2[j];
                if(valid(b,p)) {
                    derivatives(b,u,load,bm,p,k3);
                    for(int j=0;j<4;++j) c[j]=y[j]+h*k3[j];
                    if(valid(c,p)) {
                        derivatives(c,u,load,bm,p,k4);
                        for(int j=0;j<4;++j) next[j]=y[j]+h*(k1[j]+2*k2[j]+2*k3[j]+k4[j])/6;
                        if(valid(next,p)) { accepted=true; break; }
                    }
                }
            }
            h*=.5; ++rejections;
        }
        if(!accepted || h<1e-14) return -3;
        for(int j=0;j<4;++j) y[j]=next[j];
        // Diagnostics are accepted RK substep-node extrema, not a formal dense
        // continuous-time extremum certificate. High-fidelity replay is separate.
        minp=std::min({minp,y[2],y[3]}); maxp=std::max({maxp,y[2],y[3]});
        minx=std::min(minx,y[0]); maxx=std::max(maxx,y[0]);
        t+=h;
    }
    double f[4]; derivatives(y,u,load,bm,p,f);
    diagnostics[0]=f[1]; diagnostics[1]=minp;diagnostics[2]=maxp;
    diagnostics[3]=minx;diagnostics[4]=maxx;diagnostics[5]=steps;diagnostics[6]=rejections;
    return 0;
}

'''

_PARAMETERS = ("A1", "A2", "L", "V0_1", "V0_2", "m", "Ps", "Pr",
               "ambient_pressure", "beta_e", "F_c", "F_s", "v_s", "sigma",
               "gamma", "deadzone_v", "K_flow", "C_ip", "air_fraction_at_pr",
               "orifice_regularization_pa", "makeup_conductance_m3_s_pa",
               "relief_pressure_pa", "relief_conductance_m3_s_pa",
               "integration_stiffness_factor")
_LIBRARY = None


def _native_library():
    """可选加速器：源码随本 Python 文件携带；无编译器时使用同一 RHS 的 Radau。"""
    global _LIBRARY
    if _LIBRARY is not None:
        return _LIBRARY
    compiler = shutil.which("g++")
    if compiler is None or os.name != "posix":
        return None
    digest = hashlib.sha256(_NATIVE_SOURCE.encode("utf-8")).hexdigest()[:20]
    directory = Path(tempfile.gettempdir()) / ("hydraulic_ch8_rewritten_" + digest)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    library = directory / "native_twin.so"
    try:
        if not library.exists():
            source = directory / f"native_twin.{os.getpid()}.cpp"
            temporary = directory / f"native_twin.{os.getpid()}.so"
            source.write_text(_NATIVE_SOURCE, encoding="utf-8")
            try:
                result = subprocess.run([compiler, "-O3", "-std=c++17", "-shared", "-fPIC",
                                         str(source), "-o", str(temporary)],
                                        capture_output=True, text=True, check=False, timeout=60)
                if result.returncode:
                    raise RuntimeError(result.stderr[-1000:])
                temporary.replace(library)
            finally:
                source.unlink(missing_ok=True)
                temporary.unlink(missing_ok=True)
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
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        import warnings
        warnings.warn(f"可选 native 加速不可用，改用同一物理方程的 SciPy Radau: {exc}", RuntimeWarning)
        return None


def default_config():
    """返回独立副本，避免域随机化修改公开标称参数。"""
    return copy.deepcopy(DEFAULT_PHYSICS)


class HydraulicTwinCore:
    """真实液压状态 [x(m),v(m/s),P1(Pa),P2(Pa)]，瞬时阀的四阶模型。

    step 的压力/位置极值来自 RK 接受子步节点；它们不是连续时间证明。
    native 积分不修改输入状态，成功后才原子更新 self.state 和时钟。
    """
    state_units = ("m", "m/s", "Pa", "Pa")

    def __init__(self, m=None, beta_e=None, F_c=None, config=None, backend="auto"):
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
        if backend not in ("auto", "native", "radau"):
            raise ValueError("backend 应为 auto、native 或 radau")
        self._native = None if backend == "radau" else _native_library()
        if backend == "native" and self._native is None:
            raise RuntimeError("请求 native 后端，但本机没有可用的 g++ 加速器")
        self.backend = "native" if self._native is not None else "radau"
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
        for name in ("integration_min_substeps", "integration_max_internal_steps"):
            value = self.p[name]
            if not math.isfinite(float(value)) or int(value) != value or value < 1:
                raise ValueError(f"{name} 必须是正整数")

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
        if not math.isfinite(float(substeps)) or int(substeps) != substeps or substeps < 1:
            raise ValueError("substeps 必须是正整数")
        count = max(int(substeps), int(self.p["integration_min_substeps"]))
        if self._native is None:
            return self._step_radau(u, dt, load_n, bm, count)
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
                    backend="native", extrema_scope="accepted_RK_substep_nodes", load_n=load_n,
                    beta_multiplier=bm, time_s=self.t)
        return self.state.copy(), info


    def _step_radau(self, u, dt, load_n, beta_multiplier, substeps):
        """Portable stiff integration; failure is atomic and accepted states are never clipped.

        Native RK4 and Radau implement exactly the same ODE; numerical truncation
        errors can differ. Diagnostics cover accepted solver nodes, not a formal
        dense-time certificate.
        """
        from scipy.integrate import solve_ivp
        solution = solve_ivp(
            lambda _, state: self.rhs(state, u, load_n, beta_multiplier),
            (0., dt), self.state.copy(), method="Radau", rtol=1e-9,
            atol=np.array([1e-11, 1e-9, .01, .01]), max_step=dt/substeps)
        states = solution.y
        if (not solution.success or not np.isfinite(states).all()
                or np.any(states[2:] <= 0.)
                or np.any(self.p["V0_1"] + self.p["A1"]*states[0] <= 0.)
                or np.any(self.p["V0_2"] + self.p["A2"]*(self.p["L"]-states[0]) <= 0.)):
            raise RuntimeError(f"Radau 物理积分失败，状态和时钟未提交: {solution.message}")
        new_state = np.ascontiguousarray(states[:, -1], dtype=np.float64)
        acceleration = float(self.rhs(new_state, u, load_n, beta_multiplier)[1])
        self.state = new_state
        self.t += dt
        return self.state.copy(), dict(
            acceleration_m_s2=acceleration,
            interval_pmin_pa=float(states[2:].min()), interval_pmax_pa=float(states[2:].max()),
            interval_xmin_m=float(states[0].min()), interval_xmax_m=float(states[0].max()),
            integration_substeps=int(len(solution.t)-1), integration_rejections=None,
            backend="radau", extrema_scope="accepted_Radau_nodes",
            load_n=load_n, beta_multiplier=beta_multiplier, time_s=self.t)


# =============================================================================
# 模块二：Gymnasium 五维观测、固定奖励与可复现域随机化（8.1、8.4、8.5）
# =============================================================================
import gymnasium as gym
from gymnasium import spaces


class HydraulicGymEnv(gym.Env):
    """公开参考接口，内部状态为 [x,v,P1,P2]，全部 ODE 使用 SI 单位。

    网络特征为 [x-r, v-v_ref, a-a_ref, P1-A2/A1*P2, Ps]，前3项
    分别转换为 mm、mm/s、mm/s²，压力为 MPa。五维特征是课堂规定的
    压缩观测；未知参数及上一动作没有包含在内，不声称严格全可观测 MDP。
    加速度为当前负载下理想无噪传感值，网络不接收负载标签或未来信息。
    """
    metadata = {"render_modes": []}
    observation_names = ["position_error_mm", "velocity_error_mm_s",
                         "acceleration_error_mm_s2", "load_pressure_mpa", "supply_pressure_mpa"]
    obs_mean = np.array([0., 0., 0., 10., 21.], dtype=np.float32)
    obs_std = np.array([10., 50., 500., 8., 2.], dtype=np.float32)

    def __init__(self, twin_config=None, domain_randomization=True,
                 fault_augmentation=True, max_steps=2000, backend="auto"):
        super().__init__()
        self.twin = HydraulicTwinCore(config=twin_config, backend=backend)
        self.base_physics = {k: self.twin.p[k] for k in ("m", "beta_e", "F_c")}
        self.default_m = self.base_physics["m"]
        self.default_beta = self.base_physics["beta_e"]
        self.default_Fc = self.base_physics["F_c"]
        self.domain_randomization = bool(domain_randomization)
        self.fault_augmentation = bool(fault_augmentation)
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps 必须为正整数")
        self.dt = .001
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        self.step_count = 0
        self.last_u = 0.
        self._done = False

    @property
    def time(self):
        return self.step_count * self.dt

    def normalize_obs(self, obs):
        value = np.asarray(obs, dtype=np.float64)
        if value.shape != (5,) or not np.isfinite(value).all():
            raise ValueError("观测必须为五维有限值")
        return ((value - self.obs_mean) / (self.obs_std + np.float32(1e-8))).astype(np.float32)

    def current_fault(self):
        return ((self.load_after, self.beta_after) if self.time >= self.fault_time - 1e-12
                else (0., 1.))

    def raw_observation(self):
        x, v, p1, p2 = self.twin.state
        load, beta = self.current_fault()
        acceleration = self.twin.rhs(self.twin.state, self.last_u, load, beta)[1]
        reference = self.initial_x + self.v_ref * self.time
        p = self.twin.p
        return np.array([(x-reference)*1000., (v-self.v_ref)*1000., acceleration*1000.,
                         (p1-p['A2']/p['A1']*p2)/1e6, p['Ps']/1e6])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = dict(options or {})
        rng = self.np_random
        dr = self.domain_randomization and not options.get("evaluation", False)
        parameters = ({"m": self.default_m*rng.uniform(.8, 1.2),
                       "beta_e": self.default_beta*rng.uniform(.6, 1.4),
                       "F_c": self.default_Fc*rng.uniform(.7, 1.3)}
                      if dr else self.base_physics.copy())
        self.twin.update_physics(parameters)
        self.v_ref = float(options.get("velocity", .05))
        self.initial_x = float(options.get("position", rng.uniform(.0001, .005)
                                          if dr else self.twin.p['initial_position_m']))
        state, equilibrium_u = self.twin.equilibrium_for_velocity(self.v_ref, position=self.initial_x)
        self.twin.reset(x0=state)
        self.step_count, self.last_u, self._done = 0, 0., False
        augmented = self.fault_augmentation and rng.random() < .8
        self.fault_time = float(options.get("fault_time", rng.uniform(.05, .8) if augmented else np.inf))
        self.load_after = float(options.get("load_n", rng.uniform(0., 5500.) if augmented else 0.))
        self.beta_after = float(options.get("beta_multiplier", rng.uniform(.5, 1.) if augmented else 1.))
        if (np.isnan(self.fault_time) or self.fault_time < 0 or not np.isfinite(self.load_after)
                or not np.isfinite(self.beta_after) or self.beta_after <= 0):
            raise ValueError("故障参数必须有限且体积模量倍率为正")
        self._last_obs = self.normalize_obs(self.raw_observation())
        return self._last_obs.copy(), dict(**parameters, fault_time=self.fault_time,
            load_n=self.load_after, beta_multiplier=self.beta_after,
            initial_voltage_v=equilibrium_u, initial_position_m=self.initial_x,
            reference_velocity_m_s=self.v_ref)

    @staticmethod
    def reward_terms(error_mm, u_volts, previous_u):
        """式8.1，原始奖励不做缩放；仅训练优化器可另行乘比例。"""
        return {"reward_error": -100.*error_mm**2,
                "reward_energy": -.1*u_volts**2,
                "reward_smooth": -(u_volts-previous_u)**2,
                "reward_bonus": 2. if abs(error_mm) < .2 else 0.}

    def step(self, action_norm):
        if self._done:
            raise RuntimeError("Episode 已结束，请先 reset()")
        action = np.asarray(action_norm, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("动作必须是形状(1,)的有限归一化电压")
        voltage = float(np.clip(action[0], -1., 1.)*10.)
        load, beta = self.current_fault()
        previous_u = self.last_u
        failure, diagnostics = None, {}
        try:
            state, diagnostics = self.twin.step(voltage, dt=self.dt, load_n=load, beta_multiplier=beta)
        except (RuntimeError, ValueError) as error:
            failure = "integration_failure: " + str(error)
            state = self.twin.state.copy()
        self.last_u = voltage
        self.step_count += 1
        try:
            raw = self.raw_observation()
        except (ValueError, RuntimeError):
            raw = np.full(5, np.nan)
        if not np.isfinite(state).all() or not np.isfinite(raw).all():
            failure = failure or "nonfinite_physical_state"
        elif not 0. < state[0] < self.twin.p['L']:
            failure = failure or "mechanical_stroke_exit"
        elif min(state[2:]) <= 0. or max(state[2:]) > 40e6:
            failure = failure or "pressure_outside_training_envelope"
        elif abs(raw[0]) > 20. or abs(state[1]) > 1.:
            failure = failure or "tracking_outside_training_envelope"
        error_mm = float(raw[0]) if np.isfinite(raw[0]) else 20.
        terms = self.reward_terms(error_mm, voltage, previous_u)
        base_reward = float(sum(terms.values()))
        # 防止失败提早结束反而逃避未来代价；单独记录，不混称教材奖励。
        failure_penalty = -100000. if failure else 0.
        reward = base_reward + failure_penalty
        if np.isfinite(raw).all():
            self._last_obs = self.normalize_obs(raw)
        terminated = bool(failure)
        truncated = self.step_count >= self.max_steps
        self._done = terminated or truncated
        info = dict(diagnostics, **terms, base_reward=base_reward,
            failure_penalty=failure_penalty, raw_reward=reward, control_v=voltage,
            position_error_mm=error_mm, velocity_error_mms=float(raw[1]),
            physical_failure=failure, load_n=load, beta_multiplier=beta)
        return self._last_obs.copy(), float(reward), terminated, bool(truncated), info


"""Chapter 8 independent second-order HOCBF-QP classroom experiment.

The public filter accepts x in mm and velocity in mm/s to match the supplied
teaching interface. The propagated object and all internal calculations use
SI: x_dot=v, v_dot=k*u-c*v, with k in m/(s^2 V).

This is NOT the four-state hydraulic pressure ODE. That plant has relative
order three from valve voltage to position and requires another certificate.
A certificate on this explicit second-order object must not be transferred.
"""
from dataclasses import dataclass
import math
import numpy as np


class SafetyInfeasibleError(RuntimeError):
    """The constrained QP or its high-order initial set is infeasible."""


def _bisect(fn, left, right):
    """Root on an already certified monotone bracket; no SciPy dependency."""
    fl, fr = fn(left), fn(right)
    if fl == 0: return left
    if fr == 0: return right
    if fl * fr > 0: raise ValueError("root not bracketed")
    for _ in range(64):
        mid = .5 * (left + right)
        fm = fn(mid)
        if fm == 0: return mid
        if (fl < 0) == (fm < 0):
            left, fl = mid, fm
        else:
            right = mid
    return .5 * (left + right)


@dataclass(frozen=True)
class BarrierProjection:
    proposed_v: float
    applied_v: float
    lower_v: float
    upper_v: float
    barrier: float
    intervened: bool
    certificate_scope: str


class HOCBFSafetyFilter:
    """Scalar min-intervention QP: min .5*(u-u_rl)^2 over an interval.

    The default ``sampled=False`` reproduces the supplied continuous formula
    with alpha1=48 and alpha2=580. It has no intersample safety certificate.
    Those two numbers also cannot be the sum/product of positive real
    recursive HOCBF gains: their discriminant is -16.

    ``sampled=True`` requires explicit positive kappa1 and kappa2. Its effective
    coefficients are alpha1=kappa1+kappa2 and alpha2=kappa1*kappa2; the constraint
    is enforced over EVERY point in an exact zero-order-hold interval, and both
    h>=0 and psi1=h_dot+kappa1*h>=0 are required initially. Empty actuator/CBF
    intersections raise SafetyInfeasibleError, rather than returning a command
    falsely called safe. No state trajectory is clipped to a limit.
    """
    def __init__(self, x_min=0., x_max=100., u_min=-10., u_max=10.,
                 alpha1=48., alpha2=580., margin=1.2, dt=.001,
                 k_gain=1850., damp_c=14., sampled=False,
                 kappa1=None, kappa2=None):
        numbers=[x_min,x_max,u_min,u_max,alpha1,alpha2,margin,dt,k_gain,damp_c]
        if not all(math.isfinite(float(z)) for z in numbers):
            raise ValueError("All filter parameters must be finite")
        if not (x_min < x_max and u_min < u_max and 0 <= 2*margin < x_max-x_min):
            raise ValueError("Invalid physical limits or robustness margin")
        if min(alpha1,alpha2,dt,k_gain,damp_c) <= 0:
            raise ValueError("Gains, sample time and damping must be positive")
        self.x_min=float(x_min); self.x_max=float(x_max)
        self.u_min=float(u_min); self.u_max=float(u_max)
        self.margin=float(margin); self.dt=float(dt)
        self.k_gain=float(k_gain); self.damp_c=float(damp_c)
        self.sampled=bool(sampled)
        self.alpha1=float(alpha1); self.alpha2=float(alpha2)
        self.reference_alpha1=float(alpha1); self.reference_alpha2=float(alpha2)
        self.kappa1=kappa1; self.kappa2=kappa2
        if self.sampled:
            if kappa1 is None or kappa2 is None:
                raise ValueError("Sampled certificate requires explicit kappa1 and kappa2")
            if not all(math.isfinite(z) and z>0 for z in (kappa1,kappa2)):
                raise ValueError("Recursive HOCBF gains must be positive and finite")
            self.alpha1=float(kappa1+kappa2)
            self.alpha2=float(kappa1*kappa2)
        self.last_info=None

    @property
    def k_si(self):
        return self.k_gain / 1000.

    def compute_barrier_h(self, x_p):
        """The supplied dimensionless mechanical barrier (position in mm)."""
        return ((self.x_max-x_p)*(x_p-self.x_min))/(self.x_max-self.x_min)**2

    def reference_bounds(self, x_p, v_p_mms, k_gain=None, damp_c=None):
        """Exact supplied continuous-time formula, including alpha=(48,580)."""
        gain=self.k_gain if k_gain is None else float(k_gain)
        damp=self.damp_c if damp_c is None else float(damp_c)
        if not math.isfinite(gain) or gain<=0 or not math.isfinite(damp):
            raise ValueError("Invalid reference dynamics")
        hu=(self.x_max-self.margin)-float(x_p)
        hl=float(x_p)-(self.x_min+self.margin)
        v=float(v_p_mms)
        upper=((damp-self.reference_alpha1)*v+self.reference_alpha2*hu)/gain
        lower=((damp-self.reference_alpha1)*v-self.reference_alpha2*hl)/gain
        return max(self.u_min,lower),min(self.u_max,upper)

    def _held_interval_bound(self, gap_m, v_m_s, lower=False):
        """Exact extremum of a constant-command affine HOCBF bound on [0,dt].

        psi2 is C(t)-k*D(t)*u at the upper boundary and the opposite
        orientation at the lower boundary. The derivative of C/D has at most
        one turning point, so splitting there brackets every stationary ratio.
        """
        c=self.damp_c; k=self.k_si; a=self.alpha1; b=self.alpha2
        d0=a/c-b/c**2; d1=b/c; d2=1.-a/c+b/c**2
        c0=-b*(gap_m+v_m_s/c) if lower else b*(gap_m-v_m_s/c)
        c2=(c-a+b/c)*v_m_s
        e0=-c*c2*d0-c2*d1+c*c0*d2
        e1=-c*c2*d1
        def deriv_num(t):
            return math.exp(-c*t)*(e0+e1*t)-c0*d1
        nodes=[0.,self.dt]
        if e1:
            turning=(e1-c*e0)/(c*e1)
            if 0.<turning<self.dt: nodes.append(turning)
        nodes.sort(); candidates=list(nodes)
        for left,right in zip(nodes[:-1],nodes[1:]):
            if deriv_num(left)*deriv_num(right)<0:
                candidates.append(_bisect(deriv_num,left,right))
        denom_nodes=[0.,self.dt]
        if d2 and d1/(c*d2)>0:
            turning=-math.log(d1/(c*d2))/c
            if 0.<turning<self.dt: denom_nodes.append(turning)
        if min(d0+d1*t+d2*math.exp(-c*t) for t in denom_nodes)<=0:
            raise SafetyInfeasibleError("ZOH HOCBF control coefficient changes sign")
        values=[(c0+c2*math.exp(-c*t))/(k*(d0+d1*t+d2*math.exp(-c*t)))
                for t in candidates]
        return max(values) if lower else min(values)

    def filter_action(self, x_p, v_p_mms, u_rl, k_gain=None, damp_c=None):
        x=float(x_p)/1000.; v=float(v_p_mms)/1000.; proposed=float(u_rl)
        if not all(math.isfinite(z) for z in (x,v,proposed)):
            raise ValueError("State and proposed action must be finite")
        hu=(self.x_max-self.margin)/1000.-x
        hl=x-(self.x_min+self.margin)/1000.
        if min(hu,hl)<-1e-11:
            raise SafetyInfeasibleError("State outside inner travel set")
        if self.sampled:
            if k_gain is not None and not math.isclose(k_gain,self.k_gain):
                raise ValueError("Certificate and propagated object must use the same gain")
            if damp_c is not None and not math.isclose(damp_c,self.damp_c):
                raise ValueError("Certificate and propagated object must use the same damping")
            if min(self.kappa1*hu-v,self.kappa1*hl+v)<-1e-11:
                raise SafetyInfeasibleError("State outside high-order initial set")
            lo=max(self.u_min,self._held_interval_bound(hl,v,True))
            hi=min(self.u_max,self._held_interval_bound(hu,v))
            scope="exact ZOH interval of declared second-order object"
        else:
            lo,hi=self.reference_bounds(x_p,v_p_mms,k_gain,damp_c)
            scope="continuous reference inequality only; no sampled certificate"
        if lo>hi+1e-12:
            raise SafetyInfeasibleError(f"Empty actuator/CBF intersection: [{lo:g}, {hi:g}] V")
        if lo>hi: lo=hi  # floating-point coincidence only, not an infeasible QP
        applied=float(np.clip(proposed,lo,hi))
        self.last_info=BarrierProjection(proposed,applied,lo,hi,
            self.compute_barrier_h(x_p),abs(applied-proposed)>1e-10,scope)
        return applied

    def state_transition_si(self, x_m, v_m_s, u_v, duration=None):
        """Exact held-input SI object; no pressure states and no state clipping."""
        dt=self.dt if duration is None else float(duration)
        if not math.isfinite(dt) or dt<0: raise ValueError("Invalid duration")
        c=self.damp_c; k=self.k_si
        decay=math.exp(-c*dt); phi=-math.expm1(-c*dt)/c
        x=float(x_m)+phi*float(v_m_s)+(k/c)*(dt-phi)*float(u_v)
        v=decay*float(v_m_s)+k*phi*float(u_v)
        return float(x),float(v)

    def interval_minima(self, x_m, v_m_s, u_v, duration=None):
        """Exact mechanical and recursive-barrier minima over one hold interval."""
        if not self.sampled: raise ValueError("Only defined for recursive sampled filter")
        dt=self.dt if duration is None else float(duration)
        c=self.damp_c; k=self.k_si; kap=self.kappa1
        hu=(self.x_max-self.margin)/1000.-x_m
        hl=x_m-(self.x_min+self.margin)/1000.
        vinf=k*u_v/c; w=v_m_s-vinf
        def exp_affine_min(a0,a1,a2):
            nodes=[0.,dt]
            if a2 and a1/(c*a2)>0:
                t=-math.log(a1/(c*a2))/c
                if 0.<t<dt: nodes.append(t)
            return min(a0+a1*t+a2*math.exp(-c*t) for t in nodes)
        pos_nodes=[0.,dt]
        if w and -vinf/w>0:
            t=-math.log(-vinf/w)/c
            if 0.<t<dt:pos_nodes.append(t)
        positions=[self.state_transition_si(x_m,v_m_s,u_v,t)[0] for t in pos_nodes]
        a=self.alpha1; b=self.alpha2
        d0=a/c-b/c**2;d1=b/c;d2=1-a/c+b/c**2
        c2=(c-a+b/c)*v_m_s
        return {
            "h_mechanical_min":min(self.compute_barrier_h(z*1000) for z in positions),
            "h_upper_m_min":(self.x_max-self.margin)/1000.-max(positions),
            "h_lower_m_min":min(positions)-(self.x_min+self.margin)/1000.,
            "psi1_upper_m_s_min":exp_affine_min(kap*hu-kap*w/c-vinf,-kap*vinf,(kap/c-1)*w),
            "psi1_lower_m_s_min":exp_affine_min(kap*hl+kap*w/c+vinf,kap*vinf,(1-kap/c)*w),
            "psi2_upper_m_s2_min":exp_affine_min(b*(hu-v_m_s/c)-k*d0*u_v,-k*d1*u_v,c2-k*d2*u_v),
            "psi2_lower_m_s2_min":exp_affine_min(b*(hl+v_m_s/c)+k*d0*u_v,k*d1*u_v,-c2+k*d2*u_v),
        }


def run_safety_experiment(duration=1.,dt=.001):
    """Measured 8.5 V experiment, 20 mm start, 100 mm upper mechanical bound.

    The explicit classroom object k=2640 mm/(s^2 V), c=14/s and positive
    recursive gains (49/s,49/s) reproduce approximately -3.1 V braking and a
    98.8 mm asymptote. They are a declared reduced-order configuration, not
    identified hydraulic parameters. The exact reference constants 1850 and
    (48,580) remain available via HOCBFSafetyFilter.reference_bounds.
    """
    count=int(round(duration/dt))
    if count<1 or not math.isclose(count*dt,duration,abs_tol=1e-12):
        raise ValueError("duration must be an integer multiple of dt")
    filt=HOCBFSafetyFilter(k_gain=2640.,dt=dt,sampled=True,kappa1=49.,kappa2=49.)
    times=np.arange(count+1)*dt
    safe=np.empty((count+1,2));safe[0]=[.020,0.]
    controls=np.empty(count+1)
    raw=np.array([filt.state_transition_si(.020,0.,8.5,t) for t in times])
    collision=None
    if raw[-1,0]>=.100:
        collision=_bisect(lambda t:filt.state_transition_si(.020,0.,8.5,t)[0]-.100,0.,duration)
        raw[times>collision]=np.nan
    minimum={}
    for i,t in enumerate(times):
        x,v=safe[i]
        u=filt.filter_action(x*1000.,v*1000.,8.5)
        controls[i]=u
        if i<count:
            mins=filt.interval_minima(x,v,u)
            for name,value in mins.items():minimum[name]=float(min(minimum.get(name,math.inf),value))
            safe[i+1]=filt.state_transition_si(x,v,u)
    intervention=np.flatnonzero(controls<8.5-1e-10)
    checks={"all_zoh_position_barriers_nonnegative":minimum["h_mechanical_min"]>=-1e-12,
            "all_inner_position_barriers_nonnegative":min(minimum["h_upper_m_min"],minimum["h_lower_m_min"])>=-1e-11,
            "all_zoh_psi1_nonnegative":min(minimum["psi1_upper_m_s_min"],minimum["psi1_lower_m_s_min"])>=-1e-10,
            "all_zoh_psi2_nonnegative":min(minimum["psi2_upper_m_s2_min"],minimum["psi2_lower_m_s2_min"])>=-1e-9,
            "all_commands_within_voltage_limits":bool(np.max(np.abs(controls))<=10.+1e-12)}
    summary={"model":"x_dot=v; v_dot=2.64*u-14*v; x,v in SI",
             "scope":"independent second-order classroom object, not hydraulic pressure ODE",
             "sample_time_s":dt,"recursive_gains_per_s":[49.,49.],
             "effective_alpha_coefficients":[98.,2401.],
             "proposed_voltage_v":8.5,"duration_s":duration,
             "minimum_safe_voltage_v":float(controls.min()),
             "first_intervention_s":float(times[intervention[0]]) if intervention.size else None,
             "safe_final_position_mm":float(safe[-1,0]*1000.),
             "safe_max_position_mm":float(safe[:,0].max()*1000.),
             "safe_final_velocity_mm_s":float(safe[-1,1]*1000.),
             "unsafe_first_collision_s":collision,
             "unsafe_record_convention":"unprotected trajectory ends at first collision; no clipped plateau",
             "interval_minima":minimum,"checks":checks,"passed":all(checks.values())}
    if not summary["passed"]:raise AssertionError(summary)
    data={"time_s":times,"x_raw_m":raw[:,0],"v_raw_m_s":raw[:,1],
          "x_safe_m":safe[:,0],"v_safe_m_s":safe[:,1],"u_safe_v":controls,
          "h":np.array([filt.compute_barrier_h(x*1000.) for x in safe[:,0]])}
    return {"data":data,"summary":summary,"filter":filt}



# -----------------------------------------------------------------------------
# 边缘部署：浮点 actor、非结构化幅值剪枝与真实 W8/A16 整数矩阵乘法
# -----------------------------------------------------------------------------
import json
from pathlib import Path
import numpy as np


def _edge_observation(obs, width):
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim not in (1, 2) or x.shape[-1] != width or not np.isfinite(x).all():
        raise ValueError(f'Expected finite observation of shape ({width},) or (N,{width})')
    return x


class FloatActor:
    """SB3 SAC 确定性均值头：Linear/ReLU/.../Linear/tanh。

    返回归一化动作 [-1,1]；五维输入在调用前做 Z-score。
    提取的 actor 不包含 critic、优化器或训练时机理教师。
    """
    def __init__(self, weights, biases, metadata=None):
        if not weights or len(weights) != len(biases):
            raise ValueError('Nonempty matching weight and bias lists required')
        self.weights = [np.asarray(w, np.float32).copy() for w in weights]
        self.biases = [np.asarray(b, np.float32).copy() for b in biases]
        self.metadata = dict(metadata or {})
        previous = None
        for w, b in zip(self.weights, self.biases):
            if (w.ndim != 2 or min(w.shape) < 1 or b.shape != (w.shape[0],)
                    or not np.isfinite(w).all() or not np.isfinite(b).all()
                    or (previous is not None and previous != w.shape[1])):
                raise ValueError('Invalid finite feedforward actor topology')
            previous = w.shape[0]
        if previous != 1:
            raise ValueError('Hydraulic actor must have one scalar action')
        self.input_dim = self.weights[0].shape[1]

    def __call__(self, obs):
        x = _edge_observation(obs, self.input_dim)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i < len(self.weights)-1:
                x = np.maximum(x, 0.)
        return np.tanh(x)

    @classmethod
    def from_sb3(cls, model):
        import torch.nn as nn
        actor = model.actor
        modules = list(actor.latent_pi)
        if len(modules) % 2 or any(not isinstance(m, nn.Linear if i % 2 == 0 else nn.ReLU)
                                  for i, m in enumerate(modules)):
            raise ValueError('Exporter supports alternating Linear/ReLU hidden layers only')
        if not isinstance(actor.mu, nn.Linear):
            raise ValueError('SAC mean head must be Linear')
        layers = [m for m in modules if isinstance(m, nn.Linear)] + [actor.mu]
        return cls([m.weight.detach().cpu().numpy() for m in layers],
                   [m.bias.detach().cpu().numpy() for m in layers],
                   {'source': 'SB3 deterministic tanh(mean) actor', 'format_version': 1})

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f'weight_{i}': w for i, w in enumerate(self.weights)}
        arrays.update({f'bias_{i}': b for i, b in enumerate(self.biases)})
        arrays['metadata_json'] = np.array(json.dumps(self.metadata, ensure_ascii=False))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            n = len([k for k in f.files if k.startswith('weight_')])
            metadata = json.loads(str(f['metadata_json'])) if 'metadata_json' in f else {}
            return cls([f[f'weight_{i}'] for i in range(n)],
                       [f[f'bias_{i}'] for i in range(n)], metadata)


def _edge_prune(weights, ratio):
    if not np.isscalar(ratio) or not np.isfinite(ratio) or not 0. <= float(ratio) <= 1.:
        raise ValueError('prune_ratio must be finite and in [0,1]')
    w = np.array(weights, dtype=np.float32, copy=True)
    if not w.size or not np.isfinite(w).all():
        raise ValueError('Weights must be nonempty and finite')
    # 稳定秩选择消除同幅值权重的 percentile 平局歧义。
    count = int(np.floor(float(ratio)*w.size))
    order = np.argsort(np.abs(w).reshape(-1), kind='stable')
    w.reshape(-1)[order[:count]] = 0.
    return w


def _edge_affine_range(values, bits):
    if bits not in (8, 16):
        raise ValueError('Activation precision must be 8 or 16 bits')
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Calibration values must be nonempty and finite')
    qmin, qmax = -(1 << (bits-1)), (1 << (bits-1))-1
    lo, hi = min(0., float(values.min())), max(0., float(values.max()))
    if hi == lo:
        return 1., 0
    scale = (hi-lo)/(qmax-qmin)
    zero = int(np.clip(np.rint(qmin-lo/scale), qmin, qmax))
    return scale, zero


def _edge_quantize(values, scale, zero, bits):
    qmin, qmax = -(1 << (bits-1)), (1 << (bits-1))-1
    return np.clip(np.rint(np.asarray(values)/scale)+zero, qmin, qmax).astype(
        np.int8 if bits == 8 else np.int16)


class IntegerActor:
    """真实 INT8 权重、INT8/INT16 激活、INT32 累加的独立参考运行时。

    中间层重定标使用 INT64 乘法与整数舍入，末层保留 INT32 logits。
    输入仿射预处理、输入量化和最终 tanh 为浮点；不声称全整数激活
    或特定 MCU 上的耗时。加载时对满范围累加/重定标溢出作静态核查。
    """
    def __init__(self, weights, biases, weight_scales, multipliers, metadata,
                 input_center, input_channel_scale):
        self.weights = [np.array(w, copy=True) for w in weights]
        self.biases = [np.array(b, copy=True) for b in biases]
        self.weight_scales = [np.array(s, np.float64, copy=True) for s in weight_scales]
        self.multipliers = [np.array(m, copy=True) for m in multipliers]
        self.metadata = dict(metadata)
        self.layers = self.metadata.get('layers', [])
        if not self.layers or not (len(self.layers) == len(self.weights) == len(self.biases)
                                   == len(self.weight_scales) == len(self.multipliers)):
            raise ValueError('Invalid integer layer count')
        self.input_dim = self.weights[0].shape[1]
        self.input_center = np.asarray(input_center, np.float32).copy()
        self.input_channel_scale = np.asarray(input_channel_scale, np.float32).copy()
        if (self.input_center.shape != (self.input_dim,)
                or self.input_channel_scale.shape != (self.input_dim,)
                or not np.isfinite(self.input_center).all()
                or not np.isfinite(self.input_channel_scale).all()
                or np.any(self.input_channel_scale <= 0)):
            raise ValueError('Invalid affine input prefix')
        previous_width = self.input_dim
        previous_scale = previous_zero = previous_bits = None
        for i, (w, b, sw, mult, layer) in enumerate(zip(
                self.weights, self.biases, self.weight_scales, self.multipliers, self.layers)):
            bits = layer.get('activation_bits', 8)
            if bits not in (8,16) or layer.get('kind', 'per_channel') != 'per_channel':
                raise ValueError('Only symmetric per-channel W8/A8 or W8/A16 supported')
            if (w.ndim != 2 or w.dtype != np.int8 or w.shape[1] != previous_width
                    or b.dtype != np.int32 or b.shape != (w.shape[0],)
                    or sw.shape != b.shape or mult.shape != b.shape
                    or not np.isfinite(sw).all() or np.any(sw <= 0)
                    or mult.dtype.kind not in 'iu' or np.any(mult <= 0)):
                raise ValueError('Invalid per-channel integer representation')
            qmin, qmax = -(1 << (bits-1)), (1 << (bits-1))-1
            sx, sy = float(layer['input_scale']), float(layer['output_scale'])
            zx, zy, shift = layer['input_zero'], layer['output_zero'], layer['shift']
            if (not np.isfinite([sx,sy]).all() or sx <= 0 or sy <= 0
                    or not isinstance(zx, int) or not isinstance(zy, int)
                    or not qmin <= zx <= qmax or not qmin <= zy <= qmax
                    or not isinstance(shift, int) or not 1 <= shift <= 60):
                raise ValueError('Invalid affine scales, zero points or integer shift')
            if i and (sx != previous_scale or zx != previous_zero or bits != previous_bits):
                raise ValueError('Integer activation representation is disconnected between layers')
            if bool(layer.get('output_int32', False)) and (i != len(self.layers)-1 or layer['relu']):
                raise ValueError('Only the last non-ReLU layer may retain INT32 logits')
            # Python integers avoid wraparound in the safety calculation itself.
            fullrange = max(abs(qmin-zx), abs(qmax-zx))
            bound = [sum(abs(int(v)) for v in row)*fullrange + abs(int(bias))
                     for row, bias in zip(w, b)]
            if any(n >= 2**31-1 for n in bound):
                raise ValueError('Full-range INT32 accumulation may overflow')
            if not layer.get('output_int32', False) and any(
                    n*int(m)+(1 << (shift-1)) >= 2**63-1 for n,m in zip(bound,mult)):
                raise ValueError('INT64 requantization may overflow')
            previous_width, previous_scale, previous_zero, previous_bits = w.shape[0], sy, zy, bits
        if previous_width != 1:
            raise ValueError('Expected scalar hydraulic action')

    def __call__(self, obs):
        raw = _edge_observation(obs, self.input_dim)
        first = self.layers[0]
        balanced = (raw-self.input_center)/self.input_channel_scale
        q = _edge_quantize(balanced, first['input_scale'], first['input_zero'],
                           first.get('activation_bits',8))
        for w, b, mult, layer in zip(self.weights, self.biases, self.multipliers, self.layers):
            x = q.astype(np.int32)-layer['input_zero']
            accum = x @ w.astype(np.int32).T + b
            if layer.get('output_int32',False):
                q = accum
                continue
            scaled = accum.astype(np.int64)*mult
            denom = 1 << layer['shift']
            rounded = np.sign(scaled)*((np.abs(scaled)+denom//2)//denom)
            value = rounded+layer['output_zero']
            if layer['relu']:
                value = np.maximum(value, layer['output_zero'])
            bits = layer.get('activation_bits',8)
            q = np.clip(value, -(1 << (bits-1)), (1 << (bits-1))-1).astype(
                np.int8 if bits == 8 else np.int16)
        last = self.layers[-1]
        if last.get('output_int32',False):
            return np.tanh(q.astype(np.float64)*(last['input_scale']*self.weight_scales[-1])).astype(np.float32)
        return np.tanh(last['output_scale']*(q.astype(np.float32)-last['output_zero']))

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {'input_center':self.input_center, 'input_channel_scale':self.input_channel_scale,
                  'metadata_json':np.array(json.dumps(self.metadata,ensure_ascii=False))}
        for i,(w,b,sw,mult) in enumerate(zip(self.weights,self.biases,self.weight_scales,self.multipliers)):
            arrays.update({f'weight_{i}':w, f'bias_{i}':b, f'weight_scale_{i}':sw, f'multiplier_{i}':mult})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            metadata=json.loads(str(f['metadata_json']))
            n=len(metadata['layers'])
            return cls([f[f'weight_{i}'] for i in range(n)],
                       [f[f'bias_{i}'] for i in range(n)],
                       [f[f'weight_scale_{i}'] for i in range(n)],
                       [f[f'multiplier_{i}'] for i in range(n)],metadata,
                       f['input_center'],f['input_channel_scale'])


class EdgeLightweightExporter:
    """参考式(8.6)与可审查的真实整数导出。

    按幅值剔除连接是非结构化剪枝；下面的新模型直接导出不含蒸馏，
    必须独立闭环验证。权重量化后反量化的浮点模型仅标为 QDQ。
    """
    @staticmethod
    def prune_and_quantize_weights(W_fp32, prune_ratio=.70):
        w = _edge_prune(W_fp32, prune_ratio)
        scale, zero = _edge_affine_range(w,8)
        q = _edge_quantize(w,scale,zero,8)
        dequant = (scale*(q.astype(np.float32)-zero)).astype(np.float32)
        return dequant,q,scale,zero,100.*float(np.mean(w == 0.))

    @staticmethod
    def export_qdq(actor, prune_ratio=.70):
        weights = [EdgeLightweightExporter.prune_and_quantize_weights(w,prune_ratio)[0]
                   for w in actor.weights]
        return FloatActor(weights, actor.biases,
            {'method':'magnitude pruning + affine weight QDQ; float GEMM',
             'prune_ratio':float(prune_ratio),'deployment_ready':False})

    @staticmethod
    def export_integer(actor, calibration, prune_ratio=.70, activation_bits=16):
        calibration = _edge_observation(calibration,actor.input_dim)
        if calibration.ndim != 2 or not len(calibration):
            raise ValueError('Calibration must be nonempty shape (N,input_dim)')
        x=calibration
        weights=[_edge_prune(w,prune_ratio) for w in actor.weights]
        integer_w=[];integer_b=[];scales=[];multipliers=[];layers=[]
        sx,zx=_edge_affine_range(x,activation_bits)
        qx=_edge_quantize(x,sx,zx,activation_bits)
        for i,(w,b) in enumerate(zip(weights,actor.biases)):
            sw=np.max(np.abs(w),axis=1).astype(np.float64)/127.
            sw=np.where(sw > 1e-12,sw,1.)
            qw=np.clip(np.rint(w/sw[:,None]),-127,127).astype(np.int8)
            raw_b=np.rint(b/(sx*sw))
            if np.max(np.abs(raw_b),initial=0.) >= 2**31-1:
                raise ValueError('Bias requires more than INT32; revise calibration')
            qb=raw_b.astype(np.int32)
            max_acc=w.shape[1]*((1 << activation_bits)-1)*127+int(np.max(np.abs(qb.astype(np.int64)),initial=0))
            if max_acc >= 2**31-1:
                raise ValueError('INT32 dot-product overflow')
            accum=(qx.astype(np.int32)-zx) @ qw.astype(np.int32).T+qb
            y=accum.astype(np.float64)*(sx*sw)
            relu=i < len(weights)-1
            if relu:y=np.maximum(y,0.)
            sy,zy=_edge_affine_range(y,activation_bits)
            ratio=sx*sw/sy;shift=30
            while shift > 1 and max_acc*max(1,int(np.max(np.rint(ratio*(1 << shift))))) >= 2**62:
                shift-=1
            multiplier=np.rint(ratio*(1 << shift)).astype(np.int64)
            if np.any(multiplier <= 0):
                raise ValueError('Fixed-point multiplier underflow; revise calibration')
            layer=dict(kind='per_channel',input_scale=float(sx),input_zero=int(zx),
                       output_scale=float(sy),output_zero=int(zy),shift=int(shift),
                       relu=relu,output_int32=not relu,activation_bits=activation_bits)
            integer_w.append(qw);integer_b.append(qb);scales.append(sw);multipliers.append(multiplier);layers.append(layer)
            if relu:
                scaled=accum.astype(np.int64)*multiplier;denom=1 << shift
                rounded=np.sign(scaled)*((np.abs(scaled)+denom//2)//denom)
                qx=np.clip(np.maximum(rounded+zy,zy),-(1 << (activation_bits-1)),(1 << (activation_bits-1))-1).astype(
                    np.int8 if activation_bits == 8 else np.int16)
            sx,zx=sy,zy
        metadata=dict(format_version=4 if activation_bits == 16 else 3,activation_bits=activation_bits,
                      layers=layers,prune_ratio=float(prune_ratio),deployment_ready=False,
                      method='per-layer stable magnitude pruning; symmetric per-output-channel INT8 weights',
                      runtime='INT32 GEMM, INT64 requantization; float input affine and final tanh',
                      calibration_count=len(calibration),
                      actual_weight_sparsity=float(sum(np.count_nonzero(w == 0.) for w in weights)/sum(w.size for w in weights)),
                      validation='Requires independent closed-loop evaluation before deployment')
        return IntegerActor(integer_w,integer_b,scales,multipliers,metadata,
                            np.zeros(actor.input_dim,np.float32),np.ones(actor.input_dim,np.float32))


"""Self-contained training fragment for the rewritten Chapter 8 lab.

This file is concatenated into drl_hydraulic_lab.py. HydraulicGymEnv is defined
in that same file; no earlier project, policy, or teacher is imported.
"""


def train_agent(output_dir, timesteps=1_000_000, seed=2026):
    """Train original SB3 SAC from random initialization and export its mean.

    A fresh child directory is always created under ``output_dir``. The return
    value is a dictionary containing that run directory and artifact paths.
    ``actor_fp32.npz`` uses FloatActor.save: weight_0,bias_0,... and
    metadata_json containing normalization. Its output is tanh(logit), i.e. a in
    [-1,1]; voltage is 10*a. Training does not edit the actor after model.learn.

    A 10,000-step call is a smoke experiment; it is never described as the
    million-step classroom experiment or as a converged policy.
    """
    import csv
    import hashlib
    import inspect
    import json
    import math
    import platform
    import random
    import time
    import uuid
    from collections import deque
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import torch
    import gymnasium as gym
    import stable_baselines3 as sb3
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    if isinstance(timesteps, bool) or int(timesteps) != timesteps or timesteps < 4:
        raise ValueError("timesteps must be an integer of at least 4")
    if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**31:
        raise ValueError("seed must be a nonnegative 31-bit integer")
    requested_steps, seed = int(timesteps), int(seed)
    n_envs = 4
    learning_starts = min(50_000, max(1, requested_steps // 2 - 1))
    reward_scale = 0.001
    window_size = 10_000
    period = 10_000
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = Path(output_dir).resolve() / (
        f"train_{stamp}_seed{seed}_{uuid.uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir, log_dir = run_dir / "models", run_dir / "outputs"
    model_dir.mkdir()
    log_dir.mkdir()
    checkpoint_dir = model_dir / "checkpoints"
    checkpoint_dir.mkdir()

    # Every random source used here is explicitly seeded. Gymnasium reset uses
    # its own per-environment generator through VecEnv.seed(seed+i).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    def sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    source_hashes = {}
    for name in ("HydraulicGymEnv", "HydraulicTwinCore", "train_agent"):
        obj = globals().get(name)
        if obj is not None:
            try:
                source = inspect.getsource(obj)
            except (OSError, TypeError):
                continue
            source_hashes[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # Keep a source snapshot with each run. It is evidence, never executable
    # state imported from another project or another training attempt.
    try:
        current_source = Path(inspect.getfile(train_agent)).resolve()
        snapshot = run_dir / "source_snapshot.py"
        snapshot.write_bytes(current_source.read_bytes())
        source_hashes["source_snapshot.py"] = sha256(snapshot)
    except (OSError, TypeError):
        pass

    class OptimizerRewardScale(gym.RewardWrapper):
        def reward(self, reward):
            return float(reward) * reward_scale

    def make_env():
        # Monitor sits inside scaling so episode statistics retain physical
        # rewards, not the numerical scale used by the Q optimizers.
        return OptimizerRewardScale(Monitor(HydraulicGymEnv(max_steps=2000)))

    env = DummyVecEnv([make_env for _ in range(n_envs)])
    env.seed(seed)
    model = SAC(
        "MlpPolicy", env, learning_rate=3e-4, buffer_size=100_000,
        learning_starts=learning_starts, batch_size=128, tau=0.005,
        gamma=0.99, train_freq=4, gradient_steps=2,
        ent_coef="auto_0.03", target_entropy=0.35 - math.log(10.0),
        policy_kwargs={"net_arch": [64, 64]}, seed=seed, device="cpu",
        verbose=0,
    )
    # No supervised initialization, imported checkpoints, actor regularizers,
    # teacher loss, or post-training mean replacement is performed.
    untrained_path = model_dir / "sac_untrained.zip"
    model.save(untrained_path)

    first_env = env.envs[0].unwrapped
    obs_mean = np.asarray(first_env.obs_mean, dtype=np.float32)
    obs_std = np.asarray(first_env.obs_std, dtype=np.float32)
    if model.observation_space.shape != (5,) or model.action_space.shape != (1,):
        env.close()
        raise ValueError("Chapter 8 SAC requires exactly five observations and one action")
    if obs_mean.shape != (5,) or obs_std.shape != (5,) or np.any(obs_std <= 0):
        env.close()
        raise ValueError("Observation normalization must have five positive scales")

    class ActualTrainingLog(BaseCallback):
        def __init__(self):
            super().__init__()
            self.base_rewards = deque(maxlen=window_size)
            self.raw_rewards = deque(maxlen=window_size)
            self.penalties = deque(maxlen=window_size)
            self.failures = deque(maxlen=window_size)
            self.episode_per_step = deque(maxlen=30)
            self.episode_count = 0
            self.rows = []
            self.episode_rows = []
            self.last_logged = -1
            self.last_checkpoint = 0
            self.started = time.perf_counter()

        def _on_step(self):
            # rewards in callback locals have passed through the explicit
            # optimizer scale. Prefer exact raw fields from env.step info.
            for scaled_reward, info in zip(self.locals["rewards"], self.locals["infos"]):
                penalty = float(info.get("failure_penalty", 0.0))
                raw = float(info.get("raw_reward", float(scaled_reward) / reward_scale))
                base = float(info.get("base_reward", raw - penalty))
                self.base_rewards.append(base)
                self.raw_rewards.append(raw)
                self.penalties.append(penalty)
                self.failures.append(bool(info.get("physical_failure", False)))
                episode = info.get("episode")
                if episode is not None:
                    length = int(episode["l"])
                    episode_return = float(episode["r"])
                    per_step = episode_return / max(length, 1)
                    self.episode_per_step.append(per_step)
                    self.episode_count += 1
                    self.episode_rows.append({
                        "episode": self.episode_count,
                        "timesteps_when_completed": int(self.num_timesteps),
                        "length": length, "raw_episode_return": episode_return,
                        "raw_episode_return_per_step": per_step,
                    })
            if self.num_timesteps % period == 0:
                self.record()
            milestone = int(self.num_timesteps // 100_000)
            if milestone > self.last_checkpoint:
                self.model.save(checkpoint_dir / f"sac_{self.num_timesteps}.zip")
                self.last_checkpoint = milestone
            return True

        def entropy_on_replay(self):
            buffer = self.model.replay_buffer
            if buffer is None or buffer.size() < 1:
                return None, 0
            count = 1024
            # Diagnostic draws must not change subsequent training randomness.
            numpy_state = np.random.get_state()
            try:
                np.random.seed(seed + int(self.num_timesteps))
                observations = buffer.sample(count).observations
            finally:
                np.random.set_state(numpy_state)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed + int(self.num_timesteps) + 1)
                with torch.no_grad():
                    _, log_probability = self.model.actor.action_log_prob(observations)
                    entropy = float(-log_probability.mean().cpu())
            if not math.isfinite(entropy):
                raise RuntimeError("Non-finite tanh-corrected policy entropy")
            return entropy, count

        def record(self):
            if self.last_logged == self.num_timesteps or not self.base_rewards:
                return
            entropy, cohort_count = self.entropy_on_replay()
            coefficient = float(torch.exp(self.model.log_ent_coef.detach()).cpu())
            row = {
                "timesteps": int(self.num_timesteps),
                "base_reward_mean": float(np.mean(self.base_rewards)),
                "reward_window_count": len(self.base_rewards),
                "reward_window_is_full": len(self.base_rewards) == window_size,
                "reward_source": "base_reward_last_10000_actual_environment_steps",
                "raw_reward_mean": float(np.mean(self.raw_rewards)),
                "failure_penalty_mean": float(np.mean(self.penalties)),
                "window_physical_failures": int(sum(self.failures)),
                "completed_episode_per_step_mean": (
                    float(np.mean(self.episode_per_step)) if self.episode_per_step else None
                ),
                "completed_episodes": self.episode_count,
                "replay_action_entropy_nat": entropy,
                "replay_voltage_entropy_nat": (
                    entropy + math.log(10.0) if entropy is not None else None
                ),
                "entropy_replay_cohort_count": cohort_count,
                "uniform_exploration_active": self.num_timesteps <= learning_starts,
                "behavior_voltage_entropy_nat": (
                    math.log(20.0) if self.num_timesteps <= learning_starts else
                    entropy + math.log(10.0) if entropy is not None else None
                ),
                "temperature_alpha": coefficient,
                "gradient_updates": int(self.model._n_updates),
                "elapsed_s": time.perf_counter() - self.started,
            }
            self.rows.append(row)
            self.last_logged = int(self.num_timesteps)
            with (log_dir / "training.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row), lineterminator="\n")
                writer.writeheader()
                writer.writerows(self.rows)
            if self.episode_rows:
                with (log_dir / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(self.episode_rows[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(self.episode_rows)
            (model_dir / "progress.json").write_text(
                json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print("SAC actual training: " + json.dumps(row, ensure_ascii=False), flush=True)

    callback = ActualTrainingLog()
    metadata = {
        "experiment": "chapter8_vanilla_sac_from_random_initialization",
        "algorithm": "Stable-Baselines3 SAC with original actor and critic objectives",
        "requested_timesteps": requested_steps,
        "seed": seed, "n_envs": n_envs,
        "sampling_period_s": 0.001, "max_episode_steps": 2000,
        "learning_starts": learning_starts,
        "train_frequency_vector_steps": 4, "gradient_steps_per_update": 2,
        "learning_rate": 3e-4, "batch_size": 128, "buffer_size": 100_000,
        "gamma": 0.99, "tau": 0.005,
        "optimizer_reward_multiplier": reward_scale,
        "target_entropy_normalized_nat": 0.35 - math.log(10.0),
        "target_entropy_voltage_nat": 0.35,
        "policy_architecture": [5, 64, 64, 1],
        "observation_names": ["e_mm", "de_mm_s", "dde_mm_s2", "PL_MPa", "Ps_MPa"],
        "obs_mean": obs_mean.tolist(), "obs_std": obs_std.tolist(),
        "reward": "-100*e_mm**2 - 0.1*u_V**2 - (u_V-u_prev_V)**2 + 2*I(abs(e_mm)<0.2)",
        "reward_log": "Fixed last10000 actual-step base reward window; failure cost, episode return and optimizer scale are separate",
        "entropy_log": "Monte Carlo -log pi(a|s) including tanh Jacobian on a seeded current-replay cohort; H(u)=H(a)+ln(10)",
        "behavior_during_learning_starts": "Uniform a in [-1,1]; voltage entropy ln(20); actor entropy is separately diagnostic",
        "domain_randomization": "m +/-20%; nominal beta_e 600..1400 MPa; F_c +/-30%; each episode reset",
        "physical_parameters": dict(first_env.twin.p),
        "fault_augmentation": bool(first_env.fault_augmentation),
        "fault_augmentation_distribution": "80% episodes: fault time U(0.05,0.8)s, load U(0,5500)N, beta multiplier U(0.5,1)",
        "teacher_initialization": False, "teacher_loss": False,
        "post_training_actor_rewrite": False,
        "imported_model_checkpoint": False,
        "untrained_checkpoint_sha256": sha256(untrained_path),
        "source_sha256": source_hashes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": torch.__version__, "gymnasium": gym.__version__, "stable_baselines3": sb3.__version__},
        "device": "cpu", "torch_threads": 1,
        "performance_claim": "Measured results only; no promise of the schematic PPT numerical targets",
        "checkpoint_timing": "100k callback snapshots precede the scheduled gradient update; sac_model.zip is saved after learn returns",
        "status": "training",
    }
    metadata_path = model_dir / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=requested_steps, callback=callback, progress_bar=False)
        callback.record()
        final_path = model_dir / "sac_model.zip"
        model.save(final_path)

        exported_actor = FloatActor.from_sb3(model)
        exported_actor.metadata.update(
            obs_mean=obs_mean.tolist(), obs_std=obs_std.tolist(),
            action_to_voltage=10.0,
            source_checkpoint_sha256=sha256(final_path),
            deployment_origin="Exact final vanilla SAC mean; no subsequent actor update",
        )
        actor_path = model_dir / "actor_fp32.npz"
        exported_actor.save(actor_path)

        buffer = model.replay_buffer
        observation_pool = buffer.observations[:buffer.size()].reshape(-1, 5)
        if not len(observation_pool):
            raise RuntimeError("Training produced no replay observations")
        selection = np.random.default_rng(seed + 17).permutation(len(observation_pool))[:16_000]
        selected = np.asarray(observation_pool[selection], dtype=np.float32)
        split = min(12_000, max(1, len(selected) * 3 // 4))
        # Save disjoint calibration/validation rows. The validation cohort is
        # held out from quantizer fitting; both originate in actual replay.
        calibration_path = model_dir / "calibration_observations.npz"
        np.savez_compressed(calibration_path, calibration=selected[:split], validation=selected[split:])

        # Verify export against this exact final checkpoint without changing it.
        probes = selected[:min(512, len(selected))]
        # Check the serialized, independently loaded runtime, not only the
        # in-memory export object.
        reloaded_actor = FloatActor.load(actor_path)
        exported_actions = reloaded_actor(probes)
        with torch.no_grad():
            exact_actions = model.actor(torch.as_tensor(probes, device="cpu"), deterministic=True).cpu().numpy()
        maximum_difference = float(np.max(np.abs(exported_actions - exact_actions)))
        if maximum_difference > 2e-5:
            raise AssertionError(f"Exported mean actor disagrees with SAC: {maximum_difference:g}")
        metadata.update(
            status="completed", actual_timesteps=int(model.num_timesteps),
            gradient_updates=int(model._n_updates), completed_episodes=callback.episode_count,
            wall_seconds=time.perf_counter() - started,
            final_checkpoint_sha256=sha256(final_path), actor_fp32_sha256=sha256(actor_path),
            calibration_observations_sha256=sha256(calibration_path),
            exported_actor_max_normalized_action_difference=maximum_difference,
            calibration_rows=split, held_out_quantization_validation_rows=len(selected) - split,
            deployment_origin="The final SAC mean actor without any post-training weight update",
        )
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    except BaseException as error:
        metadata.update(status="failed", failure_type=type(error).__name__, failure=str(error),
                        actual_timesteps=int(model.num_timesteps), wall_seconds=time.perf_counter() - started)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
    finally:
        env.close()
    return {
        "run_dir": run_dir, "model_path": final_path,
        "actor_fp32_path": actor_path, "calibration_path": calibration_path,
        "training_csv": log_dir / "training.csv", "metadata_path": metadata_path,
        "metadata": metadata,
    }


# =============================================================================
# 模块六：共同对象的四策略闭环实验，不回填或平滑图中数据（8.5.2）
# =============================================================================
import csv
import time

POLICIES = ("pid", "nominal_ff", "sac_fp32", "edge_int8")


class DigitalPID:
    """固定位置PID，参考-实测误差，附条件积分抗饱和。"""
    def __init__(self, initial_voltage, dt=.001, kp=1., ki=40., kd=.01):
        self.u0, self.dt, self.kp, self.ki, self.kd = initial_voltage, dt, kp, ki, kd
        self.integral, self.derivative = 0., 0.

    def update(self, position_error, velocity_error):
        self.derivative = .85*self.derivative + .15*velocity_error
        trial = self.integral + position_error*self.dt
        u = self.u0+self.kp*position_error+self.ki*trial+self.kd*self.derivative
        if abs(u) <= 10. or np.sign(position_error) != np.sign(u):
            self.integral = trial
        return float(np.clip(self.u0+self.kp*position_error+self.ki*self.integral
                             + self.kd*self.derivative, -10., 10.))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("CSV数据为空")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def trajectory_metrics(rows, fault_time=.5):
    t = np.array([r['time_s'] for r in rows])
    e = np.array([r['velocity_error_mms'] for r in rows])
    u = np.array([r['control_v'] for r in rows])
    if len(t) < 2 or not np.isfinite(np.c_[t, e, u]).all() or np.any(np.diff(t) <= 0):
        raise ValueError("轨迹必须有限且时间严格递增")
    post, pre, tail = t >= fault_time-1e-12, t < fault_time-1e-12, t >= t[-1]-.1-1e-12
    if not post.any():
        raise ValueError("轨迹不含故障时段")
    peak = np.flatnonzero(post)[np.argmax(np.abs(e[post]))]
    recovery = None
    # 以最大绝对偏差后的首个时刻开始，直至记录结束均位于±1mm/s，且至少留50ms。
    for k in range(peak, len(t)):
        if t[-1]-t[k] < .05-1e-12:
            break
        if np.all(np.abs(e[k:]) <= 1.):
            recovery = float(t[k]-fault_time)
            break
    return dict(minimum_velocity_error_mms=float(e[post].min()),
        maximum_velocity_error_mms=float(e[post].max()),
        velocity_error_rms_mms=float(np.sqrt(np.mean(e[post]**2))),
        recovery_1mms_s=recovery, final_100ms_velocity_bias_mms=float(e[tail].mean()),
        peak_voltage_v=float(u[post].max()), final_voltage_v=float(u[tail].mean()),
        max_abs_voltage_v=float(np.abs(u).max()),
        prefault_velocity_error_rms_mms=float(np.sqrt(np.mean(e[pre]**2))) if pre.any() else None,
        final_position_error_mm=float(rows[-1]['position_error_mm']))


def run_policy(policy, actor=None, *, backend="auto", twin_config=None,
               start_time=.4, end_time=1., fault_time=.5, load_n=5000., beta_multiplier=.5):
    if policy not in POLICIES:
        raise ValueError("未知控制策略")
    if not start_time < fault_time < end_time:
        raise ValueError("需要 start_time < fault_time < end_time")
    n = round((end_time-start_time)/.001)
    if not np.isclose(n*.001, end_time-start_time, rtol=0, atol=1e-12):
        raise ValueError("评估时长应为1ms的整数倍")
    env = HydraulicGymEnv(twin_config, domain_randomization=False,
        fault_augmentation=False, max_steps=n+1, backend=backend)
    obs, initial = env.reset(seed=8108, options=dict(evaluation=True,
        velocity=.05, position=.0001, fault_time=fault_time-start_time,
        load_n=load_n, beta_multiplier=beta_multiplier))
    pid = DigitalPID(initial['initial_voltage_v'])
    rows, observations = [], []
    for k in range(n+1):
        raw = env.raw_observation()
        state = env.twin.state.copy()
        if policy == "pid":
            voltage = pid.update(-raw[0], -raw[1])
        elif policy == "nominal_ff":
            # 标称空载逆模型，无负载估计/积分；不把真实故障标签偷喂给控制器。
            voltage = initial['initial_voltage_v']
        else:
            if actor is None:
                raise ValueError("需要已加载的独立神经网络策略")
            action = np.asarray(actor(obs)).reshape(-1)[0]
            voltage = float(np.clip(action, -1., 1.))*10.
        applied = np.array([voltage/10.], dtype=np.float32)
        voltage = float(applied[0])*10.
        load, beta = env.current_fault()
        rows.append(dict(policy=policy, time_s=start_time+env.time,
            position_mm=float(state[0]*1000.), position_error_mm=float(raw[0]),
            velocity_mms=float(state[1]*1000.), reference_velocity_mms=50.,
            velocity_error_mms=float(raw[1]), acceleration_error_mms2=float(raw[2]),
            load_pressure_mpa=float(raw[3]), p1_mpa=float(state[2]/1e6),
            p2_mpa=float(state[3]/1e6), control_v=voltage,
            previous_control_v=float(env.last_u), load_n=load,
            beta_mpa=env.twin.p['beta_e']*beta/1e6))
        observations.append(obs.copy())
        if k == n:
            break
        obs, _, terminated, _, info = env.step(applied)
        if terminated:
            raise RuntimeError(f"{policy}在{env.time:.6f}s失败: {info['physical_failure']}")
    env.close()
    return rows, np.array(observations), initial


def load_assets(asset_dir):
    directory = Path(asset_dir)
    metadata = json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
    required = {'actor_fp32.npz', 'actor_int8.npz'}
    if not required.issubset(metadata.get('files', {})):
        raise ValueError('模型来源清单必须包含两个运行时actor及各自SHA256')
    for name, expected in ((name, item['sha256']) for name,item in metadata['files'].items()):
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"模型来源检查失败: {name}")
    fp32 = FloatActor.load(directory/'actor_fp32.npz')
    edge = IntegerActor.load(directory/'actor_int8.npz')
    return fp32, edge, metadata


def evaluate_experiment(output_dir, asset_dir, backend="auto"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fp32, edge, provenance = load_assets(asset_dir)
    # 强非线性新策略可能产生快速切换；默认使用误差控制的刚性求解器
    # 做独立评估，避免把固定RK子步的数值误差误认作策略表现。
    evaluation_backend = ('radau' if backend == 'auto' and
                          provenance.get('strategy_kind') == 'vanilla_fresh' else backend)
    rows, metrics, observations = [], {}, {}
    for policy in POLICIES:
        actor = fp32 if policy == 'sac_fp32' else edge if policy == 'edge_int8' else None
        try:
            group, obs, _ = run_policy(policy, actor, backend=evaluation_backend)
        except (ValueError, RuntimeError) as error:
            if provenance.get('strategy_kind') != 'vanilla_fresh':
                raise
            metrics[policy] = dict(failure=str(error), completed=False)
            print(policy, '独立闭环失败：'+str(error), flush=True)
            continue
        rows.extend(group)
        metrics[policy] = trajectory_metrics(group)
        observations[policy] = obs
        print(policy, json.dumps(metrics[policy], ensure_ascii=False), flush=True)
    write_csv(out/'evaluation.csv', rows)
    safety = run_safety_experiment()
    # numpy数组转CSV，用首撞后的NaN自然断开未保护曲线。
    keys = list(safety['data'])
    write_csv(out/'safety.csv', [dict(zip(keys, values)) for values in zip(*safety['data'].values())])
    write_json(out/'safety_metrics.json', safety['summary'])
    fp = np.array([r['velocity_mms'] for r in rows if r['policy'] == 'sac_fp32'])
    qi = np.array([r['velocity_mms'] for r in rows if r['policy'] == 'edge_int8'])
    # 当前机器numpy核真实计时，绝不称作MCU实测；不含传感/ODE/安全滤波。
    probe = next(iter(observations.values()))[0]
    timings = {}
    for label, actor in [('fp32_numpy', fp32), ('integer_numpy', edge)]:
        for _ in range(100):
            actor(probe)
        values = []
        for _ in range(5):
            tick = time.perf_counter_ns()
            for _ in range(500):
                actor(probe)
            values.append((time.perf_counter_ns()-tick)/500./1e6)
        timings[label+'_median_ms'] = float(np.median(values))
    result = dict(policies=metrics, safety=safety['summary'],
        model_origin=provenance, physical_config=DEFAULT_PHYSICS,
        evaluation=dict(sample_period_s=.001, start_s=.4, end_s=1., fault_s=.5,
            integration_backend=evaluation_backend,
            load_n=5000., beta_multiplier=.5, observation_sign='actual-reference',
            initial_condition='same unloaded 50mm/s equilibrium for all four policies',
            control_mode='deterministic; no artificial noise/dither or smoothing',
            nominal_ff='unloaded equilibrium inverse; no external load estimate or integral'),
        edge=dict(weight_dtype='INT8', activation_dtype='INT16',
            closed_loop_velocity_rms_difference_mms=(float(np.sqrt(np.mean((fp-qi)**2)))
                                                   if len(fp) == len(qi) and len(fp) > 0 else None),
            benchmark=timings, benchmark_scope='current-host NumPy actor kernel only; not MCU latency'),
        plot_sources=dict(a='PPT page12 teaching illustration, not training data',
            b='independent 2nd-order SI plant with sampled HOCBF',
            c_d=('new 4-state closed loops using final vanilla SAC and newly pruned integer actor'
                 if provenance.get('strategy_kind') == 'vanilla_fresh' else
                 'newly integrated 4-state hydraulic closed loops using provenance-bound guided pretrained actors')))
    write_json(out/'metrics.json', result)
    return result


# =============================================================================
# 模块七：课堂四联图及独立真实训练图；示意和实测数据分别保存
# =============================================================================
def configure_plotting():
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'chapter8_rewritten_mpl'))
    os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'chapter8_rewritten_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {item.name for item in font_manager.fontManager.ttflist}
    fonts = [name for name in ('Noto Sans CJK SC', 'Noto Sans CJK JP', 'WenQuanYi Micro Hei',
                               'Microsoft YaHei', 'SimHei', 'DejaVu Sans') if name in available]
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':fonts,
        'axes.unicode_minus':False, 'font.size':9, 'axes.titlesize':11,
        'axes.labelsize':10, 'legend.fontsize':7.2, 'pdf.fonttype':42, 'savefig.dpi':180})
    return plt


def draw_ppt_training_schematic(ax):
    """PPT第12页的阶段示意；插值锚点不是原始训练记录，不写入training.csv。"""
    from scipy.interpolate import PchipInterpolator
    from matplotlib.patches import Patch
    xk = np.array([0., 50., 100., 150., 200., 300., 400., 1000.])
    reward = np.array([-180., -170., -145., -105., -65., -15., -2., -2.])
    entropy = np.array([2.8, 2.75, 2.4, 1.9, 1.15, .5, .35, .35])
    xx = np.linspace(0., 1000., 1001)
    patches = []
    for lo, hi, color, text in [(0,50,'#eeeeee','阶段 I：随机探索（0–50k）'),
        (50,400,'#e9edfb','阶段 II：策略成型（50k–400k）'),
        (400,1000,'#e8f5e9','阶段 III：稳态微调（>400k）')]:
        ax.axvspan(lo, hi, color=color, zorder=0)
        patches.append(Patch(facecolor=color, label=text))
    line1, = ax.plot(xx, PchipInterpolator(xk, reward)(xx), color='#4169ed', lw=1.9,
                     label='平均奖励目标（左轴，示意）')
    ax.set(xlim=(0,1000), ylim=(-205,17), xlabel='训练交互步数（千步）', ylabel='平均奖励目标（示意）')
    ax.tick_params(axis='y', colors='#4169ed')
    ax.yaxis.label.set_color('#4169ed')
    right = ax.twinx()
    line2, = right.plot(xx, PchipInterpolator(xk, entropy)(xx), '--', color='#24282e', lw=1.4,
                       label='策略熵目标（右轴，示意）')
    right.set(ylim=(0,3.2), ylabel='电压策略熵目标 H(u) / nat（示意）')
    ax.set_title('(a) SAC 三阶段收敛：PPT 教学示意，非训练日志', pad=12)
    ax.legend(handles=[*patches, line1, line2], loc='center right', framealpha=.97)
    right.annotate('阶段 III 目标：H ≈ 0.35\n真实训练曲线另附', xy=(810,.35), xytext=(430,1.08),
        fontsize=8, bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#888888'),
        arrowprops=dict(arrowstyle='->', lw=1.2))
    return dict(artifact_type='PPT teaching schematic, not training measurements',
        contains_measured_data=False, source_page=12, anchors_steps_k=xk.tolist(),
        anchors_reward=reward.tolist(), anchors_entropy=entropy.tolist(),
        interpolation='PCHIP; intermediate anchors are illustration choices; no random jitter')


def plot_experiment(output_dir, training_csv=None):
    plt = configure_plotting()
    out = Path(output_dir)
    rows = read_csv(out/'evaluation.csv')
    safety = read_csv(out/'safety.csv')
    result = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    def col(group, key):
        return np.array([float(row[key]) for row in group])
    fresh = result['model_origin'].get('strategy_kind') == 'vanilla_fresh'
    labels = {'pid':'传统 PID', 'nominal_ff':'标称空载模型前馈',
              'sac_fp32':'原生 SAC · FP32' if fresh else '机理引导 SAC · FP32',
              'edge_int8':'本次剪枝 · W8/A16' if fresh else '70% 剪枝 · W8/A16 定点策略'}
    colors = {'pid':'#454545', 'nominal_ff':'#ef8b00', 'sac_fp32':'#3c66ed', 'edge_int8':'#159238'}
    styles = {'pid':'--', 'nominal_ff':':', 'sac_fp32':'-', 'edge_int8':'-.'}
    fig, axes = plt.subplots(2, 2, figsize=(17.8, 9.4))
    fig.subplots_adjust(left=.061, right=.94, top=.865, bottom=.125, wspace=.27, hspace=.37)
    fig.suptitle('第八章 深度强化学习与虚实迁移：SAC、HOCBF 与边缘量化验证', fontsize=15, y=.96)
    fig.text(.5, .913, '重写版：五维观测与物理数字孪生；各图数据来源在图下注明', ha='center', fontsize=9, color='#5e6470')
    for ax in axes.flat:
        ax.grid(True, ls=':', lw=.7, alpha=.42)
        ax.set_axisbelow(True)
    source = draw_ppt_training_schematic(axes[0,0])
    write_json(out/'ppt_schematic_source.json', source)
    ax = axes[0,1]
    ts = col(safety, 'time_s')
    ax.axhspan(100,116, color='#fae1e7', zorder=0)
    ax.axhline(100, color='#ee3b64', ls='--', lw=1.2)
    raw, = ax.plot(ts, col(safety,'x_raw_m')*1000., ':', color='#ef3c69', lw=1.9,
                   label='无安全滤波：首次碰撞后停止记录')
    protected, = ax.plot(ts, col(safety,'x_safe_m')*1000., color='#3c66ed', lw=1.9,
                         label='采样 HOCBF-QP：受保护位移')
    ax.set(xlim=(0,.25), ylim=(15,116), xlabel='时间 t / s', ylabel='活塞位移 / mm')
    right = ax.twinx()
    cmd, = right.plot(ts, col(safety,'u_safe_v'), '-.', color='#278c32', lw=1.5, label='最小干预安全电压')
    right.set(ylim=(-5.5,13.5), ylabel='安全电压 / V')
    right.yaxis.label.set_color('#278c32')
    right.tick_params(axis='y', colors='#278c32')
    ax.set_title('(b) 二阶示教对象：HOCBF-QP 行程防撞', pad=12)
    ax.legend(handles=[raw,protected,cmd], loc='lower right')
    sm = result['safety']
    ax.text(.53,.6, f"首次干预 {sm['first_intervention_s']*1000:.0f} ms\n最低电压 {sm['minimum_safe_voltage_v']:.3f} V\n1 s 位置 {sm['safe_final_position_mm']:.3f} mm",
        transform=ax.transAxes, fontsize=8.3, bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#4169ed'))
    for ax, key, ylabel, title in [
        (axes[1,0], 'velocity_error_mms', '速度误差 v − v_ref / (mm/s)', '(c) 5000 N 负载 + 体积模量减半：真实速度误差'),
        (axes[1,1], 'control_v', '实际阀控制电压 / V', '(d) FP32 与剪枝定点策略：独立闭环电压')]:
        ax.axvspan(.5,1.,color='#eeeeee',zorder=0)
        ax.axvline(.5,color='#777777',ls='--',lw=.9)
        ax.axhline(0. if key=='velocity_error_mms' else 10., color='#888888',ls=':',lw=.8)
        for policy in POLICIES:
            group = [row for row in rows if row['policy']==policy]
            if not group:
                ax.plot([], [], color=colors[policy], ls=styles[policy],
                        label=labels[policy]+'（闭环失败，未画替代曲线）')
                continue
            ax.plot(col(group,'time_s'), col(group,key), color=colors[policy], ls=styles[policy],
                    lw=1.9 if policy=='sac_fp32' else 1.5, label=labels[policy])
        ax.set(xlim=(.4,1.), xlabel='全局时间 t / s', ylabel=ylabel, title=title)
        ax.legend(loc='best',framealpha=.97)
    metrics = result['policies']
    summaries = []
    for policy, label in [('sac_fp32','FP32'),('edge_int8','W8/A16')]:
        m = metrics[policy]
        if m.get('failure'):
            summaries.append(label+'：独立闭环失败，详见metrics.json')
        else:
            recovery = f"{m['recovery_1mms_s']*1000:.0f} ms" if m['recovery_1mms_s'] is not None else '本窗口未达标'
            summaries.append(f"{label}：最低 {m['minimum_velocity_error_mms']:.2f} mm/s；恢复 {recovery}")
    text = '\n'.join(summaries)
    axes[1,0].text(.40,.29,text,transform=axes[1,0].transAxes,fontsize=7.8,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#4169ed'))
    fig.text(.061,.067,'(a) 为课件目标示意，非实测训练；(b) 为独立二阶模型，不能把该证书套用到四状态液压对象。',fontsize=8,color='#555555')
    origin_note = ('(c)(d) 本次原生 SAC 最终权重及新量化策略的独立闭环；无教师初始化、先验损失或训练后actor改写。'
                   if fresh else '(c)(d) 由重写代码重新积分：沿用来源可查的百万步机理引导预训练权重；未添加人工颤振或修改响应。')
    fig.text(.061,.041,origin_note,fontsize=8,color='#555555')
    for ext in ('png','pdf'):
        fig.savefig(out/f'chapter8_comparison.{ext}',facecolor='white')
    plt.close(fig)
    if training_csv is not None:
        plot_real_training(training_csv, out)


def plot_real_training(training_csv, output_dir):
    """绘制真实CSV；旧日志只选完整回合指标，避免与首个不完整窗口混接。"""
    plt = configure_plotting()
    rows = read_csv(training_csv)
    legacy = 'reward_mean' in rows[0]
    if legacy:
        rows = [r for r in rows if r['reward_source']=='completed_episode_per_step_mean']
        reward_key, entropy_key = 'reward_mean', 'voltage_entropy'
        label = '历史机理引导预训练：完整回合平均每步奖励'
        note = '沿用真实历史日志；不含离线教师/Jacobian精修；熵取固定探针状态。排除首个不同统计口径的数据点。'
    else:
        reward_key, entropy_key = 'base_reward_mean', 'replay_voltage_entropy_nat'
        label = '本次原生 SAC：最近至多 10k 步原始教材奖励均值'
        note = '真实环境交互日志；不足10k步时为实际部分窗口；奖励未经优化器缩放；熵来自 replay 状态并含 tanh 雅可比。'
    def get(key):
        return np.array([float(r[key]) for r in rows])
    x = get('timesteps')/1000.
    fig, axes = plt.subplots(1,2,figsize=(13.6,4.7))
    fig.subplots_adjust(left=.075,right=.96,top=.79,bottom=.2,wspace=.25)
    fig.suptitle('真实 SAC 训练结果（与 PPT 教学示意分别保存）',fontsize=14)
    axes[0].plot(x,get(reward_key),color='#4169ed',lw=1.7,label=label)
    if np.nanmin(get(reward_key)) < -1000:
        axes[0].set_yscale('symlog',linthresh=10.)
    axes[0].set(xlabel='实际环境交互步数 / 千步',ylabel='原始平均每步奖励')
    axes[1].plot(x,get(entropy_key),'--',color='#333333',lw=1.5,label='真实电压策略熵 H(u)')
    axes[1].set(xlabel='实际环境交互步数 / 千步',ylabel='H(u) / nat（可为负）')
    for ax in axes:
        ax.grid(True,ls=':',alpha=.4)
        ax.legend(loc='best',fontsize=8)
    fig.text(.075,.07,note,fontsize=8,color='#555555')
    out = Path(output_dir)
    for ext in ('png','pdf'):
        fig.savefig(out/f'sac_training_actual.{ext}',facecolor='white')
    plt.close(fig)


# =============================================================================
# 模块八：独立重放验证与命令行入口
# =============================================================================
def verify_experiment(output_dir, asset_dir):
    """用未被控制器调用的 Radau 求解器重放原始电压，检查数值和来源。"""
    out = Path(output_dir)
    recorded = read_csv(out/'evaluation.csv')
    fp32, edge, provenance = load_assets(asset_dir)
    metadata = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    recorded_backend = metadata.get('evaluation', {}).get('integration_backend', 'auto')
    checks = {}
    for policy in POLICIES:
        rows = [r for r in recorded if r['policy']==policy]
        if not rows:
            checks[policy] = dict(passed=False, reason='No completed closed-loop trajectory was recorded')
            continue
        twin = HydraulicTwinCore(backend='radau')
        twin.reset(x0=np.array([float(rows[0]['position_mm'])/1000.,
            float(rows[0]['velocity_mms'])/1000., float(rows[0]['p1_mpa'])*1e6,
            float(rows[0]['p2_mpa'])*1e6]))
        maximum = np.zeros(4)
        for k, row in enumerate(rows):
            target = np.array([float(row['position_mm'])/1000.,float(row['velocity_mms'])/1000.,
                               float(row['p1_mpa'])*1e6,float(row['p2_mpa'])*1e6])
            maximum = np.maximum(maximum, np.abs(twin.state-target))
            if k == len(rows)-1:
                break
            dt = float(rows[k+1]['time_s'])-float(row['time_s'])
            if not np.isclose(dt,.001,rtol=0,atol=1e-12):
                raise ValueError('重放要求1ms一致时钟')
            u, load, bm = float(row['control_v']), float(row['load_n']), float(row['beta_mpa'])/1000.
            if recorded_backend == 'radau':
                # 原轨迹为Radau时换用另一类独立求解器；不把同一次积分自比当校验。
                from scipy.integrate import solve_ivp
                solution = solve_ivp(lambda t,y: twin.rhs(y,u,load,bm), (0.,.001),
                    twin.state, method='DOP853', rtol=2e-10,
                    atol=np.array([1e-12,1e-10,.01,.01]), max_step=.000125)
                if not solution.success or not np.isfinite(solution.y[:,-1]).all() or np.min(solution.y[2:,-1]) <= 0:
                    raise RuntimeError('独立DOP853重放积分失败')
                twin.state = solution.y[:,-1]
                twin.t += .001
            else:
                twin.step(u, dt=.001, load_n=load, beta_multiplier=bm)
        checks[policy] = dict(maximum_absolute_state_difference=maximum.tolist(),
            units=['m','m/s','Pa','Pa'],
            independent_solver='DOP853' if recorded_backend=='radau' else 'Radau',
            passed=bool(np.all(maximum < [1e-6,1e-5,1000.,1000.])))
    # 对网络策略再做独立闭环积分，防止CSV被手动平滑/换成目标波形。
    for policy, actor in [('sac_fp32',fp32),('edge_int8',edge)]:
        old = [r for r in recorded if r['policy']==policy]
        if not old:
            continue
        try:
            current,_,_ = run_policy(policy,actor,backend=recorded_backend)
        except (ValueError, RuntimeError) as error:
            checks[policy+'_closed_loop_reproduction'] = dict(passed=False, reason=str(error))
            continue
        difference = max(abs(a['velocity_error_mms']-float(b['velocity_error_mms'])) for a,b in zip(current,old))
        checks[policy+'_closed_loop_reproduction'] = dict(max_speed_error_difference_mms=difference,
                                                          passed=difference<1e-5 and len(current)==len(old))
    safe = run_safety_experiment()['summary']
    checks['sampled_hocbf'] = dict(passed=safe['passed'], scope=safe['scope'])
    checks['model_files_match_declared_provenance'] = dict(passed=True)
    result = dict(checks=checks, all_checks_passed=all(x['passed'] for x in checks.values()),
        verification_scope='model hashes, ODE replay with a different solver, closed-loop reproduction and 2nd-order sampled barrier',
        performance_scope='PPT teaching target values are not pass/fail assertions')
    write_json(out/'verification.json',result)
    if not result['all_checks_passed']:
        raise AssertionError('独立重放校验失败，详情见verification.json')
    return result


def validate_parameter_robustness(output_dir, asset_dir, seed=8119, scenarios=12):
    """固定种子留出工况；失败如实记录，不从训练包络推断必然鲁棒。"""
    fp32, edge, _ = load_assets(asset_dir)
    rng = np.random.default_rng(seed)
    results = []
    for scenario in range(scenarios):
        config = dict(m=float(rng.uniform(200.,300.)),
            beta_e=float(rng.uniform(600e6,1400e6)), F_c=float(rng.uniform(840.,1560.)))
        for policy, actor in [('sac_fp32',fp32),('edge_int8',edge)]:
            row = dict(scenario=scenario, policy=policy, **config)
            try:
                trajectory,_,_ = run_policy(policy, actor, twin_config=config)
                row.update(**trajectory_metrics(trajectory), failure=None)
            except (RuntimeError, ValueError) as error:
                row.update(failure=str(error))
            results.append(row)
    summary = dict(seed=seed, scenarios=scenarios,
        scope='Held-out seeded reset-envelope parameters; 5000N load and bulk modulus multiplier0.5; same initial unloaded cruise. Not a general Sim2Real guarantee.',
        results=results)
    write_json(Path(output_dir)/'parameter_robustness.json', summary)
    return summary


def prepare_fresh_assets(run):
    """量化本次新权重；新量化策略必须独立评估，不继承旧策略性能。"""
    root = Path(run['run_dir'])
    assets = root/'assets'
    assets.mkdir(exist_ok=False)
    actor = FloatActor.load(run['actor_fp32_path'])
    actor.save(assets/'actor_fp32.npz')
    with np.load(run['calibration_path'],allow_pickle=False) as data:
        edge = EdgeLightweightExporter.export_integer(actor,data['calibration'],prune_ratio=.70,activation_bits=16)
        error = np.abs(edge(data['validation'])-actor(data['validation']))
    edge.save(assets/'actor_int8.npz')
    files = {name:{'sha256':hashlib.sha256((assets/name).read_bytes()).hexdigest(),
                   'bytes':(assets/name).stat().st_size} for name in ['actor_fp32.npz','actor_int8.npz']}
    write_json(assets/'provenance.json',dict(origin='New vanilla SAC training from random initialization',
        strategy_kind='vanilla_fresh',
        source_commit=None, method='No teacher or post-SAC actor update; newly pruned quantizer requires validation',
        files=files, actual_timesteps=run['metadata']['actual_timesteps'],
        held_out_action_mae=float(error.mean()),held_out_action_max_error=float(error.max()),
        deployment_ready=False))
    return assets


def main():
    import argparse
    parser = argparse.ArgumentParser(description='第八章课堂实验：物理数字孪生、SAC、安全滤波与量化')
    parser.add_argument('--mode',choices=['replay','all','train','safety','plot','verify','robustness'],default='replay')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--assets',type=Path,default=Path(__file__).resolve().parent/'assets')
    parser.add_argument('--timesteps',type=int,default=1_000_000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--backend',choices=['auto','native','radau'],default='auto')
    parser.add_argument('--show',action='store_true',help='生成后使用系统图片查看器打开PNG（需要桌面）')
    args = parser.parse_args()
    if args.mode == 'train':
        run = train_agent(args.output,timesteps=args.timesteps,seed=args.seed)
        run_out = Path(run['run_dir'])/'outputs'
        plot_real_training(run['training_csv'],run_out)
        try:
            assets = prepare_fresh_assets(run)
            evaluate_experiment(run_out,assets,args.backend)
            plot_experiment(run_out,run['training_csv'])
        except (ValueError, RuntimeError) as error:
            write_json(run_out/'closed_loop_failure.json',dict(error=str(error),
                meaning='Training finished; this policy/quantizer did not complete the independent fault test',
                deployment_ready=False))
            print('新权重闭环未通过：'+str(error),flush=True)
        print('真实训练目录：',run['run_dir'])
        return
    if args.mode == 'safety':
        result = run_safety_experiment()
        write_json(args.output/'safety_metrics.json',result['summary'])
        print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
        return
    if args.mode == 'robustness':
        result = validate_parameter_robustness(args.output,args.assets)
        print('已记录参数验证工况数：',len(result['results']))
        return
    if args.mode in ('replay','all'):
        evaluate_experiment(args.output,args.assets,args.backend)
    if args.mode in ('replay','all','plot'):
        training = Path(__file__).resolve().parent/'provenance'/'training_original.csv'
        plot_experiment(args.output,training if training.exists() else None)
    if args.mode in ('replay','all','verify'):
        result = verify_experiment(args.output,args.assets)
        print('独立重放检查：',result['all_checks_passed'])
    if args.show:
        from PIL import Image
        Image.open(args.output/'chapter8_comparison.png').show()
    print('输出目录：',args.output.resolve())


if __name__ == '__main__':
    main()
