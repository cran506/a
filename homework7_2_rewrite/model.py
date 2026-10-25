"""课件四状态局部灰盒：机械力平衡与两腔线性化连续方程。

这是固定工作点的教学近似，不是全孔口非线性数字孪生。Kq/Kc 与容积
在本次扫描中冻结，以隔离预测时域 N 的影响。压力为表压，状态单位
为 [mm, m/s, MPa, MPa]。u 是零偏置标定的等效控制电压，假定工作点
u0=0、可用范围 ±10 V；本模型不辨识真实阀芯的绝对开度。
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


class HydraulicGreyBox:
    def __init__(self, parameters):
        self.p = dict(parameters)
        p = self.p
        if not all(np.isfinite(v) and v > 0 for v in p.values()):
            raise ValueError("Physical parameters must be finite and positive")
        a1, a2, m = p['area1_m2'], p['area2_m2'], p['mass_kg']
        beta, v1, v2 = p['bulk_modulus_pa'], p['volume1_m3'], p['volume2_m3']
        self.equilibrium = np.array([0., 0., p['equilibrium_p1_mpa'],
                                     p['equilibrium_p2_mpa']])
        self.preload_n = 1e6*(a1*self.equilibrium[2]-a2*self.equilibrium[3])
        self.ac = np.array([
            [0., 1000., 0., 0.],
            [0., -p['viscous_ns_m']/m, a1*1e6/m, -a2*1e6/m],
            [0., -beta*a1/v1/1e6, -beta*p['flow_pressure1_m3_s_pa']/v1, 0.],
            [0., beta*a2/v2/1e6, 0., -beta*p['flow_pressure2_m3_s_pa']/v2]])
        self.bc = np.array([0., 0., beta*p['flow_voltage1_m3_s_v']/v1/1e6,
                            -beta*p['flow_voltage2_m3_s_v']/v2/1e6])
        self.cc = -self.ac@self.equilibrium
        self._cache = {}

    def rhs(self, state, voltage):
        return self.ac@np.asarray(state)+self.bc*voltage+self.cc

    def discretize(self, dt):
        """Exact held-input ZOH; avoids Euler error in millisecond pressure modes."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive')
        if dt not in self._cache:
            block = np.zeros((5, 5))
            block[:4, :4], block[:4, 4] = self.ac, self.bc
            transition = expm(block*dt)
            ad, bd = transition[:4, :4], transition[:4, 4]
            cd = self.equilibrium-ad@self.equilibrium
            self._cache[dt] = ad, bd, cd
        return tuple(a.copy() for a in self._cache[dt])

    def step(self, state, voltage, dt):
        ad, bd, cd = self.discretize(dt)
        return ad@state+bd*voltage+cd

    def linearize(self, state=None, voltage=None):
        """This fixed-working-point model has constant Jacobians (LTI special case)."""
        return self.ac.copy(), self.bc.copy(), self.cc.copy()

    def pressure_extrema(self, state, voltage, dt):
        """Find continuous intersample pressure extrema from the exact trajectory.

        Bracket roots of dP/dt on 32 intervals, then use Brent refinement.
        Endpoint-only constraints alone do not imply continuous pressure safety.
        """
        from scipy.optimize import brentq
        grid = np.linspace(0., dt, 33)
        stamps = set(grid.tolist())
        def at(t):
            return np.asarray(state) if t == 0. else self.step(state, voltage, float(t))
        for pressure in (2, 3):
            slopes = [self.rhs(at(t), voltage)[pressure] for t in grid]
            for left, right, fl, fr in zip(grid[:-1], grid[1:], slopes[:-1], slopes[1:]):
                if fl*fr < 0:
                    stamps.add(float(brentq(lambda t: self.rhs(at(t), voltage)[pressure],
                                            left, right, xtol=1e-13)))
        stamps = np.array(sorted(stamps))
        states = np.array([at(t) for t in stamps])
        return stamps, states
