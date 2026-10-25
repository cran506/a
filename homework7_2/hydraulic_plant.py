"""真实混合液压对象与 MPC 灰箱预测模型。

公开状态顺序为 [相对位置 mm, 速度 m/s, P1 MPa, P2 MPa, 阀位等效 V]。
真实对象保留事件驱动静摩擦、有向孔口与硬死区。预测模型为公开的光滑
近似：近零滑动摩擦、压差平方根、补油及阀死区作正则化；它不声称能精确
预测静摩擦释放时刻。rhs、数值 Jacobian 和 rollout 使用同一个预测 RHS。
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from twin_adapter import (ClosedLoopTwin, auxiliary_circuit_settings,
                          physical_config, pilot_unload_gate, return_bypass_gate)
from media_model import AeratedHydraulicModel


STATE_UNITS = ('mm', 'm/s', 'MPa', 'MPa', 'V')
_COMPACT = dict(L=.3, V0_1=.0001, V0_2=.0001, beta_e=1.8e9,
                xv_max=.005, air_fraction_at_pr=0.)


def _parameters(config):
    p = physical_config()
    path = Path(__file__).with_name('classroom_config.json')
    p.update(json.loads(path.read_text()) if path.exists() else _COMPACT)
    if config is not None:
        p.update(dict(config))
    p.setdefault('C_ep', 0.)
    p.setdefault('ambient_pressure', 1e5)
    # 使用显式时窗外载，避免上传开环例子的工件重复叠加。
    p['x_work'] = p['L']+1.
    return p


def _state(z):
    a = np.asarray(z, dtype=float)
    if a.shape != (5,) or not np.all(np.isfinite(a)):
        raise ValueError('state 必须为有限的五维教育单位向量')
    return a


def _voltage(u):
    a = np.asarray(u, dtype=float)
    if a.size != 1 or not np.all(np.isfinite(a)):
        raise ValueError('控制电压必须为一个有限标量')
    return float(a.reshape(-1)[0])


class HydraulicPlant:
    """同一个真实 hybrid twin；压力轨迹诊断保留 Pa、s 单位。"""
    state_units = STATE_UNITS

    def __init__(self, config=None, origin_m=.05, valve_tau=.0025,
                 makeup_area=.002, rtol=1e-6):
        self.origin_m = float(origin_m)
        p = _parameters(config)
        if not 0 < self.origin_m < p['L']:
            raise ValueError('物理原点必须位于缸行程内')
        self.twin = ClosedLoopTwin(
            config=p, load_force=p.get('load_force', 0.),
            load_window=tuple(p.get('load_window', (.35, .70))),
            load_ramp=p.get('load_ramp', 0.),
            unload_ramp=p.get('unload_ramp', p.get('load_ramp', 0.)),
            valve_tau=valve_tau, makeup_area=makeup_area, rtol=rtol,
            ambient_pressure=p['ambient_pressure'])
        self.reset()

    @property
    def parameters(self):
        return dict(self.twin.p, makeup_area=self.twin.makeup_area,
                    valve_tau=self.twin.valve_tau,
                    ambient_pressure=self.twin.ambient_pressure)

    @property
    def fullparams(self):
        return self.parameters

    full_params = fullparams

    @property
    def state(self):
        x, v, p1, p2 = self.twin.state
        return np.array([(x-self.origin_m)*1000., v, p1/1e6, p2/1e6,
                         10.*self.twin.valve_state])

    @property
    def t(self):
        return self.twin.t

    time_s = t

    @property
    def nfev(self):
        return self.twin.nfev

    @property
    def friction_events(self):
        return [e.copy() for e in self.twin.friction_events]

    @property
    def pressure_trace(self):
        return np.asarray(getattr(self.twin, 'pressure_trace', []), dtype=float).reshape(-1, 3).copy()

    @property
    def last_pressure_trace(self):
        return np.asarray(getattr(self.twin, 'last_pressure_trace', []), dtype=float).reshape(-1, 3).copy()

    @property
    def last_interval_peak(self):
        return dict(getattr(self.twin, 'last_interval_peak', {}))

    @property
    def internal_state_trace(self):
        """真实积分轨迹 [t_s, x_m, v_m_s, P1_Pa, P2_Pa, 阀位归一化]。"""
        return np.asarray(getattr(self.twin, 'internal_state_trace', []), dtype=float).reshape(-1, 6).copy()

    def reset(self, state=None):
        p = self.twin.p
        z = (np.array([0., 0., p['Pr']/1e6, p['Pr']/1e6, 0.])
             if state is None else _state(state))
        if min(z[2:4]) <= 0 or abs(z[4]) > 10:
            raise ValueError('初始绝对压力必须正，阀位等效电压须在 ±10 V 内')
        self.twin.reset([self.origin_m+z[0]/1000., z[1], z[2]*1e6, z[3]*1e6],
                        valve_state=z[4]/10.)
        return self.state

    def step(self, voltage, dt):
        self.twin.step_voltage(_voltage(voltage), dt)
        return self.state


class GreyBoxModel:
    """用于 RTI 的光滑滑动预测，真实闭环验证必须使用 HydraulicPlant。

    缺省正则化：压差 1 kPa，阀死区 0.1 V，方向混合 0.05 V，
    摩擦速度 0.002 m/s。死区中心有小非零预测增益；这是优化近似，
    不是修改真实阀死区。压力从不裁剪；积分失败或非正压力显式报错。
    """
    state_units = STATE_UNITS

    def __init__(self, config=None, origin_m=.05, valve_tau=.0025,
                 makeup_area=.002, rtol=1e-6):
        self.p = _parameters(config)
        self.origin_m, self.valve_tau = float(origin_m), float(valve_tau)
        self.makeup_area, self.rtol = float(makeup_area), float(rtol)
        if not 0 < self.origin_m < self.p['L'] or self.valve_tau <= 0 or self.makeup_area < 0 or self.rtol <= 0:
            raise ValueError('预测原点、阀滞后、补油面积或精度无效')
        self.model = AeratedHydraulicModel(self.p)
        self.auxiliary_circuit = auxiliary_circuit_settings(self.p)
        self.pressure_epsilon_pa = float(self.p.get('prediction_pressure_epsilon_pa', 1000.))
        self.deadzone_epsilon_v = float(self.p.get('prediction_deadzone_epsilon_v', .1))
        self.direction_epsilon_v = float(self.p.get('prediction_direction_epsilon_v', .05))
        self.friction_epsilon_m_s = float(self.p.get('prediction_friction_epsilon_m_s', .002))
        if min(self.pressure_epsilon_pa, self.deadzone_epsilon_v,
               self.direction_epsilon_v, self.friction_epsilon_m_s) <= 0:
            raise ValueError('预测正则化尺度必须正')
        self.nfev = 0

    @property
    def parameters(self):
        return dict(self.p, makeup_area=self.makeup_area, valve_tau=self.valve_tau)

    @property
    def fullparams(self):
        return self.parameters

    full_params = fullparams

    @property
    def prediction_settings(self):
        return dict(pressure_epsilon_pa=self.pressure_epsilon_pa,
                    deadzone_epsilon_v=self.deadzone_epsilon_v,
                    direction_epsilon_v=self.direction_epsilon_v,
                    friction_epsilon_m_s=self.friction_epsilon_m_s,
                    friction_mode='smooth_sliding',
                    deadzone_mode='odd_softplus_with_smooth_port_direction',
                    makeup_mode='one_sided_C1', pressure_clipping=False)

    def _phi(self, delta):
        return math.sqrt(2/self.p['rho'])*delta/(delta*delta+self.pressure_epsilon_pa**2)**.25

    def _makeup(self, pressure):
        # C1 单侧补油近似：P>=Pr 精确为零，有限过渡避免 sqrt 的无界导数。
        deficit = max(self.p['Pr']-pressure, 0.)
        return (self.p['Cd']*self.makeup_area*math.sqrt(2/self.p['rho'])*
                deficit*deficit/(deficit*deficit+self.pressure_epsilon_pa**2)**.75)

    def rhs(self, z, u):
        z, u = _state(z), _voltage(u)
        x = self.origin_m+z[0]/1000.
        v, p1, p2, valve = z[1], z[2]*1e6, z[3]*1e6, z[4]
        p = self.p
        v1, v2 = p['V0_1']+p['A1']*x, p['V0_2']+p['A2']*(p['L']-x)
        if min(v1, v2) <= 0:
            raise ValueError('预测液压腔容积非正')
        eps, dead = self.deadzone_epsilon_v, p['deadzone']*10.
        effective = eps*(np.logaddexp(0., (valve-dead)/eps)
                         -np.logaddexp(0., (-valve-dead)/eps))
        direction = math.tanh(valve/self.direction_epsilon_v)
        # Keep the auxiliary positive-spool branch exactly closed at neutral;
        # the ordinary odd softplus deadzone otherwise leaves a tiny positive
        # softplus tail that would create fictitious zero-command supply flow.
        positive_effective = effective*.5*(1.+direction)
        k = p['Cd']*p['w']*(p['xv_max']/10.)
        q = []
        for pi, forward, reverse in [(p1, p['Ps'], p['Pr']), (p2, p['Pr'], p['Ps'])]:
            ff, fr = self._phi(forward-pi), self._phi(reverse-pi)
            q.append(k*effective*.5*((ff-fr)+direction*(ff+fr))+self._makeup(pi))
        auxiliary = self.auxiliary_circuit
        if auxiliary['supply_gain'] != 1.:
            q[0] += ((auxiliary['supply_gain']-1.)
                     *k*positive_effective*self._phi(p['Ps']-p1))
        if auxiliary['return_gain'] > 1.:
            q[1] += ((auxiliary['return_gain']-1.)
                     *return_bypass_gate(valve, auxiliary)
                     *k*positive_effective*self._phi(p['Pr']-p2))
        if auxiliary['pilot_conductance']:
            pilot = auxiliary['pilot_conductance']*pilot_unload_gate(
                valve, v, auxiliary)
            q[0] += pilot*(auxiliary['p1_target']-p1)
            q[1] += pilot*(auxiliary['p2_target']-p2)
        leak = p['C_ip']*(p1-p2)
        dp1 = self.model._get_beta(p1)/v1*(q[0]-p['A1']*v-leak-p['C_ep']*p1)
        dp2 = self.model._get_beta(p2)/v2*(q[1]+p['A2']*v+leak-p['C_ep']*p2)
        friction = ((p['Fc']+(p['Fs']-p['Fc'])*math.exp(-(v/p['vs'])**2))*
                    math.tanh(v/self.friction_epsilon_m_s)+p['sigma']*v)
        force = (p1*p['A1']-p2*p['A2']-p['ambient_pressure']*(p['A1']-p['A2'])
                 -friction-self.model._env_force(x, v)-p.get('prediction_external_force_n', 0.))
        return np.array([1000.*v, force/p['mass'], dp1/1e6, dp2/1e6,
                         (u-valve)/self.valve_tau])

    def continuous_linearization(self, z, u):
        """同一光滑 RHS 的中心差分；各列按教育单位选取扰动。"""
        z, u = _state(z), _voltage(u)
        steps = np.maximum(np.abs(z)*1e-5, np.array([1e-4, 1e-6, 1e-5, 1e-5, 1e-5]))
        a = np.empty((5, 5))
        for j, h in enumerate(steps):
            perturb = np.zeros(5)
            perturb[j] = h
            a[:, j] = (self.rhs(z+perturb, u)-self.rhs(z-perturb, u))/(2*h)
        b = np.zeros((5, 1))
        b[4, 0] = 1/self.valve_tau
        c = self.rhs(z, u)-a@z-b[:, 0]*u
        return a, b, c

    def linearize_discrete(self, z, u, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('离散周期必须为正')
        a, b, c = self.continuous_linearization(z, u)
        augmented = np.zeros((7, 7))
        augmented[:5, :5], augmented[:5, 5:6], augmented[:5, 6] = a, b, c
        transition = expm(augmented*dt)
        return transition[:5, :5], transition[:5, 5:6], transition[:5, 6]

    def rollout(self, z, u_sequence, dt):
        """未来 N 个采样终点，不包含初态；ZOH 下 LSODA 自适应积分。"""
        z = _state(z).copy()
        commands = np.asarray(u_sequence, dtype=float)
        if commands.ndim == 2 and commands.shape[1:] == (1,):
            commands = commands[:, 0]
        if commands.ndim != 1 or not np.all(np.isfinite(commands)) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('预测指令须为有限一维序列，dt 必须正')
        out = np.empty((len(commands), 5))
        for i, command in enumerate(commands):
            sol = solve_ivp(lambda t, y: self.rhs(y, command), (0., dt), z,
                            method='LSODA', rtol=self.rtol,
                            atol=[1e-6, 1e-8, 1e-6, 1e-6, 1e-7])
            self.nfev += sol.nfev
            if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
                raise RuntimeError(f'预测积分在stage {i}失败：{sol.message}')
            z = sol.y[:, -1]
            if min(z[2:4]) <= 0:
                raise RuntimeError(f'预测stage {i}绝对压力非正；未进行压力裁剪')
            out[i] = z
        return out

    def get_jacobian_coeffs(self, z, u=None):
        """真实硬阀计量口的当前分支 Kq/Kc，非正则化预测导数。

        Kq 是潜在 dQ/dxv [m²/s]；Kc 是阀口 -dQ/dP [m³/(s·Pa)]。
        Kq_voltage 在硬死区内为零；压差为零且开阀时 Kc=inf。
        双腔 pilot 调压阀的导压系数另列为 Kc*_pilot，不并入主阀口。
        """
        z = _state(z)
        p, valve = self.p, z[4]
        direction = 1 if valve >= 0 else -1
        sources = (p['Ps'], p['Pr']) if direction > 0 else (p['Pr'], p['Ps'])
        active = abs(valve) > 10*p['deadzone']
        xv = p['xv_max']*max(abs(valve)/10-p['deadzone'], 0.)
        result = dict(deadzone_active=not active, valve_direction=direction)
        auxiliary = self.auxiliary_circuit
        if direction > 0:
            bypass = return_bypass_gate(valve, auxiliary)
            return_gain = (1.+(auxiliary['return_gain']-1.)*bypass)
            # The pilot gain itself changes as the main spool moves. Include
            # this derivative in the flow/spool coefficient of the bypass port.
            return_kq_gain = (return_gain+(auxiliary['return_gain']-1.)*xv
                              *bypass*(1.-bypass)*10./(
                                  p['xv_max']*auxiliary['return_width_v']))
            gains = ((auxiliary['supply_gain'], auxiliary['supply_gain']),
                     (return_kq_gain, return_gain))
        else:
            return_gain = 1.
            gains = ((1., 1.), (1., 1.))
        for i, source in enumerate(sources, 1):
            delta = source-z[i+1]*1e6
            phi = np.sign(delta)*math.sqrt(2*abs(delta)/p['rho'])
            kq_gain, kc_gain = gains[i-1]
            kq = direction*p['Cd']*p['w']*phi*kq_gain
            kc = (0. if xv == 0 else kc_gain*p['Cd']*p['w']*xv/
                  math.sqrt(2*p['rho']*abs(delta)) if delta != 0 else math.inf)
            result[f'Kq{i}_m2_s'] = float(kq)
            result[f'Kq{i}_voltage_m3_s_v'] = float(kq*p['xv_max']/10) if active else 0.
            result[f'Kc{i}_m3_s_pa'] = float(kc)
        pilot = (auxiliary['pilot_conductance']*pilot_unload_gate(
            valve, z[1], auxiliary) if auxiliary['pilot_conductance'] else 0.)
        result.update(extension_supply_gain=auxiliary['supply_gain'],
                      extension_return_gain=return_gain,
                      Kc1_pilot_m3_s_pa=pilot, Kc2_pilot_m3_s_pa=pilot,
                      Kc1_total_m3_s_pa=result['Kc1_m3_s_pa']+pilot,
                      Kc2_total_m3_s_pa=result['Kc2_m3_s_pa']+pilot)
        result.update(Kq=result['Kq1_m2_s'], Kc=result['Kc1_m3_s_pa'],
                      Kq_m2_s=result['Kq1_m2_s'], Kc_m3_s_pa=result['Kc1_m3_s_pa'])
        return result
