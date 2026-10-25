"""原始 HydraulicTwin 的闭环适配：有向节流、补油、阀动态与工艺负载。"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from media_model import AeratedHydraulicModel
from friction_model import sliding_friction, validate_friction


def auxiliary_circuit_settings(p):
    """Optional classroom return bypass and two-chamber pressure regulator.

    Zero regulator conductance and unit return gain recover the original
    closed-center four-way valve exactly.  The regulator has its own pilot
    valves and pressure supplies: it is an explicit alternative circuit, not
    a pressure reset or a change to the measured output.
    """
    p2_target = float(p.get('pilot_p2_target_pa', p.get('initial_p2_pa', 2e6)))
    ambient = float(p.get('ambient_pressure', 1e5))
    p1_target = float(p.get('pilot_p1_target_pa',
                          (p2_target*p['A2']+ambient*(p['A1']-p['A2']))/p['A1']))
    settings = dict(
        supply_gain=float(p.get('extension_supply_gain', 1.)),
        return_gain=float(p.get('extension_return_gain', 1.)),
        return_open_v=float(p.get('extension_return_open_v', 2.5)),
        return_width_v=float(p.get('extension_return_width_v', .25)),
        pilot_conductance=float(p.get('pilot_unload_conductance', 0.)),
        pilot_valve_scale_v=float(p.get('pilot_unload_valve_scale_v', .65)),
        pilot_speed_scale_m_s=float(p.get('pilot_unload_speed_scale_m_s', .08)),
        p1_target=p1_target, p2_target=p2_target)
    values = tuple(settings.values())
    if (not all(math.isfinite(value) for value in values)
            or settings['supply_gain'] <= 0. or settings['return_gain'] < 1.
            or min(settings['return_width_v'], settings['pilot_valve_scale_v'],
                   settings['pilot_speed_scale_m_s'], p1_target, p2_target) <= 0.
            or settings['pilot_conductance'] < 0.):
        raise ValueError('课堂回油旁路或压力调节回路参数无效')
    return settings


def return_bypass_gate(valve_v, settings):
    """Smooth pilot opening for the extension return bypass (voltage units)."""
    argument = (valve_v-settings['return_open_v'])/settings['return_width_v']
    if argument >= 0.:
        return 1./(1.+math.exp(-argument))
    value = math.exp(argument)
    return value/(1.+value)


def pilot_unload_gate(valve_v, velocity, settings):
    """Open the auxiliary pressure regulators only near neutral and rest."""
    return (1./(1.+(valve_v/settings['pilot_valve_scale_v'])**8)
            /(1.+(velocity/settings['pilot_speed_scale_m_s'])**4))


def physical_config():
    """上传 twin 的原始示例参数，未改成课堂整定值。"""
    return json.loads(Path(__file__).with_name('physical_config.json').read_text())


def classroom_config():
    """公开的课堂阀配置；不是第四章辨识结果。"""
    config = physical_config()
    config.update(json.loads(Path(__file__).with_name('classroom_config.json').read_text()))
    return config


class QuantizedSensor:
    """可配置的量化 ADC，LSB=量程/2**bits；第六章使用16bit。"""
    def __init__(self, bits=12, span=.3):
        if bits < 1 or bits > 24 or span <= 0:
            raise ValueError('ADC 参数无效')
        self.bits, self.span = bits, span
        self.levels = 2**bits
        self.lsb = span/self.levels

    def sample(self, position):
        code = int(np.clip(np.rint(position/self.lsb), 0, self.levels-1))
        return code*self.lsb


class ClosedLoopTwin(AeratedHydraulicModel):
    """四阶液压状态 + 一阶阀芯；1ms 周期内控制和外载保持不变。

    保留原 twin 的摩擦参数、有效模量、泄漏和端盖模型；将 tanh 摩擦
    换为有真实静摩擦区间的混合粘滑模型。节流使用有向压差，低压腔
    通过油箱补油单向阀吸油；不裁剪积分后的压力或输出位移。

    释放阈值的 1e-6 N 容差仅用于避免浮点根反复触发；零速仅在事件
    求根定位后置为精确零，不按采样周期截断速度或替换输出轨迹。
    零速释放后的 event 首点使用右侧符号，避免一次积分首步跨过
    极短滑动后将同一个初始零根重新认作停转；该保护只改事件值，
    不给实际速度添加种子、不跳过物理积分时间。
    """
    friction_force_tolerance = 1e-6
    friction_event_sign_value = 1e-10  # event 的 0+ 符号代表值，单位 m/s
    pressure_trace_subdivisions = 32
    max_friction_events_per_step = 64

    def __init__(self, config=None, load_force=180000., load_window=(.35, .70),
                 valve_tau=.0053, rtol=1e-5, load_ramp=0., makeup_area=.001,
                 ambient_pressure=1e5, unload_ramp=None):
        config = physical_config() if config is None else config.copy()
        config['x_work'] = config['L']+1.
        validate_friction(config)
        super().__init__(config)
        self.auxiliary_circuit = auxiliary_circuit_settings(config)
        unload_ramp = load_ramp if unload_ramp is None else unload_ramp
        values = (valve_tau, load_force, rtol, load_ramp, unload_ramp, makeup_area,
                  ambient_pressure, *load_window)
        if (not all(np.isfinite(v) for v in values) or valve_tau < 0
                or not 0 <= load_window[0] < load_window[1] or load_force < 0
                or rtol <= 0 or load_ramp < 0 or unload_ramp < 0
                or load_ramp+unload_ramp > load_window[1]-load_window[0]
                or makeup_area < 0 or ambient_pressure < 0 or config['Pr'] <= 0):
            raise ValueError('阀、负载、补油或压力参数无效')
        self.load_force, self.load_window = load_force, load_window
        self.load_ramp, self.unload_ramp = load_ramp, unload_ramp
        self.makeup_area = makeup_area
        self.ambient_pressure = ambient_pressure
        self.valve_tau, self.rtol = valve_tau, rtol
        self.valve_state = self.active_force = 0.
        self.nfev = 0

    def reset(self, x0=None, valve_state=0.):
        if not np.isfinite(valve_state) or abs(valve_state) > 1.:
            raise ValueError('初始归一化阀位必须在 [-1,1] 内')
        state = super().reset(x0)
        self.valve_state, self.active_force = float(valve_state), 0.
        self.nfev = 0
        self.sticking = bool(state[1] == 0.)
        self.slip_direction = 0 if self.sticking else int(np.sign(state[1]))
        self.friction_events = []
        self.release_count = self.restick_count = 0
        self._internal_state_trace_rows = [(0., *map(float, state), self.valve_state)]
        self._interval_state_trace_rows = []
        self.pressure_trace_nfev = 0
        self.last_pressure_trace = np.array([[0., state[2], state[3]]])
        self.last_interval_peak = dict(p1_pa=float(state[2]), p2_pa=float(state[3]),
                                       p1_time_s=0., p2_time_s=0.,
                                       p1_min_pa=float(state[2]), p2_min_pa=float(state[3]))
        return state

    @property
    def internal_state_trace(self):
        """真实积分轨迹：[t_s,x_m,v_m_s,P1_pa,P2_pa,阀归一化位移]。

        包含 Radau 接受节点、每采样区间最多 1/32 周期的密输出节点，
        以及压力导数变号括号内的极值根；不是重新拟合的输出曲线。
        """
        return np.asarray(self._internal_state_trace_rows, dtype=float)

    @property
    def pressure_trace(self):
        """自 reset 起的采样间压力轨迹，列为 t_s、P1_pa、P2_pa。"""
        return self.internal_state_trace[:, [0, 3, 4]]

    def _record_pressure_segment(self, solution, rhs, dt):
        """在同一真实 Radau 密输出上定位压力极值，不修改积分状态。"""
        start, end = float(solution.t[0]), float(solution.t[-1])
        count = max(1, int(np.ceil((end-start)/(dt/self.pressure_trace_subdivisions))))
        times = np.unique(np.r_[solution.t, np.linspace(start, end, count+1)])
        states = solution.sol(times).T
        derivatives = []
        for time, state in zip(times, states):
            derivatives.append(rhs(time, state)[2:4])
        self.pressure_trace_nfev += len(times)
        derivatives = np.asarray(derivatives)
        roots = []
        for chamber in range(2):
            for index in range(len(times)-1):
                left, right = derivatives[index, chamber], derivatives[index+1, chamber]
                if left*right < 0.:
                    def pressure_rate(time):
                        self.pressure_trace_nfev += 1
                        return rhs(time, solution.sol(time))[2+chamber]
                    roots.append(brentq(pressure_rate, times[index], times[index+1],
                                        xtol=1e-12, rtol=4*np.finfo(float).eps))
        if roots:
            times = np.unique(np.r_[times, roots])
            states = solution.sol(times).T
        for time, state in zip(times, states):
            valve = float(state[4]) if self.valve_tau else self.valve_state
            row = (float(time), *map(float, state[:4]), valve)
            self._interval_state_trace_rows.append(row)
            self._internal_state_trace_rows.append(row)
        if min(states[:, 2:4].min(axis=0)) <= 0:
            raise RuntimeError(f'真实采样间绝对压力非正 t=[{start:.9f},{end:.9f}]s')

    def _finish_pressure_interval(self):
        rows = np.asarray(self._interval_state_trace_rows, dtype=float)
        self.last_pressure_trace = rows[:, [0, 3, 4]]
        i1, i2 = np.argmax(rows[:, 3]), np.argmax(rows[:, 4])
        self.last_interval_peak = dict(p1_pa=float(rows[i1, 3]), p2_pa=float(rows[i2, 4]),
                                       p1_time_s=float(rows[i1, 0]), p2_time_s=float(rows[i2, 0]),
                                       p1_min_pa=float(rows[:, 3].min()),
                                       p2_min_pa=float(rows[:, 4].min()))

    def _friction(self, velocity):
        """任意非零速度的滑动摩擦；静止力平衡由显式 stick 分支处理。"""
        return sliding_friction(velocity, self.p)

    def net_drive_force(self, state=None, external_force=None):
        """除摩擦外的轴向合力，包含绝对压力的大气修正和外载，单位 N。"""
        xp, velocity, p1, p2 = self.state if state is None else state
        force = self.active_force if external_force is None else external_force
        p = self.p
        return float(p1*p['A1']-p2*p['A2']
                     -self.ambient_pressure*(p['A1']-p['A2'])
                     -self._env_force(xp, velocity)-force)

    @property
    def friction_force(self):
        """当前模式实际摩擦：stick 时平衡合力，slide 时用有向分支。"""
        if self.sticking:
            return self.net_drive_force()
        return sliding_friction(self.state[1], self.p, self.slip_direction)

    def friction_diagnostics(self, external_force=None):
        """采样时刻诊断；默认使用此刻负载，ODE 内仍使用 interval-held 负载。

        release_pending 标出新负载在采样边界已越过阈值、下一 step 即释放。
        """
        force = self.load_at(self.t) if external_force is None else external_force
        net = self.net_drive_force(external_force=force)
        pending = self.sticking and abs(net) > self.p['Fs']+self.friction_force_tolerance
        friction = (float(np.clip(net, -self.p['Fs'], self.p['Fs'])) if self.sticking else
                    sliding_friction(self.state[1], self.p, self.slip_direction))
        return dict(mode='stick' if self.sticking else 'slip', sticking=self.sticking,
                    direction=self.slip_direction, slip_direction=self.slip_direction,
                    net_drive_n=net, net_drive_force_n=net,
                    friction_n=friction, friction_force_n=friction,
                    release_pending=pending, release_count=self.release_count,
                    restick_count=self.restick_count, event_count=len(self.friction_events))

    def _friction_event(self, time, state, kind, direction):
        net = self.net_drive_force(state)
        self.friction_events.append(dict(time_s=float(time), kind=kind,
                                        direction=int(direction), net_drive_n=net,
                                        friction_n=(net if kind == 'restick' else
                                                    sliding_friction(0., self.p, direction))))
        if kind == 'restick':
            self.restick_count += 1
        else:
            self.release_count += 1

    def load_at(self, t):
        start, end = self.load_window
        if not start-1e-10 <= t < end-1e-10:
            return 0.
        rise = (t-start)/self.load_ramp if self.load_ramp else 1.
        fall = (end-t)/self.unload_ramp if self.unload_ramp else 1.
        fraction = np.clip(min(rise, fall, 1.), 0., 1.)
        return self.load_force*.5*(1-np.cos(np.pi*fraction))

    def _dynamics(self, t, x, u):
        """独立数学 RHS；按输入状态判定静止力平衡，不改写混合模式。

        实际 step 的模式切换仍由终端事件处理，此接口可独立校验数学式。
        """
        if x[1] == 0.:
            net = self.net_drive_force(x)
            if abs(net) <= self.p['Fs']:
                return self._state_derivative(t, x, u, True, 0)
            return self._state_derivative(t, x, u, False, int(np.sign(net)))
        return self._state_derivative(t, x, u, False, 0)

    def _state_derivative(self, t, x, u, sticking, direction):
        xp, velocity, p1, p2 = x
        if sticking:
            velocity = 0.
        p = self.p
        v1, v2 = p['V0_1']+p['A1']*xp, p['V0_2']+p['A2']*(p['L']-xp)
        if min(v1, v2) <= 0:
            raise RuntimeError('液压腔容积非正，停止仿真')
        ue = np.sign(u)*max(abs(u)-p['deadzone'], 0.)
        area = p['Cd']*p['w']*abs(ue*p['xv_max'])
        delta1, delta2 = ((p['Ps']-p1, p['Pr']-p2) if ue >= 0
                          else (p['Pr']-p1, p['Ps']-p2))
        q1 = area*np.sign(delta1)*np.sqrt(2/p['rho']*abs(delta1))
        q2 = area*np.sign(delta2)*np.sqrt(2/p['rho']*abs(delta2))
        auxiliary = self.auxiliary_circuit
        if ue > 0. and auxiliary['supply_gain'] != 1.:
            q1 *= auxiliary['supply_gain']
        if ue > 0. and auxiliary['return_gain'] > 1.:
            q2 += (auxiliary['return_gain']-1.)*return_bypass_gate(10.*u, auxiliary)*(
                area*np.sign(p['Pr']-p2)*np.sqrt(2/p['rho']*abs(p['Pr']-p2)))
        # 双向缸的两腔各接油箱补油单向阀；仅 P<Pr 时有流入。
        q1 += p['Cd']*self.makeup_area*np.sqrt(2/p['rho']*max(p['Pr']-p1, 0.))
        q2 += p['Cd']*self.makeup_area*np.sqrt(2/p['rho']*max(p['Pr']-p2, 0.))
        if auxiliary['pilot_conductance']:
            pilot = auxiliary['pilot_conductance']*pilot_unload_gate(
                10.*u, velocity, auxiliary)
            q1 += pilot*(auxiliary['p1_target']-p1)
            q2 += pilot*(auxiliary['p2_target']-p2)
        leak = p['C_ip']*(p1-p2)
        dp1 = self._get_beta(p1)/v1*(q1-p['A1']*velocity-leak-p['C_ep']*p1)
        dp2 = self._get_beta(p2)/v2*(q2+p['A2']*velocity+leak-p['C_ep']*p2)
        drive = self.net_drive_force([xp, velocity, p1, p2])
        dv = 0. if sticking else (drive-sliding_friction(velocity, p, direction))/p['mass']
        return np.array([velocity, dv, dp1, dp2])

    def step_voltage(self, voltage, dt):
        if not np.isfinite(voltage) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('电压必须有限，dt必须为正且有限')
        command = float(np.clip(voltage, -10., 10.))/10.
        self.active_force = self.load_at(self.t)
        if self.valve_tau == 0:
            self.valve_state = command
        y = (np.r_[self.state, self.valve_state] if self.valve_tau else self.state.copy())
        atol = [1e-9, 1e-10, .1, .1]+([1e-9] if self.valve_tau else [])
        time, end_time = self.t, self.t+dt
        switches = 0
        self._interval_state_trace_rows = []
        while time < end_time:
            net = self.net_drive_force(y[:4])
            threshold = self.p['Fs']+self.friction_force_tolerance
            # A sampled external load may release the piston exactly at a ZOH boundary.
            if self.sticking and abs(net) > threshold:
                self.sticking = False
                self.slip_direction = int(np.sign(net))
                self._friction_event(time, y[:4], 'release', self.slip_direction)
                switches += 1
            sticking, direction = self.sticking, self.slip_direction

            def fun(t, state):
                valve = state[4] if self.valve_tau else command
                rhs = self._state_derivative(t, state[:4], valve, sticking, direction)
                return np.r_[rhs, (command-valve)/self.valve_tau] if self.valve_tau else rhs

            if sticking:
                def event(t, state):
                    return abs(self.net_drive_force(state[:4]))-threshold
                event.direction = 1
            else:
                segment_start = time
                starts_at_zero = y[1] == 0.
                def event(t, state):
                    # With v(t0)=0, g0=0 and g1 of the opposite sign lets
                    # solve_ivp select t0 even if v first departed in the
                    # physical release direction and returned later in this
                    # first solver step. Represent only g(t0)'s right-side
                    # sign; all t>t0 use the actual dense-output velocity.
                    if starts_at_zero and t == segment_start:
                        return direction*self.friction_event_sign_value
                    return state[1]
                event.direction = -direction
            event.terminal = True
            sol = solve_ivp(fun, (time, end_time), y, events=event, method='Radau',
                            rtol=self.rtol, atol=atol, dense_output=True)
            if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
                raise RuntimeError(f'Radau 在 t={time:.6f}s 失败：{sol.message}')
            self.nfev += sol.nfev
            self._record_pressure_segment(sol, fun, dt)
            y, time = sol.y[:, -1], float(sol.t[-1])
            if not sol.t_events[0].size:
                break
            switches += 1
            # Set velocity to zero only at the velocity/release root, never per sample.
            y[1] = 0.
            net = self.net_drive_force(y[:4])
            if sticking:
                self.sticking = False
                self.slip_direction = int(np.sign(net))
                self._friction_event(time, y[:4], 'release', self.slip_direction)
            elif abs(net) <= self.p['Fs']:
                self.sticking = True
                self.slip_direction = 0
                self._friction_event(time, y[:4], 'restick', 0)
            else:
                self.slip_direction = int(np.sign(net))
                self._friction_event(time, y[:4], 'reverse_release', self.slip_direction)
            if switches > self.max_friction_events_per_step:
                raise RuntimeError(f'粘滑事件过密 t={time:.9f}s；请检查工况和积分精度')
        self.state = y[:4].copy()
        if self.valve_tau:
            self.valve_state = float(y[4])
        self.t = end_time
        self._finish_pressure_interval()
        if min(self.state[2:]) <= 0:
            raise RuntimeError(f'绝对压力非正 t={self.t:.6f}s，请检查负载或模型')
        return self.state.copy()

    def extension_kq(self):
        """按 P1 计算伸出支路 dQ1/dxv；反向阀位时只是潜在增益。"""
        dp = self.p['Ps']-self.state[2]
        return self.p['Cd']*self.p['w']*np.sqrt(2*dp/self.p['rho']) if dp > 0 else np.nan
