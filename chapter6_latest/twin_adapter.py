"""原始 HydraulicTwin 的闭环适配：有向节流、补油、阀动态与工艺负载。"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from hydraulic_digital_twin import HydraulicTwin


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


class ClosedLoopTwin(HydraulicTwin):
    """四阶液压状态 + 一阶阀芯；1ms 周期内控制和外载保持不变。

    保留原 twin 的摩擦、有效模量、泄漏和端盖模型。节流使用有向压差，
    低压腔通过油箱补油单向阀吸油；不裁剪积分后的压力或输出位移。
    """
    def __init__(self, config=None, load_force=180000., load_window=(.35, .70),
                 valve_tau=.0053, rtol=1e-5, load_ramp=0., makeup_area=.001,
                 ambient_pressure=1e5, unload_ramp=None):
        config = physical_config() if config is None else config.copy()
        config['x_work'] = config['L']+1.
        super().__init__(config)
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

    def reset(self, x0=None):
        state = super().reset(x0)
        self.valve_state = self.active_force = 0.
        self.nfev = 0
        return state

    def load_at(self, t):
        start, end = self.load_window
        if not start-1e-10 <= t < end-1e-10:
            return 0.
        rise = (t-start)/self.load_ramp if self.load_ramp else 1.
        fall = (end-t)/self.unload_ramp if self.unload_ramp else 1.
        fraction = np.clip(min(rise, fall, 1.), 0., 1.)
        return self.load_force*.5*(1-np.cos(np.pi*fraction))

    def _dynamics(self, t, x, u):
        xp, velocity, p1, p2 = x
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
        # 双向缸的两腔各接油箱补油单向阀；仅 P<Pr 时有流入。
        q1 += p['Cd']*self.makeup_area*np.sqrt(2/p['rho']*max(p['Pr']-p1, 0.))
        q2 += p['Cd']*self.makeup_area*np.sqrt(2/p['rho']*max(p['Pr']-p2, 0.))
        leak = p['C_ip']*(p1-p2)
        dp1 = self._get_beta(p1)/v1*(q1-p['A1']*velocity-leak-p['C_ep']*p1)
        dp2 = self._get_beta(p2)/v2*(q2+p['A2']*velocity+leak-p['C_ep']*p2)
        # P1/P2 是绝对压；活塞杆外端的大气力必须纳入力平衡。
        drive = p1*p['A1']-p2*p['A2']-self.ambient_pressure*(p['A1']-p['A2'])
        dv = (drive-self._friction(velocity)-self._env_force(xp, velocity)
              -self.active_force)/p['mass']
        return np.array([velocity, dv, dp1, dp2])

    def step_voltage(self, voltage, dt):
        if not np.isfinite(voltage) or not np.isfinite(dt) or dt <= 0:
            raise ValueError('电压必须有限，dt必须为正且有限')
        command = float(np.clip(voltage, -10., 10.))/10.
        self.active_force = self.load_at(self.t)
        if self.valve_tau == 0:
            self.valve_state = command
            fun = lambda t, y: self._dynamics(t, y, command)
            y0 = self.state
            atol = [1e-9, 1e-8, .1, .1]
        else:
            def fun(t, y):
                return np.r_[self._dynamics(t, y[:4], y[4]), (command-y[4])/self.valve_tau]
            y0 = np.r_[self.state, self.valve_state]
            atol = [1e-9, 1e-8, .1, .1, 1e-9]
        sol = solve_ivp(fun, (self.t, self.t+dt), y0, method='Radau', rtol=self.rtol, atol=atol)
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise RuntimeError(f'Radau 在 t={self.t:.6f}s 失败：{sol.message}')
        self.nfev += sol.nfev
        self.state = sol.y[:4, -1]
        if self.valve_tau:
            self.valve_state = float(sol.y[4, -1])
        self.t += dt
        if min(self.state[2:]) <= 0:
            raise RuntimeError(f'绝对压力非正 t={self.t:.6f}s，请检查负载或模型')
        return self.state.copy()

    def extension_kq(self):
        """按 P1 计算伸出支路 dQ1/dxv；反向阀位时只是潜在增益。"""
        dp = self.p['Ps']-self.state[2]
        return self.p['Cd']*self.p['w']*np.sqrt(2*dp/self.p['rho']) if dp > 0 else np.nan
