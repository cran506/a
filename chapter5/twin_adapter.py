"""保留上传的 HydraulicTwin；补齐第3/5章闭环所需的接口。"""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from hydraulic_digital_twin import HydraulicTwin


def physical_config():
    return json.loads(Path(__file__).with_name('physical_config.json').read_text())


class QuantizedSensor:
    """0~300mm 的12bit ADC，LSB=量程/4096，饱和代码0~4095。"""
    def __init__(self, bits=12, span=.3):
        if bits < 1 or bits > 24 or span <= 0:
            raise ValueError("ADC 参数无效")
        self.bits, self.span = bits, span
        self.levels = 2**bits
        self.lsb = span/self.levels

    def sample(self, position):
        code = int(np.clip(np.rint(position/self.lsb), 0, self.levels-1))
        return code*self.lsb


class ClosedLoopTwin(HydraulicTwin):
    """四阶液压模型加一阶实际阀芯滞后；控制指令在周期内 ZOH。

    原文件不改。按第3章PPT补上负压差回流符号；外载直接作用于牛顿方程。
    矩形负载按控制采样时刻锁存，使积分器不跨越未定位的切换点。
    """
    def __init__(self, config=None, load_force=180000., load_window=(.35, .70),
                 valve_tau=.015, rtol=1e-5):
        config = physical_config() if config is None else config.copy()
        # 禁用原开环例子的 x=0.4m 工件；改用显式工艺时窗外载。
        config['x_work'] = config['L']+1.
        super().__init__(config)
        if (not all(np.isfinite(v) for v in (valve_tau, load_force, rtol, *load_window))
                or valve_tau < 0 or not 0 <= load_window[0] < load_window[1]
                or load_force < 0 or rtol <= 0 or config['Pr'] <= 2000):
            raise ValueError("阀时间常数、负载幅值或时窗无效")
        self.load_force, self.load_window = load_force, load_window
        self.valve_tau, self.rtol = valve_tau, rtol
        self.valve_state = self.active_force = 0.
        self.nfev = 0

    def reset(self, x0=None):
        state = super().reset(x0)
        self.valve_state = self.active_force = 0.
        self.nfev = 0
        return state

    def _get_beta(self, pressure):
        """低压气穴等效近似：体积模量在蒸气压处趋零。

        原来的残余0.001*beta不能保证绝对压力非负。只在低于回油背压
        时增加气穴衰减，令压力趋近蒸气压而非跨越它；不裁剪最终压力。
        Pv=2kPa 是本实验补充的近似参数，非多相气穴模型。
        """
        vapor_pressure = 2000.
        factor = np.clip((pressure-vapor_pressure)/(self.p['Pr']-vapor_pressure), 0., 1.)
        return super()._get_beta(pressure)*factor

    def _dynamics(self, t, x, u):
        d = np.asarray(super()._dynamics(t, x, u), dtype=float)
        xp, _, p1, p2 = x
        p = self.p
        ue = np.sign(u)*max(abs(u)-p['deadzone'], 0.)
        area = p['Cd']*p['w']*abs(ue*p['xv_max'])
        if ue >= 0:
            delta1, delta2 = p['Ps']-p1, p['Pr']-p2
            old1, old2 = area*np.sqrt(2/p['rho']*abs(delta1)), -area*np.sqrt(2/p['rho']*abs(delta2))
        else:
            delta1, delta2 = p['Pr']-p1, p['Ps']-p2
            old1, old2 = -area*np.sqrt(2/p['rho']*abs(delta1)), area*np.sqrt(2/p['rho']*abs(delta2))
        signed1 = area*np.sign(delta1)*np.sqrt(2/p['rho']*abs(delta1))
        signed2 = area*np.sign(delta2)*np.sqrt(2/p['rho']*abs(delta2))
        v1, v2 = p['V0_1']+p['A1']*xp, p['V0_2']+p['A2']*(p['L']-xp)
        if min(v1, v2) <= 0:
            raise RuntimeError('液压腔容积非正，停止仿真')
        d[2] += self._get_beta(p1)/v1*(signed1-old1)
        d[3] += self._get_beta(p2)/v2*(signed2-old2)
        d[1] -= self.active_force/p['mass']
        return d

    def step_voltage(self, voltage, dt):
        if not np.isfinite(voltage) or dt <= 0:
            raise ValueError("电压必须有限，dt必须为正")
        command = float(np.clip(voltage, -10., 10.))/10.
        self.active_force = (self.load_force if self.load_window[0]-1e-10 <= self.t < self.load_window[1]-1e-10 else 0.)
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
        """按当前P1计算伸出支路的潜在 dQ1/dxv，Ps<=P1则NaN。

        不按实际阀方向筛选；阀反向时该值不是当前活动支路的局部增益。
        """
        dp = self.p['Ps']-self.state[2]
        return self.p['Cd']*self.p['w']*np.sqrt(2*dp/self.p['rho']) if dp > 0 else np.nan
