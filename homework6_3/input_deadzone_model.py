"""修订版课堂对象：输入硬死区 → 阀有效开口一阶 → 机械速度一阶。

u_eff=D(u) 是题图的静态死区输出；z 是动态有效阀口等效电压。
两者均为 V，但 z 不再经过第二次死区。每个 ZOH 区间解析推进。
"""
import math
from deadzone_model import deadzone


class InputDeadzoneVelocityPlant:
    def __init__(self, gain_m_s_v=.01, tau_valve_s=.0008, tau_velocity_s=.012, deadzone_v=.5):
        values = (gain_m_s_v, tau_valve_s, tau_velocity_s, deadzone_v)
        if (not all(math.isfinite(v) for v in values) or gain_m_s_v <= 0
                or tau_valve_s < 0 or tau_velocity_s <= 0 or deadzone_v < 0):
            raise ValueError('输入死区模型参数无效')
        self.gain_m_s_v, self.tau_valve_s = float(gain_m_s_v), float(tau_valve_s)
        self.tau_velocity_s, self.deadzone_v = float(tau_velocity_s), float(deadzone_v)
        self.reset()

    def reset(self, valve_voltage_v=0., velocity_m_s=0.):
        if not all(math.isfinite(v) for v in (valve_voltage_v, velocity_m_s)):
            raise ValueError('初态必须有限')
        # 通用属性名保留；此处是有效开口状态，不是死区前的原始阀位。
        self.valve_voltage_v, self.velocity_m_s = float(valve_voltage_v), float(velocity_m_s)
        self.time_s = 0.
        return self.velocity_m_s

    def step(self, voltage_v, dt):
        if not math.isfinite(voltage_v) or not math.isfinite(dt) or dt <= 0:
            raise ValueError('电压必须有限，步长必须为正')
        drive = deadzone(voltage_v, self.deadzone_v)
        tm, tv = self.tau_velocity_s, self.tau_valve_s
        em = math.exp(-dt/tm)
        if tv == 0.:
            self.velocity_m_s = self.velocity_m_s*em+self.gain_m_s_v*drive*(-math.expm1(-dt/tm))
            self.valve_voltage_v = drive
        else:
            ev, delta = math.exp(-dt/tv), 1/tm-1/tv
            if delta == 0.:
                convolution = dt/tm*em
            elif abs(delta*dt) < 1e-5:
                convolution = dt/tm*em*math.expm1(delta*dt)/(delta*dt)
            else:
                convolution = (ev-em)/(tm*delta)
            self.velocity_m_s = (self.velocity_m_s*em+self.gain_m_s_v*(
                drive*(-math.expm1(-dt/tm))+(self.valve_voltage_v-drive)*convolution))
            self.valve_voltage_v = drive+(self.valve_voltage_v-drive)*ev
        self.time_s += dt
        return self.velocity_m_s
