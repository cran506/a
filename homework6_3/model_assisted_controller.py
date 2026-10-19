"""模块三：低频阀—速度通道的机理前馈、残差反馈和双向死区逆映射。

本题隔离研究死区失配，使用 tau_m * dv/dt + v = K_v * u_eff。
不把第五/六章双腔逆动力学参数冒充为本题的等效速度增益。
"""
import math
from collections import deque
import numpy as np


def inverse_deadzone(linear_v, estimate_v, mode="hard", beta=18., epsilon=1e-10):
    """PPT 式6.3的双向逆映射，或给定的连续 tanh 改造。

    beta 的单位为 V^-1。有限 beta 不会严格消除真实硬死区。
    """
    value = np.asarray(linear_v, dtype=float)
    if (not np.all(np.isfinite(value)) or not math.isfinite(estimate_v)
            or estimate_v < 0 or not math.isfinite(epsilon) or epsilon < 0):
        raise ValueError("补偿输入、估计值和零位保护阈值必须有限且有效")
    if mode == "hard":
        result = np.where(abs(value) <= epsilon, 0., value+estimate_v*np.sign(value))
    elif mode == "tanh":
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError("beta 必须为有限正数")
        result = value+estimate_v*np.tanh(beta*value)
    else:
        raise ValueError("补偿方式仅支持 hard 或 tanh")
    return float(result) if value.ndim == 0 else result


class DigitalPID:
    """与前章相同的数字误差反馈结构；误差用等效电压归一化。

    修订实验默认关闭普通 PID，使用公开的动态残差压力测试支路。
    支持 I 和不完全微分，输出总限幅在外层执行。
    """
    def __init__(self, kp, ki=0., kd=0., ts=.001, alpha=.85):
        if not all(math.isfinite(x) for x in (kp, ki, kd, ts, alpha)) or ts <= 0 or not 0 <= alpha < 1:
            raise ValueError("无效 PID 参数")
        self.kp, self.ki, self.kd, self.ts, self.alpha = kp, ki, kd, ts, alpha
        self.reset()

    def reset(self):
        self.integral = self.derivative = 0.
        self.previous_feedback = None

    def update(self, target, feedback):
        error = target-feedback
        self.integral += error*self.ts
        derivative = 0. if self.previous_feedback is None else -(feedback-self.previous_feedback)/self.ts
        self.derivative = self.alpha*self.derivative+(1-self.alpha)*derivative
        self.previous_feedback = feedback
        return self.kp*error+self.ki*self.integral+self.kd*self.derivative


class ModelAssistedController:
    """共同线性逆前馈 + 速度残差反馈，再串联待比较的死区逆模型。

    u_lin = (v_d + tau_m*a_d)/K_v + PID(e) + K_w*HP2(e[k-d])
    u = inverse_deadzone(u_lin)。三组仅改变死区估计和映射方式。
    本压力测试将逆映射放在反馈合成后，以研究反复换向的颤振。
    若只对预定正弦的符号作补偿，开环信号每半周期仅翻转一次。
    """
    def __init__(self, pid_params, phys_params, *, estimate_v=.6, mode="tanh", beta=18.):
        self.p = phys_params.copy()
        self.pid = DigitalPID(**pid_params)
        self.estimate_v, self.mode, self.beta = estimate_v, mode, beta
        if self.p['gain_m_s_v'] <= 0 or self.p['tau_velocity_s'] <= 0:
            raise ValueError("速度增益和时间常数必须为正")
        self.washout_gain = float(self.p.get('washout_gain', 0.))
        self.washout_tau_s = float(self.p.get('washout_tau_s', .002))
        self.washout_order = self.p.get('washout_order', 2)
        self.feedback_delay_samples = self.p.get('feedback_delay_samples', 0)
        if (not math.isfinite(self.washout_gain) or not math.isfinite(self.washout_tau_s)
                or self.washout_tau_s <= 0 or self.washout_order not in (1,2)
                or not isinstance(self.feedback_delay_samples, int) or self.feedback_delay_samples < 0):
            raise ValueError('动态残差参数无效')
        self.error_history = deque(maxlen=self.feedback_delay_samples+1)
        self.lowpass1 = self.lowpass2 = 0.
        self.feedback_enabled = True
        self.last = {}

    def update(self, r_state, y_meas, P_meas=None):
        """沿用模块三接口；本题仅用 r_state=[x_d,v_d,a_d] 和 y_meas=[x,v]。

        P_meas 不参与低频等效模型。返回总电压、名义补偿前馈、非线性残差修正。
        后两者相加严格等于实际限幅电压，但残差经过非线性映射，不等于线性 PID 值。
        """
        values = [*r_state, *y_meas]
        if not np.all(np.isfinite(values)):
            raise ValueError("期望和测量状态必须有限")
        gain, tau = self.p['gain_m_s_v'], self.p['tau_velocity_s']
        vd, ad = r_state[1:3]
        ff_linear = (vd+tau*ad)/gain
        error = (vd-y_meas[1])/gain
        self.error_history.append(error)
        delayed = self.error_history[0] if len(self.error_history) == self.error_history.maxlen else 0.
        alpha = math.exp(-self.pid.ts/self.washout_tau_s)
        self.lowpass1 = alpha*self.lowpass1+(1-alpha)*delayed
        highpass1 = delayed-self.lowpass1
        self.lowpass2 = alpha*self.lowpass2+(1-alpha)*highpass1
        highpass2 = highpass1-self.lowpass2 if self.washout_order == 2 else highpass1
        pid_feedback = self.pid.update(vd/gain, y_meas[1]/gain)
        washout_feedback = self.washout_gain*highpass2
        feedback = pid_feedback+washout_feedback if self.feedback_enabled else 0.
        linear = ff_linear+feedback
        raw = inverse_deadzone(linear, self.estimate_v, self.mode, self.beta)
        voltage = float(np.clip(raw, -10., 10.))
        ff = inverse_deadzone(ff_linear, self.estimate_v, self.mode, self.beta)
        self.last = dict(linear_voltage_v=linear, feedforward_linear_v=ff_linear,
                         feedback_linear_v=feedback, raw_voltage_v=raw,
                         pid_feedback_voltage_v=pid_feedback if self.feedback_enabled else 0.,
                         washout_feedback_voltage_v=washout_feedback if self.feedback_enabled else 0.,
                         delayed_error_equivalent_v=delayed,
                         lowpass1_v=self.lowpass1, lowpass2_v=self.lowpass2,
                         highpass1_v=highpass1, highpass2_v=highpass2,
                         control_voltage_v=voltage)
        return voltage, ff, voltage-ff
