"""第五章数字 PID；输入位置为 m，内部误差为 cm，输出为 V。"""
import math


class DigitalPID:
    """Word 中的增量式 PID，保留先滤波、后增量和限幅状态记忆。

    integral_separation=True 复用 Word 的积分分离；习题5-1必须设 False。
    last_raw 是本步限幅前候选值，不是独立存储的未截断输出。
    """
    def __init__(self, kp, ki, kd, ts, out_min=-10., out_max=10.,
                 alpha=.85, integral_separation=True):
        if ts <= 0 or not 0 <= alpha < 1 or out_min >= out_max:
            raise ValueError("需 Ts>0、0<=alpha<1、out_min<out_max")
        if not all(math.isfinite(v) for v in (kp, ki, kd, ts, out_min, out_max, alpha)):
            raise ValueError("PID 参数必须有限")
        self.kp, self.ki, self.kd, self.ts = kp, ki, kd, ts
        self.out_min, self.out_max, self.alpha = out_min, out_max, alpha
        self.integral_separation = integral_separation
        self.reset()

    def reset(self):
        self.e_prev = self.d_filt_prev = self.u_prev = 0.
        self.last_raw = self.last_d = self.last_i = 0.

    def _terms(self, target, feedback):
        if not math.isfinite(target) or not math.isfinite(feedback):
            raise ValueError("参考位置和反馈必须有限")
        e = (target-feedback)*100.
        factor = .05 if self.integral_separation and abs(e) >= .15 else 1.
        di = self.ki*e*self.ts*factor
        d = self.alpha*self.d_filt_prev + (1-self.alpha)*self.kd/self.ts*(e-self.e_prev)
        return e, di, d

    def update(self, target, feedback):
        e, di, d = self._terms(target, feedback)
        self.last_raw = self.u_prev + self.kp*(e-self.e_prev) + di + d-self.d_filt_prev
        u = min(self.out_max, max(self.out_min, self.last_raw))
        self.e_prev, self.d_filt_prev, self.u_prev = e, d, u
        self.last_d, self.last_i = d, di
        return u


class PositionalPID(DigitalPID):
    """仅最终输出限幅的位置式；有意保留习题要求的积分 windup。"""
    def reset(self):
        super().reset()
        self.i_total = 0.

    def update(self, target, feedback):
        e, di, d = self._terms(target, feedback)
        self.i_total += di
        self.last_raw = self.kp*e+self.i_total+d
        u = min(self.out_max, max(self.out_min, self.last_raw))
        self.e_prev, self.d_filt_prev, self.u_prev = e, d, u
        self.last_d, self.last_i = d, self.i_total
        return u
