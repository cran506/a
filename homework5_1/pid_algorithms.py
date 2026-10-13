"""位置式/增量式 PID；输入 m，内部误差 cm，输出 V。

两组仅积分记忆方式不同，共用微分滤波。默认对测量微分以免参考阶跃
冲击；derivative_mode='error' 可复核教材的误差微分式。
没有积分分离、条件积分、反算补偿或外加输出平滑。
"""
import math


class _CommonPID:
    def __init__(self, kp=4.2, ki=22., kd=.06, ts=.001, alpha=.85,
                 out_min=-10., out_max=10., derivative_mode='measurement'):
        values = (kp, ki, kd, ts, alpha, out_min, out_max)
        if (not all(math.isfinite(v) for v in values) or ts <= 0
                or min(kp, ki, kd) < 0 or not 0 <= alpha < 1
                or not out_min < out_max
                or derivative_mode not in ('measurement', 'error')):
            raise ValueError('PID 参数无效')
        self.kp, self.ki, self.kd, self.ts = kp, ki, kd, ts
        self.alpha, self.out_min, self.out_max = alpha, out_min, out_max
        self.derivative_mode = derivative_mode
        self.reset()

    def reset(self):
        self.e_prev = self.d_input_prev = self.d_prev = self.u_prev = 0.
        self.i_total = self.last_raw = self.last_d = self.last_di = 0.
        self.last_effective_integral = self.last_delta = self.state_before = 0.

    def _terms(self, target_m, feedback_m):
        if not all(math.isfinite(v) for v in (target_m, feedback_m)):
            raise ValueError('参考位置和反馈必须有限')
        error_cm = 100.*(target_m-feedback_m)
        derivative_input = (error_cm if self.derivative_mode == 'error'
                            else -100.*feedback_m)
        di = self.ki*error_cm*self.ts
        d = (self.alpha*self.d_prev+(1-self.alpha)*self.kd/self.ts
             *(derivative_input-self.d_input_prev))
        return error_cm, derivative_input, di, d

    def _remember(self, e, derivative_input, di, d, voltage):
        self.last_d, self.last_di = d, di
        # 实际输出的等效积分分量用于诊断，不用于位置式控制计算。
        self.last_effective_integral = voltage-self.kp*e-d
        self.e_prev, self.d_input_prev = e, derivative_input
        self.d_prev, self.u_prev = d, voltage
        return voltage

    def _clip(self, voltage):
        return max(self.out_min, min(self.out_max, voltage))


class PositionalPID(_CommonPID):
    """实验 A：显式积分一直累加，只限制最终物理输出。"""
    def update(self, target_m, feedback_m):
        e, derivative_input, di, d = self._terms(target_m, feedback_m)
        self.state_before = self.u_prev
        previous_raw = self.last_raw
        self.i_total += di
        self.last_raw = self.kp*e+self.i_total+d
        self.last_delta = self.last_raw-previous_raw
        return self._remember(e, derivative_input, di, d, self._clip(self.last_raw))


class IncrementalPID(_CommonPID):
    """实验 B：截断后的实际 u 写回下一周期的 u_prev。

    限幅前候选值可以越界；受到限制的是迭代输出状态。
    i_total 不参与控制，避免增量式中暗藏独立累加积分器。
    """
    def update(self, target_m, feedback_m):
        e, derivative_input, di, d = self._terms(target_m, feedback_m)
        self.state_before = self.u_prev
        self.last_delta = self.kp*(e-self.e_prev)+di+d-self.d_prev
        self.last_raw = self.u_prev+self.last_delta
        return self._remember(e, derivative_input, di, d, self._clip(self.last_raw))
