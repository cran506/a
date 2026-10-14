"""传统流量逆前馈 + 位置式 PID；仅最终物理输出限幅。"""
import math
from pid_algorithms import PositionalPID


class BaselineFFPID:
    def __init__(self, config, parameters):
        self.config, self.p = config.copy(), parameters.copy()
        self.pid = PositionalPID(config['pid_kp'], config['pid_ki'], config['pid_kd'],
                                 config['ts'], config['pid_alpha'], derivative_mode='measurement')
        self.last = {}

    def planned_velocity(self, time):
        """公共50mm工艺计划的梯形速度，仅供传统前馈，PID参考仍为阶跃。"""
        c = self.config
        t = time-c['step_time']
        ramp, speed, distance = c['ff_accel_time'], c['ff_velocity'], c['target_mm']/1000.
        cruise = max(0., distance/speed-ramp)
        if t <= 0. or t >= 2*ramp+cruise:
            return 0.
        if t < ramp:
            return speed*t/ramp
        if t < ramp+cruise:
            return speed
        return speed*(2*ramp+cruise-t)/ramp

    def inverse_flow(self, velocity):
        """恒速标称工况的双阀口流量逆；不含压力安全限制。"""
        if velocity <= 0.:
            return 0.
        p = self.p
        ratio = (p['A1']/p['A2'])**2
        friction = p['Fc']+(p['Fs']-p['Fc'])*math.exp(-(velocity/p['vs'])**2)+p['sigma']*velocity
        atmosphere = 1e5*(p['A1']-p['A2'])
        p2 = (p['A1']*(p['Ps']+ratio*p['Pr'])-atmosphere-friction)/(ratio*p['A1']+p['A2'])
        p1 = p['Ps']-ratio*(p2-p['Pr'])
        gain = p['Cd']*p['w']*p['xv_max']/10.*math.sqrt(2*(p['Ps']-p1)/p['rho'])
        return p['A1']*velocity/gain+10.*p['deadzone']

    def update(self, time, target_mm, state):
        # .update 返回受限值，但这里只用位置式的未限幅候选与显式积分。
        self.pid.update(target_mm/1000., state[0]/1000.)
        ff = self.inverse_flow(self.planned_velocity(time))
        raw = self.pid.last_raw+ff
        output = min(10., max(-10., raw))
        self.last = dict(raw_voltage_v=raw, feedforward_v=ff,
                         integral_v=self.pid.i_total, derivative_v=self.pid.last_d,
                         solver_status='not_applicable', qp_iterations=0,
                         solve_time_ms=0., computation_ms=0.)
        return output
