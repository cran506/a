"""模块三：有量纲的双腔逆模型前馈 + 数字 PID 残差反馈。

沿用 Word 的三策略开关，补齐 PPT 的已知工艺负载、真实建压流量和阀动态。
控制器仅接收参考计划及传感器值，不读取被控对象的真实外载或积分器导数。
"""
import math
import numpy as np
from controllers import DigitalPID
from media_model import AeratedHydraulicModel
from friction_model import sliding_friction, sliding_friction_slope


class ModelAssistedController:
    def __init__(self, pid_params, phys_params, enable_ff=True,
                 enable_gain_sched=True, pressure_alpha=.2,
                 valve_tau=.0025, enable_valve_inverse=True, gain_prediction_s=0.,
                 zero_flow_band_v=.03, deadzone_slew_v_s=40.,force_error_gain_s=80.,
                 enable_startup_regularization=True,velocity_error_gain_s=60.,
                 startup_force_gain_s=None,startup_velocity_threshold=1e-4):
        if (not all(math.isfinite(v) for v in (pressure_alpha,valve_tau,gain_prediction_s,zero_flow_band_v,deadzone_slew_v_s,force_error_gain_s,velocity_error_gain_s))
                or not 0 <= pressure_alpha < 1 or valve_tau < 0 or gain_prediction_s < 0
                or zero_flow_band_v<0 or deadzone_slew_v_s<=0 or force_error_gain_s<0 or velocity_error_gain_s<0):
            raise ValueError('压力滤波或阀响应参数无效')
        if (startup_force_gain_s is not None and
                (not math.isfinite(startup_force_gain_s) or startup_force_gain_s < 0)):
            raise ValueError('起动建压力反馈增益必须为有限非负数或 None')
        if not math.isfinite(startup_velocity_threshold) or startup_velocity_threshold <= 0:
            raise ValueError('起动完成速度阈值必须为有限正数')
        self.pid = DigitalPID(**pid_params)
        self.p = phys_params.copy()
        self.model = AeratedHydraulicModel(self.p)
        self.enable_ff, self.enable_gain_sched = enable_ff, enable_gain_sched
        self.pressure_alpha = pressure_alpha
        self.gain_prediction_s = gain_prediction_s
        self.zero_flow_band_v,self.deadzone_slew_v_s=zero_flow_band_v,deadzone_slew_v_s
        self.force_error_gain_s=force_error_gain_s
        self.startup_force_gain_s=startup_force_gain_s
        self.startup_velocity_threshold=startup_velocity_threshold
        self.velocity_error_gain_s=velocity_error_gain_s
        self.enable_startup_regularization = enable_startup_regularization
        self.valve_tau, self.enable_valve_inverse = valve_tau, enable_valve_inverse
        self.k_flow = self.p['Cd']*self.p['w']*(self.p['xv_max']/10)*math.sqrt(2/self.p['rho'])
        self.deadzone_v = 10*self.p['deadzone']
        # 两个阀口共用阀芯，按面积比与力平衡计算空载巡航标称压力。
        k = (self.p['A1']/self.p['A2'])**2
        force = sliding_friction(.2,self.p)
        atmosphere = self.p.get('ambient_pressure', 1e5)*(self.p['A1']-self.p['A2'])
        p2 = (self.p['A1']*(self.p['Ps']+k*self.p['Pr'])-atmosphere-force)/(k*self.p['A1']+self.p['A2'])
        self.nominal_pressures = np.array([self.p['Ps']-k*(p2-self.p['Pr']), p2])
        self.reset()

    def reset(self):
        self.pid.reset()
        self.pressure_filtered = np.array([self.p['Pr'], self.p['Pr']], dtype=float)
        self.previous_valve_target_v = 0.
        self.previous_nominal_force = 0.
        self.deadzone_offset_v = 0.
        self.deadzone_initialized = False
        self.startup_valve_mismatch_v = None
        self.motion_launched = False
        self.last = {}

    def _valve_inverse(self, valve_target):
        """动态逆补偿允许冷启动初值误差按真实阀极点自然衰减。

        首次非零目标直接作为非零首指令，不要求零阀芯在一个周期内
        达到该目标。减去解析齐次误差项，避免参考阶跃被动态逆放大。
        FF阀记忆始终从零按真正施加且限幅后的指令更新，不伪造阀初态。
        """
        if not self.enable_valve_inverse or not self.valve_tau:
            self.last.update(startup_valve_mismatch_v=0.,startup_inverse_correction_v=0.)
            return valve_target
        a = math.exp(-self.pid.ts/self.valve_tau)
        exact = (valve_target-a*self.previous_valve_target_v)/(1-a)
        mismatch = 0.
        if self.enable_startup_regularization:
            if self.startup_valve_mismatch_v is None and abs(valve_target)>1e-8:
                self.startup_valve_mismatch_v = valve_target-self.previous_valve_target_v
            mismatch = self.startup_valve_mismatch_v or 0.
            if self.startup_valve_mismatch_v is not None:
                self.startup_valve_mismatch_v *= a
        correction = a*mismatch/(1-a)
        self.last.update(startup_valve_mismatch_v=mismatch,
                         startup_inverse_correction_v=correction)
        return exact-correction

    def _deadzone_inverse(self,linear_v,moving):
        """显著流量精确反演；近零流量时有限速度渡过死区。

        精确死区逆的两侧极限相差2δ，不可能既精确又跨零连续。
        公开的0.03V线性流量带内允许短暂小流量失配，偏置限速40V/s。
        首次非零预开不作从零幅值衰减，由 _valve_inverse 处理初值相容性。
        """
        desired=math.copysign(self.deadzone_v,linear_v) if abs(linear_v)>1e-6 else 0.
        if self.deadzone_initialized and abs(linear_v)<self.zero_flow_band_v:
            change=self.deadzone_slew_v_s*self.pid.ts
            offset=self.deadzone_offset_v+float(np.clip(desired-self.deadzone_offset_v,-change,change))
        else:
            offset=desired
        self.deadzone_offset_v=offset
        self.deadzone_initialized=self.deadzone_initialized or moving
        self.last['deadzone_offset_v']=offset
        return linear_v+offset

    def _friction_rate(self, velocity, acceleration):
        # 滑动支路的单侧解析导数；零速静摩擦方向由运动计划决定。
        return sliding_friction_slope(velocity,self.p)*acceleration

    def _inverse_force_rate(self, position, vd, desired_force_rate, known_force_rate=0.):
        p = self.p
        p1, p2 = self.pressure_filtered
        v1, v2 = p['V0_1']+p['A1']*position, p['V0_2']+p['A2']*(p['L']-position)
        if min(v1, v2) <= 0:
            raise ValueError('液压腔容积必须为正')
        c1, c2 = self.model._get_beta(p1)/v1, self.model._get_beta(p2)/v2
        leak = p['C_ip']*(p1-p2)
        # 液压力 Fh=A1*P1-A2*P2 的连续性方程：Fh_dot=G*u_lin+H。
        h = -(p['A1']**2*c1+p['A2']**2*c2)*vd-(p['A1']*c1+p['A2']*c2)*leak
        h += -p['A1']*c1*p.get('C_ep',0.)*p1+p['A2']*c2*p.get('C_ep',0.)*p2
        # 与 Twin 相同的两腔补油支路；绝不裁剪积分后的压力。
        area = p.get('makeup_area', .001)
        h += p['A1']*c1*p['Cd']*area*math.sqrt(2/p['rho']*max(p['Pr']-p1,0.))
        h -= p['A2']*c2*p['Cd']*area*math.sqrt(2/p['rho']*max(p['Pr']-p2,0.))
        numerator = desired_force_rate-h
        pressures = self.pressure_filtered.copy() if self.enable_gain_sched else self.nominal_pressures
        if self.enable_gain_sched and self.gain_prediction_s:
            ratio=(p['A1']/p['A2'])**2
            delta=self.gain_prediction_s*known_force_rate/(ratio*p['A1']+p['A2'])
            pressures += np.array([ratio*delta,-delta])
        drive1, drive2 = pressures
        if numerator >= 0:
            delta_in = max(p['Ps']-drive1, 5e5)
            b1 = self.k_flow*math.sqrt(delta_in)
            outlet = drive2-p['Pr']
            b2 = -self.k_flow*math.copysign(math.sqrt(abs(outlet)),outlet)
        else:
            delta_in = max(p['Ps']-drive2, 5e5)
            outlet = drive1-p['Pr']
            b1 = self.k_flow*math.copysign(math.sqrt(abs(outlet)),outlet)
            b2 = -self.k_flow*math.sqrt(delta_in)
        g = p['A1']*c1*b1-p['A2']*c2*b2
        if g <= 0:
            raise ValueError('局部逆模型增益非正，当前工作点不可用')
        linear_v = numerator/g
        self.last.update(inverse_gain_n_s_v=g, desired_force_rate_n_s=desired_force_rate,
                         continuity_bias_n_s=h, supply_delta_pa=delta_in,
                         linear_voltage_v=linear_v)
        return linear_v

    def update(self, r_state, y_meas, P_meas, *, jerk_d=None,
               known_force_n=0., known_force_rate_n_s=0., origin_m=0.):
        """返回总电压、实际前馈电压、反馈电压，单位均为 V。

        r_state=[位置m,速度m/s,加速度m/s²]；y_meas=[相对位置m,速度m/s]。
        known_force/rate 只允许来自公共参考计划；未知外载应传0。
        jerk_d 是规划器的解析跃度，不是实测加速度强差分。
        """
        values = [*r_state,*y_meas,*P_meas,known_force_n,known_force_rate_n_s,origin_m]
        if jerk_d is not None:
            values.append(jerk_d)
        if not np.all(np.isfinite(values)):
            raise ValueError('参考与测量必须为有限数值')
        if min(P_meas) <= 0:
            raise ValueError('实测绝对压力必须为正')
        self.pressure_filtered = self.pressure_alpha*self.pressure_filtered+(1-self.pressure_alpha)*np.asarray(P_meas)
        # 冷起动时提高建压力误差环增益，首次实测运动后恢复正常增益。
        # 转换由速度传感器触发；不读取仿真时钟或预设释放时刻，停止后也不重触发。
        self.motion_launched = self.motion_launched or abs(y_meas[1]) > self.startup_velocity_threshold
        startup_mode = self.startup_force_gain_s is not None and not self.motion_launched
        force_gain = self.startup_force_gain_s if startup_mode else self.force_error_gain_s
        self.last = dict(startup_mode=bool(startup_mode),force_error_gain_s=force_gain)
        ff_raw, ff, valve_target = 0., 0., 0.
        moving = False
        if self.enable_ff:
            _, vd, ad = r_state
            # 首个 S 轨迹采样可有 v=a=0，但非零跃度已给出起动方向。
            # 静止持位时 v=a=j=0，不凭位置误差虚构800N前馈。
            direction = next((math.copysign(1.,value) for value in
                              (vd,ad,jerk_d or 0.) if abs(value)>1e-12),0.)
            friction = sliding_friction(vd,self.p,direction=direction)
            nominal_force = self.p['mass']*ad+friction
            force = nominal_force+known_force_n
            if jerk_d is None:
                force_rate = (nominal_force-self.previous_nominal_force)/self.pid.ts+known_force_rate_n_s
            else:
                force_rate = self.p['mass']*jerk_d+self._friction_rate(vd,ad)+known_force_rate_n_s
            self.previous_nominal_force = nominal_force
            self.last.update(planned_inverse_force_n=force,reference_friction_n=friction,
                             reference_motion_direction=direction)
            p1,p2=self.pressure_filtered
            actual_force=p1*self.p['A1']-p2*self.p['A2']-self.p.get('ambient_pressure',1e5)*(self.p['A1']-self.p['A2'])
            # 两个模型辅助组共享速度误差阻尼，替代原tanh近零斜率隐含的
            # 大阻尼。只用当前速度，不对传感器速度/压力作强微分。
            damping=self.p['mass']*self.velocity_error_gain_s*(vd-y_meas[1])
            force+=damping
            hold_adjustment=0.
            if direction==0 and abs(y_meas[1])<=1e-12:
                # 真正静止时摩擦可平衡[-Fs,Fs]内任意力；不强迫持位压力
                # 从合法的静摩擦平衡降到零，以免再次释放或产生反向尖峰。
                hold_goal=known_force_n+float(np.clip(actual_force-known_force_n,-self.p['Fs'],self.p['Fs']))
                hold_adjustment=hold_goal-force
                force=hold_goal
            self.last.update(desired_force_n=force,velocity_damping_force_n=damping,
                             static_hold_force_adjustment_n=hold_adjustment)
            correction=force_gain*(force-actual_force)
            self.last.update(actual_force_n=actual_force,planned_force_rate_n_s=force_rate,
                             force_error_correction_n_s=correction,force_error_gain_s=force_gain)
            force_rate+=correction
            moving = abs(vd)>1e-8 or abs(ad)>1e-6 or abs(force_rate)>1e-6
            # 停止后仍解连续性与泄漏，不把FF关断再切换符号持位偏置。
            linear_v = self._inverse_force_rate(origin_m+y_meas[0],vd,force_rate,known_force_rate_n_s)
            valve_target = self._deadzone_inverse(linear_v,moving)
            valve_target = float(np.clip(valve_target,-10.,10.))
            ff_raw = self._valve_inverse(valve_target)
            ff = float(np.clip(ff_raw,-10.,10.))
            if self.enable_valve_inverse and self.valve_tau:
                a = math.exp(-self.pid.ts/self.valve_tau)
                self.previous_valve_target_v = a*self.previous_valve_target_v+(1-a)*ff
            else:
                self.previous_valve_target_v = ff
        # 合成输出的动态余量，防止 FF + 已独立限幅的 PID 再次溢出。
        # 持位由持续逆模型补偿泄漏，PID无额外符号死区项，消除ADC触发的±δ切换。
        hold_bias = 0.
        self.pid.out_min, self.pid.out_max = -10.-ff-hold_bias,10.-ff-hold_bias
        fb_linear = self.pid.update(target=r_state[0],feedback=y_meas[0])
        fb = fb_linear+hold_bias
        total = float(np.clip(ff+fb,-10.,10.))
        self.last.update(feedforward_requested_v=ff_raw, valve_target_v=valve_target,
                         feedback_requested_v=self.pid.last_raw,
                         requested_voltage_v=ff+self.pid.last_raw+hold_bias,
                         holding_deadzone_compensation_v=hold_bias,
                         feedforward_v=ff,feedback_v=fb,voltage_v=total)
        return total,ff,fb

