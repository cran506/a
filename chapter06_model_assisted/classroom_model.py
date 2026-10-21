"""第六章课堂约化对象：位置/速度动态 + 准静态负载压力。

本模型用于展示参考图的控制机理，不是上传 HydraulicTwin 的四阶压力动力学。
统一被控对象为
    dx/dt = v
    m dv/dt = K_u g(P1) D_delta(u) - (B_h + sigma) v - F_s(v),
其中 D_delta 是真实正遮盖死区，静止时摩擦力属于 [-Fs, Fs]。
P1 = P_light + F_ext/A1，g(P1)=sqrt((Ps-P1)/(Ps-P_light))。
B_h 为等效液压阻抗；K_u 是压力工况归一化后的净驱动力/电压增益。
外载的准静态力平衡已在工作点内消去，仅通过孔口增益 g 调制运动通道。
因此压力可在负载阶跃时瞬变；这里没有油液储能、真实建压或阀芯带宽。

公开标定：
  Fs/K_u + delta = 800/1000 + .4 = 1.2 V；
  [(B_h+sigma)*.2+Fc]/K_u + delta = 2.5 V；
  g_heavy=(2.5-.4)/(4.2-.4)，据此反求重载压力，衰减44.737%。
这三项是模型标定；闭环误差、速度跌落、起动延迟均由ODE重新计算。
本模型的60kg等效质量、.02m/s Stribeck速度与上传四阶对象参数不同。
"""
from dataclasses import asdict, dataclass
import math
import numpy as np
from scipy.integrate import solve_ivp
from trajectory import generate_process_scurve


@dataclass(frozen=True)
class ClassroomParameters:
    mass: float = 60.
    hydraulic_damping: float = 7000.
    sigma: float = 1000.
    force_gain: float = 1000.
    Fs: float = 800.
    Fc: float = 500.
    vs: float = .02
    deadzone_v: float = .4
    supply_pa: float = 21e6
    nominal_pressure_pa: float = 5e6
    nominal_cruise_voltage_v: float = 2.5
    target_heavy_voltage_v: float = 4.2
    A1: float = .01
    A2: float = .005
    load_start: float = .5
    load_end: float = .92
    kp: float = 3500.
    ki: float = 20000.
    kd: float = 5.
    derivative_tau_s: float = -.001/math.log(.85)
    voltage_limit_v: float = 10.
    minimum_delta_pa: float = 5e5

    @property
    def heavy_gain_ratio(self):
        return ((self.nominal_cruise_voltage_v-self.deadzone_v)/
                (self.target_heavy_voltage_v-self.deadzone_v))

    @property
    def heavy_pressure_pa(self):
        return self.supply_pa-(self.supply_pa-self.nominal_pressure_pa)*self.heavy_gain_ratio**2

    @property
    def load_force_n(self):
        return (self.heavy_pressure_pa-self.nominal_pressure_pa)*self.A1


def stribeck(velocity, p, direction=0.):
    """纯Stribeck项，不包含已单列的 sigma*v。"""
    sign = np.sign(velocity) if abs(velocity)>1e-15 else direction
    return float((p.Fc+(p.Fs-p.Fc)*math.exp(-(velocity/p.vs)**2))*sign)


def physical_deadzone(voltage, p):
    return math.copysign(max(abs(voltage)-p.deadzone_v, 0.), voltage)


class ClassroomPlant:
    """所有策略共享参数、初态和加载计划，分别独立积分的相同对象。"""
    def __init__(self, p, rtol=1e-8):
        self.p, self.rtol = p, rtol
        self.state = np.zeros(2)
        self.release_times = []
        self.release_forces = []
        self.sticking = True
        self.t = 0.
        self.nfev = 0

    def operating_point(self, t):
        load = self.p.load_force_n if self.p.load_start-1e-12<=t<self.p.load_end-1e-12 else 0.
        pressure = self.p.nominal_pressure_pa+load/self.p.A1
        ratio = math.sqrt((self.p.supply_pa-pressure)/(self.p.supply_pa-self.p.nominal_pressure_pa))
        return load, pressure, ratio

    def net_force(self, u, gain, v=None):
        v = self.state[1] if v is None else v
        return (self.p.force_gain*gain*physical_deadzone(u,self.p)-
                (self.p.hydraulic_damping+self.p.sigma)*v)

    def step(self, voltage, ts):
        p = self.p
        _, _, gain = self.operating_point(self.t)
        drive0 = p.force_gain*gain*physical_deadzone(voltage,p)
        end = self.t+ts
        start = self.t
        y = self.state.copy()
        # 在零速粘住时，恒定输入/准静态压力下本周期不会自行释放。
        if self.sticking and abs(drive0)<=p.Fs+1e-10:
            self.t=end
            return self.state.copy()
        if self.sticking:
            self.sticking=False
            self.release_times.append(start)
            self.release_forces.append(drive0)
        direction = math.copysign(1., y[1] if abs(y[1])>1e-14 else drive0)
        for _ in range(8):
            def rhs(t, z):
                sliding = (p.Fc+(p.Fs-p.Fc)*math.exp(-(z[1]/p.vs)**2))*direction
                force = drive0-(p.hydraulic_damping+p.sigma)*z[1]-sliding
                return [z[1], force/p.mass]
            def stop(t,z):
                return z[1]
            stop.terminal=True
            stop.direction=-direction
            sol=solve_ivp(rhs,(start,end),y,method='RK45',events=stop,
                          rtol=self.rtol,atol=[1e-12,1e-12],max_step=ts)
            if not sol.success:
                raise RuntimeError(sol.message)
            self.nfev+=sol.nfev
            y=sol.y[:,-1]
            if not sol.t_events[0].size:
                break
            start=float(sol.t[-1])
            y[1]=0.
            if abs(drive0)<=p.Fs+1e-10:
                self.sticking=True
                break
            direction=math.copysign(1.,drive0)
        else:
            raise RuntimeError('单周期零速事件过密')
        self.state=y
        self.t=end
        return self.state.copy()


class ClassroomController:
    """公共PID；标称/完整组只在实测工作点增益调度开关上有差别。"""
    def __init__(self, p, ts, enable_ff, enable_gain_sched):
        self.p,self.ts=p,ts
        self.enable_ff,self.enable_gain_sched=enable_ff,enable_gain_sched
        self.integral=self.previous_error=self.derivative=0.

    def update(self, reference, measured_position, pressure_pa):
        xd,vd,ad,jd=reference
        p=self.p
        error=xd-measured_position
        self.integral+=error*self.ts
        raw_derivative=(error-self.previous_error)/self.ts
        alpha=math.exp(-self.ts/p.derivative_tau_s)
        self.derivative=alpha*self.derivative+(1-alpha)*raw_derivative
        self.previous_error=error
        fb=p.kp*error+p.ki*self.integral+p.kd*self.derivative
        ff=0.
        inverse_force=0.
        linear_v=0.
        if self.enable_ff:
            direction=next((np.sign(z) for z in (vd,ad,jd) if abs(z)>1e-12),0.)
            # 式6.1的净运动力；准静态外载平衡在对象工作点里消去。
            inverse_force=p.mass*ad+p.sigma*vd+stribeck(vd,p,direction)
            linear_v=(inverse_force+p.hydraulic_damping*vd)/p.force_gain
            if self.enable_gain_sched:
                measured_delta=max(p.supply_pa-pressure_pa,p.minimum_delta_pa)
                linear_v/=math.sqrt(measured_delta/(p.supply_pa-p.nominal_pressure_pa))
            # 式6.3：带零位保护的分段逆函数。
            ff=linear_v+math.copysign(p.deadzone_v,linear_v) if abs(linear_v)>1e-6 else 0.
        requested=ff+fb
        u=float(np.clip(requested,-p.voltage_limit_v,p.voltage_limit_v))
        # 截断记忆式抗积分饱和；正常课堂曲线并不触发此分支。
        if u!=requested and p.ki:
            self.integral+=(u-requested)/p.ki
        return u,ff,fb,inverse_force,linear_v


def _metrics(d,p,release_times,release_forces):
    t=d['time_s'];v=d['velocity_m_s'];vd=d['reference_velocity_m_s'];e=d['error_m']*1000;u=d['voltage_v']
    start=t<=.16;contact=(t>=p.load_start)&(t<=p.load_start+.12)
    light=(t>=p.load_start-.10)&(t<p.load_start-.01)
    heavy=(t>=p.load_start+.06)&(t<p.load_start+.14)
    motion=np.flatnonzero(abs(v)>1e-4);reference_motion=np.flatnonzero(abs(vd)>1e-4)
    first_ff=np.flatnonzero(abs(d['feedforward_v'])>1e-9)
    reference_ms=float(t[reference_motion[0]]*1000) if len(reference_motion) else None
    motion_ms=float(t[motion[0]]*1000) if len(motion) else None
    return dict(max_abs_error_mm=float(max(abs(e))),rms_error_mm=float(np.sqrt(np.mean(e**2))),
        final_error_mm=float(e[-1]),motion_start_ms=motion_ms,reference_threshold_ms=reference_ms,
        startup_delay_from_reference_ms=motion_ms-reference_ms if motion_ms is not None else None,
        first_breakaway_ms=float(release_times[0]*1000) if release_times else None,
        first_breakaway_net_force_n=float(release_forces[0]) if release_forces else None,
        first_feedforward_ms=float(t[first_ff[0]]*1000) if len(first_ff) else None,
        first_feedforward_voltage_v=float(d['feedforward_v'][first_ff[0]]) if len(first_ff) else None,
        startup_initial_voltage_v=float(u[0]),startup_velocity_threshold_m_s=1e-4,
        startup_max_abs_error_mm=float(max(abs(e[start]))),
        startup_velocity_overshoot_m_s=float(max(v[start]-vd[start])),
        startup_velocity_rmse_m_s=float(np.sqrt(np.mean((v[start]-vd[start])**2))),
        contact_min_velocity_m_s=float(min(v[contact])),
        contact_speed_drop_percent=float(100*max((vd[contact]-v[contact])/vd[contact])),
        contact_max_abs_error_mm=float(max(abs(e[contact]))),
        light_median_voltage_v=float(np.median(u[light])),heavy_median_voltage_v=float(np.median(u[heavy])),
        light_median_p1_mpa=p.nominal_pressure_pa/1e6,heavy_median_p1_mpa=p.heavy_pressure_pa/1e6,
        kq_decline_percent=100*(1-p.heavy_gain_ratio),
        peak_abs_voltage_v=float(max(abs(u))),saturation_time_s=float(np.sum(abs(u[:-1])>=p.voltage_limit_v-1e-8)*(t[1]-t[0])),
        min_absolute_pressure_mpa=float(min(d['p1_pa'].min(),d['p2_pa'].min())/1e6),
        max_absolute_pressure_mpa=float(max(d['p1_pa'].max(),d['p2_pa'].max())/1e6),
        meets_full_0p05mm=bool(max(abs(e))<.05),meets_full_0p1mm=bool(max(abs(e))<.1))


def run_classroom(ts=.001, parameters=None, rtol=1e-8):
    """返回 pid/nominal/full:{data, metrics, config}，全部曲线来自独立ODE积分。"""
    if not math.isfinite(ts) or ts<=0 or ts>.002:
        raise ValueError('课堂采样周期应在(0,2ms]')
    if not math.isfinite(rtol) or not 0<rtol<1:
        raise ValueError('rtol必须在(0,1)')
    p=parameters or ClassroomParameters()
    t,xd,vd,ad,jd=generate_process_scurve(dt=ts)
    results={}
    for name,ff,gain_sched in [('pid',False,False),('nominal',True,False),('full',True,True)]:
        plant=ClassroomPlant(p,rtol)
        controller=ClassroomController(p,ts,ff,gain_sched)
        fields=['position_m','velocity_m_s','error_m','voltage_v','feedforward_v','feedback_v','p1_pa','p2_pa',
                'load_n','kq_ratio','equivalent_velocity_gain_m_s_v','net_drive_force_n','friction_force_n','sticking',
                'planned_inverse_force_n','linear_voltage_v']
        data={field:np.zeros_like(t) for field in fields}
        for i,ti in enumerate(t):
            x,v=plant.state
            load,pressure,gain=plant.operating_point(ti)
            u,uff,ufb,force,linear=controller.update((xd[i],vd[i],ad[i],jd[i]),x,pressure)
            net=plant.net_force(u,gain)
            friction=net if plant.sticking and abs(net)<=p.Fs else stribeck(v,p,np.sign(net))
            # P2仅为准静态力平衡诊断：不是独立积分的腔压。
            drive_motion=p.force_gain*gain*physical_deadzone(u,p)-p.hydraulic_damping*v
            p2=(p.A1*pressure-load-drive_motion)/p.A2
            # kq比例可用于孔口灵敏度比较；绝对尺度是定义的等效量。
            kq=(p.force_gain/(p.hydraulic_damping+p.sigma))*gain
            values=[x,v,xd[i]-x,u,uff,ufb,pressure,p2,load,gain,kq,net,friction,
                    float(plant.sticking),force,linear]
            for field,value in zip(fields,values):data[field][i]=value
            if i+1<len(t):plant.step(u,ts)
        data.update(time_s=t.copy(),reference_m=xd.copy(),reference_velocity_m_s=vd.copy(),
                    reference_acceleration_m_s2=ad.copy(),reference_jerk_m_s3=jd.copy(),
                    measured_m=data['position_m'].copy())
        config=dict(model='classroom_quasi_static_pressure_reduction',model_kind='classroom_quasi_static_pressure_reduction',name=name,ts=ts,
            enable_ff=ff,enable_gain_sched=gain_sched,known_load=False,load_start=p.load_start,
            load_end=p.load_end,load_force=p.load_force_n,load_ramp=0.,unload_ramp=0.,
            valve_tau=0.,adc_bits=0,sensor_lsb_mm=0.,physical_parameters=asdict(p),
            classroom_parameters=asdict(p),gain_ratio_heavy=p.heavy_gain_ratio,
            heavy_pressure_pa=p.heavy_pressure_pa,rtol=rtol,ode_nfev=plant.nfev,
            friction_release_times_s=plant.release_times,friction_release_net_forces_n=plant.release_forces,
            pressure_assumption='Quasi-static operating pressure; no fluid/valve energy storage',
            equivalent_velocity_gain_notice='Velocity/voltage gain [m/(s V)] of the reduced impedance; kq_ratio follows the physical orifice square-root ratio.',
            leakage_compensation='No separate laminar-leak branch in this reduction; physical mode implements C_ip.')
        results[name]=dict(data=data,metrics=_metrics(data,p,plant.release_times,plant.release_forces),config=config)
    return results


if __name__=='__main__':
    import json
    for key,result in run_classroom().items():
        print(key,json.dumps(result['metrics'],ensure_ascii=False))
