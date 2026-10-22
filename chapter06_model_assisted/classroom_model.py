"""第六章课堂约化对象：真实阀动态、静摩擦事件与代数负载压力。

流量和力学连续性为 q=A1*v+Ct*g*(m*a+sigma*v+Fstr(v))/A1，
g=sqrt((Ps-P1)/(Ps-P1_0))。由此反解滑动加速度，静止时检验
|A1*q/(Ct*g)|<=Fs。m*Ct/A1²=13.5ms，接近给定 tau_m=12ms，
因此不额外串联重复的机械惯性。P1含准静态工艺负载，不是独立压力积分。
对象流量增益以代数负载工作点压力计算；交给控制器的测量 P1 另含
运动受力所需压力，因此两者留下小幅、可在数据中核验的模型残差。
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from textbook_controller import ModelAssistedController
from textbook_trajectory import generate_textbook_trajectory

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ClassroomParameters:
    A1: float=.01
    A2: float=.005
    L: float=1.
    m: float=500.
    Ps: float=21e6
    Pr: float=.1e6
    P1_0: float=5e6
    P2_0: float=10e6
    K_flow: float=2.7027e-7
    deadzone: float=.4
    u_break: float=1.2
    Ct: float=2.7027e-9
    Fs: float=3100.
    Fc: float=400.
    vs: float=.065
    sigma: float=3000.
    gamma: float=800.
    tau_m: float=.012
    tau_v: float=.0008
    heavy_drive_pressure_pa: float=16.95e6
    load_start: float=.5
    load_end: float=.92
    kp_v_m: float=770.
    ki_v_m_s: float=25000.
    kd_v_s_m: float=5.
    derivative_alpha: float=.85

    @property
    def load_force_n(self):
        return (self.heavy_drive_pressure_pa-self.P1_0)*self.A1

    @property
    def passive_time_constant_s(self):
        return self.Ct*self.m/self.A1**2


def default_parameters():
    return ClassroomParameters(**json.loads((ROOT/'textbook_config.json').read_text(encoding='utf-8')))


def physical_deadzone(voltage, threshold):
    return math.copysign(max(abs(voltage)-threshold,0.),voltage)


class ClassroomPlant:
    """相同物理参数与初态在三组中独立积分；事件精确定位粘滑。"""
    def __init__(self,p,rtol=1e-8):
        self.p,self.rtol=p,rtol
        self.state=np.zeros(3)  # x[m], v[m/s], valve[V]
        self.t=0.
        self.sticking=True
        self.direction=0
        self.force_n=0.
        self.friction_n=0.
        self.releases=[]
        self.resticks=[]
        self.nfev=0

    def operating_point(self,t):
        p=self.p
        load=p.load_force_n if p.load_start-1e-10<=t<p.load_end-1e-10 else 0.
        base=p.P1_0+load/p.A1
        delta=p.Ps-base
        if delta<5e5:
            raise ValueError('负载工作点距供油压力不足0.5 MPa')
        return load,base,math.sqrt(delta/(p.Ps-p.P1_0))

    def pressures(self,t=None):
        _,base,_=self.operating_point(self.t if t is None else t)
        return base+self.force_n/self.p.A1,self.p.P2_0

    def flow(self,valve,base):
        return self.p.K_flow*math.sqrt(self.p.Ps-base)*physical_deadzone(valve,self.p.deadzone)

    def capacity(self,valve,base,gain):
        return self.p.A1*self.flow(valve,base)/(self.p.Ct*gain)

    def sliding_rhs(self,s,u,base,gain,direction):
        p=self.p
        v,z=s[1:]
        f=direction*(p.Fc+(p.Fs-p.Fc)*math.exp(-(v/p.vs)**2))
        a=(self.flow(z,base)-p.A1*v-p.Ct*gain*(p.sigma*v+f)/p.A1)/(p.Ct*gain*p.m/p.A1)
        return np.array([v,a,(u-z)/p.tau_v]),f

    def step(self,u,dt):
        if not math.isfinite(u) or not math.isfinite(dt) or dt<=0:
            raise ValueError('电压、步长必须有限且步长为正')
        u=float(np.clip(u,-10.,10.))
        p=self.p
        _,base,gain=self.operating_point(self.t)
        end=self.t+dt
        time=self.t
        s=self.state.copy()
        atol=[1e-11,1e-10,1e-9]
        switches=0
        while time<end-1e-14:
            if self.sticking:
                capacity=self.capacity(s[2],base,gain)
                if abs(capacity)>p.Fs+1e-7:
                    self.sticking=False
                    self.direction=int(np.sign(capacity))
                    self.releases.append({'time_s':time,'force_n':capacity})
                    switches+=1
                    continue

                def rhs(_t,y):return [0.,0.,(u-y[2])/p.tau_v]
                def release(_t,y):return abs(self.capacity(y[2],base,gain))-p.Fs
                release.terminal=True
                release.direction=1
                sol=solve_ivp(rhs,(time,end),s,method='RK45',events=release,
                              rtol=self.rtol,atol=atol)
                if not sol.success:raise RuntimeError(sol.message)
                self.nfev+=sol.nfev
                s,time=sol.y[:,-1],float(sol.t[-1])
                s[1]=0.
                self.force_n=self.capacity(s[2],base,gain)
                self.friction_n=self.force_n
                if sol.t_events[0].size:
                    self.sticking=False
                    self.direction=int(np.sign(self.force_n))
                    self.releases.append({'time_s':time,'force_n':self.force_n})
                    switches+=1
                    continue
                break

            direction=self.direction
            start=time
            def rhs(_t,y):return self.sliding_rhs(y,u,base,gain,direction)[0]
            def stop(_t,y):return y[1] if _t>start+1e-12 else direction*1e-14
            stop.terminal=True
            stop.direction=-direction
            sol=solve_ivp(rhs,(time,end),s,method='RK45',events=stop,
                          rtol=self.rtol,atol=atol)
            if not sol.success:raise RuntimeError(sol.message)
            self.nfev+=sol.nfev
            s,time=sol.y[:,-1],float(sol.t[-1])
            if not sol.t_events[0].size:
                derivatives,f=self.sliding_rhs(s,u,base,gain,direction)
                self.force_n=p.m*derivatives[1]+p.sigma*s[1]+f
                self.friction_n=f
                break
            switches+=1
            s[1]=0.
            capacity=self.capacity(s[2],base,gain)
            self.force_n=capacity
            if abs(capacity)<=p.Fs+1e-7:
                self.sticking=True
                self.direction=0
                self.friction_n=capacity
                self.resticks.append({'time_s':time,'force_n':capacity})
            else:
                self.direction=int(np.sign(capacity))
                self.releases.append({'time_s':time,'force_n':capacity})
            if switches>30:raise RuntimeError(f'静摩擦事件过密 t={time:.9f}s')
        self.state=s
        self.t=end
        if self.pressures()[0]<=0:raise RuntimeError('对象压力非正')
        return s.copy()


def _metrics(data,p,releases):
    t=data['time_s'];e=data['error_m']*1000;v=data['velocity_m_s'];vd=data['reference_velocity_m_s'];u=data['voltage_v']
    start=t<=.16
    load=(t>=p.load_start)&(t<=p.load_start+.12)
    light=(t>=p.load_start-.1)&(t<p.load_start-.01)
    heavy=(t>=p.load_start+.06)&(t<p.load_start+.10)
    moving=np.flatnonzero(abs(v)>1e-4)
    ref_moving=np.flatnonzero(abs(vd)>1e-4)
    ff_moving=np.flatnonzero(abs(data['feedforward_v'])>1e-9)
    ms=float(t[moving[0]]*1000) if len(moving) else None
    ref_ms=float(t[ref_moving[0]]*1000) if len(ref_moving) else None
    return dict(max_abs_error_mm=float(max(abs(e))),rms_error_mm=float(np.sqrt(np.mean(e**2))),
        final_error_mm=float(e[-1]),motion_start_ms=ms,reference_threshold_ms=ref_ms,
        startup_delay_from_reference_ms=ms-ref_ms if ms is not None else None,
        first_breakaway_ms=float(releases[0]['time_s']*1000) if releases else None,
        first_breakaway_net_force_n=float(releases[0]['force_n']) if releases else None,
        first_feedforward_ms=float(t[ff_moving[0]]*1000) if len(ff_moving) else None,
        first_feedforward_voltage_v=float(data['feedforward_v'][ff_moving[0]]) if len(ff_moving) else None,
        startup_initial_voltage_v=float(u[0]),startup_velocity_threshold_m_s=1e-4,
        startup_max_abs_error_mm=float(max(abs(e[start]))),
        startup_velocity_overshoot_m_s=float(max(v[start]-vd[start])),
        startup_velocity_rmse_m_s=float(np.sqrt(np.mean((v[start]-vd[start])**2))),
        contact_min_velocity_m_s=float(min(v[load])),
        contact_speed_drop_percent=float(100*max((vd[load]-v[load])/np.maximum(abs(vd[load]),1e-9))),
        contact_max_abs_error_mm=float(max(abs(e[load]))),
        light_median_voltage_v=float(np.median(u[light])),heavy_median_voltage_v=float(np.median(u[heavy])),
        light_median_p1_mpa=float(np.median(data['p1_pa'][light])/1e6),
        heavy_median_p1_mpa=float(np.median(data['p1_pa'][heavy])/1e6),
        kq_decline_percent=float(100*(1-np.median(data['kq_ratio'][heavy])/np.median(data['kq_ratio'][light]))),
        peak_abs_voltage_v=float(max(abs(u))),
        saturation_time_s=float(np.sum(abs(u[:-1])>=10-1e-8)*(t[1]-t[0])),
        min_absolute_pressure_mpa=float(min(data['p1_pa'].min(),data['p2_pa'].min())/1e6),
        max_absolute_pressure_mpa=float(max(data['p1_pa'].max(),data['p2_pa'].max())/1e6),
        meets_full_0p05mm=bool(max(abs(e))<.05),meets_full_0p1mm=bool(max(abs(e))<.1))


def run_classroom(ts=.001,parameters=None,rtol=1e-8):
    if not math.isfinite(ts) or not 0<ts<=.002:raise ValueError('课堂采样周期须在(0,2ms]')
    if not math.isfinite(rtol) or not 0<rtol<1:raise ValueError('积分容差须在(0,1)')
    p=parameters or default_parameters()
    t,xd,vd,ad=generate_textbook_trajectory(dt=ts)
    full_params=asdict(p)
    physical={key:full_params[key] for key in json.loads((ROOT/'textbook_config.json').read_text(encoding='utf-8'))}
    pid=dict(kp=p.kp_v_m/100,ki=p.ki_v_m_s/100,kd=p.kd_v_s_m/100,
             ts=ts,alpha=p.derivative_alpha,integral_separation=False)
    results={}
    for name,enable_ff,gain_sched in [('pid',False,False),('nominal',True,False),('full',True,True)]:
        plant=ClassroomPlant(p,rtol)
        controller=ModelAssistedController(pid,physical,enable_ff,gain_sched)
        fields=('position_m','velocity_m_s','error_m','voltage_v','feedforward_v','feedback_v',
                'p1_pa','p2_pa','load_n','kq_ratio','equivalent_velocity_gain_m_s_v',
                'net_drive_force_n','friction_force_n','sticking','valve_voltage_v',
                'planned_inverse_force_n','linear_voltage_v','planned_leak_m3_s')
        data={key:np.zeros_like(t) for key in fields}
        for i,ti in enumerate(t):
            x,v,valve=plant.state
            load,base,gain=plant.operating_point(float(ti))
            pmeas=plant.pressures(float(ti))
            u,uff,ufb=controller.update((xd[i],vd[i],ad[i]),(x,v),pmeas)
            qgain=p.K_flow*math.sqrt(p.Ps-base)
            values=(x,v,xd[i]-x,u,uff,ufb,*pmeas,load,gain,
                    qgain/(p.A1+p.Ct*gain*p.sigma/p.A1),plant.force_n,plant.friction_n,
                    float(plant.sticking),valve,controller.last['F_drive_n'],
                    controller.last['u_lin_v'],controller.last['Q_leak_m3_s'])
            for key,val in zip(fields,values):data[key][i]=val
            if i+1<len(t):plant.step(u,ts)
        data.update(time_s=t,reference_m=xd,reference_velocity_m_s=vd,
                    reference_acceleration_m_s2=ad,measured_m=data['position_m'].copy())
        config=dict(model='classroom_pressure_impedance_reduction',name=name,ts=ts,rtol=rtol,
            enable_ff=enable_ff,enable_gain_sched=gain_sched,known_load=False,
            load_start=p.load_start,load_end=p.load_end,load_force=p.load_force_n,
            load_ramp=0.,unload_ramp=0.,tau_v=p.tau_v,
            pressure_assumption='algebraic process operating point plus measured motion force',
            plant_equation='q=A1*v+Ct*g*(m*a+sigma*v+Fstr(v))/A1',
            passive_time_constant_s=p.passive_time_constant_s,
            uploaded_example_tau_m_s=p.tau_m,
            classroom_parameters=asdict(p),physical_parameters=physical,
            sensor_lsb_mm=0.,adc_bits=0,ode_nfev=plant.nfev,
            friction_events=dict(release=plant.releases,restick=plant.resticks))
        results[name]=dict(data=data,metrics=_metrics(data,p,plant.releases),config=config)
    return results


if __name__=='__main__':
    for name,result in run_classroom().items():print(name,json.dumps(result['metrics'],ensure_ascii=False))
