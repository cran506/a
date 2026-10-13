"""第六章：公共参考、三组相同四阶 Twin 与 1 kHz 复合闭环。"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.signal import find_peaks
from trajectory import generate_process_scurve
from twin_adapter import ClosedLoopTwin,QuantizedSensor,classroom_config
from model_assisted_controller import ModelAssistedController


@dataclass
class Experiment:
    name: str
    enable_ff: bool = True
    enable_gain_sched: bool = True
    kp: float = 4.
    ki: float = 450.
    kd: float = .08
    alpha: float = .85
    ts: float = .001
    duration: float = 1.2
    valve_tau: float = .0025
    pressure_alpha: float = .2
    gain_prediction_s: float = .001
    force_error_gain_s: float = 80.
    startup_transition_s: float = .008
    load_force: float = 153000.
    load_ramp: float = .025
    unload_ramp: float = .03
    load_start: float = .5
    load_end: float = .92
    known_load: bool = True
    load_preview_s: float = 0.
    adc_bits: int = 16
    sensor_span: float = .3
    rtol: float = 1e-6


def planned_load(t,cfg):
    """独立公开的工艺参考，既供被控对象加载也供已知负载前馈使用。"""
    if t<cfg.load_start-1e-10 or t>=cfg.load_end-1e-10:
        return 0.,0.
    rise=(t-cfg.load_start)/cfg.load_ramp if cfg.load_ramp else 1.
    fall=(cfg.load_end-t)/cfg.unload_ramp if cfg.unload_ramp else 1.
    fraction=np.clip(min(rise,fall,1.),0.,1.)
    force=cfg.load_force*.5*(1-np.cos(np.pi*fraction))
    if rise<1 and rise<=fall and cfg.load_ramp:
        rate=cfg.load_force*.5*np.pi/cfg.load_ramp*np.sin(np.pi*fraction)
    elif fall<1 and cfg.unload_ramp:
        rate=-cfg.load_force*.5*np.pi/cfg.unload_ramp*np.sin(np.pi*fraction)
    else:rate=0.
    return float(force),float(rate)


def simulate(cfg):
    p=classroom_config();p.update(ambient_pressure=1e5,makeup_area=.001)
    twin=ClosedLoopTwin(config=p,load_force=cfg.load_force,
                        load_window=(cfg.load_start,cfg.load_end),load_ramp=cfg.load_ramp,
                        unload_ramp=cfg.unload_ramp,valve_tau=cfg.valve_tau,rtol=cfg.rtol)
    sensor=QuantizedSensor(cfg.adc_bits,cfg.sensor_span)
    origin=sensor.sample(.01)
    twin.reset([origin,0.,p['Pr'],p['Pr']])
    pid=dict(kp=cfg.kp,ki=cfg.ki,kd=cfg.kd,ts=cfg.ts,alpha=cfg.alpha,integral_separation=False)
    controller=ModelAssistedController(pid,p,cfg.enable_ff,cfg.enable_gain_sched,
                                      pressure_alpha=cfg.pressure_alpha,valve_tau=cfg.valve_tau,
                                      gain_prediction_s=cfg.gain_prediction_s,force_error_gain_s=cfg.force_error_gain_s,
                                      startup_transition_s=cfg.startup_transition_s)
    t,r,rv,ra,jerk=generate_process_scurve(dist=.13,T_move=1.,T_total=cfg.duration,dt=cfg.ts)
    names=['position_m','velocity_m_s','p1_pa','p2_pa','measured_m','error_m',
           'voltage_v','feedforward_v','feedback_v','requested_voltage_v',
           'feedforward_requested_v','valve_target_v','valve_normalized',
           'load_n','planned_force_n','kq_m2_s','pressure_filtered_1_pa',
           'pressure_filtered_2_pa','inverse_gain_n_s_v','force_rate_ref_n_s',
           'holding_deadzone_compensation_v','reference_force_n','measured_force_n',
           'planned_force_rate_n_s','force_error_correction_n_s','deadzone_offset_v',
           'unshaped_valve_target_v','startup_weight']
    data={k:np.empty_like(t) for k in names}
    for i,ti in enumerate(t):
        x,v,p1,p2=twin.state
        measured=sensor.sample(x)-origin
        force,force_rate=planned_load(ti+cfg.load_preview_s,cfg) if cfg.known_load else (0.,0.)
        u,ff,fb=controller.update((r[i],rv[i],ra[i]),(measured,v),(p1,p2),jerk_d=jerk[i],
                                  known_force_n=force,known_force_rate_n_s=force_rate,origin_m=origin)
        info=controller.last
        values=[x-origin,v,p1,p2,measured,r[i]-(x-origin),u,ff,fb,info['requested_voltage_v'],
                info['feedforward_requested_v'],info['valve_target_v'],twin.valve_state,
                twin.load_at(ti),force,twin.extension_kq(),*controller.pressure_filtered,
                info.get('inverse_gain_n_s_v',0.),info.get('desired_force_rate_n_s',0.),
                info['holding_deadzone_compensation_v'],info.get('desired_force_n',0.),
                info.get('actual_force_n',0.),info.get('planned_force_rate_n_s',0.),
                info.get('force_error_correction_n_s',0.),info.get('deadzone_offset_v',0.),
                info.get('unshaped_valve_target_v',0.),info.get('startup_weight',0.)]
        for key,value in zip(names,values):data[key][i]=value
        if i+1<len(t):twin.step_voltage(u,cfg.ts)
    if not np.isclose(twin.t,t[-1],rtol=0.,atol=1e-10):
        raise RuntimeError('被控对象积分时钟与采样时钟不同步')
    data.update(time_s=t,reference_m=r,reference_velocity_m_s=rv,
                reference_acceleration_m_s2=ra,reference_jerk_m_s3=jerk)
    meta=asdict(cfg);meta.update(origin_m=origin,sensor_lsb_mm=sensor.lsb*1000,
                               k_flow_m3_s_v_sqrt_pa=controller.k_flow,
                               nominal_pressures_pa=controller.nominal_pressures.tolist(),
                               radau_nfev=twin.nfev,physical_parameters=twin.p.copy())
    return dict(data=data,config=meta,metrics=metrics(data,cfg))


def metrics(d,cfg):
    t=d['time_s'];e=d['error_m']*1000;v=d['velocity_m_s'];vr=d['reference_velocity_m_s'];u=d['voltage_v']
    startup=t<=.16;contact=(t>=.5)&(t<=.62);light=(t>=.4)&(t<.49);heavy=(t>=.56)&(t<.64)
    threshold=1e-4
    started=np.flatnonzero(abs(v)>threshold);reference_started=np.flatnonzero(abs(vr)>threshold)
    ff_started=np.flatnonzero(abs(d['feedforward_v'])>1e-8)
    atload=np.flatnonzero(np.isclose(t,.5))[0]
    peaks,_=find_peaks(v[startup]-vr[startup],prominence=.002)
    start_peaks,_=find_peaks(v[startup],prominence=.005)
    startup_indices=np.flatnonzero(startup)
    peak_index=int(startup_indices[start_peaks[0]]) if len(start_peaks) else None
    contact_indices=np.flatnonzero(contact)
    contact_error_index=int(contact_indices[np.argmax(e[contact])])
    stop=(t>=.95)&(t<=1.05)
    hold=t>=1.
    preopen=t<=max(.02,cfg.startup_transition_s+cfg.valve_tau)
    light_kq=float(np.nanmedian(d['kq_m2_s'][light]));heavy_kq=float(np.nanmedian(d['kq_m2_s'][heavy]))
    return dict(max_abs_error_mm=float(np.max(abs(e))),rms_error_mm=float(np.sqrt(np.mean(e**2))),
                final_error_mm=float(e[-1]),startup_velocity_threshold_m_s=threshold,
                motion_start_ms=float(t[started[0]]*1000) if len(started) else None,
                reference_threshold_ms=float(t[reference_started[0]]*1000) if len(reference_started) else None,
                first_feedforward_ms=float(t[ff_started[0]]*1000) if len(ff_started) else None,
                first_feedforward_voltage_v=float(d['feedforward_v'][ff_started[0]]) if len(ff_started) else None,
                startup_max_abs_error_mm=float(np.max(abs(e[startup]))),
                startup_max_voltage_step_v=float(np.max(abs(np.diff(u[preopen])))),
                startup_max_voltage_drop_v=float(max(0.,np.max(-np.diff(u[preopen])))),
                startup_peak_abs_voltage_v=float(np.max(abs(u[preopen]))),
                startup_initial_voltage_v=float(u[0]),
                startup_delay_from_reference_ms=float((t[started[0]]-t[reference_started[0]])*1000) if len(started) and len(reference_started) else None,
                startup_velocity_rmse_m_s=float(np.sqrt(np.mean((v[startup]-vr[startup])**2))),
                startup_velocity_overshoot_m_s=float(np.max(v[startup]-vr[startup])),
                startup_prominent_overshoot_peaks=int(len(peaks)),
                startup_first_peak_time_s=float(t[peak_index]) if peak_index is not None else None,
                startup_first_peak_velocity_m_s=float(v[peak_index]) if peak_index is not None else None,
                reference_velocity_at_0p16_m_s=float(vr[np.argmin(abs(t-.16))]),
                contact_min_velocity_m_s=float(np.min(v[contact])),
                contact_speed_drop_percent=float(100*np.max((vr[contact]-v[contact])/np.maximum(vr[contact],1e-6))),
                contact_max_abs_error_mm=float(np.max(abs(e[contact]))),
                contact_added_error_mm=float(np.max(e[contact]-e[atload])),
                contact_error_peak_time_s=float(t[contact_error_index]),
                error_at_0p60_mm=float(e[np.argmin(abs(t-.60))]),
                light_median_voltage_v=float(np.median(u[light])),heavy_median_voltage_v=float(np.median(u[heavy])),
                light_median_p1_mpa=float(np.median(d['p1_pa'][light])/1e6),
                heavy_median_p1_mpa=float(np.median(d['p1_pa'][heavy])/1e6),
                kq_decline_percent=float(100*(1-heavy_kq/light_kq)),
                peak_abs_voltage_v=float(np.max(abs(u))),
                stop_peak_abs_voltage_v=float(np.max(abs(u[stop]))),
                stop_max_voltage_step_v=float(np.max(abs(np.diff(u[stop])))),
                stop_voltage_total_variation_v=float(np.sum(abs(np.diff(u[stop])))),
                hold_max_abs_velocity_m_s=float(np.max(abs(v[hold]))),
                saturation_time_s=float(np.sum(abs(u[:-1])>=10-1e-8)*cfg.ts),
                min_absolute_pressure_mpa=float(min(d['p1_pa'].min(),d['p2_pa'].min())/1e6),
                max_absolute_pressure_mpa=float(max(d['p1_pa'].max(),d['p2_pa'].max())/1e6),
                meets_full_0p05mm=bool(np.max(abs(e))<.05),meets_full_0p1mm=bool(np.max(abs(e))<.1))
