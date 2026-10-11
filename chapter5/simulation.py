"""1kHz 感知—控制—ZOH—Radau闭环与当前运行指标。"""
from dataclasses import dataclass, asdict
from scipy.signal import find_peaks
import numpy as np
from controllers import DigitalPID, PositionalPID
from trajectory import generate_scurve, time_grid
from twin_adapter import ClosedLoopTwin, QuantizedSensor, classroom_config


@dataclass
class Experiment:
    name: str
    kind: str = 'scurve'
    controller: str = 'incremental'
    kp: float = 3.8
    ki: float = 9.
    kd: float = .02
    alpha: float = .85
    integral_separation: bool = True
    ts: float = .001
    duration: float = 1.4
    load_force: float = 180000.
    valve_tau: float = .0053
    load_ramp: float = .04
    makeup_area: float = .001
    ambient_pressure: float = 1e5
    rtol: float = 1e-5
    adc_bits: int = 12
    sensor_span: float = .3
    unload_ramp: float | None = .02


def simulate(cfg):
    if cfg.controller not in ('incremental', 'positional'):
        raise ValueError('controller仅支持incremental或positional')
    sensor = QuantizedSensor(cfg.adc_bits, cfg.sensor_span)
    if cfg.kind == 'scurve':
        t, r, rv, ra = generate_scurve(T_total=cfg.duration, dt=cfg.ts)
    elif cfg.kind == 'step':
        t = time_grid(cfg.duration, cfg.ts)
        r, rv, ra = np.full_like(t, .1), np.zeros_like(t), np.zeros_like(t)
    else:
        raise ValueError('kind 仅支持 scurve 或 step')
    cls = DigitalPID if cfg.controller == 'incremental' else PositionalPID
    pid = cls(cfg.kp, cfg.ki, cfg.kd, cfg.ts, alpha=cfg.alpha,
              integral_separation=cfg.integral_separation)
    twin = ClosedLoopTwin(config=classroom_config(), load_force=cfg.load_force,
                          valve_tau=cfg.valve_tau, rtol=cfg.rtol,
                          load_ramp=cfg.load_ramp, unload_ramp=cfg.unload_ramp,
                          makeup_area=cfg.makeup_area,
                          ambient_pressure=cfg.ambient_pressure)
    # 校准原点落在一个ADC代码上；初值避开端盖且液压力平衡。
    origin = sensor.sample(.05)
    p2_initial = (twin.p['Pr']*twin.p['A1']
                  -twin.ambient_pressure*(twin.p['A1']-twin.p['A2']))/twin.p['A2']
    twin.reset([origin, 0., twin.p['Pr'], p2_initial])
    names = ['position_m', 'velocity_m_s', 'p1_pa', 'p2_pa', 'measured_m',
             'error_m', 'sampled_error_m', 'voltage_v', 'raw_voltage_v',
             'derivative_v', 'integral_state_v', 'valve_normalized', 'load_n', 'kq_m2_s']
    data = {k: np.empty_like(t) for k in names}
    for i, ti in enumerate(t):
        if not np.isclose(twin.t, ti, atol=1e-10, rtol=0):
            raise RuntimeError('控制时钟与物理时钟不同步')
        x, vel, p1, p2 = twin.state
        pos = x-origin
        feedback = sensor.sample(x)-origin
        u = pid.update(r[i], feedback)
        active = twin.load_at(ti)
        values = [pos, vel, p1, p2, feedback, r[i]-pos, r[i]-feedback,
                  u, pid.last_raw, pid.last_d, pid.last_i, twin.valve_state,
                  active, twin.extension_kq()]
        for k, value in zip(names, values):
            data[k][i] = value
        if i+1 < len(t):
            twin.step_voltage(u, cfg.ts)
    data.update(time_s=t, reference_m=r, reference_velocity_m_s=rv, reference_acceleration_m_s2=ra)
    meta = asdict(cfg)
    meta.update(origin_m=origin, sensor_lsb_mm=sensor.lsb*1000, radau_nfev=twin.nfev,
                physical_parameters=twin.p.copy(),
                unload_ramp=twin.unload_ramp)
    return {'data': data, 'config': meta, 'metrics': metrics(data, cfg)}


def _first_time(t, mask):
    idx = np.flatnonzero(mask)
    return float(t[idx[0]]) if len(idx) else None


def metrics(d, cfg):
    t, e, u = d['time_s'], d['error_m']*1000, d['voltage_v']
    hold = t >= t[-1]-.2-1e-12
    out = dict(final_error_mm=float(e[-1]),
               final_200ms_max_abs_error_mm=float(np.max(abs(e[hold]))),
               final_200ms_rms_error_mm=float(np.sqrt(np.mean(e[hold]**2))),
               peak_abs_voltage_v=float(np.max(abs(u))),
               min_absolute_pressure_mpa=float(min(np.min(d['p1_pa']),np.min(d['p2_pa']))/1e6),
               max_absolute_pressure_mpa=float(max(np.max(d['p1_pa']),np.max(d['p2_pa']))/1e6),
               max_internal_voltage_v=float(np.max(d['raw_voltage_v'])),
               min_internal_voltage_v=float(np.min(d['raw_voltage_v'])),
               saturation_time_s=float(np.sum(abs(u[:-1])>=10-1e-8)*cfg.ts))
    at1 = np.flatnonzero(np.isclose(t, 1., atol=1e-10))
    out['error_at_1s_mm'] = float(e[at1[0]]) if len(at1) else None
    out['meets_1s_0p05mm'] = bool(abs(out['error_at_1s_mm']) < .05) if len(at1) else None
    if cfg.kind == 'scurve':
        heavy = (t >= .35) & (t < .70)
        # 排除加载瞬态；轻载与重载统计区间明确固定。
        light_plateau, heavy_plateau = (t >= .25)&(t < .34), (t >= .50)&(t < .65)
        kp_light = float(np.nanmedian(d['kq_m2_s'][light_plateau]))
        kp_heavy = float(np.nanmedian(d['kq_m2_s'][heavy_plateau]))
        late = t >= 1.1
        windows = []
        for start in (1., 1.1, 1.2, 1.3):
            window = (t >= start-1e-10) & (t < start+.1-1e-10)
            if not np.any(window):
                continue
            ew = e[window]
            peaks, _ = find_peaks(ew, prominence=.15, distance=max(1, round(.008/cfg.ts)))
            windows.append(dict(start_s=start, end_s=start+.1,
                                mean_error_mm=float(np.mean(ew)),
                                ac_rms_mm=float(np.std(ew)), p2p_mm=float(np.ptp(ew)),
                                prominent_peaks=int(len(peaks))))
        contact = (t >= .35)&(t < .45)
        plateau_u = float(np.median(u[heavy_plateau]))
        contact_peak_u = float(np.max(u[contact]))
        out.update(contact_peak_voltage_v=contact_peak_u,
                   heavy_plateau_median_voltage_v=plateau_u,
                   contact_voltage_above_plateau_v=max(0., contact_peak_u-plateau_u))
        out['hold_windows'] = windows
        out.update(heavy_peak_abs_error_mm=float(np.max(abs(e[heavy]))),
                   heavy_rms_error_mm=float(np.sqrt(np.mean(e[heavy]**2))),
                   unloaded_1p1_to_end_error_p2p_mm=float(np.ptp(e[late])),
                   unloaded_1p1_to_end_voltage_p2p_v=float(np.ptp(u[late])),
                   light_median_kq_m2_s=kp_light, heavy_median_kq_m2_s=kp_heavy,
                   kq_decline_percent=100*(1-kp_heavy/kp_light),
                   light_median_p1_mpa=float(np.median(d['p1_pa'][light_plateau])/1e6),
                   heavy_median_p1_mpa=float(np.median(d['p1_pa'][heavy_plateau])/1e6))
    else:
        zero = _first_time(t, d['sampled_error_m'] <= 0)
        sat = u >= 10-1e-8
        exits = np.flatnonzero(sat[:-1] & ~sat[1:])+1
        exit_t = float(t[exits[0]]) if len(exits) else None
        after = exits[t[exits] >= zero] if zero is not None else np.array([], dtype=int)
        # 明确区分首次退出，和过零之后再次退出；不假设它们相同。
        exit_after = float(t[after[0]]) if len(after) else None
        maxpos = float(np.max(d['position_m']))
        out.update(first_sampled_error_nonpositive_s=zero,
                   first_true_error_nonpositive_s=_first_time(t, d['error_m'] <= 0),
                   first_exit_positive_saturation_s=exit_t,
                   first_exit_minus_error_zero_s=exit_t-zero if exit_t is not None and zero is not None else None,
                   exit_positive_saturation_after_zero_s=exit_after,
                   voltage_at_sampled_zero_v=float(u[np.flatnonzero(d['sampled_error_m']<=0)[0]]) if zero is not None else None,
                   overshoot_mm=max(0., (maxpos-.1)*1000),
                   peak_position_mm=maxpos*1000)
    return out


def quantization_probe(result):
    """同一实际轨迹的量化残差重放，仅测量微分通道中的量化误差响应。"""
    d = result['data']
    q_error = d['position_m']-d['measured_m']
    trace = {'time_s': d['time_s'], 'quantization_error_mm': q_error*1000}
    for alpha, key in [(0., 'raw_derivative_v'), (.85, 'filtered_derivative_v')]:
        pid = DigitalPID(0., 0., result['config']['kd'], result['config']['ts'], alpha=alpha,
                         out_min=-1e6, out_max=1e6, integral_separation=False)
        trace[key] = np.array([pid.update(e, 0.) for e in q_error])
    mask = (d['time_s'] >= .1)&(d['time_s'] <= 1.)
    raw = trace['raw_derivative_v'][mask]
    filt = trace['filtered_derivative_v'][mask]
    raw_rms, filt_rms = float(np.sqrt(np.mean(raw**2))), float(np.sqrt(np.mean(filt**2)))
    reduction = 100*(1-filt_rms/raw_rms) if raw_rms > 0 else None
    return trace, dict(raw_derivative_rms_v=raw_rms, filtered_derivative_rms_v=filt_rms,
                       reduction_percent=reduction, window_s=[.1,1.],
                       method='同一保守组位移量化残差重放，Kp=Ki=0，仅微分通道')
