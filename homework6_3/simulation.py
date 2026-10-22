"""作业6-3共同采样环、原始数据及量纲明确的定量指标。"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.signal import butter, sosfiltfilt
from deadzone_model import deadzone
from input_deadzone_model import InputDeadzoneVelocityPlant
from model_assisted_controller import ModelAssistedController, inverse_deadzone

ROOT = Path(__file__).resolve().parent
STRATEGIES = ("under", "hard_over", "smooth")


def build_plant(config):
    return InputDeadzoneVelocityPlant(gain_m_s_v=config['gain_m_s_v'], tau_valve_s=config['tau_valve_s'],
        tau_velocity_s=config['tau_velocity_s'], deadzone_v=config['physical_deadzone_v'])


def load_config(path=None):
    config = json.loads((Path(path) if path else ROOT/"experiment_config.json").read_text(encoding="utf-8"))
    positive = ("duration_s", "sample_time_s", "amplitude_v", "frequency_hz", "physical_deadzone_v",
                "gain_m_s_v", "tau_valve_s", "tau_velocity_s", "beta_default", "chatter_cutoff_hz")
    if any(not math.isfinite(config[k]) or config[k] <= 0 for k in positive):
        raise ValueError("周期、时钟、模型及信号参数必须为有限正数")
    if (abs(config['duration_s']/config['sample_time_s']-round(config['duration_s']/config['sample_time_s'])) > 1e-8
            or not 0 < config['chatter_cutoff_hz'] < .5/config['sample_time_s']
            or not 0 <= config['zoom_start_s'] < config['zoom_end_s'] <= config['duration_s']
            or config['measurement_noise_std_mm_s'] < 0):
        raise ValueError("采样网格、放大窗口、噪声或高通频率设置无效")
    return config


def simulate(strategy, config, beta=None):
    if strategy not in STRATEGIES:
        raise ValueError("策略仅支持 under/hard_over/smooth")
    dt = config['sample_time_s']
    time = np.arange(round(config['duration_s']/dt)+1)*dt
    omega = 2*np.pi*config['frequency_hz']
    nominal = config['amplitude_v']*np.sin(omega*time)
    acceleration = config['gain_m_s_v']*config['amplitude_v']*omega*np.cos(omega*time)
    velocity_ref = config['gain_m_s_v']*nominal
    position_ref = config['gain_m_s_v']*config['amplitude_v']/omega*(1-np.cos(omega*time))
    plant = build_plant(config)
    estimate = config['under_estimate_v'] if strategy == 'under' else config['over_estimate_v']
    actual_beta = config['beta_default'] if beta is None else beta
    controller = ModelAssistedController(
        dict(kp=config['kp'], ki=config['ki'], kd=config['kd'], ts=dt, alpha=config['derivative_alpha']),
        config, estimate_v=estimate, mode='tanh' if strategy == 'smooth' else 'hard', beta=actual_beta)
    # 同一噪声序列用于全部策略；默认关闭噪声，以检查颤振是否能由闭环自行产生。
    noise = np.random.default_rng(config['noise_seed']).normal(
        0., config['measurement_noise_std_mm_s']/1000., len(time))
    names = ('control_voltage_v', 'valve_voltage_v', 'effective_voltage_v', 'velocity_mm_s',
             'linear_voltage_v', 'measured_velocity_mm_s', 'feedback_voltage_v', 'raw_voltage_v',
             'valve_effective_voltage_v', 'delayed_error_equivalent_v', 'lowpass1_v', 'lowpass2_v',
             'highpass1_v', 'highpass2_v', 'washout_feedback_voltage_v')
    data = {name: np.empty(len(time)) for name in names}
    for index in range(len(time)):
        measured = plant.velocity_m_s+noise[index]
        voltage, _, _ = controller.update(
            [position_ref[index], velocity_ref[index], acceleration[index]], [0., measured])
        values = (voltage, plant.valve_voltage_v, deadzone(voltage, config['physical_deadzone_v']),
                  plant.velocity_m_s*1000., controller.last['linear_voltage_v'], measured*1000.,
                  controller.last['feedback_linear_v'], controller.last['raw_voltage_v'],
                  plant.valve_voltage_v, controller.last['delayed_error_equivalent_v'],
                  controller.last['lowpass1_v'], controller.last['lowpass2_v'],
                  controller.last['highpass1_v'], controller.last['highpass2_v'],
                  controller.last['washout_feedback_voltage_v'])
        for name, value in zip(names, values):
            data[name][index] = value
        if index+1 < len(time):
            plant.step(voltage, dt)
    data = dict(time_s=time, reference_voltage_v=nominal, reference_velocity_mm_s=1000.*velocity_ref, **data)
    return dict(strategy=strategy, beta=actual_beta if strategy == 'smooth' else None,
                data=data, metrics=calculate_metrics(data, config))


def _intervals_within_band(t, values, threshold):
    """对采样信号作分段线性阈值交点统计；精度随采样周期收敛。

    不以一个零值采样点虚构1ms平台；实际ZOH阀动态在相邻采样间有曲率，
    因而该统计是采样重建指标，不冒充连续时间精确事件时长。
    """
    pieces = []
    for i in range(len(t)-1):
        a, b = values[i], values[i+1]
        fractions = [0., 1.]
        if b != a:
            fractions += [(level-a)/(b-a) for level in (-threshold, threshold)
                          if 0. < (level-a)/(b-a) < 1.]
        fractions.sort()
        for left, right in zip(fractions[:-1], fractions[1:]):
            if abs(a+(b-a)*(left+right)/2.) <= threshold:
                start, end = t[i]+(t[i+1]-t[i])*left, t[i]+(t[i+1]-t[i])*right
                if pieces and abs(start-pieces[-1][1]) < 1e-10:
                    pieces[-1][1] = float(end)
                else:
                    pieces.append([float(start), float(end)])
    return pieces


def _linear_switching_metrics(time, linear_voltage, zoom):
    """Measure hard-inverse sign switches from the nonzero linear demand.

    Interpolate each zero crossing between adjacent nonzero samples.  At least
    five switches (two complete cycles) are required for a frequency estimate;
    a single reversal is not evidence of sustained chatter.
    """
    time = np.asarray(time)[zoom]
    linear_voltage = np.asarray(linear_voltage)[zoom]
    nonzero = linear_voltage != 0.
    time, linear_voltage = time[nonzero], linear_voltage[nonzero]
    switches = np.flatnonzero(np.sign(linear_voltage[:-1]) != np.sign(linear_voltage[1:]))
    crossings = time[switches] - linear_voltage[switches] * (
        time[switches+1] - time[switches]) / (
        linear_voltage[switches+1] - linear_voltage[switches])
    frequency = (float(1. / (2. * np.median(np.diff(crossings))))
                 if len(crossings) >= 5 else None)
    return dict(linear_sign_reversals=int(len(crossings)),
                chatter_first_switch_s=float(crossings[0]) if len(crossings) else None,
                chatter_last_switch_s=float(crossings[-1]) if len(crossings) else None,
                chatter_frequency_hz=frequency)


def calculate_metrics(data, config):
    t, u = data['time_s'], data['control_voltage_v']
    zoom = (t >= config['zoom_start_s']-1e-12) & (t <= config['zoom_end_s']+1e-12)
    flat = _intervals_within_band(t[zoom], data['control_voltage_v'][zoom], config['physical_deadzone_v'])
    near_zero = _intervals_within_band(t[zoom], data['velocity_mm_s'][zoom], config['velocity_flat_threshold_mm_s'])
    sos = butter(4, config['chatter_cutoff_hz'], btype='highpass', fs=1/config['sample_time_s'], output='sos')
    high = sosfiltfilt(sos, u)
    velocity_high = sosfiltfilt(sos, data['velocity_mm_s'])
    error = data['velocity_mm_s']-data['reference_velocity_mm_s']
    switching = _linear_switching_metrics(t, data['linear_voltage_v'], zoom)
    velocity_steps = np.diff(data['velocity_mm_s'][zoom])
    nonzero_steps = np.sign(velocity_steps[abs(velocity_steps) > 1e-10])
    return dict(
        velocity_rmse_mm_s=float(np.sqrt(np.mean(error**2))),
        zoom_velocity_rmse_mm_s=float(np.sqrt(np.mean(error[zoom]**2))),
        longest_flow_closed_ms=1000.*max((b-a for a,b in flat), default=0.),
        longest_effective_zero_ms=1000.*max((b-a for a,b in flat), default=0.),
        total_flow_closed_ms=1000.*sum(b-a for a,b in flat),
        longest_velocity_near_zero_ms=1000.*max((b-a for a,b in near_zero), default=0.),
        flow_closed_intervals_s=flat,
        chatter_voltage_pp_v=float(np.ptp(high[zoom])),
        chatter_voltage_rms_v=float(np.sqrt(np.mean(high[zoom]**2))),
        chatter_velocity_pp_mm_s=float(np.ptp(velocity_high[zoom])),
        chatter_velocity_rms_mm_s=float(np.sqrt(np.mean(velocity_high[zoom]**2))),
        max_voltage_step_v=float(np.max(abs(np.diff(u[zoom])))),
        **switching,
        velocity_turning_points=int(np.count_nonzero(np.diff(nonzero_steps))),
        velocity_excess_variation_mm_s=float(np.sum(abs(velocity_steps))-
            abs(data['velocity_mm_s'][zoom][-1]-data['velocity_mm_s'][zoom][0])),
        max_abs_voltage_v=float(np.max(abs(u))),
        max_abs_effective_voltage_v=float(np.max(abs(data['effective_voltage_v']))),
        max_abs_velocity_mm_s=float(np.max(abs(data['velocity_mm_s']))),
        sampled_saturation_count=int(np.count_nonzero(abs(u) >= 10.-1e-9)),
        metric_notes=f"平台为D(u)输出零区间采样重建，不等于动态阀口零流量；legacy flow_closed字段为兼容别名；颤振为{config['chatter_cutoff_hz']:g}Hz高通诊断")


def static_thresholds(config):
    """无阀滞后、无反馈的静态解析参照，不与动态闭环平台混淆。"""
    delta, estimate = config['physical_deadzone_v'], config['over_estimate_v']
    omega, amplitude = 2*np.pi*config['frequency_hz'], config['amplitude_v']
    records = []
    for beta in config['beta_values']:
        root = brentq(lambda z: inverse_deadzone(z, estimate, 'tanh', beta)-delta, 0., delta)
        records.append(dict(beta=beta, linear_threshold_v=root,
            static_plateau_ms=1000.*2*math.asin(root/amplitude)/omega,
            inverse_zero_slope=1+estimate*beta))
    residual = max(0., delta-config['under_estimate_v'])
    return dict(under_static_plateau_ms=1000.*2*math.asin(residual/amplitude)/omega, smooth=records)


def save_outputs(results, beta_results, config, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, result in [*results.items(), *((f'beta_{beta:g}', result) for beta, result in beta_results.items())]:
        with (destination/f'{name}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(result['data'])
            writer.writerows(zip(*result['data'].values()))
    summary = dict(config=config, model='输入死区D(u)→一阶有效阀口→一阶速度；二级动态残差反馈及'
                   f"{config['feedback_delay_samples']}采样缓存延迟，非全双腔模型",
                   strategies={k:dict(metrics=v['metrics'], beta=v['beta']) for k,v in results.items()},
                   beta_scan=[dict(beta=beta, metrics=r['metrics']) for beta,r in beta_results.items()],
                   static_reference=static_thresholds(config))
    (destination/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write_metrics_table(results, beta_results, destination)
    return summary


def write_metrics_table(results, beta_results, destination):
    """将控制台同一份定量表保存为 TXT 和方便导入 Excel 的 CSV。"""
    destination = Path(destination)
    rows = [*results.items(), *((f'beta={b:g}', r) for b,r in beta_results.items())]
    keys = ('longest_effective_zero_ms', 'longest_velocity_near_zero_ms', 'chatter_voltage_pp_v',
            'chatter_velocity_pp_mm_s', 'velocity_rmse_mm_s', 'linear_sign_reversals')
    lines = ['策略 / beta       有效信号零平台(ms)  近零速度平台(ms)  高频电压pp(V)  高频速度pp(mm/s)  速度RMSE(mm/s)  符号翻转']
    for name, result in rows:
        m = result['metrics']
        lines.append(f'{name:<16} {m[keys[0]]:>10.3f} {m[keys[1]]:>17.3f} '
                     f'{m[keys[2]]:>15.4f} {m[keys[3]]:>18.4f} {m[keys[4]]:>15.5f} {m[keys[5]]:>8d}')
    (destination/'metrics_table.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    with (destination/'metrics_table.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('strategy', *keys))
        writer.writerows((name, *(r['metrics'][k] for k in keys)) for name,r in rows)
    return '\n'.join(lines)
