"""补充验证：共同测量噪声敏感性、零参考保持及去反馈反事实。

这些试验独立于默认无噪声主图。所有轨迹均由同一真实死区对象积分；
高频只可来自控制闭环与采样，不添加输出纹波，不裁剪实际速度。
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from deadzone_model import deadzone
from model_assisted_controller import ModelAssistedController
from simulation import STRATEGIES, load_config, simulate, build_plant


def _write_csv(path, data):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(data)
        writer.writerows(zip(*data.values()))


def _read_csv(path):
    table = np.genfromtxt(path, delimiter=',', names=True, encoding='utf-8')
    return {name: np.atleast_1d(table[name]).astype(float) for name in table.dtype.names}


def _controller(config, strategy, beta=None, kp=None):
    estimate = config['under_estimate_v'] if strategy == 'under' else config['over_estimate_v']
    return ModelAssistedController(
        dict(kp=config['kp'] if kp is None else kp, ki=config['ki'], kd=config['kd'],
             ts=config['sample_time_s'], alpha=config['derivative_alpha']),
        config, estimate_v=estimate, mode='tanh' if strategy == 'smooth' else 'hard',
        beta=config['beta_default'] if beta is None else beta)


def _reference_state(time, config):
    omega = 2*np.pi*config['frequency_hz']
    scale = config['gain_m_s_v']*config['amplitude_v']
    return [scale/omega*(1-np.cos(omega*time)), scale*np.sin(omega*time),
            scale*omega*np.cos(omega*time)]


def _upward_crossings(time, values):
    index = np.flatnonzero((values[:-1] <= 0.) & (values[1:] > 0.))
    return np.array([time[i]-values[i]*(time[i+1]-time[i])/(values[i+1]-values[i])
                     for i in index], dtype=float)


def _hold_metrics(data, tail_s=.1):
    time, velocity = data['time_s'], data['velocity_mm_s']
    tail = time >= time[-1]-tail_s-1e-12
    t, v = time[tail], velocity[tail]
    up = _upward_crossings(t, v)
    down = _upward_crossings(t, -v)
    pp = float(np.ptp(v))
    preceding = ((time >= time[-1]-2*tail_s-1e-12)
                 & (time <= time[-1]-tail_s+1e-12))
    previous_pp = float(np.ptp(velocity[preceding]))
    # A frequency estimate below this stated diagnostic amplitude has no physical
    # meaning here. This only selects a metric, never changes any state or data.
    meaningful = bool(pp > 1e-9 and len(up) >= 2)
    elapsed = float(t[-1]-t[0])
    return dict(
        tail_start_s=float(t[0]), tail_end_s=float(t[-1]), tail_duration_s=elapsed,
        tail_velocity_pp_mm_s=pp, tail_velocity_rms_mm_s=float(np.sqrt(np.mean(v*v))),
        tail_velocity_mean_mm_s=float(np.mean(v)),
        preceding_tail_velocity_pp_mm_s=previous_pp,
        tail_to_preceding_pp_ratio=(pp/previous_pp if previous_pp > 0. else None),
        tail_voltage_pp_v=float(np.ptp(data['control_voltage_v'][tail])),
        tail_valve_voltage_pp_v=float(np.ptp(data['valve_voltage_v'][tail])),
        tail_effective_voltage_pp_v=float(np.ptp(data['effective_voltage_v'][tail])),
        tail_speed_zero_crossings=int(len(up)+len(down)),
        tail_complete_upcrossing_cycles=max(0, int(len(up)-1)),
        frequency_estimate_meaningful=meaningful,
        frequency_from_median_upcrossing_period_hz=(float(1/np.median(np.diff(up)))
                                                   if meaningful else None),
        frequency_from_crossing_count_hz=((len(up)+len(down))/(2*elapsed)
                                          if meaningful else None),
        upward_crossing_times_s=up.tolist(), downward_crossing_times_s=down.tolist(),
        frequency_note=('Median same-direction interpolated velocity zero-crossing period; '
                        'count/(2*tail duration) is also reported and has endpoint bias. '
                        'No frequency claim when pp<=1e-9 mm/s or fewer than two upward crossings.'))


def _zero_hold(source, config, strategy, *, feedback_enabled=True, hold_start_s=5., duration_s=1.):
    """从正式轨迹的真实阀位/速度继续，改为零参考而非人为速度种子。"""
    data = source['data']
    time = np.asarray(data['time_s'])
    dt = config['sample_time_s']
    index = int(np.argmin(abs(time-hold_start_s)))
    if abs(time[index]-hold_start_s) > .5*dt+1e-12:
        raise ValueError('正式轨迹不包含要求的零参考保持起点')
    count = round(duration_s/dt)
    if abs(count*dt-duration_s) > 1e-10:
        raise ValueError('保持试验时长必须是共同采样周期的整数倍')
    plant = build_plant(config)
    initial_valve = float(data['valve_voltage_v'][index])
    initial_velocity = float(data['velocity_mm_s'][index])/1000.
    plant.reset(valve_voltage_v=initial_valve, velocity_m_s=initial_velocity)
    controller = _controller(config, strategy, source.get('beta'))
    # 逐样本回放正式测量，恢复整个误差队列及LP1/LP2，不能仅从(v,z)重置。
    measured = data.get('measured_velocity_mm_s', data['velocity_mm_s'])
    for j in range(index):
        controller.update(_reference_state(time[j], config), [0., float(measured[j])/1000.])
    initial_memory = dict(lowpass1_v=controller.lowpass1, lowpass2_v=controller.lowpass2,
                          error_history_v=list(controller.error_history))
    controller.feedback_enabled = feedback_enabled
    relative = np.arange(count+1)*dt
    names = ('velocity_mm_s', 'control_voltage_v', 'valve_voltage_v',
             'effective_voltage_v', 'linear_voltage_v', 'feedback_voltage_v',
             'lowpass1_v', 'lowpass2_v', 'delayed_error_equivalent_v')
    records = {name: np.empty(count+1) for name in names}
    for j in range(count+1):
        voltage, _, _ = controller.update([0., 0., 0.], [0., plant.velocity_m_s])
        values = (plant.velocity_m_s*1000., voltage, plant.valve_voltage_v,
                  deadzone(voltage, config['physical_deadzone_v']),
                  controller.last['linear_voltage_v'], controller.last['feedback_linear_v'],
                  controller.last['lowpass1_v'], controller.last['lowpass2_v'],
                  controller.last['delayed_error_equivalent_v'])
        for name, value in zip(names, values):
            records[name][j] = value
        if j < count:
            plant.step(voltage, dt)
    records = dict(time_s=relative, source_clock_time_s=relative+time[index], **records)
    return dict(
        strategy=strategy, feedback_enabled=feedback_enabled, controller_initial_memory=initial_memory,
        source_time_s=float(time[index]), initial_valve_voltage_v=initial_valve,
        initial_velocity_mm_s=initial_velocity*1000., hold_duration_s=float(relative[-1]),
        measurement_noise_std_mm_s=0., reference_velocity_mm_s=0.,
        metrics=_hold_metrics(records), data=records)


def run_robustness(results=None, config=None, folder=None):
    """运行一次补充试验并保存 robustness.json、噪声/保持原始 CSV。

    results 为正式三组结果字典；省略时优先读取 folder 中正式 CSV，
    缺少文件则按给定 config 重算。传入的 config 和 results 不被修改。
    反事实只在保持段停用全部反馈（含washout）；主实验Kp本就为0。
    """
    config = dict(load_config() if config is None else config)
    folder = Path(folder) if folder is not None else Path(__file__).with_name('outputs')
    folder.mkdir(parents=True, exist_ok=True)
    if results is None:
        results, sources = {}, {}
        for strategy in STRATEGIES:
            path = folder/f'{strategy}.csv'
            if path.exists():
                results[strategy] = dict(data=_read_csv(path),
                    beta=config['beta_default'] if strategy == 'smooth' else None)
                sources[strategy] = str(path.resolve())
            else:
                results[strategy] = simulate(strategy, config)
                sources[strategy] = 'resimulated_with_supplied_config'
    else:
        sources = {strategy:'passed_formal_result' for strategy in STRATEGIES}
    if any(strategy not in results for strategy in STRATEGIES):
        raise ValueError('补充保持试验需要正式的 under/hard_over/smooth 三组数据')
    noise_config = dict(config, measurement_noise_std_mm_s=.02)
    noisy, noise_arrays = [], []
    for beta in (5., 18., 50.):
        result = simulate('smooth', noise_config, beta=beta)
        filename = f'robustness_beta_{beta:g}_noise.csv'
        _write_csv(folder/filename, result['data'])
        noise_arrays.append(result['data']['measured_velocity_mm_s']-result['data']['velocity_mm_s'])
        noisy.append(dict(beta=beta, metrics=result['metrics'], raw_csv=filename))
    noise_difference = float(max(np.max(abs(a-noise_arrays[0])) for a in noise_arrays))
    hold, no_p = {}, {}
    for strategy in STRATEGIES:
        trial = _zero_hold(results[strategy], config, strategy)
        filename = f'robustness_zero_hold_{strategy}.csv'
        _write_csv(folder/filename, trial.pop('data'))
        hold[strategy] = dict(trial, raw_csv=filename)
        trial0 = _zero_hold(results[strategy], config, strategy, feedback_enabled=False)
        filename0 = f'robustness_zero_hold_{strategy}_feedback_disabled.csv'
        _write_csv(folder/filename0, trial0.pop('data'))
        no_p[strategy] = dict(trial0, raw_csv=filename0)
    hard_pp = hold['hard_over']['metrics']['tail_velocity_pp_mm_s']
    disabled_pp = no_p['hard_over']['metrics']['tail_velocity_pp_mm_s']
    beta_holds = {}
    for beta in config['beta_values']:
        path = folder/f'beta_{beta:g}.csv'
        source = dict(data=_read_csv(path), beta=beta) if path.exists() else simulate('smooth',config,beta)
        trial = _zero_hold(source, config, 'smooth')
        filename = f'robustness_zero_hold_beta_{beta:g}.csv'
        _write_csv(folder/filename, trial.pop('data'))
        beta_holds[f'{beta:g}'] = dict(trial, beta=beta, raw_csv=filename)
    criteria = dict(lc_min_velocity_pp_mm_s=.5, decay_velocity_pp_mm_s=1e-6,
                    decay_velocity_rms_mm_s=1e-6, lc_min_complete_cycles=3,
                    lc_stationary_pp_ratio_min=.8, lc_stationary_pp_ratio_max=1.2,
                    reconstructed_noise_equality_tolerance_mm_s=1e-11)
    hard_metrics = hold['hard_over']['metrics']
    ratio = hard_metrics['tail_to_preceding_pp_ratio']
    def has_decayed(record):
        return bool(record['metrics']['tail_velocity_pp_mm_s'] <= criteria['decay_velocity_pp_mm_s']
                    and record['metrics']['tail_velocity_rms_mm_s'] <= criteria['decay_velocity_rms_mm_s'])
    checks = dict(
        shared_measurement_noise_verified=bool(noise_difference < criteria['reconstructed_noise_equality_tolerance_mm_s']),
        hard_hold_velocity_limit_cycle_observed=bool(
            hard_pp > criteria['lc_min_velocity_pp_mm_s']
            and hard_metrics['tail_complete_upcrossing_cycles'] >= criteria['lc_min_complete_cycles']
            and ratio is not None and criteria['lc_stationary_pp_ratio_min'] <= ratio <= criteria['lc_stationary_pp_ratio_max']),
        under_hold_velocity_decayed=has_decayed(hold['under']),
        smooth_hold_velocity_decayed=has_decayed(hold['smooth']),
        feedback_disabled_hard_hold_limit_cycle_absent=has_decayed(no_p['hard_over']),
        feedback_disabled_preserves_initial_memory=all(
            hold[k]['controller_initial_memory'] == no_p[k]['controller_initial_memory'] for k in STRATEGIES))
    report = dict(
        scope='Supplementary robustness experiments; default main figure remains unchanged and noiseless.',
        source_config=config, formal_state_sources=sources,
        noise_sensitivity=dict(std_mm_s=.02, seed=config['noise_seed'],
            shared_additive_noise_verified=bool(noise_difference < 1e-11),
            maximum_reconstructed_noise_difference_mm_s=noise_difference, beta_runs=noisy,
            note='Same seed and additive measurement noise; actual velocities and valve signals remain unfiltered.'),
        zero_reference_hold=dict(start_s=5., duration_s=1., tail_s=.1,
            nominal_feedback_runs=hold, feedback_disabled_counterfactual_runs=no_p,
            beta_hold_runs=beta_holds,
            all_error_feedback_disabled=True,
            hard_over_tail_pp_reduction_factor=(hard_pp/disabled_pp if disabled_pp > 0 else None),
            note=('All runs begin at the actual valve and velocity recorded at 5s. '
                  'Controller history is reconstructed from formal measurements; '
                  'The complete delayed-error queue and both lowpass states are preserved. '
                  'The counterfactual disables every feedback contribution only during hold. No new noise.')),
        interpretation_limits=[
            'Finite beta leaves a residual zero-flow band; smooth does not mean zero tracking error.',
            'The noiseless sine crossing is transient; persistence is assessed separately under constant zero reference.',
            'Chattering command voltage alone is not evidence of a velocity limit cycle; actual velocity metrics are provided.',
            'Observed noise sensitivity and oscillation apply to this sampled equivalent velocity model, not every hydraulic system.'
        ], check_criteria=criteria, checks=checks, passed=bool(all(checks.values())),
        check_scope='Checks cover only the configured supplemental trials, not global stability.')
    (folder/'robustness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                                       encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='共同实验 JSON 配置')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).with_name('outputs'))
    args = parser.parse_args()
    report = run_robustness(config=load_config(args.config), folder=args.output_dir)
    print(f'Saved supplemental robustness data: {(args.output_dir/"robustness.json").resolve()}')
    return report


if __name__ == '__main__':
    main()
