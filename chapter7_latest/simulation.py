"""三组同对象闭环；位移/电压1kHz记录，压力保留采样间真实积分极值。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'chapter7_mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'chapter7_cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import numpy as np
from threadpoolctl import threadpool_limits
from hydraulic_plant import HydraulicPlant, GreyBoxModel
from mpc_core import HydraulicMPC
from baseline_controller import BaselineFFPID

ROOT = Path(__file__).resolve().parent
KINDS = ('pid', 'input_mpc', 'full_mpc')


def load_config(path=None):
    config = json.loads((Path(path) if path else ROOT/'experiment_config.json').read_text(encoding='utf-8'))
    if (config['ts'] <= 0 or not 0 < config['step_time'] < config['duration']
            or config['target_mm'] <= 0 or config['horizon'] < 1
            or config['p_safe'] <= config['pressure_margin']
            or abs(config['duration']/config['ts']-round(config['duration']/config['ts'])) > 1e-8):
        raise ValueError('实验时钟、目标、时域或安全压力无效')
    return config


@threadpool_limits.wrap(limits=1)
def simulate(kind, config, progress=False):
    if kind not in KINDS:
        raise ValueError('策略仅支持 pid/input_mpc/full_mpc')
    start = time.perf_counter()
    arguments = dict(origin_m=config['origin_m'], valve_tau=config['valve_tau'],
                     makeup_area=config['makeup_area'], rtol=config['rtol'])
    plant = HydraulicPlant(**arguments)
    params = plant.parameters
    # 共同机械平衡预充初态。内泄漏仍存在，不能将其称为零流量恒压平衡。
    p2 = params.get('initial_p2_pa', params['Pr'])
    p1 = (p2*params['A2']+params['ambient_pressure']*(params['A1']-params['A2']))/params['A1']
    plant.reset([0., 0., p1/1e6, p2/1e6, 0.])
    if kind == 'pid':
        controller = BaselineFFPID(config, params)
        predictor = None
    else:
        predictor = GreyBoxModel(params, **arguments)
        controller = HydraulicMPC(horizon=config['horizon'], dt=config['ts'], model=predictor,
            enable_pressure_constraint=kind == 'full_mpc', delta_u_max=config['max_du_step'],
            p_safe=config['p_safe'], pressure_margin=config['pressure_margin'],
            sequential_iterations=config['mpc_outer_iterations'],
            q_position=config.get('q_position', 50000.), q_velocity=config.get('q_velocity', 50.),
            move_weight=config.get('move_weight', .01))
    time_grid = np.arange(round(config['duration']/config['ts'])+1)*config['ts']
    reference = np.where(time_grid >= config['step_time']-1e-12, config['target_mm'], 0.)
    names = ('position_mm', 'velocity_m_s', 'p1_mpa', 'p2_mpa', 'valve_v',
             'control_v', 'raw_control_v', 'delta_voltage_v', 'pid_integral_v',
             'feedforward_v', 'qp_iterations', 'qp_run_time_ms', 'computation_ms',
             'predicted_p2_max_mpa', 'prediction_violation_mpa', 'solver_status_code',
             'move_weight', 'settling_cost_active')
    data = {key: np.empty(len(time_grid)) for key in names}
    logs, previous = [], 0.
    for index, ti in enumerate(time_grid):
        if not np.isclose(plant.t, ti, atol=1e-10, rtol=0):
            raise RuntimeError('物理时钟与控制时钟不同步')
        state = plant.state
        settling = (kind != 'pid' and abs(reference[index]-state[0])
                    <= config.get('settling_error_mm', .5)
                    and abs(state[1]) <= config.get('settling_velocity_m_s', .1))
        if kind != 'pid':
            # 两组共用同一状态调度，运动期保留快速制动；定位后增加增量惩罚。
            controller.move_weight = (config.get('settling_move_weight', .1) if settling
                                      else config.get('move_weight', .01))
            controller.W = controller.move_weight
        tic = time.perf_counter()
        if kind == 'pid':
            voltage = controller.update(float(ti), float(reference[index]), state)
            info = controller.last.copy()
            raw, integral, ff = info['raw_voltage_v'], info['integral_v'], info['feedforward_v']
            status = 0
        elif ti < config['step_time']-1e-12:
            voltage, raw, integral, ff, status = 0., 0., 0., 0., 0
            info = dict(status='before_command', solver_time_ms=0., iterations=0)
        else:
            try:
                voltage = controller.solve(state, previous, float(reference[index]))
            except Exception:
                ROOT.joinpath('last_failure.json').write_text(json.dumps(
                    dict(kind=kind, time_s=float(ti), state=state.tolist(),
                         previous_voltage_v=previous, config=config,
                         nominal_sequence=(controller.nominal_sequence.tolist()
                                           if controller.nominal_sequence is not None else None),
                         last=controller.last), indent=2, ensure_ascii=False), encoding='utf-8')
                raise
            info = controller.last.copy()
            candidate = info.get('candidate_voltage_v')
            raw = voltage if candidate is None else candidate
            integral, ff = np.nan, 0.
            status = 2 if info.get('feasible_shifted_plan_used', False) else 1
            if info.get('osqp_status', '').lower() == 'solved inaccurate':
                status = 3
        computation = (time.perf_counter()-tic)*1000.
        values = (*state, voltage, raw, voltage-previous, integral, ff,
                  info.get('iterations', 0), info.get('solver_time_ms', 0.), computation,
                  info.get('predicted_peak_p2_mpa', state[3]),
                  info.get('prediction_violation_mpa', 0.), status,
                  controller.move_weight if kind != 'pid' else 0., int(settling))
        for key, value in zip(names, values):
            data[key][index] = value
        logs.append(dict(time_s=float(ti), active_move_weight=(controller.move_weight
                          if kind != 'pid' else None), settling_cost_active=bool(settling), **info))
        if progress and kind != 'pid' and index % 10 == 0:
            print(f'{kind} t={ti:.3f} x={state[0]:.2f}mm P2={state[3]:.2f}MPa '
                  f'u={voltage:+.2f}V controller={computation:.1f}ms', flush=True)
        if index+1 < len(time_grid):
            plant.step(voltage, config['ts'])
        previous = voltage
    trace = plant.pressure_trace
    pressure_trace = dict(time_s=trace[:, 0], p1_mpa=trace[:, 1]/1e6, p2_mpa=trace[:, 2]/1e6)
    data = dict(time_s=time_grid, reference_mm=reference, **data)
    metadata = dict(config, strategy=kind, physical_parameters=params,
                    state_units=['mm', 'm/s', 'MPa', 'MPa', 'V'],
                    n_physical_states=5, n_augmented_states=6,
                    blas_threads=1,
                    initial_state=[0., 0., p1/1e6, p2/1e6, 0.],
                    pressure_constraint_scope='P2 upper only' if kind == 'full_mpc' else 'none',
                    integrator='Radau with hybrid friction events and dense pressure extrema',
                    radau_nfev=plant.nfev, friction_event_count=len(plant.friction_events),
                    simulation_wall_s=time.perf_counter()-start,
                    source_twin_sha256=hashlib.sha256((ROOT/'hydraulic_digital_twin.py').read_bytes()).hexdigest())
    if predictor is not None:
        metadata['mpc_solver_settings'] = dict(
            q_position=controller.q_position, q_velocity=controller.q_velocity,
            motion_move_weight=config.get('move_weight', .01),
            settling_move_weight=config.get('settling_move_weight', .1),
            primal_tolerance=controller.primal_tolerance,
            nonlinear_tolerance=controller.nonlinear_tolerance,
            linearization_reserve_mpa=controller.linearization_reserve_mpa,
            polishing=controller.polishing,
            eps_abs=controller.eps_abs, eps_rel=controller.eps_rel,
            max_iterations=controller.max_iter)
        metadata['prediction_regularization'] = dict(
            pressure_epsilon_pa=predictor.pressure_epsilon_pa,
            deadzone_epsilon_v=predictor.deadzone_epsilon_v,
            direction_epsilon_v=predictor.direction_epsilon_v,
            friction_epsilon_m_s=predictor.friction_epsilon_m_s)
    result_metrics = metrics(data, pressure_trace, config)
    attempts = [attempt for log in logs for attempt in log.get('qp_diagnostics', [])]
    result_metrics.update(
        qp_attempt_count=len(attempts),
        fallback_solver_attempt_count=sum(bool(attempt.get('fallback_solver_used'))
                                          for attempt in attempts),
        solver_failure_retained_plan_count=sum(bool(log.get('solver_failure'))
                                               and bool(log.get('feasible_shifted_plan_used'))
                                               for log in logs))
    return dict(data=data, pressure_trace=pressure_trace, config=metadata,
                metrics=result_metrics, solver_logs=logs,
                friction_events=plant.friction_events)


def metrics(data, pressure, config):
    t, x, velocity, u = (data[key] for key in ('time_s', 'position_mm', 'velocity_m_s', 'control_v'))
    target = config['target_mm']
    after = t >= config['step_time']-1e-12
    active = after & (data['solver_status_code'] != 0)
    crossings = np.flatnonzero(after & (x >= target))
    first_crossing = None
    if len(crossings):
        k = crossings[0]
        first_crossing = float(t[k-1]+config['ts']*(target-x[k-1])/(x[k]-x[k-1]))
    peak_index = int(np.argmax(x))
    rail_tolerance = config.get('saturation_tolerance_v', 1e-4)
    saturated = np.flatnonzero(after & (u >= 10.-rail_tolerance))
    initial_exit = None
    if len(saturated):
        exits = np.flatnonzero((np.arange(len(t)) > saturated[0]) & (u < 10.-rail_tolerance))
        initial_exit = float(t[exits[0]]) if len(exits) else None
    p2_index = int(np.argmax(pressure['p2_mpa']))
    late = t >= config['duration']-.02-1e-12
    return dict(
        peak_position_mm=float(x[peak_index]), peak_position_time_s=float(t[peak_index]),
        overshoot_percent=float(max(0., x[peak_index]-target)/target*100.),
        first_target_crossing_s=first_crossing, final_error_mm=float(target-x[-1]),
        first_positive_saturation_exit_s=initial_exit,
        saturation_tolerance_v=rail_tolerance,
        saturation_delay_after_crossing_ms=(max(0., 1000*(initial_exit-first_crossing))
            if first_crossing is not None and initial_exit is not None else None),
        last_20ms_max_abs_error_mm=float(np.max(abs(target-x[late]))),
        peak_p2_mpa=float(pressure['p2_mpa'][p2_index]),
        peak_p2_time_s=float(pressure['time_s'][p2_index]),
        peak_p1_mpa=float(np.max(pressure['p1_mpa'])),
        min_absolute_pressure_mpa=float(min(np.min(pressure['p1_mpa']),np.min(pressure['p2_mpa']))),
        sampled_peak_p2_mpa=float(data['p2_mpa'].max()),
        constraint_violation_p2_mpa=float(max(0., pressure['p2_mpa'].max()-config['p_safe'])),
        max_abs_voltage_v=float(np.max(abs(u))), min_voltage_v=float(np.min(u)),
        max_internal_voltage_v=float(data['raw_control_v'].max()),
        max_abs_control_increment_v=float(np.max(abs(data['delta_voltage_v']))),
        peak_velocity_m_s=float(np.max(velocity)),
        min_velocity_m_s=float(np.min(velocity)),
        qp_runtime_median_ms=float(np.median(data['qp_run_time_ms'][active])) if active.any() else 0.,
        qp_runtime_max_ms=float(np.max(data['qp_run_time_ms'][active])) if active.any() else 0.,
        total_controller_median_ms=float(np.median(data['computation_ms'][after])),
        total_controller_max_ms=float(np.max(data['computation_ms'][after])),
        controller_within_1ms_fraction=float(np.mean(data['computation_ms'][after] <= 1.)),
        validated_shifted_plan_count=int(np.sum(data['solver_status_code'] == 2)),
        solved_inaccurate_count=int(np.sum(data['solver_status_code'] == 3)),
        pressure_trace_points=len(pressure['time_s']))


def save_results(results, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, result in results.items():
        for suffix, data in (('', result['data']), ('_pressure_trace', result['pressure_trace'])):
            with (folder/f'{kind}{suffix}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(data)
                writer.writerows(zip(*data.values()))
        for suffix, records in (('solver_logs', result['solver_logs']), ('friction_events', result['friction_events'])):
            (folder/f'{kind}_{suffix}.json').write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {kind: dict(config=result['config'], metrics=result['metrics']) for kind, result in results.items()}
    summary['reference_comparison'] = {
        'pid': dict(reference_overshoot_percent=16., reference_peak_p2_mpa=24.50),
        'input_mpc': dict(reference_overshoot_percent=.8, reference_peak_p2_mpa=34.06),
        'full_mpc': dict(reference_overshoot_percent=.9, reference_peak_p2_mpa=21.01)}
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
