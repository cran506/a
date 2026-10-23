"""同一真实液压对象上的逐采样 CVXPY MPC 预测时域扫描。"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from hydraulic_plant import GreyBoxModel, HydraulicPlant
from mpc_core import MPCSolveError


ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    config = json.loads((Path(path) if path else ROOT / 'experiment_config.json').read_text(encoding='utf-8'))
    horizons = config['horizons']
    if (not horizons or any(int(n) != n or n < 1 for n in horizons)
            or len(set(horizons)) != len(horizons)
            or any(n not in horizons for n in config['highlight_horizons'])
            or config['ts'] <= 0 or not 0 <= config['step_time'] < config['duration']
            or config['target_mm'] <= 0 or not 0 <= config['pressure_margin'] < config['p_safe']
            or abs(config['duration'] / config['ts'] - round(config['duration'] / config['ts'])) > 1e-9):
        raise ValueError('Invalid horizon scan, clock, target, or pressure limit')
    return config


def _stats(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(mean_ms=float(np.mean(a)) if len(a) else None,
                median_ms=float(np.median(a)) if len(a) else None,
                p95_ms=float(np.percentile(a, 95)) if len(a) else None,
                max_ms=float(np.max(a)) if len(a) else None,
                samples=int(len(a)),
                over_1ms_fraction=float(np.mean(a > 1.)) if len(a) else None)


def _metrics(data, pressure_trace, config, horizon, failure, solver_logs):
    x = np.asarray(data['x_mm'])
    t = np.asarray(data['t_s'])
    u = np.asarray(data['u_v'])
    p2 = pressure_trace['p2_mpa']
    active = np.asarray(data['status_code']) > 0
    reached = np.flatnonzero(x >= config['target_mm'])
    # Mean timing excludes warmup, idle intervals, and restoration-only rows.
    solved = np.isin(data['status_code'], [1, 2])
    solver_stats = _stats(np.asarray(data['solver_time_ms'])[solved])
    attempt_times = [diagnostic['backend_solve_time_ms']
                     for log in solver_logs
                     for diagnostic in log.get('qp_diagnostics', [])
                     if diagnostic.get('backend_solve_time_ms') is not None]
    attempt_stats = _stats(attempt_times)
    cvxpy_stats = _stats(np.asarray(data['cvxpy_wall_ms'])[solved])
    controller_stats = _stats(np.asarray(data['controller_wall_ms'])[active])
    target = config['target_mm']
    return dict(
        horizon=horizon, prediction_ms=horizon*config['ts']*1000.,
        status='completed' if failure is None else 'failed', failure=failure,
        final_time_s=float(t[-1]), samples=int(len(t)),
        peak_position_mm=float(np.max(x)),
        peak_position_time_s=float(t[np.argmax(x)]),
        overshoot_pct=float(max(0., np.max(x)-target)/target*100.),
        first_target_sample_s=float(t[reached[0]]) if len(reached) else None,
        final_error_mm=float(target-x[-1]) if failure is None else None,
        peak_pressure_mpa=float(np.max(p2)),
        peak_pressure_time_s=float(pressure_trace['t_s'][np.argmax(p2)]),
        sampled_peak_pressure_mpa=float(np.max(data['p2_mpa'])),
        pressure_violation_mpa=float(max(0., np.max(p2)-config['p_safe'])),
        max_abs_voltage_v=float(np.nanmax(np.abs(u))),
        max_abs_increment_v=float(np.nanmax(np.abs(np.diff(np.r_[0., u[np.isfinite(u)]])))),
        restoration_count=int(np.sum(np.asarray(data['status_code']) == 3)),
        refinement_count=int(np.sum(np.asarray(data['status_code']) == 4)),
        shifted_count=int(np.sum(np.asarray(data['status_code']) == 2)),
        solver=solver_stats, solver_attempt=attempt_stats,
        cvxpy_call=cvxpy_stats, whole_controller=controller_stats)


@threadpool_limits.wrap(limits=1)
def simulate(horizon, config, progress=False):
    """Receding horizon: one measured state, one CVXPY QP family, one applied input.

    Infeasible local optimizations may use a *documented*, freshly nonlinear-
    validated restoration plan. No pressure state is clipped and no invalid
    command is applied. If restoration fails, the run ends with failed status.
    """
    from cvxpy_mpc import CVXPYHydraulicMPC
    from safety_restoration import refine_shifted_plan, restore_pressure_feasible

    horizon = int(horizon)
    if horizon not in config['horizons']:
        raise ValueError('Requested horizon is not listed in the experiment config')
    arguments = dict(origin_m=config['origin_m'], valve_tau=config['valve_tau'],
                     makeup_area=config['makeup_area'], rtol=config['rtol'])
    plant = HydraulicPlant(**arguments)
    params = plant.parameters
    p2 = params.get('initial_p2_pa', params['Pr'])
    p1 = (p2*params['A2']+params['ambient_pressure']*(params['A1']-params['A2']))/params['A1']
    initial_state = [0., 0., p1/1e6, p2/1e6, 0.]
    plant.reset(initial_state)
    predictor = GreyBoxModel(params, **arguments)
    controller = CVXPYHydraulicMPC(
        horizon=horizon, dt=config['ts'], model=predictor,
        enable_pressure_constraint=True, delta_u_max=config['max_du_step'],
        p_safe=config['p_safe'], pressure_margin=config['pressure_margin'],
        sequential_iterations=config['mpc_outer_iterations'],
        q_position=config['q_position'], q_velocity=config['q_velocity'],
        move_weight=config['move_weight'])
    grid = np.arange(round(config['duration']/config['ts'])+1)*config['ts']
    fields = ('t_s', 'reference_mm', 'x_mm', 'velocity_m_s', 'p1_mpa',
              'p2_mpa', 'valve_v', 'u_v', 'delta_u_v', 'solver_time_ms',
              'cvxpy_wall_ms', 'controller_wall_ms', 'solver_iterations',
              'predicted_peak_p2_mpa', 'status_code', 'settling_cost_active')
    rows = {key: [] for key in fields}
    logs = []
    previous = 0.
    failure = None
    tic_run = time.perf_counter()
    for index, current_t in enumerate(grid):
        if not np.isclose(plant.t, current_t, atol=1e-10, rtol=0.):
            raise RuntimeError('Physical and control clocks diverged')
        state = plant.state
        reference = config['target_mm'] if current_t >= config['step_time']-1e-12 else 0.
        settling = (abs(reference-state[0]) <= config['settling_error_mm']
                    and abs(state[1]) <= config['settling_velocity_m_s'])
        controller.move_weight = config['settling_move_weight'] if settling else config['move_weight']
        controller.W = controller.move_weight
        solver_ms = cvxpy_ms = controller_ms = 0.
        iterations = 0
        predicted_peak = state[3]
        status = 0
        info = dict(status='before_command' if current_t < config['step_time'] else 'terminal_observation')
        if index == len(grid)-1:
            # The last row observes the final physical state; no extra unused QP.
            voltage = previous
        elif current_t < config['step_time']-1e-12:
            voltage = 0.
        else:
            start = time.perf_counter()
            try:
                voltage = controller.solve(state, previous, reference)
                if controller.last.get('feasible_shifted_plan_used'):
                    voltage = refine_shifted_plan(controller, state, previous, reference,
                                                  grid_points=21)
            except MPCSolveError as exc:
                try:
                    voltage = restore_pressure_feasible(controller, state, previous,
                                                        reference, failure=exc)
                except MPCSolveError as restoration_error:
                    failure = dict(time_s=float(current_t), reason=str(restoration_error),
                                   solver_failure=str(exc), state=state.tolist(),
                                   previous_voltage_v=float(previous))
                    voltage = float('nan')
            controller_ms = (time.perf_counter()-start)*1000.
            info = controller.last.copy()
            diagnostics = info.get('qp_diagnostics', [])
            solver_ms = sum(float(item.get('backend_solve_time_ms', 0.) or 0.)
                            for item in diagnostics)
            cvxpy_ms = sum(float(item.get('solver_time_ms', 0.) or 0.)
                           for item in diagnostics)
            iterations = int(info.get('iterations', 0) or 0)
            predicted_peak = float(info.get('predicted_peak_p2_mpa', state[3]))
            status = (-1 if failure else 3 if info.get('restoration_used')
                      else 4 if info.get('plan_refinement_used')
                      else 2 if info.get('feasible_shifted_plan_used') else 1)
            logs.append(dict(time_s=float(current_t), status_code=status,
                             active_move_weight=controller.move_weight, **info))
        values = (current_t, reference, *state, voltage,
                  voltage-previous if np.isfinite(voltage) else np.nan,
                  solver_ms, cvxpy_ms, controller_ms, iterations,
                  predicted_peak, status, int(settling))
        for key, value in zip(fields, values):
            rows[key].append(float(value))
        if progress and (index % 20 == 0 or failure):
            print(f'N={horizon:2d} t={current_t:.3f}s x={state[0]:.2f}mm '
                  f'P2={state[3]:.2f}MPa u={voltage:+.2f}V '
                  f'control={controller_ms:.2f}ms status={status}', flush=True)
        if failure is not None:
            break
        if index+1 < len(grid):
            plant.step(voltage, config['ts'])
        previous = voltage
    data = {key: np.asarray(value) for key, value in rows.items()}
    trace = plant.pressure_trace
    pressure = dict(t_s=trace[:,0], p1_mpa=trace[:,1]/1e6, p2_mpa=trace[:,2]/1e6)
    metrics = _metrics(data, pressure, config, horizon, failure, logs)
    metadata = dict(config, horizon=horizon, physical_parameters=params,
                    initial_state=initial_state, n_physical_states=5,
                    n_augmented_states=6, state_units=['mm','m/s','MPa','MPa','V'],
                    optimizer='CVXPY with OSQP',
                    pressure_constraint_scope='predicted P2 upper bound only',
                    prediction_p2_limit_mpa=config['p_safe']-config['pressure_margin'],
                    physical_integrator='Radau with hybrid friction events and dense pressure trace',
                    physical_rhs_sha256=hashlib.sha256((ROOT/'hydraulic_digital_twin.py').read_bytes()).hexdigest(),
                    physical_function_evaluations=plant.nfev,
                    simulation_wall_s=time.perf_counter()-tic_run,
                    blas_threads=1)
    return dict(data=data, pressure_trace=pressure, metrics=metrics,
                config=metadata, solver_logs=logs,
                friction_events=plant.friction_events)


def save_results(results, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for horizon, result in results.items():
        prefix = f'N{horizon:02d}'
        for suffix, data in (('', result['data']), ('_pressure_trace', result['pressure_trace'])):
            with (folder/f'{prefix}{suffix}.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.writer(stream, lineterminator='\n')
                writer.writerow(data.keys())
                writer.writerows(zip(*data.values()))
        for suffix, entries in (('solver_logs', result['solver_logs']),
                                ('friction_events', result['friction_events'])):
            (folder/f'{prefix}_{suffix}.json').write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = dict(config=next(iter(results.values()))['config'] if results else {},
                   results={str(h): result['metrics'] for h,result in results.items()},
                   timing_definitions={
                       'solver': 'OSQP solver_stats.solve_time of recorded QP attempts, summed per control step',
                       'solver_attempt': 'OSQP native solve_time for each recorded QP attempt, including attempts not ultimately applied',
                       'cvxpy_call': 'wall time inside each cvxpy Problem.solve, summed per control step',
                       'whole_controller': 'wall time for linearization, QP, validation and any restoration',
                       'deadline': 'nominal sample period 1.0 ms; Linux desktop timing is illustrative, not RT-certified'},
                   matrix_dimensions={str(h): dict(augmented_states=6, condensed_phi=[6*h,h],
                                                    hessian=[h,h], dense_hessian_entries=h*h)
                                      for h in results})
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    return summary
