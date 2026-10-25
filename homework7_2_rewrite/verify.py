#!/usr/bin/env python3
"""独立核验习题 7-2 的记录、QP 可行性与连续时间压力。

记录输入逐区间交由独立力学/流量方程和 solve_ivp 重放。压力极值由
密集输出上的 dP/dt 零点求得，不把预测节点的上界视为采样间保证。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
REQUIRED_HORIZONS = (5, 10, 15, 20, 30, 40, 50)
STATE_FIELDS = ('position_mm', 'velocity_m_s', 'p1_mpa', 'p2_mpa')


def independent_rhs(p, state, voltage):
    """SI 力平衡与线性化净流量；不调用 model.py 的矩阵或 rhs。"""
    _, velocity, p1_mpa, p2_mpa = state
    p1, p2 = p1_mpa*1e6, p2_mpa*1e6
    p10, p20 = p['equilibrium_p1_mpa']*1e6, p['equilibrium_p2_mpa']*1e6
    a1, a2 = p['area1_m2'], p['area2_m2']
    load = a1*p10-a2*p20
    q1 = p['flow_voltage1_m3_s_v']*voltage-p['flow_pressure1_m3_s_pa']*(p1-p10)
    q2 = p['flow_voltage2_m3_s_v']*voltage+p['flow_pressure2_m3_s_pa']*(p2-p20)
    return np.array([1000.*velocity,
                     (a1*p1-a2*p2-load-p['viscous_ns_m']*velocity)/p['mass_kg'],
                     p['bulk_modulus_pa']/p['volume1_m3']*(q1-a1*velocity)/1e6,
                     p['bulk_modulus_pa']/p['volume2_m3']*(a2*velocity-q2)/1e6])


def _csv(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def _stats(values):
    values = np.asarray(values)
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(np.max(values)))


def _replay(data, config):
    p, dt = config['physical'], config['sample_time_s']
    state = np.array([0., 0., p['equilibrium_p1_mpa'], p['equilibrium_p2_mpa']])
    states = [state.copy()]
    pressure_max, pressure_min = state[2:].copy(), state[2:].copy()
    for voltage in data['voltage_v'][:-1]:
        rhs = lambda t, z: independent_rhs(p, z, voltage)
        solution = solve_ivp(rhs, (0., dt), state, method='DOP853', dense_output=True,
                             rtol=2e-11, atol=2e-12, max_step=dt/8.)
        if not solution.success:
            raise RuntimeError(f'Independent replay failed: {solution.message}')
        samples = np.linspace(0., dt, 65)
        candidates = set(samples.tolist())
        for axis in (2, 3):
            slopes = np.array([rhs(t, solution.sol(t))[axis] for t in samples])
            for left, right, fl, fr in zip(samples[:-1], samples[1:], slopes[:-1], slopes[1:]):
                if fl*fr < 0.:
                    candidates.add(float(brentq(lambda t: rhs(t, solution.sol(t))[axis],
                                                 left, right, xtol=1e-13)))
        interval_states = solution.sol(np.array(sorted(candidates))).T
        pressure_max = np.maximum(pressure_max, interval_states[:, 2:].max(axis=0))
        pressure_min = np.minimum(pressure_min, interval_states[:, 2:].min(axis=0))
        state = solution.y[:, -1]
        states.append(state.copy())
    recorded = np.column_stack([data[name] for name in STATE_FIELDS])
    error = np.max(np.abs(np.array(states)-recorded), axis=0)
    return dict(max_state_error=error.tolist(), peak_pressure_mpa=pressure_max.tolist(),
                min_pressure_mpa=pressure_min.tolist())


def verify(output=ROOT/'outputs'):
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    config, records = summary['config'], summary['results']
    report = dict(passed=False, independent_replay='DOP853 from physical force/flow equations',
                  checks=[], cases={}, errors=[])
    def check(name, passed, detail=None, case=None):
        entry = dict(name=name, passed=bool(passed))
        if detail is not None:
            entry['detail'] = detail
        if case is None:
            report['checks'].append(entry)
        else:
            report['cases'].setdefault(str(case), dict(checks=[]))['checks'].append(entry)
        if not passed:
            report['errors'].append(f'{"global" if case is None else f"N={case}"}: {name}')
    check('seven required horizons present', set(records) == {str(n) for n in REQUIRED_HORIZONS})
    check('fixed horizon-independent weights and plant declared',
          set(config['horizons']) == set(REQUIRED_HORIZONS) and
          all(not isinstance(value, (dict, list)) for value in config['controller'].values()))
    p, dt = config['physical'], config['sample_time_s']
    expected_t = np.arange(round(config['duration_s']/dt)+1)*dt
    bounds = config['controller']
    tolerance = bounds.get('feasibility_tolerance', 2e-4)
    from model import HydraulicGreyBox
    model = HydraulicGreyBox(p)
    ad, bd, cd = model.discretize(dt)
    for n in REQUIRED_HORIZONS:
        if str(n) not in records:
            continue
        metric = records[str(n)]
        data = _csv(output/f'N{n:02d}.csv')
        trace = _csv(output/f'N{n:02d}_pressure_trace.csv')
        logs = json.loads((output/f'N{n:02d}_qp.json').read_text(encoding='utf-8'))
        state = np.column_stack([data[name] for name in STATE_FIELDS])
        u, t = data['voltage_v'], data['t_s']
        check('complete finite sample grid', len(t) == len(expected_t) and
              np.all(np.isfinite(np.column_stack(list(data.values())))) and
              np.allclose(t, expected_t, atol=1e-12, rtol=0.), case=n)
        check('initial force-balanced state', np.allclose(state[0], model.equilibrium,
                                                        atol=1e-12, rtol=0.), case=n)
        reference = np.where(t >= config['step_time_s']-1e-12, config['target_mm'], 0.)
        check('step reference matches task', np.array_equal(data['reference_mm'], reference), case=n)
        check('one QP per applied interval', len(logs) == len(t)-1 == metric['qp_calls'] and
              all(log['qp_calls'] == 1 for log in logs), case=n)
        check('hardware voltage and slew bounds', np.max(abs(u)) <= 10.+1e-8 and
              np.max(abs(np.diff(np.r_[0., u[:-1]]))) <= bounds['du_max']+tolerance, case=n)
        check('recorded increments match applied inputs',
              np.allclose(data['increment_v'], np.diff(np.r_[0., u]), atol=1e-10, rtol=0.), case=n)
        qp_errors = []
        for k, log in enumerate(logs):
            plan = np.array(log['command_sequence_v'])
            predicted = np.array(log['prediction'])
            if plan.shape != (n,) or predicted.shape != (n, 4):
                qp_errors.append(f'k={k} bad prediction shape'); continue
            if log['status'] not in ('optimal', 'optimal_inaccurate') or not log['dpp']:
                qp_errors.append(f'k={k} invalid solve status')
            if not np.isclose(log['time_s'], t[k], atol=1e-12) or not np.isclose(plan[0], u[k], atol=1e-10):
                qp_errors.append(f'k={k} mismatched applied command')
            if not np.allclose(log['state'], state[k], atol=1e-10, rtol=0.):
                qp_errors.append(f'k={k} measurement reset mismatch')
            if log['max_constraint_violation'] > tolerance+1e-10:
                qp_errors.append(f'k={k} residual exceeds declared tolerance')
            if (np.max(abs(plan)) > bounds['voltage_limit']+tolerance or
                np.max(abs(np.diff(np.r_[u[k-1] if k else 0., plan]))) > bounds['du_max']+tolerance or
                np.max(predicted[:, [2, 3]]) > bounds['pmax']+tolerance or
                np.min(predicted[:, [2, 3]]) < bounds['pressure_lower']-tolerance):
                qp_errors.append(f'k={k} future plan exceeds declared bounds')
            replay_plan, point = [], state[k].copy()
            for command in plan:
                point = ad@point+bd*command+cd
                replay_plan.append(point.copy())
            if not np.allclose(predicted, replay_plan, atol=max(5e-4, n*tolerance), rtol=0.):
                qp_errors.append(f'k={k} future dynamics inconsistent')
        check('QP plans, bounds and feedback reset consistent', not qp_errors,
              detail=qp_errors[:12], case=n)
        pressure = np.column_stack((trace['p1_mpa'], trace['p2_mpa']))
        check('saved intersample pressures within physical limits',
              np.min(pressure) >= -1e-8 and np.max(pressure) <= config['physical_pressure_limit_mpa']+1e-8,
              detail=dict(min_mpa=float(pressure.min()), max_mpa=float(pressure.max())), case=n)
        expected_metrics = dict(peak_position_mm=float(data['position_mm'].max()),
            overshoot_pct=float(max(0.,data['position_mm'].max()-config['target_mm'])/config['target_mm']*100),
            final_error_mm=float(config['target_mm']-data['position_mm'][-1]),
            peak_p1_mpa=float(pressure[:, 0].max()), peak_p2_mpa=float(pressure[:, 1].max()),
            min_pressure_mpa=float(pressure.min()))
        check('summary extrema match raw data', all(np.isclose(metric[key], value, atol=1e-8, rtol=1e-8)
              for key, value in expected_metrics.items()), case=n)
        active = t[:-1] >= config['step_time_s']-1e-12
        timing_errors = []
        for field, key in [('backend_solve_ms','backend'), ('cvxpy_wall_ms','cvxpy'),
                           ('controller_wall_ms','controller')]:
            values = data[field][:-1][active]
            expected = _stats(values)
            if not all(np.isclose(metric[key][name], value, atol=1e-8, rtol=1e-8)
                       for name, value in expected.items()):
                timing_errors.append(key)
            if not np.allclose(data[field][:-1], [log[field] for log in logs], atol=1e-10, rtol=0.):
                timing_errors.append(key+' log mismatch')
        check('timing statistics match all post-command QPs', not timing_errors,
              detail=timing_errors, case=n)
        if 'backend_run' in metric:
            run_times = np.array([log['backend_run_ms'] for log in logs])[active]
            check('native OSQP run-time statistics include all post-command QPs',
                  all(np.isclose(metric['backend_run'][name], value,
                                 atol=1e-8, rtol=1e-8)
                      for name, value in _stats(run_times).items()), case=n)
        replay = _replay(data, config)
        report['cases'][str(n)]['independent_replay'] = replay
        check('independent SI-equation replay matches CSV states',
              max(replay['max_state_error']) < 2e-6,
              detail=replay['max_state_error'], case=n)
        check('independent continuous pressure extrema match trace',
              np.allclose(replay['peak_pressure_mpa'], pressure.max(axis=0), atol=2e-6, rtol=0.) and
              np.allclose(replay['min_pressure_mpa'], pressure.min(axis=0), atol=2e-6, rtol=0.), case=n)
        check('independent continuous pressure stays within 0..21 MPa',
              min(replay['min_pressure_mpa']) >= -1e-8 and
              max(replay['peak_pressure_mpa']) <= config['physical_pressure_limit_mpa']+1e-8, case=n)
    matrices = np.load(output/'textbook_matrices_N20.npz')
    check('textbook five-dimensional augmented matrices', matrices['A_aug'].shape == (5, 5) and
          matrices['B_aug'].shape == (5,) and matrices['F'].shape == (100, 5) and
          matrices['Phi'].shape == (100, 20) and matrices['H'].shape == (20, 20))
    check('condensed Hessian is positive definite', np.linalg.eigvalsh(matrices['H']).min() > 0.)
    report['passed'] = not report['errors']
    (output/'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    report = verify(args.output)
    print(('PASS' if report['passed'] else 'FAIL')+f': {len(report["cases"])} horizon cases independently verified')
    for error in report['errors']:
        print(error)
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
