#!/usr/bin/env python3
"""习题 7-2 重写版：固定四状态灰盒、五维增广、单次 QP 的时域扫描。"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/work72_rewrite_mpl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/work72_rewrite_cache')
import cvxpy
import numpy as np
import scipy
import osqp
from threadpoolctl import threadpool_limits

from model import HydraulicGreyBox
from pressure_metrics import pressure_exposure
from mpc import CondensedHydraulicMPC, augment_model, condensed_objective, prediction_matrices

ROOT = Path(__file__).resolve().parent


def cpu_name():
    name=platform.processor()
    if not name and Path('/proc/cpuinfo').is_file():
        name=next((line.split(':',1)[1].strip() for line in
                   Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), '')
    return name or platform.machine()


def load_config(path=ROOT/'config.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values, 95)), max_ms=float(np.max(values)))


@threadpool_limits.wrap(limits=1)
def simulate(n, config):
    model = HydraulicGreyBox(config['physical'])
    dt = config['sample_time_s']
    ad, bd, cd = model.discretize(dt)
    substeps = [model.discretize(dt*f) for f in config['pressure_check_fractions']]
    controller = CondensedHydraulicMPC(n, dt, pressure_substeps=substeps, **config['controller'])
    state, previous = model.equilibrium.copy(), 0.
    rows, logs, trace = [], [], []
    grid = np.arange(round(config['duration_s']/dt)+1)*dt
    start_run = time.perf_counter()
    for k, t in enumerate(grid):
        target = config['target_mm'] if t >= config['step_time_s']-1e-12 else 0.
        tic = time.perf_counter()
        if k == len(grid)-1:
            u, info, wall = previous, None, 0.
        else:
            u, info = controller.solve(state, previous, target, ad, bd, cd)
            wall = (time.perf_counter()-tic)*1000
            info.update(time_s=float(t), reference_mm=target, state=state.tolist(),
                        previous_voltage_v=previous, controller_wall_ms=wall)
            logs.append(info)
        rows.append([t, target, *state, u, u-previous,
                     0. if info is None else info['backend_solve_ms'],
                     0. if info is None else info['cvxpy_wall_ms'], wall,
                     0 if info is None else info['iterations']])
        if k < len(grid)-1:
            local_t, local_x = model.pressure_extrema(state, u, dt)
            trace.extend(np.column_stack((t+local_t, local_x[:, 2:])).tolist())
            state = model.step(state, u, dt)
        previous = u
    data, pressure = np.array(rows), np.array(trace)
    active = np.array([log['time_s'] >= config['step_time_s']-1e-12 for log in logs])
    # First solve is compilation warm-up. Time it separately; steady stats use
    # all actual post-command solves, with no cherry-picking of slow iterations.
    backend = np.array([log['backend_solve_ms'] for log in logs])[active]
    backend_run = np.array([log['backend_run_ms'] for log in logs])[active]
    cvxwall = np.array([log['cvxpy_wall_ms'] for log in logs])[active]
    whole = np.array([log['controller_wall_ms'] for log in logs])[active]
    peak = np.max(data[:, 2]); target = config['target_mm']
    reached = np.flatnonzero(data[:, 2] >= target)
    band = np.abs(data[:, 2]-target) <= .02*target
    after_command = data[:, 0] >= config['step_time_s']
    invalid = np.flatnonzero(~band & after_command)
    settling = None if len(invalid) and invalid[-1] == len(data)-1 else float(
        data[(invalid[-1]+1) if len(invalid) else np.flatnonzero(after_command)[0], 0])
    metrics = dict(horizon=n, prediction_ms=n*dt*1000, samples=len(data),
        peak_position_mm=float(peak), overshoot_pct=float(max(0, peak-target)/target*100),
        first_target_s=None if not len(reached) else float(data[reached[0], 0]),
        settling_2pct_s=settling, final_error_mm=float(target-data[-1, 2]),
        peak_p1_mpa=float(np.max(pressure[:, 1])), peak_p2_mpa=float(np.max(pressure[:, 2])),
        min_pressure_mpa=float(np.min(pressure[:, 1:])),
        max_abs_voltage_v=float(np.max(abs(data[:, 6]))),
        max_abs_increment_v=float(np.max(abs(data[:-1, 7]))),
        backend=stats(backend), backend_run=stats(backend_run), cvxpy=stats(cvxwall), controller=stats(whole),
        backend_deadline_miss_pct=float(np.mean(backend > dt*1000)*100),
        controller_deadline_miss_pct=float(np.mean(whole > dt*1000)*100),
        warmup_cvxpy_ms=logs[0]['cvxpy_wall_ms'],
        max_qp_violation=max(log['max_constraint_violation'] for log in logs),
        qp_calls=len(logs), run_wall_s=time.perf_counter()-start_run)
    metrics.update(pressure_exposure(pressure))
    return dict(data=data, trace=pressure, logs=logs, metrics=metrics)


FIELDS = ['t_s','reference_mm','position_mm','velocity_m_s','p1_mpa','p2_mpa',
          'voltage_v','increment_v','backend_solve_ms','cvxpy_wall_ms',
          'controller_wall_ms','iterations']


def save(results, config, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for n, result in results.items():
        prefix = destination/f'N{n:02d}'
        for suffix, fields, array in [('',FIELDS,result['data']),
            ('_pressure_trace',['t_s','p1_mpa','p2_mpa'],result['trace'])]:
            with Path(str(prefix)+suffix+'.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.writer(stream,lineterminator='\n');writer.writerow(fields);writer.writerows(array)
        Path(str(prefix)+'_qp.json').write_text(json.dumps(result['logs'],ensure_ascii=False),encoding='utf-8')
    summary=dict(config=config,results={str(n):r['metrics'] for n,r in results.items()},
                 environment=dict(python=platform.python_version(),numpy=np.__version__,
                                  scipy=scipy.__version__,cvxpy=cvxpy.__version__,osqp=osqp.__version__,
                                  cpu=cpu_name(),system=platform.platform(),threads=1))
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    with (destination/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.writer(stream,lineterminator='\n');writer.writerow(['N','视野ms','超调%','P2峰MPa','最小压力MPa','最终误差mm',
                                                  'P2不低于20MPa持续ms','底层QP均值ms','完整控制均值ms','完整控制P95ms'])
        for n,r in results.items():
            m=r['metrics'];writer.writerow([n,m['prediction_ms'],m['overshoot_pct'],m['peak_p2_mpa'],
                 m['min_pressure_mpa'],m['final_error_mm'],m['p2_above_20mpa_ms'],m['backend_run']['mean_ms'],
                 m['controller']['mean_ms'],m['controller']['p95_ms']])
    model=HydraulicGreyBox(config['physical']);ad,bd,cd=model.discretize(config['sample_time_s'])
    n=20;h,g,f,phi,bias=condensed_objective(ad,bd,cd,n,model.equilibrium,0,config['target_mm'],
                   **{k:config['controller'][k] for k in ('q_position','q_velocity','move_weight')})
    aa,ba,ca=augment_model(ad,bd,cd)
    np.savez(destination/'textbook_matrices_N20.npz',Ac=model.ac,Bc=model.bc,Ad=ad,Bd=bd,cd=cd,
             A_aug=aa,B_aug=ba,c_aug=ca,F=f,Phi=phi,bias=bias,H=h,g=g)
    return summary


def print_table(summary):
    print('\n N | 视野ms | 超调% | P2峰MPa | ≥20MPa持续ms | QP均值ms | QP P95ms | 完整控制均值ms')
    print('-'*98)
    for n, m in summary['results'].items():
        print(f'{int(n):2d} | {m["prediction_ms"]:6.0f} | {m["overshoot_pct"]:5.3f} | '
              f'{m["peak_p2_mpa"]:7.4f} | {m["p2_above_20mpa_ms"]:11.3f} | {m["backend_run"]["mean_ms"]:8.3f} | '
              f'{m["backend_run"]["p95_ms"]:8.3f} | {m["controller"]["mean_ms"]:14.3f}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs')
    parser.add_argument('--horizon',type=int)
    args=parser.parse_args();config=load_config(args.config)
    results={}
    for n in ([args.horizon] if args.horizon else config['horizons']):
        print(f'Running N={n}',flush=True);results[n]=simulate(n,config);m=results[n]['metrics']
        print(f'  overshoot={m["overshoot_pct"]:.3f}% P2max={m["peak_p2_mpa"]:.5f}MPa '
              f'QPmean={m["backend_run"]["mean_ms"]:.3f}ms fullmean={m["controller"]["mean_ms"]:.3f}ms',flush=True)
    summary=save(results,config,args.output)
    print_table(summary)
    if all(n in results for n in config['highlight_horizons']):
        from plotting import plot_results
        plot_results(results,summary,args.output/'horizon_comparison.png')
    print(args.output.resolve(),flush=True)


if __name__ == '__main__':
    main()
