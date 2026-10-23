#!/usr/bin/env python3
"""习题 7-2：CVXPY 液压 MPC 预测时域 N 的安全性与计算量实验。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'work72_matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'work72_cache'))

from simulation import ROOT, load_config, save_results, simulate


def _fmt(value, decimals=3):
    return '—' if value is None else f'{value:.{decimals}f}'


def print_table(results):
    print('\n预测时域 | 超调 % | P2 峰值 MPa | 单次QP中位 ms | 周期QP均值 ms | CVXPY 均值 ms | 控制中位 ms | 控制超时 % | 恢复/改进次数 | 状态')
    print('-'*153)
    for n, result in results.items():
        metric = result['metrics']
        print(f'{n:>8d} | {_fmt(metric["overshoot_pct"]):>6} | '
              f'{_fmt(metric["peak_pressure_mpa"]):>12} | '
              f'{_fmt(metric["solver_attempt"]["median_ms"]):>13} | '
              f'{_fmt(metric["solver"]["mean_ms"]):>12} | '
              f'{_fmt(metric["cvxpy_call"]["mean_ms"]):>13} | '
              f'{_fmt(metric["whole_controller"]["median_ms"]):>11} | '
              f'{_fmt((metric["whole_controller"]["over_1ms_fraction"] or 0.)*100.,1):>10} | '
              f'{metric["restoration_count"]:>2}/{metric["refinement_count"]:<2} | {metric["status"]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiment_config.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--horizon', type=int, help='仅运行所列 N，便于调试')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--progress', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    horizons = [args.horizon] if args.horizon is not None else config['horizons']
    if any(n not in config['horizons'] for n in horizons):
        parser.error('--horizon must be in experiment_config.json')
    results = {}
    for n in horizons:
        print(f'Running N={n}, T_pred={n*config["ts"]*1000:.0f} ms ...', flush=True)
        results[n] = simulate(n, config, progress=args.progress)
        metric = results[n]['metrics']
        print(f'  {metric["status"]}: overshoot={metric["overshoot_pct"]:.3f}% '
              f'P2_peak={metric["peak_pressure_mpa"]:.4f} MPa, '
              f'solver_mean={_fmt(metric["solver"]["mean_ms"])} ms, '
              f'restored={metric["restoration_count"]}, '
              f'refined={metric["refinement_count"]}', flush=True)
    save_results(results, args.output_dir)
    print_table(results)
    if not args.no_plot and all(n in results for n in config['highlight_horizons']):
        from plotting import plot_results
        summary_rows = [dict(horizon=n, **dict(
            peak_pressure_mpa=result['metrics']['peak_pressure_mpa'],
            solver_attempt_median_ms=result['metrics']['solver_attempt']['median_ms'],
            solver_attempt_mean_ms=result['metrics']['solver_attempt']['mean_ms'],
            overshoot_pct=result['metrics']['overshoot_pct'],
            status=result['metrics']['status']))
            for n, result in results.items()]
        plot_data = {n: dict(t_s=result['data']['t_s'],
                             x_mm=result['data']['x_mm'],
                             u_v=result['data']['u_v'],
                             pressure_t_s=result['pressure_trace']['t_s'],
                             p2_mpa=result['pressure_trace']['p2_mpa'],
                             failure_time_s=(None if result['metrics']['failure'] is None
                                             else result['metrics']['failure']['time_s']))
                     for n, result in results.items()}
        plot_results(plot_data, summary_rows, config,
                     args.output_dir/'horizon_comparison.png')
    print(f'Outputs: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
