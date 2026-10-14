#!/usr/bin/env python3
"""第七章：50 mm 极限定位、输入约束与回油腔限压 MPC 三策略对比。"""
import argparse
from pathlib import Path
from simulation import ROOT, KINDS, load_config, simulate, save_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiment_config.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--strategy', choices=KINDS, default=None, help='只运行一组供调试')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    results = {}
    for kind in ([args.strategy] if args.strategy else KINDS):
        print(f'Simulating {kind} ...', flush=True)
        results[kind] = simulate(kind, config, progress=True)
        print(results[kind]['metrics'], flush=True)
    save_results(results, args.output_dir)
    if not args.no_plot and len(results) == 3:
        from plotting import plot_comparison
        plot_comparison(results, args.output_dir/'mpc_comparison.png')
    print(f'Saved: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
