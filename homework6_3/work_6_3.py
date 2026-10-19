"""《智能流体传动与控制》习题6-3：死区逆补偿鲁棒边界与tanh光滑化。

运行：python work_6_3.py
默认 beta=18；额外比较 beta=5、18、50，输出一张3×2图及原始CSV。
模型和结论边界详见 README.md；不注入人为振荡或事后修改曲线。
"""
import argparse
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'work63_matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(tempfile.gettempdir())/'work63_cache'))

from simulation import load_config, simulate, save_outputs
from verify_experiment import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='自定义JSON配置，替代默认实验配置')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent/'outputs'))
    parser.add_argument('--no-plot', action='store_true', help='仅输出原始数据与指标')
    parser.add_argument('--skip-robustness', action='store_true', help='跳过独立的噪声/零参考保持补充试验')
    args = parser.parse_args()
    config = load_config(args.config)
    results = {key:simulate(key,config) for key in ('under','hard_over','smooth')}
    beta_results = {beta:(results['smooth'] if beta == config['beta_default'] else simulate('smooth',config,beta))
                    for beta in config['beta_values']}
    folder = Path(args.output_dir)
    summary = save_outputs(results,beta_results,config,folder)
    validation = validate(results,beta_results,config,folder/'validation.json')
    if not args.skip_robustness:
        from robustness_checks import run_robustness
        extra = run_robustness(results,config,folder)
        print(f'补充鲁棒性试验：{folder / "robustness.json"}')
        if extra.get('passed') is False:
            print('补充现象检查未全部满足，已保存实际结果，见 robustness.json。')
    print((folder/'metrics_table.txt').read_text(encoding='utf-8'), end='')
    print('平台为D(u)有效信号零区间的阈值重建；近零速度另按0.5mm/s统计。')
    print('高频pp仅由20Hz高通诊断，图中绘制原始数据；有限beta仍有残余平台。')
    print(f'功能检查：{sum(validation["checks"].values())}/{len(validation["checks"])}')
    if not args.no_plot:
        from plotting import plot_comparison
        plot_comparison(results,config,folder/'deadzone_comparison.png')
    if not validation['passed']:
        print('部分默认现象未出现，已保留原始结果；请查 validation.json，禁止冒充全部通过。')
        raise SystemExit(1)
    return summary


if __name__ == '__main__':
    main()
