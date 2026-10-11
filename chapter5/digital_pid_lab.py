#!/usr/bin/env python3
"""第五章完整实验。运行 python digital_pid_lab.py；无交互弹窗。"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

# 云环境与本地均可运行，图表缓存不依赖用户主目录可写。
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'chapter5-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'chapter5-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from simulation import Experiment, simulate, quantization_probe


COLORS = {'conservative': '#1264b5', 'aggressive': '#d73b3e', 'unfiltered': '#25a579',
          'positional': '#d73b3e', 'incremental': '#1264b5'}


def style():
    for name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'SimHei']:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('提示：未找到中文字体，可安装 Noto Sans CJK。')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'savefig.dpi': 180})


def finish(fig, axes, output, name):
    for ax in np.atleast_1d(axes).flat:
        ax.grid(True, ls=':', alpha=.35)
        ax.set_axisbelow(True)
    fig.tight_layout(rect=(0, .025, 1, .96))
    fig.savefig(output/(name+'.png'), bbox_inches='tight')
    fig.savefig(output/(name+'.pdf'), bbox_inches='tight')
    plt.close(fig)


def classroom_plot(results, output):
    a, b, n = (results[k]['data'] for k in ['conservative', 'aggressive', 'unfiltered'])
    t = a['time_s']
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('第五章：S 型轨迹跟踪、量化噪声与变负载性能权衡', fontsize=16)
    ax[0].plot(t, a['reference_m']*1000, 'k:', lw=2.2, label='S 型参考轨迹：100 mm / 1 s')
    for key, label in [('conservative', '保守增益 Kp=3.8，α=0.85'), ('aggressive', '激进增益 Kp=16，α=0.85')]:
        d = results[key]['data']
        kw = dict(color=COLORS[key], ls='--' if key=='aggressive' else '-', lw=1.8)
        ax[0].plot(t, d['position_m']*1000, label=label, **kw)
        ax[1].plot(t, d['error_m']*1000, label=label, **kw)
    ax[2].plot(t, n['voltage_v'], color=COLORS['unfiltered'], lw=.65, alpha=.75, label='无滤波 α=0')
    ax[2].plot(t, a['voltage_v'], color=COLORS['conservative'], lw=1.4, label='保守增益 + 不完全微分')
    ax[2].plot(t, b['voltage_v'], color=COLORS['aggressive'], ls='--', lw=1., label='激进增益 + 不完全微分')
    for panel in ax:
        panel.axvspan(.35, .70, color='#777777', alpha=.12)
        panel.legend(loc='upper left', fontsize=9)
    ax[0].set_ylabel('相对位移 (mm)')
    ax[1].set_ylabel('真实跟踪误差 (mm)')
    ax[1].axhline(0, color='k', ls=':', lw=1.)
    ax[2].set_ylabel('控制电压 (V)')
    ax[2].set_xlabel('仿真时间 (s)；Ts = 1 ms')
    ax[2].set_ylim(-11, 11)
    ma, mb = results['conservative']['metrics'], results['aggressive']['metrics']
    fig.text(.07, .008, f"重载区间 [0.35, 0.70) s；真实误差 RMS：保守 {ma['heavy_rms_error_mm']:.2f} mm，激进 {mb['heavy_rms_error_mm']:.2f} mm。曲线均为实际数值积分结果。", fontsize=9)
    finish(fig, ax, output, 'classroom_comparison')


def hydraulic_plot(results, output):
    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    fig.suptitle('非线性机理验证：负载 → 腔压 → 阀口有效压差 → 流量增益', fontsize=15)
    for key, label in [('conservative', '保守组'), ('aggressive', '激进组')]:
        d = results[key]['data']; t = d['time_s']
        kw = dict(color=COLORS[key], ls='-' if key=='conservative' else '--')
        ax[0].plot(t, d['p1_pa']/1e6, label=label+' P1', **kw)
        ax[1].plot(t, d['p2_pa']/1e6, label=label+' P2', **kw)
        # Ps-P1<=0 表示伸出进油工作点失效，不伪造实数正向Kq。
        ax[2].plot(t, d['kq_m2_s']/results[key]['metrics']['light_median_kq_m2_s'], label=label+' Kq / 轻载中位值', **kw)
    a = results['conservative']['data']
    ax[0].axhline(21, color='k', ls=':', label='供油压力 21 MPa')
    ax[3].step(a['time_s'], a['load_n']/1000, where='post', color='#694ca0', label='显式工艺阻力')
    ax[0].set_ylabel('P1 (MPa)'); ax[1].set_ylabel('P2 (MPa)')
    ax[2].set_ylabel('伸出支路潜在增益比'); ax[3].set_ylabel('负载 (kN)'); ax[3].set_xlabel('仿真时间 (s)')
    for panel in ax:
        panel.axvspan(.35, .70, color='gray', alpha=.10)
        panel.legend(loc='upper right', fontsize=9)
    finish(fig, ax, output, 'hydraulic_mechanism')


def noise_plot(probe, metrics, output):
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('量化噪声隔离实验：同一条实际轨迹的量化残差重放', fontsize=15)
    t = probe['time_s']; mask = (t>=.15)&(t<=.24)
    ax[0].plot(t[mask], probe['quantization_error_mm'][mask], color='#694ca0')
    ax[0].set_ylabel('量化残差 (mm)')
    ax[1].plot(t[mask], probe['raw_derivative_v'][mask], color='#25a579', lw=1.1, label='理想差分 α=0')
    ax[1].plot(t[mask], probe['filtered_derivative_v'][mask], color='#1264b5', lw=1.7, label='不完全微分 α=0.85')
    ax[1].set_ylabel('量化噪声微分响应 (V)'); ax[1].set_xlabel('仿真时间 (s)'); ax[1].legend()
    text = f"相同输入，统计窗口 0.10–1.00 s：RMS {metrics['raw_derivative_rms_v']:.4f} → {metrics['filtered_derivative_rms_v']:.4f} V，下降 {metrics['reduction_percent']:.1f}%。"
    fig.text(.07, .01, text, fontsize=10)
    finish(fig, ax, output, 'quantization_noise')


def homework_plot(results, suffix, output):
    a, b = results['positional_'+suffix]['data'], results['incremental_'+suffix]['data']
    t = a['time_s']; alpha = results['positional_'+suffix]['config']['alpha']
    fig, ax = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    fig.suptitle(f'习题 5-1：位置式与增量式 PID 抗饱和对比（α={alpha:g}）', fontsize=15)
    ax[0].axhline(100, color='k', ls=':', label='100 mm 阶跃指令')
    for data, name, color in [(a, '位置式：仅输出限幅', COLORS['positional']), (b, '增量式：限幅状态回写', COLORS['incremental'])]:
        for i, field, scale in [(0,'position_m',1000), (1,'sampled_error_m',1000),
                                 (2,'voltage_v',1), (3,'raw_voltage_v',1)]:
            ax[i].plot(t, data[field]*scale, color=color, lw=1.2, label=name)
        zero_idx = np.flatnonzero(data['sampled_error_m']<=0)
        if len(zero_idx):
            ax[1].axvline(t[zero_idx[0]], color=color, ls=':', alpha=.8)
    ax[1].axhline(0, color='k', ls=':', lw=1.)
    ax[2].axhline(10, color='gray', ls=':'); ax[2].axhline(-10, color='gray', ls=':')
    ax[2].set_ylim(-11,11)
    ax[3].set_yscale('symlog', linthresh=10)
    ax[3].axhline(10, color='gray', ls=':'); ax[3].axhline(-10, color='gray', ls=':')
    ax[4].plot(t, a['integral_state_v'], color=COLORS['positional'], label='位置式历史积分 Ki·∑eTs')
    ax[4].plot(t, b['voltage_v'], color=COLORS['incremental'], label='增量式历史记忆 u_prev（已截断）')
    for axis, label in zip(ax, ['位移 (mm)', '采样误差 (mm)', '实际指令 (V)', '未限幅候选 (V)\n对称对数坐标', '内部状态 (V)']):
        axis.set_ylabel(label)
        axis.legend(loc='upper right', fontsize=9)
    ax[4].set_xlabel('仿真时间 (s)；两组完全相同增益，积分分离关闭')
    finish(fig, ax, output, 'anti_windup_'+suffix)


def write_csv(data, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = ['time_s']+[k for k in data if k!='time_s']
        writer.writerow(keys)
        writer.writerows(zip(*(data[k] for k in keys)))


def run(args):
    if not np.isfinite(args.homework_duration) or args.homework_duration < 1.2:
        raise ValueError('课后仿真时长至少为1.2s')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    shared = dict(load_force=args.load_force, valve_tau=args.valve_tau,
                  ki=args.classroom_ki, kd=args.classroom_kd)
    experiments = [Experiment('conservative', **shared), Experiment('aggressive', kp=16, **shared),
                   Experiment('unfiltered', alpha=0, **shared)]
    for suffix, alpha in [('raw',0.), ('filtered',.85)]:
        for controller in ['positional','incremental']:
            experiments.append(Experiment(controller+'_'+suffix, kind='step', controller=controller,
                                          kp=4.2, ki=22, kd=.06, alpha=alpha, integral_separation=False,
                                          duration=args.homework_duration, load_force=0., valve_tau=args.valve_tau))
    results = {}
    for cfg in experiments:
        print(f'运行 {cfg.name}：Kp={cfg.kp:g}, Ki={cfg.ki:g}, Kd={cfg.kd:g}, α={cfg.alpha:g} ...', flush=True)
        result = simulate(cfg)
        results[cfg.name] = result
        write_csv(result['data'], output/(cfg.name+'.csv'))
        print(f"  完成；末端真实误差 {result['metrics']['final_error_mm']:.4f} mm", flush=True)
    probe, probe_metrics = quantization_probe(results['conservative'])
    write_csv(probe, output/'quantization_probe.csv')
    summary = {k:{'config':r['config'],'metrics':r['metrics']} for k,r in results.items()}
    summary['quantization_probe'] = probe_metrics
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    style()
    classroom_plot(results, output); hydraulic_plot(results, output); noise_plot(probe, probe_metrics, output)
    homework_plot(results, 'raw', output); homework_plot(results, 'filtered', output)
    from reporting import write_reports
    reports = write_reports(results, probe_metrics, output)
    print('已生成图表、8份CSV、summary.json和实验报告：'+str(output), flush=True)
    return results, reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).with_name('outputs')))
    parser.add_argument('--load-force', type=float, default=180000., help='课堂工艺阻力 (N)，非图片增益缩放系数')
    parser.add_argument('--valve-tau', type=float, default=.015, help='阀一阶滞后 (s)，0为理想阀')
    parser.add_argument('--classroom-ki', type=float, default=22.)
    parser.add_argument('--classroom-kd', type=float, default=.06)
    parser.add_argument('--homework-duration', type=float, default=3.)
    args = parser.parse_args()
    run(args)
