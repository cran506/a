#!/usr/bin/env python3
"""第五章课堂实验：只生成一张位移—误差—控制电压三联结果图。"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'chapter5-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'chapter5-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from simulation import Experiment, simulate, quantization_probe

BLUE, RED, GREEN = '#0000ff', '#ff0000', '#26b876'


def style():
    for name in ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Microsoft YaHei', 'SimHei']:
        if any(f.name == name for f in font_manager.fontManager.ttflist):
            plt.rcParams['font.sans-serif'] = [name, 'DejaVu Sans']
            break
    else:
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('提示：未找到中文字体，可安装 Noto Sans CJK。')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'figure.dpi': 120, 'savefig.dpi': 180})


def classroom_plot(results, noise, output):
    a, b, n = (results[k]['data'] for k in ['conservative', 'aggressive', 'unfiltered'])
    ma, mb = (results[k]['metrics'] for k in ['conservative', 'aggressive'])
    ca, cb = (results[k]['config'] for k in ['conservative', 'aggressive'])
    t = a['time_s']
    fig, ax = plt.subplots(3, 1, figsize=(14.8, 8.6), sharex=True)
    fig.suptitle('第五章 数字 PID 闭环控制：S 型轨迹跟踪、量化噪声抑制与变负载性能权衡',
                 fontsize=15, y=.975)
    ax[0].plot(t, a['reference_m']*1000, 'k:', lw=2.4, label='S 型规划参考轨迹 r(t)（目标 100 mm）')
    ax[0].plot(t, a['position_m']*1000, color=BLUE, lw=2,
               label=f"情形 A：保守增益 Kp={ca['kp']:g}（重载滞后，卸载后平稳收敛）")
    ax[0].plot(t, b['position_m']*1000, color=RED, ls='--', lw=1.5,
               label=f"情形 B：激进增益 Kp={cb['kp']:g}（重载误差较小，卸载后持续振荡）")
    ax[1].plot(t, a['error_m']*1000, color=BLUE, lw=2, label='情形 A 真实跟踪误差：保守增益')
    ax[1].plot(t, b['error_m']*1000, color=RED, ls='--', lw=1.5, label='情形 B 真实跟踪误差：激进增益')
    ax[1].axhline(0, color='k', ls=':', lw=1.)
    ax[2].plot(t, n['voltage_v'], color=GREEN, lw=.65, alpha=.8,
               label='无滤波 PID 电压（α=0；12 位 ADC 量化）')
    ax[2].plot(t, a['voltage_v'], color=BLUE, lw=1.55,
               label='保守增益 + 不完全微分（α=0.85）')
    ax[2].plot(t, b['voltage_v'], color=RED, ls='--', lw=1.1,
               label='激进增益 + 不完全微分（α=0.85）')
    for panel in ax:
        panel.axvspan(.35, .70, color='gray', alpha=.16)
        panel.grid(True, ls=':', alpha=.5)
        panel.set_axisbelow(True)
    ax[0].legend(loc='lower right', fontsize=9, framealpha=.96)
    ax[1].legend(loc='upper right', fontsize=9, framealpha=.96)
    ax[2].legend(loc='upper right', fontsize=9, framealpha=.96)
    ax[0].set_ylabel('液压缸相对位移 $x_p$ (mm)')
    ax[1].set_ylabel('真实跟踪误差 e (mm)')
    ax[2].set_ylabel('控制电压 u (V)')
    ax[2].set_xlabel('仿真时间 t (s)   [采样周期 $T_s$ = 1 ms]')
    ax[0].set_ylim(-5, 120)
    ax[1].set_ylim(-20, 26)
    ax[2].set_ylim(-10.5, 10.5)
    ax[2].set_xlim(-.07, 1.47)
    ax[2].set_xticks(np.arange(0., 1.41, .2))
    ax[0].text(.525, 110, f"重载冲压区间\nKq 实测下降 {ma['kq_decline_percent']:.1f}%",
               ha='center', va='center', fontsize=9, color='#666666',
               bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#999999', alpha=.94))
    heavy = (t >= .35)&(t < .70)
    peak = np.flatnonzero(heavy)[np.argmax(a['error_m'][heavy])]
    ax[1].annotate(f"【看点 1：重载跟踪滞后】\nP1 升高，Ps−P1 缩小，Kq 下降\n"
                   f"保守组最大滞后 {ma['heavy_peak_abs_error_mm']:.1f} mm",
                   xy=(t[peak], a['error_m'][peak]*1000), xytext=(.02, 23),
                   va='top', fontsize=9, color=BLUE,
                   arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2),
                   bbox=dict(boxstyle='round,pad=.35', fc='white', ec=BLUE, alpha=.96))
    late_b = mb['hold_windows'][-1]
    ax[1].annotate(f"【看点 2：卸载后高增益振荡】\nKq 恢复后，激进组围绕目标反复波动\n"
                   f"末 0.1 s 误差均值 {late_b['mean_error_mm']:+.2f} mm，峰峰值 {late_b['p2p_mm']:.2f} mm",
                   xy=(1.12, b['error_m'][1120]*1000), xytext=(.75, -6.),
                   va='top', fontsize=9, color=RED,
                   arrowprops=dict(arrowstyle='->', color=RED, lw=1.2),
                   bbox=dict(boxstyle='round,pad=.35', fc='white', ec=RED, alpha=.96))
    ax[2].annotate(f"【看点 3：量化噪声与不完全微分】\n12 位 ADC：δx={ca['sensor_lsb_mm']:.5f} mm\n"
                   f"同轨迹量化残差的微分 RMS 降低 {noise['reduction_percent']:.1f}%",
                   xy=(.20, n['voltage_v'][200]), xytext=(.02, -4.2),
                   va='top', fontsize=9, color='#008000',
                   arrowprops=dict(arrowstyle='->', color='#008000', lw=1.2),
                   bbox=dict(boxstyle='round,pad=.35', fc='white', ec='#008000', alpha=.96))
    # 同一张图内放大电压细节，使 ADC 毛刺在 ±10V 主坐标中仍可辨认。
    detail = ax[2].inset_axes([.64, .07, .20, .28])
    window = (t >= .16)&(t <= .22)
    detail.plot(t[window], n['voltage_v'][window], color=GREEN, lw=.8)
    detail.plot(t[window], a['voltage_v'][window], color=BLUE, lw=1.1)
    detail.set_title('控制电压局部放大 (V)', fontsize=7, pad=2)
    detail.set_xlim(.16, .22)
    detail.set_xticks([.16, .19, .22])
    detail.tick_params(labelsize=6, length=2, pad=1)
    detail.grid(True, ls=':', alpha=.4)
    fig.subplots_adjust(left=.08, right=.985, top=.90, bottom=.13, hspace=.14)
    fig.text(.08, .035,
             f"真实闭环积分：两组均使用 Ki={ca['ki']:g}、Kd={ca['kd']:g}；"
             f"阀 τ={ca['valve_tau']*1000:g} ms；重载 {ca['load_force']/1000:g} kN，加载过渡 {ca['load_ramp']*1000:g} ms，卸载过渡 {ca['unload_ramp']*1000:g} ms。\n"
             f"重载误差 RMS：保守 {ma['heavy_rms_error_mm']:.2f} mm，激进 {mb['heavy_rms_error_mm']:.2f} mm；"
             f"保守组末端真实误差 {ma['final_error_mm']:+.3f} mm。课堂模型补充参数见 README。",
             fontsize=9, color='#333333')
    path = output/'classroom_comparison.png'
    fig.savefig(path, facecolor='white')
    plt.close(fig)
    return path


def write_csv(data, path):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        keys = ['time_s']+[k for k in data if k!='time_s']
        writer.writerow(keys)
        writer.writerows(zip(*(data[k] for k in keys)))


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shared = dict(load_force=args.load_force, valve_tau=args.valve_tau,
                  load_ramp=args.load_ramp, unload_ramp=args.unload_ramp,
                  ki=args.classroom_ki,
                  kd=args.classroom_kd, rtol=args.rtol)
    experiments = [Experiment('conservative', **shared),
                   Experiment('aggressive', kp=16., **shared),
                   Experiment('unfiltered', alpha=0., **shared)]
    results = {}
    for cfg in experiments:
        print(f'运行 {cfg.name}：Kp={cfg.kp:g}, Ki={cfg.ki:g}, Kd={cfg.kd:g}, α={cfg.alpha:g} ...', flush=True)
        result = simulate(cfg)
        results[cfg.name] = result
        m = result['metrics']
        print(f"  完成；重载最大误差 {m['heavy_peak_abs_error_mm']:.3f} mm；"
              f"末端真实误差 {m['final_error_mm']:+.4f} mm", flush=True)
        if args.save_data:
            write_csv(result['data'], output/(cfg.name+'.csv'))
    _, noise = quantization_probe(results['conservative'])
    summary = {k:{'config':r['config'], 'metrics':r['metrics']} for k,r in results.items()}
    summary['quantization_probe'] = noise
    if args.save_data:
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                      allow_nan=False)+'\n', encoding='utf-8')
    style()
    path = classroom_plot(results, noise, output)
    print('已生成唯一结果图：'+str(path), flush=True)
    return results, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(__file__).with_name('outputs')))
    parser.add_argument('--load-force', type=float, default=180000., help='课堂工艺阻力 (N)')
    parser.add_argument('--load-ramp', type=float, default=.04, help='重载接触时的加载过渡 (s)')
    parser.add_argument('--unload-ramp', type=float, default=.02, help='脱离接触时的卸载过渡 (s)')
    parser.add_argument('--valve-tau', type=float, default=.0053, help='阀芯一阶响应时间常数 (s)')
    parser.add_argument('--classroom-ki', type=float, default=9.)
    parser.add_argument('--classroom-kd', type=float, default=.02)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--save-data', action='store_true', help='另外保存三组CSV与指标JSON，默认只输出PNG')
    run(parser.parse_args())
