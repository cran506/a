"""四联图全部取自闭环记录与求解器实测耗时。"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/work72_rewrite_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pressure_metrics import pressure_dwell


def braking_window(results, config):
    """Select one common physical time window from the three pressure traces."""
    limits=[]
    for n in config['highlight_horizons']:
        trace=results[n]['trace']
        active=trace[(trace[:,0]>config['step_time_s']+.005)&(trace[:,2]>=15.)]
        if len(active):
            limits.extend([float(active[0,0]),float(active[-1,0])])
    if not limits:
        return max(config['step_time_s'],.04),min(config['duration_s'],.10)
    left=max(0.,min(limits)-.006)
    right=min(config['duration_s'],max(limits)+.012)
    if right-left<.024:
        midpoint=(left+right)/2
        left=max(0.,midpoint-.012);right=min(config['duration_s'],midpoint+.012)
    return left,right


def plot_pressure_detail(results, summary, output, styles):
    """Separate pressure/input braking figure, using the same unaltered data."""
    config=summary['config'];window=braking_window(results,config)
    fig,(axp,axu)=plt.subplots(2,1,figsize=(11.5,7.5),sharex=True,
                              gridspec_kw={'height_ratios':[1.3,1.]})
    for n in config['highlight_horizons']:
        d=results[n]['data'];p=results[n]['trace'];color,ls=styles[n]
        dwell=pressure_dwell(p)
        axp.plot(p[:,0],p[:,2],color=color,ls=ls,lw=2.1,
                 label=fr'N={n}：$P_2\geq20$ MPa 持续 {dwell:.2f} ms')
        axu.plot(d[:,0],d[:,6],color=color,ls=ls,lw=2.,label=f'N={n}')
    axp.axhspan(21,23,color='#ef476f',alpha=.10)
    axp.axhline(21,color='#e41d48',ls='--',lw=1.1,label='物理安全线 21 MPa')
    axp.axhline(20,color='#6b7280',ls=':',lw=1.,label='统一计时阈值 20 MPa')
    axp.set(ylim=(-.5,22.4),ylabel='有杆腔压力 $P_2$ (MPa)',
            title='制动窗口：比较接近压力边界的持续时间与卸压时刻')
    axp.legend(fontsize=9,loc='lower left',framealpha=.95)
    axu.axhline(0,color='#888888',lw=.8)
    axu.set(ylabel='施加控制电压 $u$ (V)',xlabel='共同物理时间 $t$ (s)',
            title='对应的实际控制电压：只执行每次 QP 的首步')
    axu.legend(fontsize=9,loc='best')
    for ax in (axp,axu):
        ax.set_xlim(*window);ax.grid(True,ls=':',alpha=.4)
    fig.suptitle('习题 7-2：压力制动过程局部对照',fontsize=16,y=.98)
    fig.subplots_adjust(left=.09,right=.98,top=.90,bottom=.12,hspace=.24)
    fig.text(.09,.035,'所有曲线共用原始时间轴与纵轴；持续时间从连续轨迹计算，未平移、偏置或裁剪压力。',
             fontsize=9,color='#536173')
    fig.savefig(output,dpi=180,facecolor='white')
    fig.savefig(output.with_suffix('.pdf'),facecolor='white');plt.close(fig)


def plot_results(results, summary, output):
    installed={font.name for font in font_manager.fontManager.ttflist}
    font=next((x for x in ('Noto Sans CJK SC','Noto Sans CJK JP','SimHei') if x in installed),'DejaVu Sans')
    with plt.rc_context({'font.family':font,'axes.unicode_minus':False,'font.size':10,
                         'axes.edgecolor':'#444444','axes.linewidth':.8}):
        fig, axes=plt.subplots(2,2,figsize=(18,9.5))
        axx,axu,axp,axt=axes.ravel()
        fig.suptitle('作业 7-2：MPC 预测时域 $N$ 对制动限压与计算开销的影响',fontsize=18,y=.98)
        styles={5:('#e41d48','--'),20:('#2463eb','-'),50:('#218449','-.')}
        config=summary['config'];duration=config['duration_s'];step=config['step_time_s'];target=config['target_mm']
        axx.plot([0,step,step,duration],[0,0,target,target],':',color='#333333',label='参考阶跃 50 mm')
        for n in config['highlight_horizons']:
            d=results[n]['data'];p=results[n]['trace'];m=results[n]['metrics'];color,ls=styles[n]
            axx.plot(d[:,0],d[:,2],color=color,ls=ls,lw=1.8,label=f'N={n}，超调 {m["overshoot_pct"]:.2f}%')
            axu.plot(d[:,0],d[:,6],color=color,ls=ls,lw=1.8,label=f'N={n}')
            axp.plot(p[:,0],p[:,2],color=color,ls=ls,lw=1.8,
                     label=f'N={n}，≥20 MPa：{pressure_dwell(p):.2f} ms')
        axx.set(title='(a) 不同预测时域下的位移跟踪',ylabel='活塞位移 $x_p$ (mm)',ylim=(-2,54))
        zoom=axx.inset_axes([.64,.20,.33,.31])
        for n in config['highlight_horizons']:
            d=results[n]['data'];color,ls=styles[n]
            zoom.plot(d[:,0],d[:,2],color=color,ls=ls,lw=1.3)
        zoom.axhline(50,color='#666666',ls=':',lw=.8)
        end_times=[]
        for n in config['highlight_horizons']:
            d=results[n]['data'];near=d[d[:,2]>=target-.5]
            if len(near):end_times.extend([near[0,0],d[np.argmax(d[:,2]),0]])
        zoom_left=max(step,min(end_times)-.003) if end_times else .067
        zoom_right=min(duration,max(end_times)+.008) if end_times else .083
        peak=max(results[n]['metrics']['peak_position_mm'] for n in config['highlight_horizons'])
        zoom.set(xlim=(zoom_left,zoom_right),ylim=(target-.2,max(target+.52,peak+.12)),title='终点超调局部放大')
        zoom.tick_params(labelsize=6);zoom.title.set_fontsize(7);zoom.grid(True,ls=':',alpha=.3)
        axu.set(title='(b) 伺服阀控制电压与输入约束',ylabel='控制电压 $u$ (V)',ylim=(-15,15))
        for lo,hi in [(-15,-10),(10,15)]:axu.axhspan(lo,hi,color='#888888',alpha=.12)
        axu.axhline(10,color='#999999',ls=':',lw=1,label='电压边界 ±10 V');axu.axhline(-10,color='#999999',ls=':',lw=1)
        axp.set(title='(c) 有杆腔制动压力：近边界持续时间与卸压时刻',ylabel='有杆腔压力 $P_2$ (MPa)',ylim=(-.8,23.5))
        axp.axhspan(21,23.5,color='#ef476f',alpha=.12)
        axp.axhline(21,color='#e41d48',ls='--',lw=1.4,label='物理安全线 21 MPa')
        axp.axhline(config['controller']['pmax'],color='#888888',ls=':',lw=1,
                    label=f'预测节点上限 {config["controller"]["pmax"]:g} MPa')
        pressure_zoom=axp.inset_axes([.045,.43,.44,.49])
        for n in config['highlight_horizons']:
            p=results[n]['trace'];color,ls=styles[n]
            pressure_zoom.plot(p[:,0],p[:,2],color=color,ls=ls,lw=1.4)
        pressure_zoom.axhline(21,color='#e41d48',ls='--',lw=.8)
        pressure_zoom.axhline(20,color='#6b7280',ls=':',lw=.7)
        pressure_zoom.set(xlim=braking_window(results,config),ylim=(-.3,22.),title='制动窗口共用坐标放大')
        pressure_zoom.tick_params(labelsize=6);pressure_zoom.title.set_fontsize(7)
        pressure_zoom.grid(True,ls=':',alpha=.3)
        for ax in (axx,axu,axp):
            ax.set_xlim(0,duration);ax.set_xlabel('仿真时间 $t$ (s)');ax.grid(True,ls=':',alpha=.4)
            location='lower right' if ax is axp else ('upper left' if ax is axx else 'upper right')
            ax.legend(fontsize=8,loc=location)
        ns=sorted(results);ms=[results[n]['metrics'] for n in ns]
        qp=np.array([m['backend_run']['mean_ms'] for m in ms]);pressure=np.array([m['peak_p2_mpa'] for m in ms])
        axt.plot(ns,qp,'s-',color='#f18700',lw=1.8,label='底层单次 QP 平均耗时')
        axt.axhline(1,color='#f18700',ls='--',label='采样周期 1 ms')
        axt.axvspan(15,25,color='#b7e7b4',alpha=.35,label='课件候选窗口 15–25')
        axt.set(title='(d) 峰值压力—单次 QP 实测耗时的双轴权衡',xlabel='预测时域步数 $N$',ylabel='底层 QP 平均求解时间 (ms)')
        axt.tick_params(axis='y',colors='#d97706');axt.yaxis.label.set_color('#d97706')
        axt.grid(True,ls=':',alpha=.4)
        right=axt.twinx();right.plot(ns,pressure,'o-',color='#2463eb',lw=1.7,label='连续轨迹 $P_2$ 峰值')
        right.axhline(21,color='#e41d48',ls=':',label='压力安全线 21 MPa')
        right.set(ylabel='有杆腔峰值压力 (MPa)',ylim=(min(20.4,float(min(pressure))-.08),21.1))
        right.tick_params(axis='y',colors='#2463eb');right.yaxis.label.set_color('#2463eb')
        lines,labels=axt.get_legend_handles_labels();other,olabels=right.get_legend_handles_labels()
        axt.legend(lines+other,labels+olabels,fontsize=8,loc='upper left')
        fig.subplots_adjust(left=.06,right=.94,top=.90,bottom=.15,hspace=.32,wspace=.18)
        fig.text(.06,.075,'固定四状态局部灰盒与固定权重，仅改变 N；五维增广，每周期执行一次 CVXPY/OSQP 求解。\n'
                 '图 (d) 为本次实测底层耗时；CVXPY 与完整控制步骤耗时另列。局部线性模型不等同于全孔口非线性实机。',
                 fontsize=9,color='#536173',linespacing=1.6)
        fig.savefig(output,dpi=180,facecolor='white');fig.savefig(output.with_suffix('.pdf'),facecolor='white');plt.close(fig)
        plot_pressure_detail(results,summary,output.with_name('pressure_braking_detail.png'),styles)


if __name__=='__main__':
    import json
    from pathlib import Path
    folder=Path(__file__).resolve().parent/'outputs'
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    results={int(n):dict(data=np.loadtxt(folder/f'N{int(n):02d}.csv',delimiter=',',skiprows=1),
              trace=np.loadtxt(folder/f'N{int(n):02d}_pressure_trace.csv',delimiter=',',skiprows=1),
              metrics=metric) for n,metric in summary['results'].items()}
    plot_results(results,summary,folder/'horizon_comparison.png')
