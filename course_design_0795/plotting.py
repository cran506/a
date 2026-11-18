"""全部图来自本次参数或原始结果；没有预设性能曲线。"""
from pathlib import Path
import csv,json,os
os.environ.setdefault('MPLCONFIGDIR','/tmp/course_design_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from physics import Parameters,friction,effective_voltage,trajectory

LABELS={'PID':'增量式 PID','ModelAssisted':'二自由度模型辅助','InputOnlyMPC':'仅输入约束 MPC','LTV_MPC':'双腔限压 LTV-MPC'}
COLORS={'PID':'#555555','ModelAssisted':'#e68613','InputOnlyMPC':'#15925b','LTV_MPC':'#345de5'}
STYLES={'PID':'--','ModelAssisted':'-.','InputOnlyMPC':':','LTV_MPC':'-'}


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def column(rows,name):return np.array([float(r[name]) for r in rows])


def setup():
    names={f.name for f in font_manager.fontManager.ttflist}
    fonts=[n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','SimHei','DejaVu Sans'] if n in names]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':fonts,'axes.unicode_minus':False,
                         'font.size':10,'axes.titlesize':11,'legend.fontsize':8,'pdf.fonttype':42})


def save(fig,root,name):
    for ax in fig.axes:ax.grid(ls=':',alpha=.3)
    fig.tight_layout()
    for suffix in ['png','pdf']:fig.savefig(root/f'{name}.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def build_figures(results):
    setup();root=Path(results);figdir=root/'figures';figdir.mkdir(exist_ok=True)
    data=np.load(root/'identification_dataset.npz');signals=np.genfromtxt(root/'identification_signals.csv',delimiter=',',names=True)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for j,(name,label,scale) in enumerate([('x','位移 / mm',1000),('p1','P1 / MPa',1e-6),('p2','P2 / MPa',1e-6)]):
        axes[j].plot(signals['time_s'],signals[f'raw_{name}_'+('m' if j==0 else 'pa')]*scale,color='#bbbbbb',lw=.7,label='带噪合成观测')
        axes[j].plot(signals['time_s'],signals[f'filtered_{name}_'+('m' if j==0 else 'pa')]*scale,color='#345de5',lw=1.1,label='因果级联滤波')
        axes[j].plot(data['t'],data['clean'][:,[0,2,3][j]]*scale,'--',color='#d36d17',lw=.8,label='无噪仿真真值')
        axes[j].set_ylabel(label)
    axes[0].set_title('多源信号：中值窗口5 + 二阶Butterworth 120 Hz（保留真实相位滞后）');axes[0].legend(ncol=3)
    axes[-1].set_xlabel('时间 / s');save(fig,figdir,'filtering')
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
    axes[0].plot(data['t'][:-1],data['u'],color='#345de5');axes[0].set_ylabel('辨识激励 / V')
    axes[1].plot(data['t'],data['clean'][:,2]/1e6,label='P1');axes[1].plot(data['t'],data['clean'][:,3]/1e6,label='P2')
    axes[1].set(xlabel='时间 / s',ylabel='无噪压力 / MPa');axes[1].legend();save(fig,figdir,'excitation')
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));p=Parameters();vel=np.linspace(-.2,.2,600);u=np.linspace(-2,2,600)
    axes[0].plot(vel,[friction(v,p.vector()) for v in vel]);axes[0].set(xlabel='速度 / (m/s)',ylabel='摩擦力 / N',title='Stribeck + tanh滑动近似')
    axes[1].plot(u,[effective_voltage(ui) for ui in u]);axes[1].axvspan(-.5,.5,color='#dddddd',alpha=.5)
    axes[1].set(xlabel='驱动电压 / V',ylabel='有效阀电压 / V',title='±0.5 V正遮盖死区');save(fig,figdir,'nonlinearities')
    history=read(root/'de_history.csv');summary=json.loads((root/'identification_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.3));generation=column(history,'generation')
    axes[0].semilogy(generation,column(history,'loss'));axes[0].set(xlabel='DE代数',ylabel='无量纲均方目标')
    axes[1].plot(generation,column(history,'beta_pa')/1e9);axes[1].axhline(summary['truth']['beta']/1e9,color='gray',ls='--')
    axes[1].set(xlabel='DE代数',ylabel='体积模量参数 / GPa')
    axes[2].plot(generation,column(history,'Ct_m3_s_pa')/1e-12);axes[2].axhline(summary['truth']['Ct']/1e-12,color='gray',ls='--')
    axes[2].set(xlabel='DE代数',ylabel='Ct / (10^-12 m³/(s·Pa))');save(fig,figdir,'identification')
    fig,axes=plt.subplots(2,1,figsize=(10,5.5),sharex=True)
    tc=data['validation_t'];truth=data['validation_clean'];pred=data['validation_prediction']
    axes[0].plot(tc,truth[:,0]*1000,label='独立激励下的真值');axes[0].plot(tc,pred[:,0]*1000,'--',label='辨识模型预测')
    axes[0].set_ylabel('位移 / mm');axes[0].legend()
    axes[1].plot(tc,(pred[:,2]-truth[:,2])/1e3,label='P1预测残差');axes[1].plot(tc,(pred[:,3]-truth[:,3])/1e3,label='P2预测残差')
    axes[1].set(xlabel='时间 / s',ylabel='残差 / kPa');axes[1].legend();save(fig,figdir,'independent_validation')
    t=np.arange(1201)*.001;r=trajectory(t)
    fig,axes=plt.subplots(3,1,figsize=(9,5.5),sharex=True)
    for i,(label,scale) in enumerate([('位置 / mm',1000),('速度 / (m/s)',1),('加速度 / (m/s²)',1)]):
        axes[i].plot(t,r[:,i]*scale);axes[i].set_ylabel(label)
    axes[0].set_title('统一参考：0.8 s内完成80 mm位移，随后保持');axes[-1].set_xlabel('时间 / s');save(fig,figdir,'trajectory')
    for scenario in ['nominal','extreme','braking']:
        records=read(root/f'{scenario}_traces.csv');policies=list(dict.fromkeys(r['policy'] for r in records))
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        first=[r for r in records if r['policy']==policies[0]]
        axes[0,0].plot(column(first,'time_s'),column(first,'reference_m')*1000,'k:',label='位置参考')
        for name in policies:
            group=[row for row in records if row['policy']==name];ts=column(group,'time_s')
            for ax,values in [(axes[0,0],column(group,'x_m')*1000),(axes[0,1],column(group,'error_mm')),
                              (axes[1,0],column(group,'u_v')),
                              (axes[1,1],np.maximum(column(group,'p1_max_mpa'),column(group,'p2_max_mpa')))]:
                ax.plot(ts,values,color=COLORS[name],ls=STYLES[name],label=LABELS[name])
        axes[0,0].set(ylabel='位移 / mm');axes[0,1].set(ylabel='跟踪误差 / mm')
        axes[1,0].set(ylabel='实际控制电压 / V');axes[1,1].set(ylabel='两腔子步峰压包络 / MPa')
        axes[1,1].axhline(21,color='crimson',ls='--',label='21 MPa安全线')
        for ax in axes.flat:
            ax.set_xlabel('时间 / s');ax.legend(fontsize=7,loc='best')
            if scenario=='extreme':ax.axvline(.5,color='gray',ls=':',lw=1.)
        fig.suptitle({'nominal':'标称工况：同一对象、初态与参考','extreme':'统一极限工况：0.50 s突加5000 N，同时体积模量减半','braking':'附加制动安全工况：相同非零初速度下的急停'}[scenario],y=1.02,fontsize=13)
        save(fig,figdir,scenario)
    records=read(root/'extreme_traces.csv');fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for name in ['PID','ModelAssisted','LTV_MPC']:
        group=[r for r in records if r['policy']==name and .45<=float(r['time_s'])<=.75]
        ts=column(group,'time_s')
        axes[0].plot(ts,column(group,'error_mm'),label=LABELS[name],color=COLORS[name],ls=STYLES[name])
        axes[1].plot(ts,column(group,'v_m_s'),color=COLORS[name],ls=STYLES[name])
        if name!='PID':axes[2].plot(ts,column(group,'estimated_load_n'),label=LABELS[name],color=COLORS[name])
    axes[2].plot([.45,.5,.5,.75],[0,0,5000,5000],'k:',label='仅用于绘图的真实外载')
    axes[0].set_ylabel('位置误差 / mm');axes[0].legend();axes[1].set_ylabel('实际速度 / (m/s)')
    axes[2].set(xlabel='时间 / s',ylabel='因果负载估计 / N');axes[2].legend()
    for ax in axes:ax.axvline(.5,color='gray',ls='--')
    save(fig,figdir,'disturbance_detail')
    met=json.loads((root/'extreme_metrics.json').read_text());names=list(met)
    fig,axes=plt.subplots(1,4,figsize=(12,3.2))
    for ax,key,title in zip(axes,['rmse_mm','disturbance_mae_mm','TV_V','p_peak_mpa'],['RMSE / mm','扰动MAE / mm','总变差 TV / V','峰压 / MPa']):
        ax.bar(np.arange(len(names)),[met[n][key] or 0 for n in names],color=[COLORS[n] for n in names])
        ax.set_xticks(np.arange(len(names)),['PID','模型辅助','MPC']);ax.set_title(title)
        if key=='p_peak_mpa':ax.axhline(21,color='crimson',ls='--')
    save(fig,figdir,'metrics')
    # 隔离积分状态的控制器台架测试，不冒充液压对象的闭环响应。
    from controllers import IncrementalPID
    pid=IncrementalPID(kp=250,ki=2000,kd=0);t=np.arange(501)*.001;e=np.where(t<.25,.05,-.001)
    inc=[];position=[];integral=0.
    for error in e:
        inc.append(pid.update([error,0,0,0],[0,0,0,0]));integral+=2000*error*.001
        position.append(np.clip(250*error+integral,-10,10))
    fig,ax=plt.subplots(figsize=(9,3.4));ax.plot(t,position,'--',color='#666666',label='仅末端限幅的位置PI')
    ax.plot(t,inc,color='#345de5',label='饱和输出记忆的增量PI');ax.axvline(.25,color='gray',ls=':')
    ax.set(xlabel='时间 / s',ylabel='实际限幅输出 / V',title='隔离积分状态测试：e由+50 mm切换为−1 mm（非液压闭环）');ax.legend()
    save(fig,figdir,'antiwindup')
    return figdir
