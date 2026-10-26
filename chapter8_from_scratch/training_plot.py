"""训练曲线只来自原始 CSV；完整范围与线性局部视图同时保留。"""
from pathlib import Path
import csv
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter8_scratch_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/chapter8_scratch_cache')
import numpy as np
import matplotlib.pyplot as plt


def read_training(root):
    with (Path(root)/'outputs/training.csv').open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def draw_training(ax,rows,*,compact=False):
    steps=np.array([int(r['timesteps']) for r in rows])/1000.
    reward=np.array([float(r['reward_mean']) for r in rows])
    entropy=np.array([float(r['replay_voltage_entropy']) for r in rows])
    counts=np.array([int(r['reward_window_count']) for r in rows])
    if np.any(np.diff(steps)<=0):raise ValueError('Training steps must increase')
    # Start the reward curve only when the same 10000-step window is full.
    # Entropy can already be measured before that time.
    full=counts==10000
    for lo,hi,color in ((0,50,'#eeeeee'),(50,400,'#e8edfc'),(400,steps[-1],'#e7f5e9')):
        if min(hi,steps[-1])>lo:ax.axvspan(lo,min(hi,steps[-1]),color=color,zorder=0)
    line,=ax.plot(steps[full],reward[full],color='#3567ee',lw=1.65,label='式 (8.1) 基础奖励，固定 10000 步窗口')
    ax.set_yscale('symlog',linthresh=10.)
    ax.set_ylabel('平均每步基础奖励（对称对数轴）',color='#3567ee')
    ax.tick_params(axis='y',colors='#3567ee')
    ax.set_xlim(0,steps[-1]);ax.set_xlabel('实际训练交互步数 / 千步')
    ax.grid(True,ls=':',alpha=.35);ax.set_axisbelow(True)
    second=ax.twinx()
    ent,=second.plot(steps,entropy,'--',color='#333333',lw=1.3,label='策略熵 H(u)：replay 状态平均')
    target=second.axhline(.35,color='#777777',lw=.8,ls=':',label='目标熵 0.35 nat')
    second.set_ylim(min(-.8,float(entropy.min())-.15),3.15)
    second.set_ylabel('H(u) / nat（tanh 后，u 以 V 计）')
    ax.axvline(50,color='#999999',ls=':',lw=.8)
    ax.set_title(f'(a) 从随机策略开始的真实 SAC 训练（{int(steps[-1]*1000):,} 步）',pad=12)
    ax.legend(handles=[line,ent,target],loc='lower right',fontsize=6.6 if compact else 8,framealpha=.95)
    ax.text(.015,.965,'0–50k：均匀探索\n50k 后：原生 SAC 更新，无教师预热',
            transform=ax.transAxes,va='top',fontsize=7 if compact else 8,
            bbox=dict(fc='white',ec='none',alpha=.8))
    inset=second.inset_axes([.42,.12,.48,.30],zorder=20)
    inset.plot(steps[full],reward[full],color='#3567ee',lw=1.)
    inset.set(xlim=(0,steps[-1]),ylim=(-200,5))
    inset.grid(True,ls=':',alpha=.3);inset.tick_params(labelsize=6)
    inset.set_title('同一原始奖励：线性局部放大 [−200, 5]',fontsize=7)
    return second


def generate_training_figure(root=None):
    from reporting import _font
    root=Path(root or Path(__file__).resolve().parent);_font()
    fig,ax=plt.subplots(figsize=(13.8,6.7))
    draw_training(ax,read_training(root))
    ax.set_title('SAC 从零训练：高熵探索 → 策略学习 → 稳态优化',pad=16)
    fig.subplots_adjust(left=.09,right=.90,bottom=.17,top=.89)
    fig.text(.09,.075,'奖励为真实固定步窗口均值；失败终止成本另存。完整早期数据保留，插图只放大显示范围。',fontsize=8)
    fig.text(.09,.035,'PPT 的 −180→−2 为对照目标；本图不修改回报、不插值成 S 形，也不以固定故障探针代替训练状态熵。',fontsize=8)
    paths=[]
    for suffix in ('.png','.pdf'):
        p=root/'outputs'/('sac_training_actual'+suffix);fig.savefig(p,facecolor='white',dpi=180);paths.append(str(p))
    plt.close(fig);return paths


if __name__=='__main__':generate_training_figure()
