"""固定重载稳态短窗验证：真实执行各训练检查点，不拟合课件回报。"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from hydraulic_env import HydraulicGymEnv


def validate_training(root=None,episodes=16,horizon_steps=50):
    root=Path(root or Path(__file__).resolve().parent);out=root/'outputs'
    torch.set_num_threads(1)
    checkpoints=[(0,root/'models/sac_untrained.zip')]
    checkpoints.extend((step,root/f'models/checkpoints/sac_{step}.zip') for step in range(100000,1000000,100000))
    checkpoints.append((1000000,root/'models/sac_model.zip'))
    metrics=[];trace=[];sources={}
    for step,path in checkpoints:
        model=SAC.load(path,device='cpu');episode_returns=[];all_rewards=[]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2029)
            for ep in range(episodes):
                env=HydraulicGymEnv(domain_randomization=True,fault_augmentation=False,max_steps=horizon_steps)
                env.reset(seed=8100+ep,options=dict(position=.05,velocity=.05,fault_time=0.,load_n=5000.,beta_multiplier=.5))
                state,steady=env.twin.equilibrium_for_velocity(.05,load_n=5000.,position=.05)
                env.twin.reset(state);env.last_u=steady
                obs=env.normalize_obs(env.raw_observation());rewards=[]
                for k in range(horizon_steps):
                    action,_=model.predict(obs,deterministic=False)
                    obs,reward,terminated,_,info=env.step(action)
                    if terminated:raise RuntimeError('Fixed validation terminated: '+str(info['physical_failure']))
                    rewards.append(info['base_reward'])
                    trace.append(dict(timesteps=step,episode=ep,step=k+1,
                        control_v=info['control_v'],position_error_mm=info['position_error_mm'],
                        velocity_error_mms=info['velocity_error_mms'],base_reward=info['base_reward']))
                env.close();episode_returns.append(float(np.mean(rewards)));all_rewards.extend(rewards)
        metrics.append(dict(timesteps=step,mean_step_reward=float(np.mean(all_rewards)),
            episode_mean_std=float(np.std(episode_returns)),episodes=episodes,horizon_steps=horizon_steps,
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        sources[str(path.relative_to(root))]=metrics[-1]['model_sha256']
    for name,rows in (('training_validation.csv',metrics),('training_validation_trace.csv',trace)):
        with (out/name).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    description=dict(experiment='Fixed loaded steady-state short-window stochastic-policy validation',
        training_reward_replacement=False,horizon_s=horizon_steps*.001,episodes_per_checkpoint=episodes,
        initial_position_m=.05,initial_velocity_m_s=.05,load_n=5000.,beta_multiplier=.5,
        initial_pressure='Force/flow equilibrium of each randomized physical plant at 5000N and50mm/s',
        previous_voltage='Actual equilibrium voltage immediately before the validation window; not an actor input',
        reward='Unchanged textbook Eq8.1 base_reward, averaged per actual environment step',
        domain_randomization='m±20%, beta600..1400MPa, Fc±30%, same fixed episode seeds8100...',
        actual_extra_validation_interactions=len(trace),validation_not_added_to_replay=True,
        raw_trace='training_validation_trace.csv',checkpoint_sha256=sources)
    (out/'training_validation_metadata.json').write_text(json.dumps(description,ensure_ascii=False,indent=2)+'\n')
    return metrics


def plot_validation(root=None):
    from reporting import _font
    import matplotlib.pyplot as plt
    root=Path(root or Path(__file__).resolve().parent);_font()
    with (root/'outputs/training_validation.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
    with (root/'outputs/training.csv').open(newline='') as stream:training=list(csv.DictReader(stream))
    x=np.array([int(r['timesteps']) for r in rows])/1000.
    y=np.array([float(r['mean_step_reward']) for r in rows])
    fig,ax=plt.subplots(figsize=(13.5,6.6))
    for lo,hi,color in ((0,50,'#eeeeee'),(50,400,'#e8edfc'),(400,1000,'#e7f5e9')):
        ax.axvspan(lo,hi,color=color,zorder=0)
    left,=ax.plot(x,y,'o-',lw=1.8,ms=3.2,color='#3567ee',label='固定重载工况验证：实际平均每步奖励')
    ax.set(xlim=(0,1000),xlabel='实际 SAC 训练交互步数 / 千步',ylabel='固定 50 ms 验证平均每步奖励')
    ax.tick_params(axis='y',colors='#3567ee');ax.yaxis.label.set_color('#3567ee');ax.grid(True,ls=':',alpha=.4)
    second=ax.twinx()
    tx=np.array([int(r['timesteps']) for r in training])/1000.
    hy=np.array([float(r['replay_voltage_entropy']) for r in training])
    right,=second.plot(tx,hy,'--',lw=1.35,color='#333333',label='策略电压熵：训练 replay 状态平均')
    second.axhline(.35,ls=':',lw=.8,color='#777777');second.set_ylabel('H(u) / nat');second.set_ylim(min(-.8,hy.min()-.1),3.15)
    ax.legend(handles=[left,right],loc='center right',fontsize=8)
    ax.set_title('SAC 从零训练：固定重载工况验证奖励与策略熵',pad=15)
    fig.subplots_adjust(left=.09,right=.90,top=.88,bottom=.18)
    fig.text(.09,.08,'每个奖励点均实际运行同一组 16 个随机化对象、每回合 50 ms；折线只连接已测检查点。',fontsize=8)
    fig.text(.09,.035,'此图为短窗验证指标，不等同于 2000 步完整训练回报；完整训练日志及完整显示范围另见 sac_training_actual.png。',fontsize=8)
    for extension in ('.png','.pdf'):fig.savefig(root/'outputs'/('training_validation'+extension),dpi=180,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    validate_training();plot_validation()
