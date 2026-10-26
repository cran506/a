"""原生 SAC 从随机初始策略学习：无教师预热、先验损失或事后策略精修。

基础奖励始终按固定真实交互步窗口统计。物理失败终止成本另列。
策略熵同时在固定物理探针与当前 replay 上计算，保留 tanh 雅可比和 V 单位变换。
"""
from __future__ import annotations
import argparse
from collections import deque
import csv
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from hydraulic_env import HydraulicGymEnv


class VanillaSAC(SAC):
    """Original SAC losses; only use a separate temperature optimizer rate."""
    temperature_learning_rate=1e-3

    def _update_learning_rate(self,optimizers):
        super()._update_learning_rate(optimizers)
        if self.ent_coef_optimizer is not None:
            for group in self.ent_coef_optimizer.param_groups:
                group['lr']=self.temperature_learning_rate


class RewardScale(gym.RewardWrapper):
    def __init__(self,env,multiplier=.001):
        super().__init__(env);self.multiplier=float(multiplier)
    def reward(self,reward):return float(reward)*self.multiplier


def fixed_physical_probe(seed=8201,count=64):
    """Reset-derived, fixed measured observations; no fitted reward/entropy points."""
    env=HydraulicGymEnv(domain_randomization=True,fault_augmentation=False)
    observations=[]
    for index in range(count):
        options={} if index%2==0 else dict(fault_time=0.,load_n=5000.,beta_multiplier=.5)
        obs,_=env.reset(seed=seed+index,options=options)
        observations.append(obs.copy())
    env.close()
    return np.array(observations,dtype=np.float32)


class TrainingLog(BaseCallback):
    def __init__(self,root,probes,period=10000,window=10000,seed=2026):
        super().__init__();self.root=Path(root);self.period=int(period);self.seed=int(seed)
        self.base=deque(maxlen=window);self.raw=deque(maxlen=window)
        self.terminal=deque(maxlen=window);self.failure=deque(maxlen=window)
        self.completed=deque(maxlen=30);self.episodes=0;self.rows=[]
        self.probes=np.asarray(probes,dtype=np.float32)
        self.started=time.perf_counter();self.last_logged=-1

    def _on_step(self):
        for info in self.locals['infos']:
            self.base.append(float(info['base_reward']))
            self.raw.append(float(info['raw_reward']))
            self.terminal.append(float(info['failure_penalty']))
            self.failure.append(bool(info['physical_failure']))
            if 'episode' in info:
                episode=info['episode'];self.completed.append(float(episode['r'])/episode['l'])
                self.episodes+=1
        if (self.num_timesteps%self.period==0 or self.num_timesteps==1000) and self.last_logged!=self.num_timesteps:
            self.record()
        return True

    def _entropy(self,observations,seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            with torch.no_grad():
                x=torch.as_tensor(observations,dtype=torch.float32,device=self.model.device)
                _,logp=self.model.actor.action_log_prob(x)
            return float(-logp.mean())

    def record(self):
        if not self.base:return
        fixed_obs=np.tile(self.probes,(16,1))
        fixed_entropy=self._entropy(fixed_obs,self.seed+100)
        numpy_state=np.random.get_state()
        try:replay=self.model.replay_buffer.sample(1024).observations.detach().cpu().numpy()
        finally:np.random.set_state(numpy_state)
        replay_entropy=self._entropy(replay,self.seed+self.num_timesteps)
        row=dict(timesteps=int(self.model.num_timesteps),reward_mean=float(np.mean(self.base)),
            reward_source='base_reward_fixed_last_10000_real_environment_steps',
            reward_window_count=len(self.base),raw_reward_mean=float(np.mean(self.raw)),
            terminal_penalty_mean=float(np.mean(self.terminal)),window_physical_failures=int(sum(self.failure)),
            completed_episode_per_step_raw_mean=float(np.mean(self.completed)) if self.completed else None,
            normalized_action_entropy=fixed_entropy,voltage_entropy=fixed_entropy+math.log(10.),
            fixed_probe_voltage_entropy=fixed_entropy+math.log(10.),
            replay_voltage_entropy=replay_entropy+math.log(10.),
            behavior_voltage_entropy=math.log(20.) if self.num_timesteps<=self.model.learning_starts else replay_entropy+math.log(10.),
            target_voltage_entropy=float(self.model.target_entropy+math.log(10.)),
            ent_coef=float(torch.exp(self.model.log_ent_coef.detach())),episodes=self.episodes,
            updates=int(self.model._n_updates),elapsed_s=time.perf_counter()-self.started)
        self.rows.append(row);self.last_logged=row['timesteps']
        with (self.root/'outputs/training.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');writer.writeheader();writer.writerows(self.rows)
        checkpoint=self.root/'models/sac_checkpoint_current.zip'
        self.model.save(checkpoint);checkpoint.replace(self.root/'models/sac_checkpoint.zip')
        if row['timesteps']%100000==0:
            directory=self.root/'models/checkpoints';directory.mkdir(exist_ok=True)
            self.model.save(directory/f'sac_{row["timesteps"]}.zip')
            from evaluation import run_policy,trajectory_metrics
            try:
                trajectory,_,_=run_policy('sac_fp32',model=self.model)
                probe=dict(timesteps=row['timesteps'],success=True,**trajectory_metrics(trajectory))
            except (ValueError,RuntimeError) as error:
                probe=dict(timesteps=row['timesteps'],success=False,error=str(error))
            with (self.root/'outputs/milestone_evaluations.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(probe,ensure_ascii=False)+'\n')
            print('milestone',json.dumps(probe,ensure_ascii=False),flush=True)
        (self.root/'models/progress.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)


def _export_replay(model,root,seed):
    buffer=model.replay_buffer
    observations=buffer.observations[:buffer.size()].reshape(-1,5)
    indices=np.random.default_rng(seed+1).permutation(len(observations))[:16000]
    chosen=observations[indices]
    np.savez_compressed(root/'models/calibration_observations.npz',calibration=chosen[:12000],validation=chosen[12000:])


def train(root=None,timesteps=1_000_000,seed=2026,n_envs=4,train_freq=4,
          gradient_steps=2,reward_scale=.001,learning_rate=3e-4):
    root=Path(root or Path(__file__).resolve().parent)
    for name in ('models','outputs'):(root/name).mkdir(parents=True,exist_ok=True)
    sources=('train_sac.py','hydraulic_env.py','hydraulic_twin.py','native_twin.cpp','physical_config.json')
    source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in sources}
    for name in ('training.csv','milestone_evaluations.jsonl'):(root/'outputs'/name).unlink(missing_ok=True)
    torch.set_num_threads(1)
    env=DummyVecEnv([lambda:RewardScale(Monitor(HydraulicGymEnv()),reward_scale) for _ in range(int(n_envs))])
    env.seed(seed)
    model=VanillaSAC('MlpPolicy',env,learning_rate=learning_rate,buffer_size=100000,
        learning_starts=50000,batch_size=128,gamma=.99,tau=.005,train_freq=int(train_freq),
        gradient_steps=int(gradient_steps),ent_coef='auto_0.03',target_entropy=.35-math.log(10.),
        policy_kwargs=dict(net_arch=[64,64]),seed=seed,device='cpu',verbose=0)
    # An uninformed high-entropy actor. No expert targets, trajectories or weights.
    with torch.no_grad():
        model.actor.mu.weight.zero_();model.actor.mu.bias.zero_()
        model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-.1)
    model.save(root/'models/sac_untrained.zip')
    probes=fixed_physical_probe(seed+101)
    np.savez_compressed(root/'models/entropy_probe_observations.npz',observations=probes)
    callback=TrainingLog(root,probes,seed=seed);started=time.perf_counter()
    model.learn(total_timesteps=int(timesteps),callback=callback,progress_bar=False)
    if callback.last_logged!=model.num_timesteps:callback.record()
    model.save(root/'models/sac_model.zip');_export_replay(model,root,seed)
    metadata=dict(algorithm='Vanilla SAC from uninformed random initial policy; original actor and critic losses',
        actual_timesteps=int(model.num_timesteps),gradient_updates=int(model._n_updates),seed=seed,
        wall_seconds=time.perf_counter()-started,torch_version=torch.__version__,
        policy_architecture=[5,64,64,1],n_envs=int(n_envs),learning_starts=50000,
        train_frequency_vector_steps=int(train_freq),gradient_steps_per_update=int(gradient_steps),batch_size=128,
        gamma=.99,learning_rate=learning_rate,temperature_learning_rate=model.temperature_learning_rate,
        optimizer_reward_multiplier=reward_scale,target_entropy_normalized=.35-math.log(10.),target_entropy_voltage=.35,
        observation_names=HydraulicGymEnv.observation_names,obs_mean=HydraulicGymEnv.obs_mean.tolist(),
        obs_std=HydraulicGymEnv.obs_std.tolist(),
        reward='-100*e_mm^2-.1*u_V^2-(u_V-u_prev_V)^2+2*I(|e_mm|<.2); physical failure cost separately -100000',
        reward_log='Consistent last10000 actual step base_reward window from first row; terminal cost and raw episode returns separate',
        entropy_log='Tanh-corrected voltage entropy H(u)=H(a)+ln10; fixed physical probe and replay separately recorded',
        initial_actor='Random hidden layers; mu output weight/bias zero; log_std output weight zero, bias -0.1. No teacher pretraining or inherited checkpoint.',
        teacher_initialization=False,teacher_prior=False,post_training_teacher_consolidation=False,
        uniform_exploration_steps=50000,domain_randomization='m±20%; beta600..1400MPa; Fc±30%',
        fault_augmentation='80% episodes: time U(.05,.8)s, load U(0,5500)N, beta multiplier U(.5,1)',
        training_initial_position_m=[.0001,.005],training_reference_velocity_m_s=.05,
        physical_config_sha256=source_hashes['physical_config.json'],model_source_sha256=source_hashes,
        deployed_model_sha256=hashlib.sha256((root/'models/sac_model.zip').read_bytes()).hexdigest(),
        deployment_origin='Exactly the final 1M SAC checkpoint, without offline actor modification')
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in source_hashes.items()):
        raise RuntimeError('Training sources changed during execution')
    (root/'models/training_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    env.close();return model,metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps',type=int,default=1_000_000);parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args();train(timesteps=args.timesteps,seed=args.seed)
