"""真实五维 SAC 训练：向量环境加速交互，日志保留原始奖励及真实策略熵。"""
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


class GuidedInitializationSAC(SAC):
    """SAC with a declared physics-teacher initialization and training prior.

    The actor stays unchanged during the first 1000 critic warm-up updates.
    Afterwards its SAC objective also includes a measured-state teacher prior.
    No teacher action is mixed into the deployed network output.
    """
    physics_prior_weight=100.

    def _setup_model(self):
        super()._setup_model()
        self.actor.optimizer.register_step_pre_hook(self._physics_prior_gradient)

    def _physics_prior_gradient(self,optimizer,args,kwargs):
        if self.replay_buffer is None or self.replay_buffer.size()<1:return
        from teacher_initialization import teacher_action
        observations=self.replay_buffer.sample(256).observations
        targets=torch.as_tensor(teacher_action(observations.detach().cpu().numpy()),
                                device=self.device)
        with torch.enable_grad():
            mean=self.actor.get_action_dist_params(observations)[0]
            consistency=((torch.tanh(mean)-targets)**2).mean()
            (self.physics_prior_weight*consistency).backward()
        self.last_physics_prior_mse=float(consistency.detach())

    def _sample_action(self,learning_starts,action_noise=None,n_envs=1):
        # Collect the first replay samples with the initialized stochastic actor
        # instead of uniform unsafe voltages; learning_starts still delays SGD.
        return super()._sample_action(0,action_noise,n_envs)

    def _update_learning_rate(self,optimizers):
        super()._update_learning_rate(optimizers)
        rate=self.lr_schedule(self._current_progress_remaining)
        actor_rate=0. if self._n_updates<1000 else rate*.03
        for group in self.actor.optimizer.param_groups:group['lr']=actor_rate


class RewardScale(gym.RewardWrapper):
    def __init__(self,env,multiplier=.01):
        super().__init__(env);self.multiplier=float(multiplier)
    def reward(self,reward):return reward*self.multiplier


class TrainingLog(BaseCallback):
    def __init__(self,root,period=20000):
        super().__init__();self.root=Path(root);self.period=int(period)
        self.recent=deque(maxlen=30);self.step_rewards=deque(maxlen=4000)
        self.episodes=0;self.started=time.perf_counter();self.rows=[];self.last_logged=-1

    def _on_step(self):
        for info in self.locals['infos']:
            if 'raw_reward' in info:self.step_rewards.append(float(info['raw_reward']))
            if 'episode' in info:
                ep=info['episode'];self.recent.append(float(ep['r'])/ep['l']);self.episodes+=1
        if (self.num_timesteps%self.period==0 or self.num_timesteps==1000) and self.num_timesteps!=self.last_logged:
            self.record()
        return True

    def record(self):
        buffer=self.model.replay_buffer
        if buffer.size()<2:return
        sample=buffer.sample(512)
        with torch.no_grad():
            _,logp=self.model.actor.action_log_prob(sample.observations)
            entropy=float(-logp.mean())
        row=dict(timesteps=int(self.model.num_timesteps),
                 reward_mean=float(np.mean(self.recent if self.recent else self.step_rewards)),
                 reward_source='completed_episode_per_step_mean' if self.recent else 'actual_step_window_before_first_completed_episode',
                 normalized_action_entropy=entropy,voltage_entropy=entropy+math.log(10.),
                 ent_coef=float(torch.exp(self.model.log_ent_coef.detach())),episodes=self.episodes,
                 updates=int(self.model._n_updates),elapsed_s=time.perf_counter()-self.started)
        row['physics_prior_action_mse']=float(getattr(self.model,'last_physics_prior_mse',0.))
        self.rows.append(row);self.last_logged=row['timesteps']
        with (self.root/'outputs/training.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n')
            writer.writeheader();writer.writerows(self.rows)
        temporary=self.root/'models/sac_checkpoint_current.zip'
        self.model.save(temporary);temporary.replace(self.root/'models/sac_checkpoint.zip')
        if row['timesteps']%100000==0:
            checkpoints=self.root/'models/checkpoints';checkpoints.mkdir(exist_ok=True)
            self.model.save(checkpoints/f'sac_{row["timesteps"]}.zip')
            from evaluation import run_policy,trajectory_metrics
            try:
                trajectory,_,_=run_policy('sac_fp32',model=self.model)
                probe=dict(timesteps=row['timesteps'],success=True,**trajectory_metrics(trajectory))
            except (RuntimeError,ValueError) as error:
                probe=dict(timesteps=row['timesteps'],success=False,error=str(error))
            with (self.root/'outputs/milestone_evaluations.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(probe,ensure_ascii=False)+'\n')
            print('milestone',json.dumps(probe,ensure_ascii=False),flush=True)
        (self.root/'models/progress.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)


def _export_replay(model,root,seed):
    rb=model.replay_buffer
    observations=rb.observations[:rb.size()].reshape(-1,5)
    rng=np.random.default_rng(seed+1);indices=rng.permutation(len(observations))[:16000]
    obs=observations[indices]
    split=min(12000,max(1,len(obs)*3//4))
    np.savez_compressed(root/'models/calibration_observations.npz',calibration=obs[:split],validation=obs[split:])


def train(root=None,timesteps=1_000_000,seed=2026,n_envs=8,train_freq=4,
          reward_scale=.01,learning_rate=3e-4,pilot_steps=None):
    root=Path(root or Path(__file__).resolve().parent)
    source_names=('train_sac.py','hydraulic_env.py','hydraulic_twin.py','native_twin.cpp',
                  'physical_config.json','teacher_initialization.py')
    source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_names}
    (root/'models').mkdir(parents=True,exist_ok=True);(root/'outputs').mkdir(exist_ok=True)
    for filename in ('training.csv','milestone_evaluations.jsonl'):
        (root/'outputs'/filename).unlink(missing_ok=True)
    torch.set_num_threads(1)
    env=DummyVecEnv([lambda:RewardScale(Monitor(HydraulicGymEnv()),reward_scale)
                     for _ in range(int(n_envs))])
    env.seed(seed)
    model=GuidedInitializationSAC('MlpPolicy',env,learning_rate=learning_rate,buffer_size=100000,
              learning_starts=10000,batch_size=256,gamma=.995,tau=.005,
              train_freq=int(train_freq),gradient_steps=1,ent_coef='auto_0.03',
              target_entropy=.35-math.log(10.),policy_kwargs=dict(net_arch=[64,64]),
              seed=seed,device='cpu',verbose=0)
    from teacher_initialization import initialize_actor
    initialization=initialize_actor(model,root,seed,steps=6000)
    (root/'models/initialization_metadata.json').write_text(json.dumps(initialization,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    model.save(root/'models/sac_initialized.zip')
    from evaluation import run_policy,trajectory_metrics
    initial_rows,_,_=run_policy('sac_fp32',model=model)
    pre=[row for row in initial_rows if row['time_s']<.5-1e-12]
    initial_probe=trajectory_metrics(initial_rows)
    initial_probe.update(
        prefault_velocity_rms_mms=float(np.sqrt(np.mean([row['velocity_error_mms']**2 for row in pre]))),
        prefault_voltage_std_v=float(np.std([row['control_v'] for row in pre])),
        prefault_max_voltage_step_v=float(np.max(np.abs(np.diff([row['control_v'] for row in pre])))))
    (root/'outputs/initialization_evaluation.json').write_text(json.dumps(initial_probe,indent=2)+'\n')
    print('initialized_actor',json.dumps(initial_probe),flush=True)
    callback=TrainingLog(root);started=time.perf_counter()
    model.learn(total_timesteps=int(timesteps),callback=callback,progress_bar=False)
    if callback.last_logged!=model.num_timesteps:callback.record()
    model.save(root/'models/sac_model.zip');_export_replay(model,root,seed)
    metadata=dict(algorithm='stable-baselines3 SAC with explicit physics-teacher actor consistency prior',actual_timesteps=int(model.num_timesteps),
        gradient_updates=int(model._n_updates),seed=seed,wall_seconds=time.perf_counter()-started,
        torch_version=torch.__version__,target_entropy_normalized=.35-math.log(10.),
        target_entropy_voltage=.35,policy_architecture=[5,64,64,1],n_envs=int(n_envs),
        train_frequency_vector_steps=int(train_freq),gradient_steps_per_update=1,
        optimizer_reward_multiplier=reward_scale,learning_rate=learning_rate,
        observation_names=HydraulicGymEnv.observation_names,
        obs_mean=HydraulicGymEnv.obs_mean.tolist(),obs_std=HydraulicGymEnv.obs_std.tolist(),
        reward='-100*e_mm^2-.1*u_V^2-(u_V-u_prev_V)^2+2*I(|e_mm|<.2); explicit physical failure penalty -100000',
        domain_randomization='m ±20%; base beta 600..1400 MPa; Fc ±30%',
        fault_augmentation='80% episodes: time U(.05,.8)s, load U(0,5500)N, beta multiplier U(.5,1)',
        training_initial_position_m=[.0001,.005],training_reference_velocity_m_s=.05,
        physical_config_sha256=hashlib.sha256((root/'physical_config.json').read_bytes()).hexdigest(),
        observation_signs='e=actual-reference, de=actual-reference, dde=actual-reference; PL=P1-(A2/A1)P2 in MPa',
        normalization='User-supplied fixed mean/std Z-score formula; prior engineering statistics, not fitted empirical unit variance',
        policy_initialization='New five-input actor initialized by explicit physics-teacher supervised learning; no ten-dimensional checkpoint reuse. Teacher absent from runtime policy.',
        teacher_initialization=initialization,
        actor_learning_rate_multiplier=.03,critic_warmup_updates=1000,
        actor_objective='SAC actor objective + 100*mean((tanh(mu(s))-five_feature_physics_teacher(s))^2), sampled from the replay buffer; training only',
        physics_prior_weight=float(model.physics_prior_weight),
        model_source_sha256=source_hashes,
        source_hash_capture='Before teacher data collection and SAC interactions; checked again after training',
        initialization_deterministic_evaluation=initial_probe)
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest for name,digest in source_hashes.items()):
        raise RuntimeError('Training source changed during execution; refusing ambiguous model metadata')
    (root/'models/training_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    env.close();return model,metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timesteps',type=int,default=1_000_000);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--n-envs',type=int,default=8);p.add_argument('--train-freq',type=int,default=4)
    p.add_argument('--reward-scale',type=float,default=.01)
    args=p.parse_args();train(timesteps=args.timesteps,seed=args.seed,n_envs=args.n_envs,
                            train_freq=args.train_freq,reward_scale=args.reward_scale)
