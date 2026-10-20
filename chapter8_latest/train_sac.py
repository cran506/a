"""Actual SB3 SAC training and measured tanh-corrected policy entropy."""
from __future__ import annotations
import csv
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
from hydraulic_env import HydraulicGymEnv


class RewardScale(gym.RewardWrapper):
    """Optimizer-only constant scaling; Monitor/CSV keep the original reward."""
    def reward(self, reward):
        return reward*.01


class TrainingLog(BaseCallback):
    def __init__(self, root, period=20000):
        super().__init__()
        self.root=Path(root);self.period=period;self.recent=[];self.episodes=0
        self.started=time.perf_counter();self.rows=[]

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                ep=info['episode'];self.recent.append(float(ep['r'])/ep['l'])
                self.episodes+=1
        if self.num_timesteps % self.period==0 or self.num_timesteps==1000:
            buffer=self.model.replay_buffer
            if buffer.size()>10:
                sample=buffer.sample(512)
                with torch.no_grad():
                    _,logp=self.model.actor.action_log_prob(sample.observations)
                    entropy=float(-logp.mean().item())
                coef=float(torch.exp(self.model.log_ent_coef.detach()).cpu())
                row=dict(timesteps=self.num_timesteps,
                    reward_mean=float(np.mean(self.recent[-30:])) if self.recent else float('nan'),
                    normalized_action_entropy=entropy,voltage_entropy=entropy+math.log(10),
                    ent_coef=coef,episodes=self.episodes,updates=self.model._n_updates,
                    elapsed_s=time.perf_counter()-self.started)
                self.rows.append(row)
                with (self.root/'outputs/training.csv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row),lineterminator='\n');writer.writeheader();writer.writerows(self.rows)
                print(json.dumps(row),flush=True)
                self.model.save(self.root/'models/sac_checkpoint.zip')
        return True


def train(root, timesteps=1_000_000, seed=2026):
    root=Path(root);(root/'models').mkdir(exist_ok=True);(root/'outputs').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    env=RewardScale(Monitor(HydraulicGymEnv()))
    # A target for physical-voltage entropy .35 corresponds to normalized
    # tanh-action entropy .35-ln(10); the units must not be interchanged.
    model=SAC('MlpPolicy',env,learning_rate=3e-4,buffer_size=100000,
        learning_starts=10000,batch_size=256,gamma=.995,tau=.005,
        train_freq=4,gradient_steps=1,ent_coef='auto_0.03',
        target_entropy=.35-math.log(10),policy_kwargs=dict(net_arch=[64,64]),
        seed=seed,device='cpu',verbose=0)
    callback=TrainingLog(root)
    started=time.perf_counter()
    model.learn(total_timesteps=timesteps,callback=callback,progress_bar=False)
    model.save(root/'models/sac_model.zip')
    rb=model.replay_buffer
    rng=np.random.default_rng(seed+1)
    n=rb.size();indices=rng.permutation(n)[:16000]
    obs=rb.observations[indices,0,:]
    np.savez_compressed(root/'models/calibration_observations.npz',calibration=obs[:12000],validation=obs[12000:])
    metadata=dict(algorithm='stable-baselines3 SAC',actual_timesteps=model.num_timesteps,
        gradient_updates=model._n_updates,seed=seed,wall_seconds=time.perf_counter()-started,
        torch_version=torch.__version__,target_entropy_normalized=.35-math.log(10),
        target_entropy_voltage=.35,policy_architecture=[10,64,64,1],
        deterministic_evaluation=True,observation_names=HydraulicGymEnv.observation_names,
        optimizer_reward_multiplier=.01,
        obs_mean=HydraulicGymEnv.obs_mean.tolist(),obs_std=HydraulicGymEnv.obs_std.tolist(),
        reward='-100*e_mm^2-0.1*u_V^2-(u_V-u_prev_V)^2+2*I(|e_mm|<0.2); failure penalty -500',
        fault_augmentation='80% episodes: event U(.04,.8)s; load U(1000,5500)N; bulk multiplier U(.5,1)',
        domain_randomization='mass ±20%; base bulk .6–1.4GPa; Coulomb friction ±30%')
    (root/'models/training_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    env.close()
    return model,metadata
