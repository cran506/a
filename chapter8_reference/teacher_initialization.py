"""可审计的五维机理教师初始化；其后仍执行真实 SAC 交互与梯度更新。

教师只读取五维已测状态，不读取故障时刻、负载标签、随机质量或隐藏腔压。
通过 P_L 与两腔面积比估计准稳态回油压力；反馈增益只由实测量平滑调度。
监督预热不是 SAC 训练回报，不计入交互步数；分别保存元数据。
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from hydraulic_env import HydraulicGymEnv


def teacher_action(observation):
    """Normalized obs [...,5] -> normalized absolute voltage [...,1]."""
    obs=np.asarray(observation,dtype=np.float64)
    if obs.shape[-1]!=5:raise ValueError('Teacher requires exactly five measured features')
    raw=obs*np.array([10.,50.,500.,8.,2.])+np.array([0.,0.,0.,10.,21.])
    e,dv,acc,pl,ps=np.moveaxis(raw,-1,0)
    ratio=.00038/.00021
    # Q1/Q2≈A1/A2 and PL=P1-(A2/A1)*P2 close the quasi-steady pressure estimate.
    p2=(ps*1e6+ratio**2*.1e6-pl*1e6)/(ratio**2+1./ratio)
    p1=pl*1e6+p2/ratio
    q=.00038*.05+1e-14*(p1-p2)
    nominal=.5+q/(1.64e-9*np.sqrt(np.maximum(ps*1e6-p1,.5e6)))
    sigmoid=lambda value: 1./(1.+np.exp(-np.clip(value,-40.,40.)))
    # Unloaded stiff dynamics need low acceleration feedback at 1 ms.
    # Measured negative acceleration/pressure increases it during heavy-load transients.
    gate=np.maximum(sigmoid((pl-10.)*2.),sigmoid((-acc-3000.)/1000.))
    ka=.00005+.00045*gate
    voltage=np.clip(nominal-3.*e-.2*dv-ka*acc,-10.,10.)
    return (voltage/10.)[...,None].astype(np.float32)


def build_teacher_data(root,seed=2026,episodes=48,steps_per_episode=800):
    """Collect actual DR trajectories plus explicitly labelled regression coverage."""
    root=Path(root);rng=np.random.default_rng(seed)
    env=HydraulicGymEnv(domain_randomization=True,fault_augmentation=True,max_steps=steps_per_episode)
    actual=[];failures=0;minima=[];maxima=[]
    for ep in range(episodes):
        obs,info=env.reset(seed=seed+ep,options=dict(velocity=.05))
        min_error=0.;max_error=0.
        for k in range(steps_per_episode):
            actual.append(obs.copy())
            action=teacher_action(obs)
            # Measured state coverage comes from small real command perturbations,
            # never from editing the position, velocity or chamber pressures.
            noisy=np.clip(action+rng.normal(0.,.004,size=(1,)),-1.,1.).astype(np.float32)
            obs,_,terminated,truncated,info=env.step(noisy)
            min_error=min(min_error,info['velocity_error_mms'])
            max_error=max(max_error,info['velocity_error_mms'])
            if terminated or truncated:
                failures+=int(terminated);break
        minima.append(min_error);maxima.append(max_error)
    env.close()
    physical=np.asarray(actual,dtype=np.float32)
    # Independent regression coverage is not claimed as ODE interaction or replay.
    count=max(30000,len(physical))
    raw=np.column_stack([rng.uniform(-2.,2.,count),rng.uniform(-60.,60.,count),
                         rng.uniform(-30000.,30000.,count),rng.uniform(.5,20.5,count),
                         np.full(count,21.)])
    synthetic=((raw-HydraulicGymEnv.obs_mean)/HydraulicGymEnv.obs_std).astype(np.float32)
    data=np.concatenate([physical,synthetic],axis=0)
    target=teacher_action(data)
    (root/'models').mkdir(exist_ok=True,parents=True)
    np.savez_compressed(root/'models/teacher_dataset.npz',observations=data,actions=target,
                        physical_observations=physical)
    metadata=dict(method='Five-feature approximate inverse plus fixed position/velocity/acceleration feedback',
        teacher_version='v2_quasisteady_pressure_measured_gain_scheduling',
        reference_velocity_m_s=.05,return_pressure_estimate='P2=(Ps+r^2*Pr-PL)/(r^2+1/r), r=A1/A2',
        gains=dict(kp_v_per_mm=3.,kv_v_per_mms=.2,ka_base_v_per_mms2=.00005,ka_increment_v_per_mms2=.00045),
        gain_schedule='ka=0.00005+0.00045*max(sigmoid(2*(PL_MPa-10)),sigmoid((-acc_mms2-3000)/1000))',
        physical_priors=dict(A1_m2=.00038,A2_m2=.00021,Pr_pa=.1e6,K_flow=1.64e-9,C_ip=1e-14,deadzone_v=.5),
        teacher_uses_only_measured_five_features=True,future_fault_or_random_parameters_as_input=False,
        seed=seed,episodes=episodes,physical_steps=len(physical),physical_failures=failures,
        regression_coverage_samples=count,
        training_only_minimum_velocity_error_mms=float(min(minima)),
        training_only_maximum_velocity_error_mms=float(max(maxima)),
        simulated_command_noise_std_v=.04,
        scope='Actor behavior initialization only; neither supervisory samples nor teacher trajectories are SAC training-return samples.')
    return data,target,metadata


def initialize_actor(model,root,seed=2026,steps=8000):
    """Fit the actor mean to a declared physics teacher before genuine SAC."""
    root=Path(root);started=time.perf_counter()
    data,target,metadata=build_teacher_data(root,seed)
    rng=np.random.default_rng(seed+1);order=rng.permutation(len(data))
    split=int(.9*len(order));train_indices=order[:split];test_indices=order[split:]
    device=model.device
    x=torch.as_tensor(data,device=device);y=torch.as_tensor(target,device=device)
    # Inverse tanh logits cannot represent exact hard saturation; leave 1e-4 margin.
    logits=torch.atanh(y.clamp(-.9999,.9999))
    params=list(model.actor.latent_pi.parameters())+list(model.actor.mu.parameters())
    optimizer=torch.optim.Adam(params,lr=8e-4)
    log=[]
    for step in range(steps):
        indices=rng.choice(train_indices,size=512,replace=True)
        mean=model.actor.get_action_dist_params(x[indices])[0]
        # Action error dominates; a mild logit term preserves saturated tails.
        loss=((torch.tanh(mean)-y[indices])**2).mean()+.01*((mean-logits[indices])**2).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if step%1000==0 or step==steps-1:
            with torch.no_grad():
                pred=torch.tanh(model.actor.get_action_dist_params(x[test_indices])[0])
                mae=float(torch.mean(torch.abs(pred-y[test_indices])))
            log.append(dict(step=step+1,loss=float(loss.detach()),held_out_action_mae=mae))
    with torch.no_grad():
        model.actor.log_std.weight.zero_();model.actor.log_std.bias.fill_(-3.)
    metadata.update(behavior_gradient_steps=steps,behavior_training_samples=len(train_indices),
        behavior_validation_samples=len(test_indices),validation_split_seed=seed+1,
        initial_log_std=-3.,wall_seconds=time.perf_counter()-started,log=log,
        final_held_out_voltage_mae_v=log[-1]['held_out_action_mae']*10.)
    model.save(root/'models/sac_behavior_initialized.zip')
    (root/'models/teacher_initialization.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return metadata
