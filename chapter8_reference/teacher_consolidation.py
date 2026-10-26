"""SAC 之后单独记录的机理教师雅可比整形。

神经网络依旧直接输出绝对阀电压；部署没有外接前馈或教师。该步骤把
过强的局部加速度敏感度约束到五维机理教师，避免微小权重/INT8 扰动
触发 1 ms 采样下的电压极限环。此处梯度步不算 SAC 环境交互。
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from teacher_initialization import teacher_action


def _teacher_torch(obs):
    raw=obs*obs.new_tensor([10.,50.,500.,8.,2.])+obs.new_tensor([0.,0.,0.,10.,21.])
    e,dv,acc,pl,ps=raw.unbind(dim=-1)
    ratio=.00038/.00021
    p2=(ps*1e6+ratio**2*.1e6-pl*1e6)/(ratio**2+1./ratio)
    p1=pl*1e6+p2/ratio
    nominal=.5+(.00038*.05+1e-14*(p1-p2))/(1.64e-9*torch.sqrt(torch.clamp(ps*1e6-p1,min=.5e6)))
    gate=torch.maximum(torch.sigmoid(2.*(pl-10.)),torch.sigmoid((-acc-3000.)/1000.))
    ka=.00005+.00045*gate
    return ((nominal-3.*e-.2*dv-ka*acc).clamp(-10.,10.)/10.).unsqueeze(-1)


def _mean_action(model,obs):
    return torch.tanh(model.actor.get_action_dist_params(obs)[0])


def _check(model):
    from hydraulic_env import HydraulicGymEnv
    from evaluation import trajectory_metrics
    env=HydraulicGymEnv(domain_randomization=False,fault_augmentation=False,max_steps=1000)
    obs,_=env.reset(options=dict(evaluation=True,velocity=.05,position=.0001,fault_time=.1,load_n=5000.,beta_multiplier=.5))
    rows=[];failure=None
    for step in range(601):
        raw=env.raw_observation();action,_=model.predict(obs,deterministic=True)
        voltage=float(np.asarray(action).reshape(-1)[0])*10.
        rows.append(dict(time_s=.4+env.time,velocity_error_mms=float(raw[1]),control_v=voltage,position_error_mm=float(raw[0])))
        if step==600:break
        obs,_,terminated,_,info=env.step(np.asarray(action,dtype=np.float32))
        if terminated:failure=info.get('physical_failure');break
    env.close()
    metrics=trajectory_metrics(rows)
    pre=rows[20:100]
    metrics.update(prefault_voltage_std_v=float(np.std([r['control_v'] for r in pre])),
                   prefault_velocity_rms_mms=float(np.sqrt(np.mean([r['velocity_error_mms']**2 for r in pre]))),
                   failure=failure)
    return metrics


def _jacobian_report(model,points):
    x=torch.tensor(points,dtype=torch.float32,device=model.device,requires_grad=True)
    pred=_mean_action(model,x)
    student=torch.autograd.grad(pred.sum(),x,create_graph=False)[0].detach().cpu().numpy()
    t=torch.tensor(points,dtype=torch.float32,device=model.device,requires_grad=True)
    expected=torch.autograd.grad(_teacher_torch(t).sum(),t)[0].detach().cpu().numpy()
    return dict(normalized_student=student.tolist(),normalized_teacher=expected.tolist(),
                acceleration_gain_v_per_mms2_student=(student[:,2]*10./500.).tolist(),
                acceleration_gain_v_per_mms2_teacher=(expected[:,2]*10./500.).tolist())


def consolidate(model,root,seed=2028,steps=12000):
    root=Path(root);started=time.perf_counter();torch.set_num_threads(1)
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    directory=root/'models';directory.mkdir(exist_ok=True)
    model.save(directory/'sac_before_consolidation.zip')
    before_hash=hashlib.sha256((directory/'sac_before_consolidation.zip').read_bytes()).hexdigest()
    before=_check(model)
    rng=np.random.default_rng(seed)
    with np.load(directory/'teacher_dataset.npz') as f:
        wide=np.asarray(f['observations'],dtype=np.float32)
    # Explicitly labelled regression anchors, not invented physical trajectories.
    count=40000
    raw=np.column_stack([rng.normal(0.,.1,count),rng.normal(0.,2.,count),
                         rng.normal(0.,1000.,count),rng.uniform(2.4,5.8,count),
                         np.full(count,21.)])
    local=((raw-np.array([0.,0.,0.,10.,21.]))/np.array([10.,50.,500.,8.,2.])).astype(np.float32)
    # Fine anchors around unloaded tracking equilibrium fix the 1 ms local loop slope.
    fine_raw=np.column_stack([rng.normal(0.,.015,count),rng.normal(0.,.25,count),
                              rng.normal(0.,100.,count),rng.uniform(2.6,5.4,count),
                              np.full(count,21.)])
    fine=((fine_raw-np.array([0.,0.,0.,10.,21.]))/np.array([10.,50.,500.,8.,2.])).astype(np.float32)
    np.savez_compressed(directory/'consolidation_observations.npz',
                        unloaded=local,fine_unloaded=fine)
    xwide=torch.tensor(wide,device=model.device);ywide=torch.tensor(teacher_action(wide),device=model.device)
    xlocal=torch.tensor(local,device=model.device);xfine=torch.tensor(fine,device=model.device)
    parameters=list(model.actor.latent_pi.parameters())+list(model.actor.mu.parameters())
    optimizer=torch.optim.Adam(parameters,lr=2e-4)
    weights=torch.tensor([1.,1.,100.,2.,2.],device=model.device)
    nominal_points=np.array([[0.,0.,0.,(3.597-10.)/8.,0.],
                              [0.,0.,0.,(17.-10.)/8.,0.]],dtype=np.float32)
    jac_before=_jacobian_report(model,nominal_points)
    logs=[];best=None;best_state=None
    for step in range(int(steps)):
        phase=step/max(steps-1,1)
        for group in optimizer.param_groups:group['lr']=2e-4*(.1+.9*.5*(1.+np.cos(np.pi*phase)))
        wi=rng.integers(0,len(wide),256);li=rng.integers(0,count,128);fi=rng.integers(0,count,128)
        anchors=torch.cat([xlocal[li],xfine[fi]],dim=0).detach().requires_grad_(True)
        target=_teacher_torch(anchors)
        expected_jac=torch.autograd.grad(target.sum(),anchors,create_graph=False)[0].detach()
        target=target.detach()
        predicted=_mean_action(model,anchors)
        student_jac=torch.autograd.grad(predicted.sum(),anchors,create_graph=True)[0]
        wide_loss=((_mean_action(model,xwide[wi])-ywide[wi])**2).mean()
        anchor_loss=((predicted-target)**2).mean()
        jac_loss=(((student_jac-expected_jac)*weights)**2).mean()
        loss=wide_loss+20.*anchor_loss+2.*jac_loss
        optimizer.zero_grad();loss.backward();optimizer.step()
        if (step+1)%2000==0 or step==steps-1:
            measured=_check(model)
            record=dict(step=step+1,loss=float(loss.detach()),wide_loss=float(wide_loss.detach()),
                        anchor_loss=float(anchor_loss.detach()),jacobian_loss=float(jac_loss.detach()),commissioning=measured)
            logs.append(record);print(json.dumps(record),flush=True)
            score=(1000. if measured['failure'] else 0.)+100.*measured['prefault_voltage_std_v']+10.*measured['prefault_velocity_rms_mms']+abs(measured['minimum_velocity_error_mms'])+10.*(measured['recovery_1mms_s'] or 1.)+abs(measured['final_100ms_velocity_bias_mms'])*5.
            if best is None or score<best:
                best=score;best_state=copy.deepcopy(model.actor.state_dict());selected_step=step+1
    model.actor.load_state_dict(best_state)
    after=_check(model)
    metadata=dict(method='Post-SAC five-feature teacher action and input-Jacobian consolidation',
        environment_interactions=0,gradient_steps=int(steps),seed=seed,
        mean_actor_parameters_only=True,teacher_source_sha256=hashlib.sha256((root/'teacher_initialization.py').read_bytes()).hexdigest(),
        input_checkpoint_sha256=before_hash,
        regression_anchor_counts=dict(broad_existing=len(wide),unloaded_neighborhood=count,fine_unloaded_neighborhood=count),
        loss='wide_action_MSE +20*unloaded_anchor_action_MSE+2*mean((jac_student-jac_teacher)*[1,1,100,2,2])^2',
        checkpoint_selection='Minimum recorded commissioning score; these nominal conditions are calibration, not held-out robustness validation',
        selected_gradient_step=selected_step,before=before,after=after,
        jacobian_before=jac_before,jacobian_after=_jacobian_report(model,nominal_points),
        wall_seconds=time.perf_counter()-started,log=logs,
        deployment='Direct five-input neural actor; no external physics controller, teacher or action interpolation at runtime')
    metadata['source_sha256']=source_hash
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=source_hash:
        raise RuntimeError('Consolidation source changed during execution')
    (directory/'teacher_consolidation.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    model.save(directory/'sac_model.zip')
    deployed_hash=hashlib.sha256((directory/'sac_model.zip').read_bytes()).hexdigest()
    training_path=directory/'training_metadata.json'
    training=json.loads(training_path.read_text(encoding='utf-8'))
    training['post_training_consolidation']=metadata
    training['consolidation_source_sha256']=source_hash
    training['deployed_model_sha256']=deployed_hash
    training['deployment_origin']='Recorded 1M physics-prior SAC checkpoint, followed by the separately logged offline teacher/Jacobian consolidation stage; actor parameters change after SAC.'
    training_path.write_text(json.dumps(training,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    import argparse
    from stable_baselines3 import SAC
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--steps',type=int,default=12000)
    args=parser.parse_args();root=Path(__file__).resolve().parent
    model=SAC.load(root/'models/sac_model.zip',device='cpu');consolidate(model,root,steps=args.steps)
