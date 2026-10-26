"""五维参考接口下的共同对象故障对比与真实整数 actor 部署。

从全局 .4 s 的同一个 50 mm/s 巡航平衡点开始，.5 s 同时施加
5000 N 与体积模量倍率 .5。四组均重新积分各自真实闭环，只有控制律
不同；不以 FP32 输出替代整数策略，也不对速度/电压作事后平滑。
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from hydraulic_env import HydraulicGymEnv
from edge_exporter import EdgeLightweightExporter
from safety_experiment import run_safety_experiment


POLICIES=('pid','nominal_ff','sac_fp32','edge_int8')
START_TIME=.4
END_TIME=1.
FAULT_TIME=.5
INITIAL_POSITION=.0001
REFERENCE_VELOCITY=.05


def run_policy(policy,model=None,edge=None,seed=8108,*,start_time_s=START_TIME,
               end_time_s=END_TIME,fault_time_s=FAULT_TIME,
               initial_position_m=INITIAL_POSITION,
               reference_velocity_m_s=REFERENCE_VELOCITY):
    """Return raw rows, five-dimensional network observations and reset info."""
    from baseline_pid import Controller
    if policy not in POLICIES:raise ValueError('Unknown policy: '+str(policy))
    if not start_time_s<fault_time_s<end_time_s:
        raise ValueError('Expected start < fault < end')
    dt=.001;n_steps=int(round((end_time_s-start_time_s)/dt))
    if not np.isclose(n_steps*dt,end_time_s-start_time_s,rtol=0.,atol=1e-12):
        raise ValueError('Evaluation duration must be a multiple of 1 ms')
    env=HydraulicGymEnv(domain_randomization=False,fault_augmentation=False,
                        max_steps=n_steps+1)
    obs,initial=env.reset(seed=seed,options=dict(evaluation=True,
        velocity=reference_velocity_m_s,position=initial_position_m,
        fault_time=fault_time_s-start_time_s,load_n=5000.,beta_multiplier=.5))
    pid=Controller(initial['initial_voltage_v'])
    nominal_voltage=float(initial['initial_voltage_v'])
    rows=[];observations=[]
    try:
        for k in range(n_steps+1):
            state=env.twin.state.copy();raw=env.raw_observation()
            if np.asarray(obs).shape!=(5,):
                raise RuntimeError('Reference evaluation requires exactly five observations')
            if policy=='pid':
                # The user-provided Gym coordinate uses actual-reference.
                voltage=pid.update(-float(raw[0]),-float(raw[1]))
            elif policy=='nominal_ff':
                voltage=nominal_voltage
            elif policy=='sac_fp32':
                if model is None:raise ValueError('SAC model is required')
                action,_=model.predict(obs,deterministic=True)
                voltage=float(np.clip(np.asarray(action).reshape(-1)[0],-1.,1.))*10.
            else:
                if edge is None:raise ValueError('Integer actor is required')
                action=edge.action(obs,'int8') if hasattr(edge,'action') else edge(obs)
                voltage=float(np.clip(np.asarray(action).reshape(-1)[0],-1.,1.))*10.
            # Record the exact float32 Gym action that actually reaches the
            # plant, so an independent raw-voltage replay is bit-consistent.
            applied_action=np.array([voltage/10.],dtype=np.float32)
            voltage=float(np.clip(applied_action[0],-1.,1.))*10.
            load,beta=env.current_fault()
            rows.append(dict(policy=policy,time_s=start_time_s+env.time,
                position_mm=float(state[0]*1000.),position_error_mm=float(raw[0]),
                velocity_mms=float(state[1]*1000.),
                reference_velocity_mms=float(env.v_ref*1000.),
                velocity_error_mms=float((state[1]-env.v_ref)*1000.),
                acceleration_error_mms2=float(raw[2]),load_pressure_mpa=float(raw[3]),
                control_v=float(voltage),previous_control_v=float(env.last_u),
                p1_mpa=float(state[2]/1e6),p2_mpa=float(state[3]/1e6),
                load_n=float(load),beta_mpa=float(env.twin.p['beta_e']*beta/1e6)))
            observations.append(obs.copy())
            if k==n_steps:break
            obs,_,terminated,_,info=env.step(applied_action)
            if terminated:
                raise RuntimeError(f'Evaluation {policy} failed at {env.time}: {info}')
    finally:
        env.close()
    return rows,np.asarray(observations,dtype=np.float32),initial


def trajectory_metrics(rows,fault_time_s=FAULT_TIME):
    time=np.array([r['time_s'] for r in rows]);error=np.array([r['velocity_error_mms'] for r in rows])
    voltage=np.array([r['control_v'] for r in rows])
    if len(time)<2 or not np.isfinite(np.c_[time,error,voltage]).all() or np.any(np.diff(time)<=0.):
        raise ValueError('Trajectory must have finite, strictly increasing samples')
    post=time>=fault_time_s-1e-12;pre=~post;tail=time>=time[-1]-.1-1e-12
    if not post.any():raise ValueError('Trajectory does not include the fault')
    start=int(np.flatnonzero(post)[0]);peak_index=start+int(np.argmax(abs(error[post])))
    recovery=None
    for index in range(peak_index,len(time)):
        if time[-1]-time[index]<.05-1e-12:break
        if np.all(abs(error[index:])<=1.):
            recovery=float(time[index]-fault_time_s);break
    minimum_index=start+int(np.argmin(error[post]))
    return dict(minimum_velocity_error_mms=float(error[post].min()),
        minimum_velocity_error_time_s=float(time[minimum_index]),
        maximum_velocity_error_mms=float(error[post].max()),
        velocity_error_rms_mms=float(np.sqrt(np.mean(error[post]**2))),
        final_100ms_velocity_bias_mms=float(np.mean(error[tail])),
        recovery_1mms_s=recovery,peak_voltage_v=float(voltage[post].max()),
        final_voltage_v=float(np.mean(voltage[tail])),max_abs_voltage_v=float(abs(voltage).max()),
        max_voltage_step_v=float(np.max(abs(np.diff(voltage)))),
        prefault_velocity_error_rms_mms=float(np.sqrt(np.mean(error[pre]**2))) if pre.any() else None,
        prefault_voltage_std_v=float(np.std(voltage[pre])) if pre.any() else None,
        prefault_max_voltage_step_v=float(np.max(abs(np.diff(voltage[pre])))) if pre.sum()>1 else None,
        final_position_error_mm=float(rows[-1]['position_error_mm']))


def evaluate(root):
    root=Path(root);out=root/'outputs';out.mkdir(exist_ok=True)
    training=json.loads((root/'models/training_metadata.json').read_text(encoding='utf-8'))
    current_physics=hashlib.sha256((root/'physical_config.json').read_bytes()).hexdigest()
    if training['physical_config_sha256']!=current_physics:
        raise ValueError('物理配置在训练后已改变；请重新训练，不能重写训练来源哈希')
    torch.set_num_threads(1)
    model=SAC.load(root/'models/sac_model.zip',device='cpu')
    if model.observation_space.shape!=(5,):
        raise ValueError('A newly trained five-input SAC checkpoint is required; ten-input models are incompatible')
    with np.load(root/'models/calibration_observations.npz') as recorded:
        cal=np.asarray(recorded['calibration'],dtype=np.float32)
        validation=np.asarray(recorded['validation'],dtype=np.float32)
    if cal.ndim!=2 or cal.shape[1]!=5 or validation.shape[1]!=5:
        raise ValueError('Calibration and held-out observations must use the five-input model')
    fp_rows,fp_obs,initial=run_policy('sac_fp32',model=model)
    # One declared commissioning trajectory joins calibration. It is not a
    # held-out Sim2Real experiment; integer deployment still uses its own states.
    cal=np.concatenate([cal,fp_obs])
    # Export calibration anchors do not modify the SAC checkpoint. They are
    # labelled regression inputs, rather than additional physical interactions.
    rng=np.random.default_rng(2028)
    def local_anchors(count,error_std,velocity_std,accel_std):
        raw=np.column_stack([rng.normal(0.,error_std,count),rng.normal(0.,velocity_std,count),
            rng.normal(0.,accel_std,count),rng.uniform(2.4,5.8,count),np.full(count,21.)])
        return ((raw-HydraulicGymEnv.obs_mean)/HydraulicGymEnv.obs_std).astype(np.float32)
    local=local_anchors(4000,.1,2.,1000.)
    fine=local_anchors(4000,.015,.25,100.)
    cal=np.concatenate([cal,local,fine])
    edge=EdgeLightweightExporter.from_sb3(model,cal,prune_ratio=.70)
    initial_distillation=edge.distill_pruned(cal,steps=8000,batch_size=256)
    polishing=edge.polish_pruned(cal,steps=15000,batch_size=256,
                                learning_rate=3e-4,min_learning_rate=1e-5,seed=2027)
    from edge_consolidation import consolidate_edge
    jacobian_refinement=consolidate_edge(edge,cal,steps=8000)
    distillation=dict(method='fixed-mask actor distillation, cosine Adam polishing and local input-Jacobian preservation',
        steps=initial_distillation['steps']+polishing['steps']+jacobian_refinement['steps'],
        initial_action_mse=initial_distillation['initial_action_mse'],
        final_action_mse=jacobian_refinement['final_action_mse'],
        stages=[initial_distillation,polishing,jacobian_refinement])
    candidates=[]
    for alpha in (.25,.5,.75):
        candidate=EdgeLightweightExporter(edge.weights,edge.biases,cal,prune_ratio=.70,
            pruned_weights=edge.pruned_weights,pruned_biases=edge.pruned_biases,
            smoothquant_alpha=alpha,quantization_mode='per_channel',
            equalize_hidden=True,final_int32=True,activation_bits=16)
        score=float(np.mean(abs(candidate.action(cal,'int8')-candidate.action(cal,'pruned_fp32'))))
        candidates.append((score,alpha,candidate))
    _,selected_alpha,edge=min(candidates,key=lambda item:item[0])
    selection=dict(selection_metric='calibration normalized-action MAE against fixed pruned FP32 actor',
        calibration_count=len(cal),selected_alpha=selected_alpha,
        weight_quantization='per-output-channel symmetric INT8',
        activation_precision='INT16; W8/A8 calibration produced an unloaded limit cycle, so only activation storage/precision is increased',
        hidden_equalization='positive diagonal ReLU cross-layer equalization, float-function preserving',
        final_logits='INT32 dot-product output, dequantized only at final tanh',
        candidates=[dict(alpha=alpha,calibration_action_mae=score) for score,alpha,_ in candidates])
    (out/'quantization_calibration_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    validation_report=edge.validation_report(validation);sizes=edge.save(root/'models')
    standalone=EdgeLightweightExporter.load_int8(root/'models/actor_int8.npz')
    if not np.array_equal(standalone(validation[:50]),edge.action(validation[:50],'int8')):
        raise AssertionError('Serialized integer actor differs from live integer inference')
    benchmarks=edge.benchmark(fp_obs[0])
    rows=[];metrics={};policy_obs={}
    for policy in POLICIES:
        trajectory,observations,_=(fp_rows,fp_obs,initial) if policy=='sac_fp32' else run_policy(policy,model=model,edge=edge)
        rows.extend(trajectory);metrics[policy]=trajectory_metrics(trajectory);policy_obs[policy]=observations
        print(policy,json.dumps(metrics[policy]),flush=True)
    with (out/'evaluation.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    np.savez_compressed(root/'models/commissioning_observations.npz',fp32=fp_obs,
                        int8=policy_obs['edge_int8'])
    safety=run_safety_experiment(out)['summary']
    training=dict(training,learning_starts=int(model.learning_starts),
        train_frequency_vector_steps=int(model.train_freq.frequency),
        gradient_steps_per_update=int(model.gradient_steps),batch_size=int(model.batch_size),
        gamma=float(model.gamma),learning_rate=float(model.learning_rate))
    speed_fp=np.array([r['velocity_mms'] for r in fp_rows])
    speed_q=np.array([r['velocity_mms'] for r in rows if r['policy']=='edge_int8'])
    difference=float(np.sqrt(np.mean((speed_fp-speed_q)**2)))
    edge_metrics=dict(validation=validation_report,distillation=distillation,
        quantization_selection=selection,artifact_bytes=sizes,benchmark=benchmarks,
        commissioning_calibration_count=len(fp_obs),calibration_not_held_out=True,
        synthetic_unloaded_anchor_count=len(local)+len(fine),
        benchmark_scope='Single-observation NumPy actor kernels, including input affine preprocessing and tanh; excludes ODE, sensing, SB3 API and HOCBF.',
        closed_loop_velocity_rms_difference_mms=difference,
        displayed_window_velocity_rms_difference_mms=difference,
        displayed_window_rms_percent_of_50mms=100.*difference/50.)
    result=dict(training=training,policies=metrics,edge=edge_metrics,safety=safety,
        evaluation=dict(start_time_s=START_TIME,display_start_time_s=START_TIME,end_time_s=END_TIME,
            sample_count_per_policy=601,initial_position_m=INITIAL_POSITION,
            initial_velocity_m_s=REFERENCE_VELOCITY,initial_voltage_v=initial['initial_voltage_v'],
            fault_time_s=FAULT_TIME,load_step_n=5000.,bulk_multiplier=.5,
            pid_gains=dict(kp_v_per_mm=1.,ki_v_per_mm_s=40.,kd_v_per_mms=.01,derivative_alpha=.85),
            observations='Five reference features [actual-reference position, actual-reference velocity, analytic acceleration error, P1-(A2/A1)*P2, Ps]; units mm,mm/s,mm/s²,MPa,MPa. No future fault label.',
            position_error_definition='actual minus reference',
            nominal_ff='Unloaded equilibrium inverse held constant; no oracle load signal, pressure gain scheduling or integral correction.',
            physical_model='Four internal physical states, non-linear orifice flow and force/continuity equations; teaching parameters explicitly declared.',
            reference_comparison='This checkpoint is trained from scratch by ordinary SAC with no teacher initialization, auxiliary actor prior, or post-training FP32 modification. Compression distillation uses the same final FP32 actor as its target. No response curves or training rewards are fitted to schematic values.'))
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':evaluate(Path(__file__).resolve().parent)
