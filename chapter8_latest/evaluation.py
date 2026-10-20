"""Same-plant, same-state fault comparison and real integer-policy deployment."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from hydraulic_env import HydraulicGymEnv
from edge_exporter import EdgeLightweightExporter
from safety_experiment import run_safety_experiment


def run_policy(policy, model=None, edge=None, seed=8108):
    from baseline_pid import Controller
    env=HydraulicGymEnv(domain_randomization=False,fault_augmentation=False,max_steps=1000)
    obs,initial=env.reset(seed=seed,options=dict(evaluation=True,velocity=.05,
        position=.001,fault_time=.2,load_n=5000.,beta_multiplier=.5))
    pid=Controller(initial['initial_voltage_v'])
    nominal_voltage=initial['initial_voltage_v']
    rows=[];observations=[]
    # All controllers start from the same physical cruise state at t=.3s.
    # Keep the settling samples in CSV; the comparison figure shows .4..1s.
    for k in range(701):
        state=env.twin.state.copy();raw=env.raw_observation()
        if policy=='pid':
            u=float(pid.update(float(raw[0]),float(raw[1])))
        elif policy=='nominal_ff':
            u=nominal_voltage
        elif policy=='sac_fp32':
            action,_=model.predict(obs,deterministic=True);u=float(action[0])*10.
        elif policy=='edge_int8':
            u=float(np.asarray(edge.action(obs,'int8')).reshape(-1)[0])*10.
        else:
            raise ValueError(policy)
        load,bm=env.current_fault()
        rows.append(dict(policy=policy,time_s=.3+env.time,position_mm=state[0]*1000,
            position_error_mm=raw[0],velocity_mms=state[1]*1000,
            reference_velocity_mms=env.v_ref*1000,velocity_error_mms=(state[1]-env.v_ref)*1000,
            control_v=u,p1_mpa=state[2]/1e6,p2_mpa=state[3]/1e6,load_n=load,
            beta_mpa=env.twin.p['beta_e']*bm/1e6))
        observations.append(obs.copy())
        if k==700:break
        obs,_,terminated,_,info=env.step(np.array([u/10.],dtype=np.float32))
        if terminated:
            raise RuntimeError(f'Evaluation {policy} failed at {env.time}: {info}')
    env.close()
    return rows,np.array(observations),initial


def trajectory_metrics(rows):
    t=np.array([r['time_s'] for r in rows]);e=np.array([r['velocity_error_mms'] for r in rows])
    u=np.array([r['control_v'] for r in rows]);fault=t>=.5-1e-12;tail=t>=.9-1e-12
    peak_index=np.argmax(np.abs(e[fault]))+np.flatnonzero(fault)[0]
    recovery=None
    # Recovery must persist through the rest of the displayed experiment,
    # with at least 50 ms of evidence; an isolated zero crossing is insufficient.
    for index in range(peak_index,len(t)-49):
        if np.all(np.abs(e[index:])<=1.):
            recovery=float(t[index]-.5);break
    return dict(minimum_velocity_error_mms=float(e[fault].min()),
        maximum_velocity_error_mms=float(e[fault].max()),
        velocity_error_rms_mms=float(np.sqrt(np.mean(e[fault]**2))),
        final_100ms_velocity_bias_mms=float(np.mean(e[tail])),
        recovery_1mms_s=recovery,peak_voltage_v=float(u[fault].max()),
        final_voltage_v=float(np.mean(u[tail])),max_abs_voltage_v=float(np.abs(u).max()),
        max_voltage_step_v=float(np.max(np.abs(np.diff(u)))),
        final_position_error_mm=float(rows[-1]['position_error_mm']))


def evaluate(root):
    root=Path(root);out=root/'outputs';out.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    model=SAC.load(root/'models/sac_model.zip',device='cpu')
    calibration=np.load(root/'models/calibration_observations.npz')
    # Calibration and held-out replay observations are disjoint recorded data.
    cal=calibration['calibration'];validation=calibration['validation']
    fp_rows,fp_obs,initial=run_policy('sac_fp32',model=model)
    # Include one declared commissioning trajectory in activation calibration;
    # it is calibration data, not a held-out claim. Never substitute its outputs
    # for integer deployment: INT8 receives its own evolving observations.
    cal=np.concatenate([cal,fp_obs])
    edge=EdgeLightweightExporter.from_sb3(model,cal,prune_ratio=.70)
    distillation=edge.distill_pruned(cal,steps=8000,batch_size=256)
    # Choose quantization balancing using calibration actions only. Held-out
    # states and the integer closed-loop trajectory do not select the exponent.
    candidates=[]
    for alpha in [.25,.5,.75]:
        candidate=EdgeLightweightExporter(edge.weights,edge.biases,cal,prune_ratio=.70,
            pruned_weights=edge.pruned_weights,pruned_biases=edge.pruned_biases,
            smoothquant_alpha=alpha)
        score=float(np.mean(np.abs(candidate.action(cal,'int8')-candidate.action(cal,'pruned_fp32'))))
        candidates.append((score,alpha,candidate))
    _,selected_alpha,edge=min(candidates,key=lambda item:item[0])
    quantization_selection=dict(selection_metric='calibration normalized-action MAE against fixed pruned FP32 actor',
        calibration_count=len(cal),selected_alpha=selected_alpha,
        candidates=[dict(alpha=alpha,calibration_action_mae=score) for score,alpha,_ in candidates])
    (out/'quantization_calibration_selection.json').write_text(json.dumps(quantization_selection,indent=2)+'\n')
    validation_report=edge.validation_report(validation)
    sizes=edge.save(root/'models')
    restored=EdgeLightweightExporter.load_int8(root/'models/actor_int8.npz')
    if not np.array_equal(restored(validation[:50]),edge.action(validation[:50],'int8')):
        raise AssertionError('Serialized integer actor does not reproduce live integer inference')
    benchmarks=edge.benchmark(fp_obs[0])
    rows=[];metrics={};policy_obs={}
    for policy in ['pid','nominal_ff','sac_fp32','edge_int8']:
        trajectory,observations,_=run_policy(policy,model=model,edge=edge)
        rows.extend(trajectory);metrics[policy]=trajectory_metrics(trajectory);policy_obs[policy]=observations
        print(policy,json.dumps(metrics[policy]),flush=True)
    with (out/'evaluation.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(root/'models/commissioning_observations.npz',fp32=fp_obs,
                        int8=policy_obs['edge_int8'])
    safety=run_safety_experiment(out)['summary']
    training=json.loads((root/'models/training_metadata.json').read_text())
    training.update(reward='-100*e_mm^2-0.1*u_V^2-(u_V-u_prev_V)^2+2*I(|e_mm|<0.2); failure -500',
        optimizer_reward_multiplier=.01,learning_starts=model.learning_starts,
        train_frequency_steps=model.train_freq.frequency,gradient_steps_per_update=model.gradient_steps,
        batch_size=model.batch_size,gamma=model.gamma,learning_rate=float(model.learning_rate),
        physical_config_sha256=hashlib.sha256((root/'physical_config.json').read_bytes()).hexdigest())
    (root/'models/training_metadata.json').write_text(json.dumps(training,ensure_ascii=False,indent=2)+'\n')
    edge_metrics=dict(validation=validation_report,distillation=distillation,
        quantization_selection=quantization_selection,
        artifact_bytes=sizes,benchmark=benchmarks,commissioning_calibration_count=len(fp_obs),
        benchmark_scope='Single-observation NumPy actor kernels only; excludes hydraulic integration, sensing, SB3 Python API and HOCBF. Input affine preprocessing and final tanh included.',
        calibration_not_held_out=True,
        closed_loop_velocity_rms_difference_mms=float(np.sqrt(np.mean((np.array([
            r['velocity_mms'] for r in rows if r['policy']=='sac_fp32'])-
            np.array([r['velocity_mms'] for r in rows if r['policy']=='edge_int8']))**2))))
    displayed={name:np.array([r['velocity_mms'] for r in rows
        if r['policy']==name and r['time_s']>=.4-1e-12]) for name in ['sac_fp32','edge_int8']}
    edge_metrics['displayed_window_velocity_rms_difference_mms']=float(np.sqrt(np.mean(
        (displayed['sac_fp32']-displayed['edge_int8'])**2)))
    edge_metrics['displayed_window_rms_percent_of_50mms']=100.*edge_metrics['displayed_window_velocity_rms_difference_mms']/50.
    result=dict(training=training,policies=metrics,edge=edge_metrics,safety=safety,
        evaluation=dict(start_time_s=.3,display_start_time_s=.4,end_time_s=1.,initial_position_m=.001,
            initial_velocity_m_s=.05,initial_voltage_v=initial['initial_voltage_v'],
            fault_time_s=.5,load_step_n=5000.,bulk_multiplier=.5,
            pid_gains=dict(kp_v_per_mm=1.,ki_v_per_mm_s=4.,kd_v_per_mms=.01,derivative_alpha=.85),
            observations='Ideal sampled position, velocity, acceleration and both chamber pressures; no noise/delay. The acceleration channel senses the force change immediately.',
            nominal_ff='Unloaded equilibrium inverse of the actual nominal orifice/force equations; no load adaptation or integral correction.',
            physical_model='Declared compact-cylinder classroom scaling, not the prior .01 m² identified apparatus.',
            reference_comparison='Reproduces mechanism trends; schematic target peaks/timings are not fitted or injected. No physical hardware/Sim2Real few-shot data.'))
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result
