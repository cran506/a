"""独立重放保存的动作，核验模型来源、SAC权重和四组各自闭环。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import drl_hydraulic_lab as lab


def verify(root):
    root=Path(root);source=Path(lab.__file__)
    meta=json.loads((root/'training_metadata.json').read_text())
    train=lab.read_csv(root/'training.csv');rows=lab.read_csv(root/'evaluation.csv')
    checks={
        'source_sha256_matches_training':meta['source_sha256']==hashlib.sha256(source.read_bytes()).hexdigest(),
        'actual_steps_match_final_log':int(train[-1]['steps'])==meta['actual_steps'],
        'training_clock_strictly_increases':bool(np.all(np.diff([int(r['steps']) for r in train])>0)),
        'newly_initialized_model_provenance':'fresh random initialization' in meta['origin'],
    }
    errors={};performance={}
    with np.load(root/'actor_fp32.npz') as data:fp32=lab.FloatActor(data)
    with np.load(root/'actor_int8.npz') as data:edge=lab.Int8Actor(data)
    from stable_baselines3 import SAC
    import torch
    torch.set_num_threads(1)
    model=SAC.load(root/'sac.zip',device='cpu')
    cal=np.load(root/'calibration_observations.npy')
    prediction,_=model.predict(cal,deterministic=True)
    mean_error=float(np.max(abs(prediction-fp32(cal))))
    checks['numpy_actor_matches_new_sac_checkpoint']=mean_error<2e-5
    checks['sb3_step_counter_matches_metadata']=model.num_timesteps==meta['actual_steps']
    for policy in ['PID','nominal_feedforward','SAC_FP32','pruned_INT8']:
        group=[r for r in rows if r['policy']==policy]
        plant=lab.HydraulicTwin();initial,bias=plant.equilibrium();plant.reset(initial)
        error=np.zeros(4);proposal_error=0.;previous=bias
        for i,row in enumerate(group):
            actual=np.array([float(row[key]) for key in ['x_m','v_m_s','p1_pa','p2_pa']])
            error=np.maximum(error,abs(actual-plant.state))
            time=float(row['time_s']);load=float(row['load_n']);beta=float(row['beta_ratio'])
            obs=lab.observation(plant,time,u=previous,load=load,beta_ratio=beta)
            if policy in ['SAC_FP32','pruned_INT8']:
                actor=fp32 if policy=='SAC_FP32' else edge
                proposal_error=max(proposal_error,abs(float(actor(obs).item())*10-float(row['proposed_v'])))
            if i<len(group)-1:
                voltage=float(row['applied_v']);plant.step(voltage,load=load,beta_ratio=beta);previous=voltage
        errors[policy]=dict(max_state_error_si=error.tolist(),own_actor_proposal_error_v=proposal_error)
        checks[policy+'_saved_voltage_replays_own_states']=bool(np.all(error<=np.array([1e-10,1e-9,.05,.05])))
        if policy in ['SAC_FP32','pruned_INT8']:checks[policy+'_uses_own_policy']=proposal_error<2e-4
        checks[policy+'_complete_common_clock']=len(group)==1001 and all(abs(float(r['time_s'])-i*.001)<1e-10 for i,r in enumerate(group))
        post=[r for r in group if float(r['time_s'])>=.5]
        speed=np.array([float(r['velocity_error_mms']) for r in post])
        performance[policy]=dict(min_speed_error_mms=float(speed.min()),post_fault_rmse_mms=float(np.sqrt(np.mean(speed**2))))
    summary=dict(checks=checks,all_method_checks_pass=all(checks.values()),
                 maximum_numpy_vs_torch_action_error=mean_error,replay=errors,
                 recomputed_performance=performance,
                 performance_scope='method checks do not imply convergence or textbook performance')
    lab.save_json(root/'independent_verification.json',summary)
    if not summary['all_method_checks_pass']:raise AssertionError(json.dumps(summary,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    args=parser.parse_args();verify(args.run_dir)
