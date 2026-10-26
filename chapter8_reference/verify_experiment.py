"""从保存策略和原始数据重新演算，检查五维接口与独立部署证据。"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def hashes(root,names):
    return {str(n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in sorted(names)}


def verify(root=None):
    from stable_baselines3 import SAC
    import torch
    from hydraulic_env import HydraulicGymEnv
    from hydraulic_twin import HydraulicTwinCore
    from edge_exporter import EdgeLightweightExporter
    from evaluation import run_policy,trajectory_metrics,POLICIES
    from test_physics_safety import validate_physics
    from safety_experiment import run_safety_experiment
    root=Path(root or Path(__file__).resolve().parent);out=root/'outputs'
    torch.set_num_threads(1)
    meta=json.loads((root/'models/training_metadata.json').read_text())
    metrics=json.loads((out/'metrics.json').read_text())
    model=SAC.load(root/'models/sac_model.zip',device='cpu')
    actor=EdgeLightweightExporter.load_int8(root/'models/actor_int8.npz')
    with (out/'evaluation.csv').open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    groups={p:[{k:(v if k=='policy' else float(v)) for k,v in row.items()}
               for row in rows if row['policy']==p] for p in POLICIES}
    checks={
        'exactly_five_policy_inputs':model.observation_space.shape==(5,),
        'reference_fixed_standardizer':bool(np.array_equal(HydraulicGymEnv.obs_mean,[0,0,0,10,21]) and
                                           np.array_equal(HydraulicGymEnv.obs_std,[10,50,500,8,2])),
        'one_million_genuine_interactions':int(meta['actual_timesteps'])==1_000_000,
        'model_step_counter_matches_metadata':int(model.num_timesteps)==int(meta['actual_timesteps']),
        'same_physics_as_training':meta['physical_config_sha256']==hashes(root,['physical_config.json'])['physical_config.json'],
        'training_sources_unchanged':all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
                                         for name,digest in meta.get('model_source_sha256',{}).items()) and bool(meta.get('model_source_sha256')),
        'deployment_stage_checkpoint_matches_metadata':meta.get('deployed_model_sha256')==hashes(root,['models/sac_model.zip'])['models/sac_model.zip'],
        'consolidation_source_unchanged':meta.get('consolidation_source_sha256')==hashes(root,['teacher_consolidation.py'])['teacher_consolidation.py'],
    }
    replay_errors={};initials=[]
    for policy,records in groups.items():
        time=np.array([r['time_s'] for r in records]);states=np.array([[r['position_mm']/1000,
            r['velocity_mms']/1000,r['p1_mpa']*1e6,r['p2_mpa']*1e6] for r in records])
        checks[policy+'_601_actual_samples']=len(time)==601
        checks[policy+'_common_clock']=bool(np.allclose(time,np.arange(601)*.001+.4,atol=1e-12,rtol=0))
        checks[policy+'_voltage_bounds']=max(abs(r['control_v']) for r in records)<=10.+1e-6
        fault=time>=.5-1e-12
        checks[policy+'_shared_fault_schedule']=bool(
            np.allclose([r['load_n'] for r in records],np.where(fault,5000.,0.),rtol=0,atol=1e-12)
            and np.allclose([r['beta_mpa'] for r in records],np.where(fault,500.,1000.),rtol=0,atol=1e-9))
        twin=HydraulicTwinCore();twin.reset(states[0]);history=[twin.state.copy()]
        for r in records[:-1]:
            state,_=twin.step(r['control_v'],load_n=r['load_n'],beta_multiplier=r['beta_mpa']*1e6/twin.p['beta_e'])
            history.append(state.copy())
        error=np.max(abs(np.array(history)-states),axis=0)
        replay_errors[policy]=error.tolist();initials.append(states[0])
        checks[policy+'_voltage_replays_physical_states']=bool(np.all(error<[1e-10,1e-9,.01,.01]))
        recomputed=trajectory_metrics(records)
        checks[policy+'_metrics_recomputed']=all((value is None and metrics['policies'][policy][key] is None)
            or (value is not None and np.isclose(value,metrics['policies'][policy][key],rtol=1e-9,atol=1e-10))
            for key,value in recomputed.items())
    checks['identical_initial_states']=bool(np.all(np.array(initials)==initials[0]))
    for policy in ('sac_fp32','edge_int8'):
        reproduced,_,_=run_policy(policy,model=model,edge=actor)
        expected=np.array([r['control_v'] for r in groups[policy]])
        actual=np.array([r['control_v'] for r in reproduced])
        checks[policy+'_policy_reproduces_own_closed_loop']=bool(np.allclose(actual,expected,atol=1e-6,rtol=0))
        measured=metrics['policies'][policy]
        checks[policy+'_prefault_stability']=bool(measured['prefault_velocity_error_rms_mms']<1.
            and measured['prefault_voltage_std_v']<.5 and measured['prefault_max_voltage_step_v']<2.)
        checks[policy+'_fault_recovery_and_final_bias']=bool(measured['recovery_1mms_s'] is not None
            and measured['recovery_1mms_s']<=.1 and abs(measured['final_100ms_velocity_bias_mms'])<.2)
    physics=validate_physics();safety=run_safety_experiment(out)['summary']
    (out/'physics_validation.json').write_text(json.dumps(physics,ensure_ascii=False,indent=2)+'\n')
    checks['independent_radau_physics']=physics['passed']
    checks['sampled_hocbf_intervals_and_independent_flow']=safety['passed']
    with (out/'training.csv').open(newline='') as f:logs=list(csv.DictReader(f))
    steps=np.array([int(x['timesteps']) for x in logs])
    checks['actual_monotone_training_log']=bool(len(steps)>10 and np.all(np.diff(steps)>0) and steps[-1]==meta['actual_timesteps'])
    all_passed=bool(all(checks.values()));checks['all_checks_passed']=all_passed
    sources=[p.name for p in root.glob('*.py')]+[p.name for p in root.glob('*.json') if p.name!='manifest.json']+['native_twin.cpp','requirements.txt']
    artifacts=['models/sac_model.zip','models/actor_int8.npz','models/actor_fp32.npz','models/training_metadata.json',
               'models/teacher_consolidation.json','models/sac_before_consolidation.zip']
    data=['outputs/evaluation.csv','outputs/training.csv','outputs/metrics.json','outputs/safety.csv','outputs/safety_summary.json',
          'outputs/physics_validation.json','outputs/chapter8_comparison.png','outputs/chapter8_comparison.pdf']
    result=dict(checks=checks,raw_voltage_replay_max_state_error_si=replay_errors,
        source_sha256=hashes(root,sources),artifact_sha256=hashes(root,artifacts),data_sha256=hashes(root,data))
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Verification:',checks,flush=True)
    if not all_passed:raise AssertionError('实验验证失败，详见 outputs/validation.json')
    return result


if __name__=='__main__':verify()
