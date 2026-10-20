"""Cross-check saved models, raw data and reported metrics of the finished run."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from edge_exporter import EdgeLightweightExporter
from evaluation import trajectory_metrics


def verify(root=None,expected_steps=1_000_000):
    root=Path(root or Path(__file__).resolve().parent);out=root/'outputs'
    metrics=json.loads((out/'metrics.json').read_text())
    training=list(csv.DictReader((out/'training.csv').open()))
    rows=list(csv.DictReader((out/'evaluation.csv').open()))
    meta=json.loads((root/'models/training_metadata.json').read_text())
    model=SAC.load(root/'models/sac_model.zip',device='cpu')
    torch.set_num_threads(1)
    with np.load(root/'models/calibration_observations.npz') as cal:
        observations=cal['validation'][:512]
    with np.load(root/'models/actor_fp32.npz') as data:
        weights=[data[f'weight_{k}'] for k in range(3)]
        biases=[data[f'bias_{k}'] for k in range(3)]
    sb3_action=model.predict(observations,deterministic=True)[0]
    numpy_action=EdgeLightweightExporter._float_forward(observations,weights,biases)
    action_error=float(np.max(np.abs(sb3_action-numpy_action)))
    fields=[name for name in rows[0] if name!='policy']
    numeric=np.array([[float(r[name]) for name in fields] for r in rows])
    groups={policy:[{name:float(value) if name!='policy' else value for name,value in row.items()}
        for row in rows if row['policy']==policy] for policy in metrics['policies']}
    ref_times=np.array([r['time_s'] for r in groups['pid']])
    same_grid=all(np.array_equal(ref_times,np.array([r['time_s'] for r in group])) for group in groups.values())
    same_initial=all(np.allclose([g[0][n] for n in ['position_mm','velocity_mms','p1_mpa','p2_mpa']],
                                [groups['pid'][0][n] for n in ['position_mm','velocity_mms','p1_mpa','p2_mpa']],rtol=0,atol=1e-10)
                     for g in groups.values())
    metric_errors=[]
    for policy,group in groups.items():
        recomputed=trajectory_metrics(group)
        for name,value in recomputed.items():
            reported=metrics['policies'][policy][name]
            metric_errors.append(value==reported if value is None else abs(value-reported)<1e-10)
    checks=dict(real_million_steps=int(training[-1]['timesteps'])==meta['actual_timesteps']==model.num_timesteps==expected_steps,
        quantization_exponent_selected_by_calibration=metrics['edge']['quantization_selection']['selected_alpha']==min(
            metrics['edge']['quantization_selection']['candidates'],key=lambda row:row['calibration_action_mae'])['alpha'],
        no_synthetic_entropy=bool(np.allclose([float(r['voltage_entropy'])-float(r['normalized_action_entropy']) for r in training],np.log(10))),
        finite_training_reward=all(np.isfinite(float(r['reward_mean'])) for r in training),
        all_four_policies_present=set(groups)=={'pid','nominal_ff','sac_fp32','edge_int8'},
        complete_shared_time_grid=same_grid and len(ref_times)==701,
        identical_initial_physical_states=same_initial,
        finite_physical_records=bool(np.isfinite(numeric).all()),
        all_actual_inputs_within_limits=all(abs(float(r['control_v']))<=10+1e-10 for r in rows),
        identical_fault_inputs=all(r['load_n']==(5000. if r['time_s']>=.5-1e-12 else 0.) and
            r['beta_mpa']==(500. if r['time_s']>=.5-1e-12 else 1000.) for g in groups.values() for r in g),
        reported_metrics_recomputed_from_raw_csv=all(metric_errors),
        exported_fp32_matches_sb3=action_error<2e-6,
        pruning_at_least_69_percent=metrics['edge']['validation']['actual_weight_sparsity']>=.69,
        independent_sampled_safety_passed=metrics['safety']['passed'],
        independent_physics_validation_present=(out/'physics_validation.json').is_file())
    checks['all_checks_passed']=all(checks.values())
    result=dict(checks=checks,expected_timesteps=expected_steps,
        fp32_export_max_normalized_action_error=action_error,
        model_sha256=hashlib.sha256((root/'models/sac_model.zip').read_bytes()).hexdigest(),
        int8_sha256=hashlib.sha256((root/'models/actor_int8.npz').read_bytes()).hexdigest(),
        scope='Software/numerical consistency, not hardware certification. Performance values are measured, not thresholded to match a reference image.')
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not checks['all_checks_passed']:raise AssertionError(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':verify()
