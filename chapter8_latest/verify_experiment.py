"""Cross-check saved models, raw data and reported metrics of the finished run."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


PHYSICS_SOURCE_FILES = (
    'physical_config.json', 'hydraulic_twin.py', 'native_twin.cpp',
    'safety_filter.py', 'safety_experiment.py', 'test_physics_safety.py')
RUNTIME_SOURCE_FILES = PHYSICS_SOURCE_FILES + (
    'hydraulic_env.py', 'train_sac.py', 'evaluation.py', 'baseline_pid.py',
    'edge_exporter.py', 'edge_inference.py', 'drl_hydraulic_lab.py',
    'observation_normalization.py', 'verify_experiment.py')


def _file_hashes(root, names):
    root=Path(root)
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def physics_source_files(root):
    """Bind an explicit safety-demonstration configuration when one is used."""
    return PHYSICS_SOURCE_FILES + (('safety_config.json',)
        if (Path(root)/'safety_config.json').is_file() else ())


def _passed_evidence(document):
    """Require explicit boolean results, not a truthy string or an empty table."""
    return (isinstance(document,dict) and document.get('passed') is True and
            isinstance(document.get('checks'),dict) and bool(document['checks']) and
            all(value is True for value in document['checks'].values()))


def _load_evidence(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError) as error:
        return None, str(error)
    return value, None


def independent_evidence_checks(root, metrics):
    """Reject failed, missing, mismatched or stale numerical safety evidence.

    The independent validation writer binds its results to the model/config
    sources and to the exact safety CSV/summary it just produced. A leftover
    file alone is therefore insufficient to mark the current run verified.
    """
    root=Path(root);out=root/'outputs'
    physics,physics_error=_load_evidence(out/'physics_validation.json')
    safety,safety_error=_load_evidence(out/'safety_summary.json')
    normalization,normalization_error=_load_evidence(out/'normalization_validation.json')
    try:
        current_sources=_file_hashes(root,physics_source_files(root))
        current_safety=_file_hashes(root,('outputs/safety.csv','outputs/safety_summary.json'))
    except OSError as error:
        current_sources=current_safety=None
        source_error=str(error)
    else:
        source_error=None
    physics_dict=physics if isinstance(physics,dict) else {}
    safety_dict=safety if isinstance(safety,dict) else {}
    normalization_dict=normalization if isinstance(normalization,dict) else {}
    coordinate_error=normalization_dict.get('max_held_out_normalized_action_difference')
    coordinate_error_valid=(isinstance(coordinate_error,(int,float)) and
        not isinstance(coordinate_error,bool) and np.isfinite(coordinate_error) and
        0.<=coordinate_error<1e-8)
    recorded_safety=metrics.get('safety') if isinstance(metrics,dict) else None
    checks=dict(
        independent_physics_validation_passed=_passed_evidence(physics),
        independent_sampled_safety_passed=_passed_evidence(safety),
        independent_random_safety_checks_passed=_passed_evidence(
            safety_dict.get('independent_random_validation')),
        safety_summary_matches_evaluation_metrics=(
            isinstance(recorded_safety,dict) and recorded_safety==safety_dict),
        independent_validation_uses_current_sources=(current_sources is not None and
            physics_dict.get('source_sha256')==current_sources),
        independent_validation_binds_current_safety_records=(current_safety is not None and
            physics_dict.get('safety_evidence_sha256')==current_safety),
        empirical_normalization_validation_passed=_passed_evidence(normalization),
        empirical_normalization_coordinate_error_below_1e_minus8=bool(coordinate_error_valid))
    return checks,dict(physics_read_error=physics_error,safety_read_error=safety_error,
        normalization_read_error=normalization_error,
        source_read_error=source_error,
        current_physics_source_sha256=current_sources,
        current_safety_evidence_sha256=current_safety)


def verify(root=None,expected_steps=1_000_000):
    import torch
    from stable_baselines3 import SAC
    from edge_exporter import EdgeLightweightExporter
    from evaluation import trajectory_metrics
    from observation_normalization import actor_forward
    from hydraulic_env import HydraulicGymEnv
    root=Path(root or Path(__file__).resolve().parent);out=root/'outputs'
    metrics=json.loads((out/'metrics.json').read_text())
    training=list(csv.DictReader((out/'training.csv').open()))
    rows=list(csv.DictReader((out/'evaluation.csv').open()))
    meta=json.loads((root/'models/training_metadata.json').read_text())
    model=SAC.load(root/'models/sac_model.zip',device='cpu')
    torch.set_num_threads(1)
    with np.load(root/'models/calibration_observations.npz') as cal:
        observations=cal['validation'][:512]
        normalization_calibration=np.asarray(cal['calibration'],dtype=np.float64)
        normalization_validation=np.asarray(cal['validation'],dtype=np.float64)
    with np.load(root/'models/actor_fp32.npz') as data:
        weights=[data[f'weight_{k}'] for k in range(3)]
        biases=[data[f'bias_{k}'] for k in range(3)]
    sb3_action=model.predict(observations,deterministic=True)[0]
    numpy_action=EdgeLightweightExporter._float_forward(observations,weights,biases)
    action_error=float(np.max(np.abs(sb3_action-numpy_action)))
    # Independently check the saved coordinate-transformed artifact against
    # this loaded checkpoint, rather than trusting a cached diagnostic scalar.
    normalizer_error=None
    normalization_artifact_error=float('inf')
    normalization_stats_match=False
    try:
        with np.load(root/'models/actor_empirical_zscore.npz') as saved:
            new_mean=saved['mean'];new_scale=saved['scale']
            constant_channels=saved['constant_channels']
            folded_w=[saved[f'W{k}'] for k in range(3)]
            folded_b=[saved[f'b{k}'] for k in range(3)]
        old_mean=np.asarray(HydraulicGymEnv.obs_mean,dtype=np.float64)
        old_scale=np.asarray(HydraulicGymEnv.obs_std+np.float32(1e-8),dtype=np.float64)
        raw_cal=normalization_calibration*old_scale+old_mean
        std=raw_cal.std(axis=0);constant=std<=1e-8
        normalization_stats_match=(
            np.allclose(new_mean,raw_cal.mean(axis=0),rtol=0.,atol=1e-10) and
            np.allclose(new_scale,np.where(constant,1.,std),rtol=0.,atol=1e-10) and
            np.array_equal(constant_channels,constant))
        checkpoint_w,checkpoint_b=EdgeLightweightExporter.extract_actor(model.actor)
        raw_validation=normalization_validation*old_scale+old_mean
        expected=actor_forward(normalization_validation,checkpoint_w,checkpoint_b)
        actual=actor_forward((raw_validation-new_mean)/new_scale,folded_w,folded_b)
        normalization_artifact_error=float(np.max(np.abs(actual-expected)))
    except (OSError,KeyError,ValueError) as error:
        normalizer_error=str(error)
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
        empirical_normalization_statistics_match_recorded_calibration=bool(normalization_stats_match),
        empirical_normalization_artifact_matches_current_checkpoint=bool(
            np.isfinite(normalization_artifact_error) and normalization_artifact_error<1e-8),
        empirical_normalization_uses_declared_training_coordinates=(
            np.array_equal(HydraulicGymEnv.obs_mean,np.asarray(meta['obs_mean'],np.float32)) and
            np.array_equal(HydraulicGymEnv.obs_std,np.asarray(meta['obs_std'],np.float32))),
        pruning_at_least_69_percent=metrics['edge']['validation']['actual_weight_sparsity']>=.69)
    evidence_checks,evidence_details=independent_evidence_checks(root,metrics)
    checks.update(evidence_checks)
    extra_sources=tuple(name for name in physics_source_files(root)
                        if name not in RUNTIME_SOURCE_FILES)
    source_hashes=_file_hashes(root,RUNTIME_SOURCE_FILES+extra_sources)
    checks['declared_training_config_matches_current_config']=(
        meta.get('physical_config_sha256')==source_hashes['physical_config.json'])
    checks['all_checks_passed']=all(checks.values())
    artifacts=_file_hashes(root,('models/sac_model.zip','models/sac_checkpoint.zip',
        'models/actor_fp32.npz','models/actor_pruned_fp32.npz','models/actor_int8.npz',
        'models/actor_empirical_zscore.npz','models/calibration_observations.npz',
        'models/training_metadata.json'))
    result=dict(checks=checks,expected_timesteps=expected_steps,
        fp32_export_max_normalized_action_error=action_error,
        empirical_zscore_artifact_max_action_error=(normalization_artifact_error
            if np.isfinite(normalization_artifact_error) else None),
        empirical_zscore_artifact_read_error=normalizer_error,
        model_sha256=artifacts['models/sac_model.zip'],
        int8_sha256=artifacts['models/actor_int8.npz'],
        physical_config_sha256=source_hashes['physical_config.json'],
        source_sha256=source_hashes,artifact_sha256=artifacts,
        data_sha256=_file_hashes(root,('outputs/training.csv','outputs/evaluation.csv',
            'outputs/metrics.json','outputs/physics_validation.json',
            'outputs/safety.csv','outputs/safety_summary.json',
            'outputs/normalization_validation.json')),
        independent_evidence=evidence_details,
        scope='Software/numerical consistency, not hardware certification. Performance values are measured, not thresholded to match a reference image.')
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if not checks['all_checks_passed']:raise AssertionError(result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':verify()
