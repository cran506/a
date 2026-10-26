"""数据拟合的 Z-score 与已训练 actor 的等价坐标变换。

百万步历史模型使用固定工程尺度。这里用独立声明的校准行拟合经验
均值/标准差，并折叠 actor 首层，展示真正 Z-score 而不破坏已学策略。
这不是重新训练记录；常量通道不可能同时具有单位方差。
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class EmpiricalZScore:
    mean: np.ndarray
    scale: np.ndarray
    constant_channels: np.ndarray

    @classmethod
    def fit(cls, samples, floor=1e-8):
        values=np.asarray(samples,dtype=np.float64)
        if values.ndim!=2 or len(values)<2 or not np.isfinite(values).all():
            raise ValueError('Calibration must be a finite matrix with at least two rows')
        mean=values.mean(axis=0);std=values.std(axis=0)
        constant=std<=floor
        return cls(mean,np.where(constant,1.,std),constant)

    def transform(self, samples):
        values=np.asarray(samples,dtype=np.float64)
        if values.shape[-1]!=len(self.mean) or not np.isfinite(values).all():
            raise ValueError('Observation dimensions or finite values invalid')
        return (values-self.mean)/self.scale


def fold_actor_first_layer(weights,biases,old_mean,old_scale,normalizer):
    """W'(raw-mu_new)/sigma_new+b' = W(raw-mu_old)/sigma_old+b."""
    w=[np.asarray(a,dtype=np.float64).copy() for a in weights]
    b=[np.asarray(a,dtype=np.float64).copy() for a in biases]
    old_mean=np.asarray(old_mean,dtype=np.float64)
    old_scale=np.asarray(old_scale,dtype=np.float64)
    if old_mean.shape!=normalizer.mean.shape or np.any(old_scale<=0.):
        raise ValueError('Invalid original normalization')
    w[0]=w[0]*(normalizer.scale/old_scale)[None,:]
    b[0]+=np.asarray(weights[0],dtype=np.float64)@((normalizer.mean-old_mean)/old_scale)
    return w,b


def actor_forward(obs,weights,biases):
    x=np.asarray(obs,dtype=np.float64)
    for index,(w,b) in enumerate(zip(weights,biases)):
        x=x@np.asarray(w).T+b
        if index<len(weights)-1:x=np.maximum(x,0.)
    return np.tanh(x)


def run_normalization_diagnostic(root,model=None):
    from stable_baselines3 import SAC
    from edge_exporter import EdgeLightweightExporter
    from hydraulic_env import HydraulicGymEnv
    root=Path(root)
    if model is None:model=SAC.load(root/'models/sac_model.zip',device='cpu')
    data=np.load(root/'models/calibration_observations.npz')
    cal=np.asarray(data['calibration'],dtype=np.float64)
    validation=np.asarray(data['validation'],dtype=np.float64)
    mean=np.asarray(HydraulicGymEnv.obs_mean,dtype=np.float64)
    # Reconstruct the denominator actually evaluated by the float32 defaults.
    scale=np.asarray(HydraulicGymEnv.obs_std+np.float32(1e-8),dtype=np.float64)
    raw_cal=cal*scale+mean;raw_validation=validation*scale+mean
    normalizer=EmpiricalZScore.fit(raw_cal)
    w,b=EdgeLightweightExporter.extract_actor(model.actor)
    fw,fb=fold_actor_first_layer(w,b,mean,scale,normalizer)
    transformed=normalizer.transform(raw_cal)
    expected=actor_forward(validation,w,b)
    actual=actor_forward(normalizer.transform(raw_validation),fw,fb)
    error=float(np.max(abs(actual-expected)))
    varying=~normalizer.constant_channels
    checks=dict(calibration_mean_zero=bool(np.max(abs(transformed.mean(axis=0)))<1e-8),
                varying_calibration_variance_one=bool(np.max(abs(transformed[:,varying].std(axis=0)-1.))<1e-8),
                held_out_actor_coordinate_equivalence=error<1e-8)
    arrays=dict(mean=normalizer.mean,scale=normalizer.scale,constant_channels=normalizer.constant_channels)
    arrays.update({f'W{i}':a for i,a in enumerate(fw)})
    arrays.update({f'b{i}':a for i,a in enumerate(fb)})
    np.savez_compressed(root/'models/actor_empirical_zscore.npz',**arrays)
    report=dict(scope='Post-training equivalent coordinate transform; original million-step training used fixed engineering scales.',
        source='Recorded training-replay observations inverted through the declared original affine normalization.',
        calibration_rows=len(cal),held_out_rows=len(validation),observation_names=HydraulicGymEnv.observation_names,
        empirical_mean=normalizer.mean.tolist(),empirical_scale=normalizer.scale.tolist(),
        constant_channels=normalizer.constant_channels.tolist(),
        max_held_out_normalized_action_difference=error,checks=checks,passed=all(checks.values()))
    (root/'outputs/normalization_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if not report['passed']:raise RuntimeError('Z-score coordinate equivalence failed')
    return report


if __name__=='__main__':
    result=run_normalization_diagnostic(Path(__file__).resolve().parent)
    print(json.dumps(result,ensure_ascii=False,indent=2))
