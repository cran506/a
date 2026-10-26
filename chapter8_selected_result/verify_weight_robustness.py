"""对最终五维策略做独立权重小扰动闭环检查；不修改保存的策略。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import torch
from stable_baselines3 import SAC
from evaluation import run_policy,trajectory_metrics


def verify(root=None,trials=5,radius=1e-4):
    root=Path(root or Path(__file__).resolve().parent)
    if trials<1 or not 0.<radius<.01:
        raise ValueError('Expected positive trials and a small positive weight radius')
    torch.set_num_threads(1)
    checkpoint=root/'models/sac_model.zip';rows=[]
    for index in range(trials):
        seed=8200+index;model=SAC.load(checkpoint,device='cpu')
        generator=torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name,parameter in model.actor.named_parameters():
                if name.startswith(('latent_pi','mu')):
                    perturbation=(torch.rand(parameter.shape,generator=generator)*2.-1.)*radius
                    parameter.add_(perturbation)
        trajectory,_,_=run_policy('sac_fp32',model=model)
        metrics=trajectory_metrics(trajectory)
        rows.append(dict(seed=seed,**metrics))
        if metrics['prefault_velocity_error_rms_mms']>.5 or metrics['prefault_max_voltage_step_v']>.5:
            raise AssertionError(f'Perturbed actor develops unloaded oscillation: {rows[-1]}')
    result=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        scope='Independent uniform mean-network parameter perturbations, followed by complete closed-loop re-integration; finite commissioning checks, not a stability proof.',
        perturbation_radius=radius,trials=trials,results=rows)
    (root/'outputs/consolidation_weight_perturbation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=5)
    parser.add_argument('--radius',type=float,default=1e-4)
    args=parser.parse_args();print(json.dumps(verify(trials=args.trials,radius=args.radius),indent=2))
