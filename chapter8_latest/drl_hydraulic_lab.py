"""第八章：真实 SAC 训练、HOCBF、安全对比及边缘整数推理。"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter8_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/chapter8_cache')
import argparse
from pathlib import Path


def require_learning_runtime():
    try:
        import torch
        import gymnasium
        import stable_baselines3
    except ImportError as error:
        raise SystemExit('缺少强化学习依赖，请按 README 安装 CPU PyTorch 和 requirements.txt。') from error
    torch.set_num_threads(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','train','evaluate','report','safety','normalize'],default='all')
    parser.add_argument('--timesteps',type=int,default=1_000_000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--reuse-model',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    if args.mode in ['all','train','evaluate','normalize']:
        require_learning_runtime()
    if args.mode in ['all','train'] and not args.reuse_model:
        from train_sac import train
        train(root,args.timesteps,args.seed)
    if args.mode in ['all','evaluate']:
        from evaluation import evaluate
        evaluate(root)
    if args.mode in ['all','evaluate','train','normalize']:
        from observation_normalization import run_normalization_diagnostic
        run_normalization_diagnostic(root)
    if args.mode=='safety':
        from safety_experiment import run_safety_experiment
        result=run_safety_experiment(root/'outputs')
        print(result['summary'])
    if args.mode=='all':
        from test_physics_safety import write_independent_validation
        write_independent_validation(root)
    if args.mode in ['all','report']:
        from reporting import build_report
        build_report(root)
    if args.mode=='all':
        import json
        from verify_experiment import verify
        from package_delivery import package
        steps=json.loads((root/'models/training_metadata.json').read_text())['actual_timesteps']
        verify(root,expected_steps=steps)
        package(root)

if __name__=='__main__':main()
