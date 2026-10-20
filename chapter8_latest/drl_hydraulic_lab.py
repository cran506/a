"""第八章：真实 SAC 训练、HOCBF、安全对比及边缘整数推理。"""
from __future__ import annotations
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter8_mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/chapter8_cache')
try:
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print('\n[环境缺失] 请先安装必需的深度强化学习基础库：')
    print('pip install stable-baselines3[extra] gymnasium\n')
    import sys
    sys.exit(1)

import argparse
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','train','evaluate','report'],default='all')
    parser.add_argument('--timesteps',type=int,default=1_000_000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--reuse-model',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    torch.set_num_threads(1)
    if args.mode in ['all','train'] and not args.reuse_model:
        from train_sac import train
        train(root,args.timesteps,args.seed)
    if args.mode in ['all','evaluate']:
        from evaluation import evaluate
        evaluate(root)
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
