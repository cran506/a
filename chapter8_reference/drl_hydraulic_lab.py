"""第八章：参考五维接口、真实 SAC、采样 HOCBF 和独立 INT8 推理。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/chapter8_reference_mpl')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['all','train','consolidate','evaluate','plot','safety','verify','package'],default='all')
    parser.add_argument('--timesteps',type=int,default=1_000_000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--reuse-model',action='store_true',help='使用本目录已训练的五输入策略')
    args=parser.parse_args();root=Path(__file__).resolve().parent
    if args.mode in ('all','train') and not args.reuse_model:
        from train_sac import train
        model,_=train(root,timesteps=args.timesteps,seed=args.seed)
        from teacher_consolidation import consolidate
        consolidate(model,root)
    if args.mode=='consolidate':
        from stable_baselines3 import SAC
        from teacher_consolidation import consolidate
        consolidate(SAC.load(root/'models/sac_model.zip',device='cpu'),root)
    if args.mode in ('all','evaluate'):
        from evaluation import evaluate
        evaluate(root)
    if args.mode=='safety':
        from safety_experiment import run_safety_experiment
        print(json.dumps(run_safety_experiment(root/'outputs')['summary'],ensure_ascii=False,indent=2))
    if args.mode in ('all','evaluate','plot'):
        from reporting import generate_figures
        generate_figures(root)
    if args.mode in ('all','verify'):
        from verify_experiment import verify
        verify(root)
    if args.mode in ('all','package'):
        from package_delivery import package
        package(root)


if __name__=='__main__':main()
