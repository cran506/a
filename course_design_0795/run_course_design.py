#!/usr/bin/env python3
"""《智能流体传动与控制》课程设计一键入口；默认学号后四位0795。"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/course_design_mpl')
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=795,help='本人学号后四位，0795按整数795处理')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'results')
    parser.add_argument('--mode',choices=['all','identify','simulate','verify','report','plot'],default='all')
    args=parser.parse_args()
    if not 0<=args.seed<=9999:parser.error('seed必须是0—9999的学号后四位')
    root=args.output;root.mkdir(parents=True,exist_ok=True);np.random.seed(args.seed);start=time.perf_counter()
    if args.mode in ['all','identify']:
        from identification import identify
        identified,_=identify(root,args.seed)
    else:
        from physics import Parameters
        identified=Parameters(**json.loads((root/'identification_summary.json').read_text())['identified'])
    if args.mode in ['all','simulate']:
        from experiments import run_scenario
        for scenario in ['nominal','extreme','braking']:run_scenario(root,identified,scenario)
    if args.mode in ['all','verify']:
        from verification import verify
        verify(root,args.seed)
    if args.mode in ['all','plot']:
        from plotting import build_figures
        build_figures(root)
    if args.mode in ['all','report']:
        from report import build_report
        build_report(root)
    metadata=dict(student_id_last_four=f'{args.seed:04d}',seed_integer=args.seed,python=platform.python_version(),
                  mode=args.mode,wall_seconds=time.perf_counter()-start,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
                  scope='course design through chapter7; chapter8 deferred by explicit user request')
    (root/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
    print('课程设计执行完成：',root.resolve(),flush=True)


if __name__=='__main__':main()
