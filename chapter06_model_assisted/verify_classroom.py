"""复核课堂约化模型的数值精度和加载因果性，保存 numerical_checks.json。

从已保存的 summary.json 和 full.csv 读取基准，而不是以复跑结果替换基准。
严格积分、改变控制采样率、无载和移动加载时刻分别检验不同问题。
采样减半改变数字控制器，不等同于只收紧ODE求解容差。
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from classroom_model import ClassroomParameters, run_classroom


ROOT = Path(__file__).resolve().parent


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory)
    summary_path, csv_path = directory/'summary.json', directory/'full.csv'
    baseline = json.loads(summary_path.read_text(encoding='utf-8'))
    saved = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=float, encoding='utf-8')
    config = baseline['full']['config']
    parameters = ClassroomParameters(**config['classroom_parameters'])
    ts = float(config['ts'])
    checks = []

    def record(name, passed, **evidence):
        checks.append(dict(name=name, passed=bool(passed), **evidence))

    saved_error_mm = float(np.max(abs(saved['error_m']))*1000)
    baseline_error_mm = float(baseline['full']['metrics']['max_abs_error_mm'])
    record('saved_csv_matches_summary', abs(saved_error_mm-baseline_error_mm)<1e-10,
           csv_max_abs_error_mm=saved_error_mm, summary_max_abs_error_mm=baseline_error_mm)
    record('saved_error_is_reference_minus_position',
           np.allclose(saved['error_m'], saved['reference_m']-saved['position_m'],rtol=0.,atol=1e-15),
           max_identity_residual_m=float(np.max(abs(saved['error_m']-(saved['reference_m']-saved['position_m'])))))

    strict = run_classroom(ts=ts, parameters=parameters, rtol=1e-10)
    strict_evidence = {}
    for name in ('pid','nominal','full'):
        original = baseline[name]['metrics']
        rerun = strict[name]['metrics']
        strict_evidence[name] = dict(
            baseline_max_abs_error_mm=float(original['max_abs_error_mm']),
            strict_max_abs_error_mm=float(rerun['max_abs_error_mm']),
            max_error_metric_difference_mm=abs(original['max_abs_error_mm']-rerun['max_abs_error_mm']),
            contact_min_velocity_difference_m_s=abs(original['contact_min_velocity_m_s']-rerun['contact_min_velocity_m_s']))
    strict_full = strict['full']['data']
    time_equal = np.array_equal(saved['time_s'], strict_full['time_s'])
    if not time_equal:
        raise ValueError('保存CSV与严格积分复跑的时间网格不同')
    error_trace_difference_mm = float(np.max(abs(saved['error_m']-strict_full['error_m']))*1000)
    velocity_trace_difference_m_s = float(np.max(abs(saved['velocity_m_s']-strict_full['velocity_m_s'])))
    record('strict_ode_tolerance',
           all(item['max_error_metric_difference_mm']<1e-5 and
               item['contact_min_velocity_difference_m_s']<1e-6 for item in strict_evidence.values()) and
           error_trace_difference_mm<1e-5 and velocity_trace_difference_m_s<1e-6,
           baseline_rtol=float(config['rtol']), strict_rtol=1e-10,
           strategies=strict_evidence, full_max_error_trace_difference_mm=error_trace_difference_mm,
           full_max_velocity_trace_difference_m_s=velocity_trace_difference_m_s,
           note='仅收紧ODE容差；控制周期和所有物理参数保持相同。')

    half = run_classroom(ts=ts/2, parameters=parameters, rtol=1e-10)
    half_evidence = {name:dict(max_abs_error_mm=float(result['metrics']['max_abs_error_mm']),
                             contact_min_velocity_m_s=float(result['metrics']['contact_min_velocity_m_s']),
                             first_breakaway_ms=result['metrics']['first_breakaway_ms'])
                     for name,result in half.items()}
    record('half_control_period',
           half['full']['metrics']['max_abs_error_mm']<.05 and
           abs(half['full']['metrics']['max_abs_error_mm']-baseline_error_mm)<.03,
           baseline_period_s=ts, half_period_s=ts/2, strategies=half_evidence,
           note='改变数字控制周期的敏感性试验；要求完整控制仍达到0.05mm，不预设缩短周期必然单调改善。')

    no_load_parameters = replace(parameters,heavy_drive_pressure_pa=parameters.P1_0)
    no_load = run_classroom(ts=ts,parameters=no_load_parameters,rtol=1e-10)
    no_load_nominal, no_load_full = no_load['nominal']['data'],no_load['full']['data']
    no_load_is_zero = all(np.all(result['data']['load_n']==0.) for result in no_load.values())
    step_index = int(np.flatnonzero(np.isclose(no_load_full['time_s'], parameters.load_start))[0])
    voltage_jumps = {name:float(abs(data['voltage_v'][step_index]-data['voltage_v'][step_index-1]))
                     for name,data in (('nominal',no_load_nominal),('full',no_load_full))}
    pressure_jump_pa = float(abs(no_load_full['p1_pa'][step_index]-no_load_full['p1_pa'][step_index-1]))
    record('no_load_counterfactual',
           no_load_is_zero and max(voltage_jumps.values())<.05 and pressure_jump_pa<5e4,
           no_load_heavy_drive_pressure_pa=no_load_parameters.heavy_drive_pressure_pa,
           all_load_samples_zero=no_load_is_zero,voltage_jump_at_0p50s_v=voltage_jumps,
           pressure_jump_at_0p50s_pa=pressure_jump_pa,
           max_nominal_full_position_difference_m=float(np.max(abs(no_load_nominal['position_m']-no_load_full['position_m']))),
           note='移除工艺负载后，两种前馈均不得在原加载时刻出现人工电压阶跃；完整组仍按实测运动压力微调。')

    shifted_parameters = replace(parameters,load_start=.55,load_end=.97)
    shifted = run_classroom(ts=ts,parameters=shifted_parameters,rtol=1e-10)
    shifted_evidence = {}
    shift_ok = True
    for name,result in shifted.items():
        data=result['data']
        active=np.flatnonzero(data['load_n']>0.)
        first=float(data['time_s'][active[0]]) if len(active) else None
        unloaded=np.flatnonzero((data['time_s']>=shifted_parameters.load_end)&(data['load_n']==0.))
        first_unloaded=float(data['time_s'][unloaded[0]]) if len(unloaded) else None
        shift_ok &= first is not None and abs(first-.55)<1e-12 and first_unloaded is not None and abs(first_unloaded-.97)<1e-12
        shifted_evidence[name]=dict(first_loaded_sample_s=first,first_unloaded_sample_s=first_unloaded,
            max_abs_error_mm=result['metrics']['max_abs_error_mm'],
            contact_min_velocity_m_s=result['metrics']['contact_min_velocity_m_s'])
    prefix = saved['time_s']<parameters.load_start-1e-12
    before_old_load_difference = float(np.max(abs(saved['position_m'][prefix]-shifted['full']['data']['position_m'][prefix])))
    shifted_data=shifted['full']['data']
    middle=(saved['time_s']>=parameters.load_start)&(saved['time_s']<shifted_parameters.load_start)
    no_early_load=bool(np.all(shifted_data['load_n'][middle]==0.))
    record('shifted_load_counterfactual',shift_ok and no_early_load and before_old_load_difference<1e-10,
           requested_load_start_s=.55,requested_load_end_s=.97,strategies=shifted_evidence,
           full_position_difference_before_original_load_m=before_old_load_difference,
           no_load_at_original_0p50s=True if no_early_load else False,
           note='负载时刻由参数移动；控制器及ODE自动响应，未固定响应曲线时刻。')

    report=dict(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        model='classroom_pressure_impedance_reduction',
        scope='只验证明确公开的约化模型；不证明上传四阶液压Twin具有同样动态性能。',
        inputs=dict(summary_sha256=_hash(summary_path),full_csv_sha256=_hash(csv_path),
                    classroom_model_sha256=_hash(ROOT/'classroom_model.py')),
        all_passed=all(check['passed'] for check in checks),checks=checks)
    path=directory/'numerical_checks.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return path,report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'outputs'/'classroom',help='已生成的课堂结果目录')
    args=parser.parse_args()
    path,report=verify(args.directory)
    for check in report['checks']:
        print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}")
    print(path)
    if not report['all_passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
