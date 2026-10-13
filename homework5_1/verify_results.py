"""从 CSV 独立复核控制/采样记录，--strict 另做更严格的真实积分复算。"""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np

from work_5_1 import ROOT, ExperimentConfig, simulate


def verify(folder, strict=False):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    checks, details, data = {}, {}, {}
    for kind in ('positional', 'incremental'):
        d = np.genfromtxt(folder/f'{kind}.csv', delimiter=',', names=True)
        data[kind] = d
        meta = summary[kind]['config']
        local = ExperimentConfig(**{field.name: meta[field.name]
                                    for field in fields(ExperimentConfig)})
        valid_fields = [name for name in d.dtype.names
                        if not (kind == 'incremental' and name == 'explicit_integral_v')]
        checks[kind+'_finite'] = all(bool(np.isfinite(d[name]).all()) for name in valid_fields)
        checks[kind+'_pressure_positive'] = bool(min(d['p1_pa'].min(), d['p2_pa'].min()) > 0)
        checks[kind+'_clock'] = bool(np.allclose(np.diff(d['time_s']), local.ts,
                                                atol=1e-12, rtol=0.))
        expected_reference = np.where(d['time_s'] >= local.step_time-1e-12,
                                      local.step_amplitude_m, 0.)
        checks[kind+'_reference'] = bool(np.array_equal(d['reference_m'], expected_reference))
        checks[kind+'_true_error'] = bool(np.allclose(d['true_error_m'],
                    d['reference_m']-d['position_m'], atol=1e-15, rtol=0.))
        checks[kind+'_output_limits'] = bool(np.max(abs(d['voltage_v'])) <= local.voltage_limit_v)
        checks[kind+'_clip'] = bool(np.allclose(d['voltage_v'],
                    np.clip(d['raw_voltage_v'], -local.voltage_limit_v, local.voltage_limit_v),
                    atol=1e-12, rtol=0.))
        checks[kind+'_actual_output_memory'] = bool(
            np.array_equal(d['state_after_v'], d['voltage_v'])
            and np.array_equal(d['state_before_v'][1:], d['voltage_v'][:-1]))
        origin, lsb = meta['origin_m'], meta['adc_lsb_mm']/1000.
        expected_feedback = np.clip(np.rint((d['position_m']+origin)/lsb),
                                    0, 2**local.adc_bits-1)*lsb-origin
        checks[kind+'_adc_from_true_state'] = bool(np.allclose(
                    d['measured_m'], expected_feedback, atol=1e-14, rtol=0.))
        if kind == 'positional':
            expected_integral = np.cumsum(local.ki*100*d['sampled_error_m']*local.ts)
            checks['positional_unlimited_integral'] = bool(np.allclose(
                d['explicit_integral_v'], expected_integral, atol=1e-10, rtol=0.))
        else:
            ie = d['effective_integral_v']
            prior = np.r_[0., ie[:-1]]
            expected_ie = (prior+local.ki*100*d['sampled_error_m']*local.ts
                           +d['voltage_v']-d['raw_voltage_v'])
            checks['incremental_implicit_tracking_integral'] = bool(np.allclose(
                ie, expected_ie, atol=1e-10, rtol=0.))
            checks['incremental_no_explicit_integrator'] = bool(np.isnan(d['explicit_integral_v']).all())
        peak = d['position_m'].max()*1000.
        checks[kind+'_reported_peak'] = bool(abs(peak-summary[kind]['metrics']['peak_position_mm']) < 1e-10)
        if strict:
            rerun = simulate(kind, replace(local, rtol=1e-7))
            difference = abs(rerun['data']['position_m']-d['position_m'])*1000.
            details[kind] = dict(
                baseline_rtol=local.rtol, strict_rtol=1e-7,
                max_position_difference_mm=float(difference.max()),
                overshoot_difference_percentage_points=float(abs(
                    rerun['metrics']['overshoot_percent']-summary[kind]['metrics']['overshoot_percent'])),
                strict_metrics=rerun['metrics'],
                strict_release_count=rerun['config']['release_count'],
                strict_restick_count=rerun['config']['restick_count'])
            # 容许量化闭环分支的一两个 ADC 计数差，不能用参考目标当积分误差标准。
            checks[kind+'_strict_position_agreement'] = bool(difference.max() < .15)
    checks['common_reference'] = bool(np.array_equal(
        data['positional']['reference_m'], data['incremental']['reference_m']))
    shared = ('kp', 'ki', 'kd', 'alpha', 'ts', 'duration', 'step_time', 'step_amplitude_m',
              'derivative_mode', 'adc_bits', 'origin_m', 'valve_tau_s', 'physical_parameters')
    checks['common_physics_gains_sensor'] = all(
        summary['positional']['config'][key] == summary['incremental']['config'][key] for key in shared)
    ma, mb = (summary[kind]['metrics'] for kind in ('positional', 'incremental'))
    checks['positional_windup_observed'] = bool(
        ma['desaturation_delay_after_crossing_ms'] is not None
        and ma['desaturation_delay_after_crossing_ms'] > 0)
    checks['incremental_lower_overshoot'] = bool(mb['overshoot_percent'] < ma['overshoot_percent'])
    checks['incremental_exits_before_error_reversal'] = bool(
        mb['first_nonpositive_sampled_error_s'] is None
        or mb['initial_positive_saturation_exit_s'] < mb['first_nonpositive_sampled_error_s'])
    result = dict(passed=all(checks.values()), checks=checks, numerical_comparison=details,
                  reference_targets_are_not_assertions=True)
    (folder/'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    outcome = verify(args.output_dir, args.strict)
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    raise SystemExit(0 if outcome['passed'] else 1)
