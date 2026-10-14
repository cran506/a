"""核对实际数据、预测约束与采样间压力；--strict 用相同控制序列严格复算对象。"""
import argparse
import json
from pathlib import Path

import numpy as np
from simulation import ROOT, KINDS
from hydraulic_plant import HydraulicPlant


def verify(folder, strict=False):
    folder = Path(folder)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    checks, comparisons = {}, {}
    for kind in KINDS:
        rows = np.genfromtxt(folder/f'{kind}.csv', names=True, delimiter=',')
        pressure = np.genfromtxt(folder/f'{kind}_pressure_trace.csv', names=True, delimiter=',')
        logs = json.loads((folder/f'{kind}_solver_logs.json').read_text(encoding='utf-8'))
        config, reported = summary[kind]['config'], summary[kind]['metrics']
        fields = [key for key in rows.dtype.names if not (kind != 'pid' and key == 'pid_integral_v')]
        checks[kind+'_finite'] = all(bool(np.isfinite(rows[key]).all()) for key in fields)
        checks[kind+'_clock'] = bool(np.allclose(np.diff(rows['time_s']), config['ts'], atol=1e-12, rtol=0.))
        checks[kind+'_voltage_bound'] = bool(np.max(abs(rows['control_v'])) <= 10.+1e-12)
        checks[kind+'_stored_increment'] = bool(np.allclose(rows['delta_voltage_v'],
            np.diff(rows['control_v'], prepend=0.), atol=1e-12, rtol=0.))
        checks[kind+'_trace_order'] = bool(np.all(np.diff(pressure['time_s']) >= -1e-12))
        checks[kind+'_positive_real_pressure'] = bool(min(pressure['p1_mpa'].min(),pressure['p2_mpa'].min()) > 0.)
        checks[kind+'_reported_real_peak'] = bool(abs(pressure['p2_mpa'].max()-reported['peak_p2_mpa']) < 1e-10)
        checks[kind+'_intersample_peak_retained'] = bool(reported['peak_p2_mpa']+1e-10 >= rows['p2_mpa'].max())
        checks[kind+'_reported_position_peak'] = bool(abs(rows['position_mm'].max()-reported['peak_position_mm']) < 1e-10)
        checks[kind+'_same_step_reference'] = bool(np.array_equal(rows['reference_mm'],
            np.where(rows['time_s'] >= config['step_time']-1e-12, config['target_mm'], 0.)))
        if kind != 'pid':
            checks[kind+'_increment_bound'] = bool(np.max(abs(rows['delta_voltage_v'])) <= config['max_du_step']+1e-12)
            active = rows['time_s'] >= config['step_time']-1e-12
            checks[kind+'_all_commands_validated'] = bool(np.isin(rows['solver_status_code'][active], [1,2,3]).all())
            successful = [item for item in logs if item.get('status') in ('solved','validated_shifted_plan')]
            checks[kind+'_future_sequence_input_bounds'] = all(
                max(abs(value) for value in item['command_sequence_v']) <= 10.+1e-12 for item in successful)
            if kind == 'full_mpc':
                checks['full_prediction_pressure_constraint'] = all(
                    item['predicted_peak_p2_mpa'] <= config['p_safe']-config['pressure_margin']+2e-6
                    for item in successful)
                checks['full_actual_intersample_pressure_constraint'] = bool(
                    pressure['p2_mpa'].max() <= config['p_safe'])
        if strict:
            plant = HydraulicPlant(config=config['physical_parameters'], origin_m=config['origin_m'],
                valve_tau=config['valve_tau'], makeup_area=config['makeup_area'], rtol=1e-8)
            plant.reset(config['initial_state'])
            states = []
            for k, voltage in enumerate(rows['control_v']):
                states.append(plant.state)
                if k+1 < len(rows):
                    plant.step(float(voltage), config['ts'])
            states = np.asarray(states)
            peak = float(plant.pressure_trace[:,2].max()/1e6)
            difference = float(np.max(abs(states[:,0]-rows['position_mm'])))
            comparisons[kind] = dict(mode='recorded-input replay, not a new MPC optimization run',
                baseline_rtol=config['rtol'], strict_rtol=1e-8,
                max_position_difference_mm=difference,
                strict_peak_p2_mpa=peak,
                peak_pressure_difference_mpa=abs(peak-reported['peak_p2_mpa']))
            checks[kind+'_strict_position_agreement'] = bool(difference < .02)
            checks[kind+'_strict_peak_agreement'] = bool(abs(peak-reported['peak_p2_mpa']) < .02)
            if kind == 'full_mpc':
                checks['full_strict_actual_pressure_constraint'] = bool(peak <= config['p_safe'])
    configs = [summary[kind]['config'] for kind in KINDS]
    shared = ('physical_parameters', 'initial_state', 'ts', 'duration', 'step_time', 'target_mm',
              'origin_m', 'valve_tau', 'makeup_area')
    checks['common_plant_initial_condition_reference'] = all(
        all(config[key] == configs[0][key] for key in shared) for config in configs[1:])
    checks['input_mpc_reduces_overshoot'] = bool(summary['input_mpc']['metrics']['overshoot_percent']
                                               < summary['pid']['metrics']['overshoot_percent'])
    checks['pid_braking_overpressure_observed'] = bool(summary['pid']['metrics']['peak_p2_mpa'] > configs[0]['p_safe'])
    checks['input_only_braking_overpressure_observed'] = bool(summary['input_mpc']['metrics']['peak_p2_mpa'] > configs[0]['p_safe'])
    checks['input_only_peak_more_severe_than_pid'] = bool(summary['input_mpc']['metrics']['peak_p2_mpa']
                                                         > summary['pid']['metrics']['peak_p2_mpa'])
    outcome = dict(passed=all(checks.values()), checks=checks, strict_replay=comparisons,
                   pressure_safety_scope='P2 upper bound in this nominal classroom experiment',
                   realtime_requirement_passed=all(summary[kind]['metrics']['controller_within_1ms_fraction'] == 1.
                                                   for kind in ('input_mpc','full_mpc')),
                   reference_numbers_are_comparison_targets=True)
    (folder/'validation.json').write_text(json.dumps(outcome, indent=2, ensure_ascii=False), encoding='utf-8')
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    result = verify(args.output_dir, args.strict)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
