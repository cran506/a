"""独立检查原始数据映射、共同模型和三策略现象，不修改计算结果。"""
import json
from pathlib import Path
import numpy as np
from deadzone_model import deadzone
from model_assisted_controller import inverse_deadzone


def validate(results, beta_results, config, destination=None):
    checks = {}
    for name, result in [*results.items(), *((f'beta_{beta:g}', r) for beta,r in beta_results.items())]:
        d = result['data']
        checks[f'{name}:finite'] = all(np.all(np.isfinite(x)) for x in d.values())
        checks[f'{name}:clock'] = bool(np.allclose(np.diff(d['time_s']), config['sample_time_s'], atol=1e-12, rtol=0))
        checks[f'{name}:physical_input_deadzone'] = bool(np.allclose(
            d['effective_voltage_v'], deadzone(d['control_voltage_v'], config['physical_deadzone_v']), atol=1e-12, rtol=0))
        alpha = np.exp(-config['sample_time_s']/config['tau_valve_s'])
        checks[f'{name}:causal_valve_update'] = bool(np.allclose(
            d['valve_voltage_v'][1:], alpha*d['valve_voltage_v'][:-1]+(1-alpha)*d['effective_voltage_v'][:-1], atol=1e-12,rtol=0))
        estimate = config['under_estimate_v'] if name == 'under' else config['over_estimate_v']
        smooth = name == 'smooth' or name.startswith('beta_')
        beta = result['beta'] if smooth else config['beta_default']
        wanted = np.clip(inverse_deadzone(d['linear_voltage_v'], estimate, 'tanh' if smooth else 'hard', beta), -10,10)
        checks[f'{name}:inverse_formula'] = bool(np.allclose(d['control_voltage_v'], wanted, atol=1e-12, rtol=0))
        checks[f'{name}:hardware_bounds'] = bool(np.max(abs(d['control_voltage_v'])) <= 10.+1e-12)
        checks[f'{name}:reference'] = bool(np.allclose(d['reference_voltage_v'], config['amplitude_v']*
            np.sin(2*np.pi*config['frequency_hz']*d['time_s']), atol=1e-12, rtol=0))
    under, hard, smooth = (results[k]['metrics'] for k in ('under','hard_over','smooth'))
    checks['under_has_visible_residual_platform'] = under['longest_effective_zero_ms'] > 50.
    checks['under_has_visible_near_zero_velocity_platform'] = under['longest_velocity_near_zero_ms'] > 80.
    checks['hard_velocity_chattering_is_visible'] = hard['chatter_velocity_pp_mm_s'] > .8
    checks['hard_over_has_repeated_feedback_switches'] = hard['linear_sign_reversals'] > 3
    checks['smooth_has_single_reversal'] = smooth['linear_sign_reversals'] == 1
    checks['smooth_reduces_high_frequency_pp'] = smooth['chatter_voltage_pp_v'] < hard['chatter_voltage_pp_v']
    checks['smooth_residual_plateau_is_disclosed'] = smooth['longest_flow_closed_ms'] > 0.
    checks['smooth_reduces_plateau_vs_under'] = smooth['longest_flow_closed_ms'] < under['longest_flow_closed_ms']
    checks['all_requested_betas_run'] = set(beta_results) == set(config['beta_values'])
    report = dict(passed=all(checks.values()), checks=checks,
                  scope='默认参数下功能及现象检查；不证明任意beta、噪声及反馈增益下稳定',
                  noise_std_mm_s=config['measurement_noise_std_mm_s'])
    if destination is not None:
        Path(destination).write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    from simulation import load_config, simulate
    c = load_config()
    results = {k:simulate(k,c) for k in ('under','hard_over','smooth')}
    beta_results = {b:(results['smooth'] if b == c['beta_default'] else simulate('smooth',c,b)) for b in c['beta_values']}
    validation = validate(results,beta_results,c,Path(__file__).resolve().parent/'outputs/validation.json')
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    raise SystemExit(0 if validation['passed'] else 1)
