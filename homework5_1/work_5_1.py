#!/usr/bin/env python3
"""习题 5-1：相同对象/增益/采样条件下的 PID 抗积分饱和对比。

直接运行生成一张三联图、两份 CSV 和由本次积分计算的 summary.json。
默认共同对测量微分；--derivative-mode error 复核教材误差微分式。
"""
import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'homework5_1_mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'homework5_1_cache'))
Path(os.environ['XDG_CACHE_HOME']).mkdir(parents=True, exist_ok=True)
import numpy as np

from pid_algorithms import IncrementalPID, PositionalPID
from twin_adapter import ClosedLoopTwin, QuantizedSensor, classroom_config

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    kp: float = 4.2
    ki: float = 22.
    kd: float = .06
    alpha: float = .85
    derivative_mode: str = 'measurement'
    ts: float = .001
    duration: float = 1.2
    step_time: float = .05
    step_amplitude_m: float = .1
    voltage_limit_v: float = 10.
    adc_bits: int = 12
    sensor_span_m: float = .3
    initial_position_m: float = .05
    valve_tau_s: float = .0053
    rtol: float = 1e-5

    def validate(self):
        values = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not all(np.isfinite(v) for v in values):
            raise ValueError('实验参数必须有限')
        if (self.ts <= 0 or not 0 < self.step_time < self.duration
                or self.step_amplitude_m <= 0 or not 0 < self.voltage_limit_v <= 10
                or self.valve_tau_s < 0 or self.rtol <= 0
                or self.initial_position_m < 0
                or self.initial_position_m+self.step_amplitude_m >= self.sensor_span_m
                or abs(self.duration/self.ts-round(self.duration/self.ts)) > 1e-9
                or abs(self.step_time/self.ts-round(self.step_time/self.ts)) > 1e-9):
            raise ValueError('实验时钟、量程或执行器参数无效')
        QuantizedSensor(self.adc_bits, self.sensor_span_m)
        PositionalPID(self.kp, self.ki, self.kd, self.ts, self.alpha,
                      derivative_mode=self.derivative_mode)


def load_config(path=None):
    config = ExperimentConfig(**json.loads(
        (Path(path) if path else ROOT/'experiment_config.json').read_text(encoding='utf-8')))
    config.validate()
    return config


def simulate(kind, config):
    """采样时刻先记物理状态，随后计算 u[k] 并保持至下一周期。"""
    config.validate()
    cls = {'positional': PositionalPID, 'incremental': IncrementalPID}[kind]
    controller = cls(config.kp, config.ki, config.kd, config.ts, config.alpha,
                     -config.voltage_limit_v, config.voltage_limit_v,
                     config.derivative_mode)
    sensor = QuantizedSensor(config.adc_bits, config.sensor_span_m)
    plant = ClosedLoopTwin(classroom_config(), load_force=0.,
                           valve_tau=config.valve_tau_s, rtol=config.rtol)
    origin = sensor.sample(config.initial_position_m)
    p = plant.p
    initial_p2 = (p['Pr']*p['A1']-plant.ambient_pressure*(p['A1']-p['A2']))/p['A2']
    plant.reset([origin, 0., p['Pr'], initial_p2])
    time = np.arange(round(config.duration/config.ts)+1)*config.ts
    reference = np.where(np.arange(len(time)) >= round(config.step_time/config.ts),
                         config.step_amplitude_m, 0.)
    names = ('position_m', 'velocity_m_s', 'p1_pa', 'p2_pa', 'measured_m',
             'true_error_m', 'sampled_error_m', 'voltage_v', 'raw_voltage_v',
             'derivative_v', 'integral_increment_v', 'explicit_integral_v',
             'effective_integral_v', 'delta_voltage_v', 'state_before_v',
             'state_after_v', 'valve_normalized', 'net_drive_n', 'friction_n',
             'sticking', 'saturated')
    data = {key: np.empty(len(time)) for key in names}
    for index, ti in enumerate(time):
        if not np.isclose(plant.t, ti, rtol=0., atol=1e-10):
            raise RuntimeError('物理时钟与控制采样时钟不同步')
        x, velocity, p1, p2 = plant.state
        position = x-origin
        feedback = sensor.sample(x)-origin
        voltage = controller.update(float(reference[index]), feedback)
        diag = plant.friction_diagnostics()
        values = (position, velocity, p1, p2, feedback, reference[index]-position,
                  reference[index]-feedback, voltage, controller.last_raw,
                  controller.last_d, controller.last_di,
                  controller.i_total if kind == 'positional' else np.nan,
                  controller.last_effective_integral, controller.last_delta,
                  controller.state_before, controller.u_prev, plant.valve_state,
                  diag['net_drive_n'], diag['friction_n'], int(diag['sticking']),
                  int(abs(voltage) >= config.voltage_limit_v-1e-9))
        for key, value in zip(names, values):
            data[key][index] = value
        if index+1 < len(time):
            plant.step_voltage(voltage, config.ts)
    data = {'time_s': time, 'reference_m': reference, **data}
    metadata = dict(controller=kind, **asdict(config), origin_m=origin,
                    adc_lsb_mm=sensor.lsb*1000., physical_parameters=plant.p.copy(),
                    integrator='Radau + friction events', radau_nfev=plant.nfev,
                    release_count=plant.release_count, restick_count=plant.restick_count,
                    source_twin_sha256=hashlib.sha256(
                        (ROOT/'hydraulic_digital_twin.py').read_bytes()).hexdigest())
    return dict(data=data, config=metadata, metrics=metrics(data, config),
                friction_events=plant.friction_events)


def _first_index(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def metrics(data, config):
    t, x, e, u = (data[key] for key in
                  ('time_s', 'position_m', 'true_error_m', 'voltage_v'))
    after_step = t >= config.step_time-1e-12
    positive = after_step & (u >= config.voltage_limit_v-1e-9)
    entry = _first_index(positive)
    exit_index = (_first_index((np.arange(len(t)) > entry)
                              & (u < config.voltage_limit_v-1e-9))
                  if entry is not None else None)
    crossing_index = _first_index(after_step & (e <= 0.))
    sampled_crossing = _first_index(after_step & (data['sampled_error_m'] <= 0.))
    crossing_time = None
    if crossing_index is not None:
        k = crossing_index
        if k > 0 and e[k-1] > 0:
            crossing_time = float(t[k-1]+config.ts*e[k-1]/(e[k-1]-e[k]))
        else:
            crossing_time = float(t[k])
    peak = int(np.argmax(x))
    exit_time = float(t[exit_index]) if exit_index is not None else None
    end = t >= config.duration-.1-1e-12
    return dict(
        peak_position_mm=float(x[peak]*1000.), peak_time_s=float(t[peak]),
        overshoot_percent=float(max(0., x[peak]-config.step_amplitude_m)
                                /config.step_amplitude_m*100.),
        max_internal_voltage_v=float(np.max(data['raw_voltage_v'])),
        first_positive_saturation_s=float(t[entry]) if entry is not None else None,
        initial_positive_saturation_exit_s=exit_time,
        crossing_time_interpolated_s=crossing_time,
        first_nonpositive_true_error_sample_s=(float(t[crossing_index])
                                              if crossing_index is not None else None),
        first_nonpositive_sampled_error_s=(float(t[sampled_crossing])
                                          if sampled_crossing is not None else None),
        desaturation_delay_after_crossing_ms=(max(0., (exit_time-crossing_time)*1000.)
                                             if crossing_time is not None and exit_time is not None
                                             else None),
        exit_relative_to_crossing_ms=((exit_time-crossing_time)*1000.
                                     if crossing_time is not None and exit_time is not None else None),
        sampled_error_desaturation_delay_ms=(max(0., (exit_time-t[sampled_crossing])*1000.)
                                            if sampled_crossing is not None and exit_time is not None
                                            else None),
        max_explicit_integral_v=(float(np.max(data['explicit_integral_v']))
                                 if np.any(np.isfinite(data['explicit_integral_v'])) else None),
        final_error_mm=float(e[-1]*1000.),
        last_100ms_max_abs_error_mm=float(np.max(abs(e[end]))*1000.),
        physical_saturation_duration_ms=float(np.sum(data['saturated'][:-1])*config.ts*1000.),
        output_max_abs_v=float(np.max(abs(u))),
        state_max_abs_v=float(np.max(abs(data['state_after_v']))),
        min_pressure_pa=float(min(data['p1_pa'].min(), data['p2_pa'].min())),
        max_pressure_pa=float(max(data['p1_pa'].max(), data['p2_pa'].max())))


def reference_comparison(results):
    # 以下只在表格中比较，不参与 ODE、控制律或图线计算。
    targets = {
        'positional': {'max_internal_voltage_v': 42.5,
                       'crossing_time_interpolated_s': .204,
                       'initial_positive_saturation_exit_s': .216,
                       'overshoot_percent': 31.4},
        'incremental': {'initial_positive_saturation_exit_s': .061,
                        'overshoot_percent': 1.0},
    }
    comparison = {}
    for kind, expectations in targets.items():
        comparison[kind] = {}
        for metric, target in expectations.items():
            actual = results[kind]['metrics'][metric]
            comparison[kind][metric] = dict(reference=target, actual=actual,
                difference=(actual-target if actual is not None else None))
    return comparison


def save_results(results, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for kind, result in results.items():
        data = result['data']
        with (folder/f'{kind}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(data)
            writer.writerows(zip(*data.values()))
        (folder/f'{kind}_friction_events.json').write_text(
            json.dumps(result['friction_events'], indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {kind: dict(config=result['config'], metrics=result['metrics'])
               for kind, result in results.items()}
    summary['reference_comparison'] = reference_comparison(results)
    summary['derivative_note'] = (
        'Both controllers share derivative-on-measurement (no setpoint kick).'
        if results['positional']['config']['derivative_mode'] == 'measurement'
        else 'Both controllers use the textbook derivative-on-error formula; setpoint kick is retained.')
    (folder/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                       encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'experiment_config.json')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--derivative-mode', choices=('measurement', 'error'), default=None)
    parser.add_argument('--no-plot', action='store_true', help='仅导出数据')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.derivative_mode:
        config = replace(config, derivative_mode=args.derivative_mode)
    folder = args.output_dir or ROOT/('outputs' if config.derivative_mode == 'measurement'
                                      else 'outputs_error_derivative')
    results = {}
    for kind in ('positional', 'incremental'):
        print(f'Simulating {kind}, D={config.derivative_mode} ...', flush=True)
        results[kind] = simulate(kind, config)
    save_results(results, folder)
    if not args.no_plot:
        from plotting import plot_comparison
        plot_comparison(results, folder/'anti_windup_comparison.png')
    for kind, result in results.items():
        print(kind+': '+json.dumps(result['metrics'], ensure_ascii=False), flush=True)
    print(f'Saved: {folder.resolve()}', flush=True)


if __name__ == '__main__':
    main()
