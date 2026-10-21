"""重积分与无载反事实检查；先运行 model_assisted_lab.py 生成原始 CSV。

python verify_numerics.py
python verify_numerics.py --unknown   # 额外观察未知负载，结果不作为精度承诺

检查从 summary.json 恢复实际实验参数，不使用预期曲线或手写目标数据。
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np

from simulation import Experiment, simulate


ROOT = Path(__file__).resolve().parent


def reconstruct_experiment(summary):
    """额外元数据不是 Experiment 参数；禁止用当前默认值覆盖已保存参数。"""
    strategies = summary.get("strategies", summary)
    saved = strategies["full"]
    parameters = saved["config"]
    names = {field.name for field in fields(Experiment)}
    missing = names.difference(parameters)
    if missing:
        raise ValueError(f"summary 缺少实验参数：{sorted(missing)}；请重新生成结果")
    config = Experiment(**{name: parameters[name] for name in names})
    if not config.enable_ff or not config.enable_gain_sched:
        raise ValueError("full 配置必须开启前馈与实测压力增益调度")
    return config, saved


def compare_reintegration(baseline, saved, result, position_tolerance_mm):
    data = result["data"]
    same_grid = (baseline.shape == data["time_s"].shape and np.allclose(
        baseline["time_s"], data["time_s"], rtol=0., atol=1e-12))
    if not same_grid:
        raise ValueError("保存的 CSV 与重积分时间网格不一致，拒绝错位比较")
    baseline_max_mm = float(np.max(np.abs(baseline["error_m"])) * 1000)
    baseline_identity = bool(np.allclose(
        baseline["error_m"], baseline["reference_m"] - baseline["position_m"],
        rtol=0., atol=1e-12))
    difference_mm = float(np.max(np.abs(
        data["position_m"] - baseline["position_m"])) * 1000)
    strict_max_mm = float(np.max(np.abs(data["error_m"])) * 1000)
    recorded_physics = saved["config"].get("physical_parameters")
    same_physics = (recorded_physics is not None and
                    recorded_physics == result["config"]["physical_parameters"])
    same_reference = bool(np.allclose(
        baseline["reference_m"], data["reference_m"], rtol=0., atol=1e-12))
    reported_max = saved["metrics"]["max_abs_error_mm"]
    summary_matches_csv = bool(np.isclose(reported_max, baseline_max_mm, rtol=0., atol=1e-9))
    checks = {
        "saved_error_equals_reference_minus_state": baseline_identity,
        "saved_summary_matches_csv": summary_matches_csv,
        "same_time_grid": same_grid,
        "same_reference": same_reference,
        "same_physical_parameters": same_physics,
        "position_difference_within_tolerance": difference_mm <= position_tolerance_mm,
        "strict_max_error_below_0p05mm": strict_max_mm < .05,
    }
    return {
        "saved_rtol": saved["config"]["rtol"],
        "recomputed_rtol": result["config"]["rtol"],
        "saved_max_abs_error_mm": baseline_max_mm,
        "recomputed_max_abs_error_mm": strict_max_mm,
        "change_in_max_abs_error_mm": strict_max_mm - baseline_max_mm,
        "max_position_difference_from_csv_mm": difference_mm,
        "position_difference_tolerance_mm": position_tolerance_mm,
        "recomputed_first_breakaway_ms": result["metrics"]["first_breakaway_ms"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def check_no_load(result, event_time, jump_limit_v):
    """无载时原负载时刻仍处巡航；检查是否存在按时间拼接的电压跳变。"""
    data = result["data"]
    time, voltage = data["time_s"], data["voltage_v"]
    window = np.flatnonzero(np.abs(time - event_time) <= .005 + 1e-12)
    if len(window) < 3:
        raise ValueError("负载时刻前后需至少三个采样点用于无载反事实检查")
    near_voltage = voltage[window]
    max_step = float(np.max(np.abs(np.diff(near_voltage))))
    checks = {
        "actual_load_is_zero": bool(np.all(data["load_n"] == 0.)),
        "planned_load_is_zero": bool(np.all(data["planned_force_n"] == 0.)),
        "no_event_time_voltage_jump": max_step < jump_limit_v,
        "all_states_are_finite": bool(all(np.all(np.isfinite(value)) for value in data.values())),
    }
    return {
        "description": "Full controller, load_force=0; the former load time is not a controller trigger.",
        "original_event_time_s": event_time,
        "window_start_s": float(time[window[0]]),
        "window_end_s": float(time[window[-1]]),
        "max_adjacent_voltage_step_v": max_step,
        "endpoint_voltage_change_v": float(near_voltage[-1] - near_voltage[0]),
        "jump_limit_v": jump_limit_v,
        "metrics": result["metrics"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "twin",
                        help="包含 full.csv 和 summary.json 的四阶孪生结果目录")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "numerical_checks.json")
    parser.add_argument("--unknown", action="store_true", help="另外运行外载未告知控制器的边界实验")
    parser.add_argument("--position-tolerance-mm", type=float, default=.005)
    parser.add_argument("--no-load-jump-limit-v", type=float, default=.1)
    args = parser.parse_args()
    if (not np.isfinite(args.position_tolerance_mm) or args.position_tolerance_mm <= 0 or
            not np.isfinite(args.no_load_jump_limit_v) or args.no_load_jump_limit_v <= 0):
        parser.error("位置差异和无载电压跳变阈值必须为有限正数")
    try:
        summary = json.loads((args.input / "summary.json").read_text(encoding="utf-8"))
        baseline = np.genfromtxt(args.input / "full.csv", delimiter=",", names=True, encoding="utf-8")
        config, saved = reconstruct_experiment(summary)
    except (OSError, KeyError, ValueError) as error:
        parser.error(f"无法读取原始四阶孪生结果：{error}")

    print("以 rtol=1e-8 重积分 full……", flush=True)
    strict = simulate(replace(config, rtol=1e-8))
    convergence = compare_reintegration(baseline, saved, strict, args.position_tolerance_mm)
    print("运行 load_force=0 的无载反事实……", flush=True)
    no_load = simulate(replace(config, name="full_no_load", load_force=0.))
    counterfactual = check_no_load(no_load, config.load_start, args.no_load_jump_limit_v)
    report = {
        "source": {"summary": "summary.json", "csv": "full.csv"},
        "reintegration": convergence,
        "no_load_counterfactual": counterfactual,
        "unknown_load_was_requested": args.unknown,
        "passed": convergence["passed"] and counterfactual["passed"],
    }
    if args.unknown:
        print("运行已知负载前馈关闭的边界实验……", flush=True)
        unknown = simulate(replace(config, name="full_unknown_load", known_load=False))
        report["unknown_load"] = {
            "description": "Boundary experiment; no known force/rate is passed to the controller. Not a 0.05 mm acceptance test.",
            "metrics": unknown["metrics"],
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"重积分 max|e|={convergence['recomputed_max_abs_error_mm']:.9f} mm；"
          f"与 CSV 最大位置差={convergence['max_position_difference_from_csv_mm']:.9f} mm", flush=True)
    print(f"无载事件窗口最大相邻电压差={counterfactual['max_adjacent_voltage_step_v']:.6f} V；"
          f"检查通过={report['passed']}；{args.output}", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
