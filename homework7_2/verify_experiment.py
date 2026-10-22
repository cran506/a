"""独立核验习题 7-2 的七组闭环记录及压力安全证据。

默认检查全部 CSV、逐步求解日志和汇总表。``--strict`` 额外以更紧的
Radau 相对误差重放 N=5、20、50 的原始控制指令；任何失败均写入
``outputs/validation.json`` 并返回非零退出码。验证以保存的数据为准，
不会把预测模型的压力上界误称为真实对象的安全保证。
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"work72_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir())/"work72_cache"))

from hydraulic_plant import HydraulicPlant


HERE = Path(__file__).resolve().parent
EXPECTED_HORIZONS = (5, 10, 15, 20, 30, 40, 50)
STATE_FIELDS = ("x_mm", "velocity_m_s", "p1_mpa", "p2_mpa", "valve_v")
TIME_FIELDS = ("solver_time_ms", "cvxpy_wall_ms", "controller_wall_ms")


def _load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        columns = {field: [] for field in reader.fieldnames}
        for row in reader:
            for field in reader.fieldnames:
                columns[field].append(float(row[field]))
    if not columns or len(next(iter(columns.values()))) == 0:
        raise ValueError(f"CSV has no data: {path}")
    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _close(actual, expected, atol=1e-8, rtol=1e-8):
    if actual is None or expected is None:
        return actual is None and expected is None
    try:
        return bool(np.isclose(float(actual), float(expected), atol=atol, rtol=rtol,
                               equal_nan=False))
    except (ValueError, TypeError):
        return False


def _finite_float(value):
    try:
        answer = float(value)
        return answer if math.isfinite(answer) else math.nan
    except (TypeError, ValueError):
        return math.nan


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _stat(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return dict(mean_ms=float(np.mean(finite)) if len(finite) else None,
                median_ms=float(np.median(finite)) if len(finite) else None,
                p95_ms=float(np.percentile(finite, 95)) if len(finite) else None,
                max_ms=float(np.max(finite)) if len(finite) else None,
                samples=int(len(finite)),
                over_1ms_fraction=float(np.mean(finite > 1.)) if len(finite) else None)


class Audit:
    def __init__(self, strict):
        self.report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                           strict_replay=bool(strict), passed=False, checks=[],
                           cases={}, errors=[])

    def check(self, name, passed, *, case=None, detail=None):
        item = dict(check=name, passed=bool(passed))
        if detail is not None:
            item["detail"] = detail
        if case is None:
            self.report["checks"].append(item)
        else:
            self.report["cases"].setdefault(str(case), dict(checks=[]))["checks"].append(item)
        if not passed:
            where = "global" if case is None else f"N={case}"
            self.report["errors"].append(f"{where}: {name}" +
                                         (f" ({detail})" if detail is not None else ""))
        return bool(passed)


def _compare_metric(audit, case, metrics, name, expected, *, atol=1e-8):
    actual = metrics.get(name)
    audit.check("summary." + name, _close(actual, expected, atol=atol),
                case=case, detail={"reported": actual, "derived": expected})


def _validate_summary(audit, case, metrics, data, trace, config):
    t, x, u = data["t_s"], data["x_mm"], data["u_v"]
    p2 = trace["p2_mpa"]
    target = float(config["target_mm"])
    peak_x = int(np.argmax(x))
    peak_p2 = int(np.argmax(p2))
    finite_u = u[np.isfinite(u)]
    reached = np.flatnonzero(x >= target)
    comparisons = dict(
        prediction_ms=case*config["ts"]*1000.,
        final_time_s=float(t[-1]), samples=len(t),
        peak_position_mm=float(x[peak_x]), peak_position_time_s=float(t[peak_x]),
        overshoot_pct=float(max(0., np.max(x)-target)/target*100.),
        first_target_sample_s=float(t[reached[0]]) if len(reached) else None,
        final_error_mm=float(target-x[-1]),
        peak_pressure_mpa=float(p2[peak_p2]),
        peak_pressure_time_s=float(trace["t_s"][peak_p2]),
        sampled_peak_pressure_mpa=float(np.max(data["p2_mpa"])),
        pressure_violation_mpa=float(max(0., np.max(p2)-config["p_safe"])),
        max_abs_voltage_v=float(np.max(np.abs(finite_u))) if len(finite_u) else None,
        max_abs_increment_v=float(np.max(np.abs(np.diff(np.r_[0., finite_u]))))
        if len(finite_u) else None,
        restoration_count=int(np.sum(data["status_code"] == 3)),
        shifted_count=int(np.sum(data["status_code"] == 2)),
        refinement_count=int(np.sum(data["status_code"] == 4)),
    )
    for name, expected in comparisons.items():
        _compare_metric(audit, case, metrics, name, expected)
    for key, values in (
        ("solver", data["solver_time_ms"][np.isin(data["status_code"], (1, 2))]),
        ("cvxpy_call", data["cvxpy_wall_ms"][np.isin(data["status_code"], (1, 2))]),
        ("whole_controller", data["controller_wall_ms"][data["status_code"] > 0]),
    ):
        summary = metrics.get(key, {})
        for name, expected in _stat(values).items():
            actual = summary.get(name) if isinstance(summary, dict) else None
            audit.check(f"summary.{key}.{name}", _close(actual, expected), case=case,
                        detail={"reported": actual, "derived": expected})


def _validate_logs(audit, case, data, logs, config):
    t, u, codes = data["t_s"], data["u_v"], data["status_code"]
    controlled = np.flatnonzero(codes != 0)
    audit.check("one solver log per attempted control step", len(logs) == len(controlled),
                case=case, detail={"logs": len(logs), "steps": len(controlled)})
    bound = float(config["p_safe"]-config["pressure_margin"])
    limit, slew = 10., float(config["max_du_step"])
    max_logged_prediction = -math.inf
    restoration_count = 0
    refinement_count = 0
    failures = []
    for index, log in zip(controlled, logs):
        stamp = float(t[index])
        kind = int(codes[index])
        context = f"t={stamp:.3f}s"
        if not _close(log.get("time_s"), stamp, atol=1e-10):
            failures.append(f"{context}: log time mismatch")
        if not _close(log.get("status_code"), kind):
            failures.append(f"{context}: status code mismatch")
        if kind < 0:
            failures.append(f"{context}: solver/restoration failed")
            continue
        if kind not in (1, 2, 3, 4):
            failures.append(f"{context}: unknown applied-command status {kind}")
            continue
        label = {1: "solved", 2: "validated_shifted_plan",
                 3: "restored_pressure_feasible",
                 4: "refined_pressure_feasible"}[kind]
        if log.get("status") != label:
            failures.append(f"{context}: undeclared or mismatched control mode")
        restored = kind == 3
        refined = kind == 4
        if bool(log.get("restoration_used", False)) != restored:
            failures.append(f"{context}: restoration flag mismatch")
        if bool(log.get("plan_refinement_used", False)) != refined:
            failures.append(f"{context}: plan refinement flag mismatch")
        if restored:
            restoration_count += 1
            feasible_candidates = _finite_float(log.get("restoration_feasible_candidates"))
            if (not log.get("restoration_failure") or
                    not math.isfinite(feasible_candidates) or feasible_candidates < 1):
                failures.append(f"{context}: missing restoration evidence")
        if refined:
            refinement_count += 1
            original_cost = _finite_float(log.get("plan_refinement_original_cost"))
            selected_cost = _finite_float(log.get("plan_refinement_selected_cost"))
            improvement = _finite_float(log.get("plan_refinement_absolute_improvement"))
            threshold = _finite_float(log.get("plan_refinement_required_improvement"))
            candidates = _finite_float(log.get("plan_refinement_feasible_candidates"))
            if (not log.get("plan_refinement_checked") or
                    not log.get("shifted_plan_replaced") or
                    not log.get("feasible_shifted_plan_used") or
                    log.get("plan_refinement_source") != "constant_grid" or
                    not all(math.isfinite(value) for value in
                            (original_cost, selected_cost, improvement, threshold, candidates)) or
                    candidates < 2 or improvement <= threshold or
                    not _close(original_cost-selected_cost, improvement, atol=1e-5)):
                failures.append(f"{context}: missing or inconsistent finite-grid improvement evidence")
        try:
            command = np.asarray(log["command_sequence_v"], dtype=float)
            prediction = np.asarray(log["prediction"], dtype=float)
        except (KeyError, ValueError, TypeError):
            failures.append(f"{context}: missing/non-numeric sequence or prediction")
            continue
        if (command.shape != (case,) or prediction.shape != (case, 5)
                or not np.all(np.isfinite(command)) or not np.all(np.isfinite(prediction))):
            failures.append(f"{context}: invalid sequence/prediction shape or value")
            continue
        previous = float(u[index-1]) if index else 0.
        if not (_close(command[0], u[index], atol=1e-8) and
                _close(log.get("voltage_v"), u[index], atol=1e-8)):
            failures.append(f"{context}: applied voltage differs from validated sequence")
        if refined:
            move_weight = _finite_float(log.get("active_move_weight"))
            target = np.array([data["reference_mm"][index], 0.])
            if not math.isfinite(move_weight):
                failures.append(f"{context}: refinement move weight missing")
            else:
                state_error = prediction[:, :2]-target
                increments = np.diff(np.r_[previous, command])
                calculated_cost = float(np.sum(state_error**2*np.array(
                    [config["q_position"], config["q_velocity"]]))+
                    move_weight*np.dot(increments, increments))
                if (not _close(calculated_cost, selected_cost, atol=1e-4,
                               rtol=1e-8) or
                        not _close(log.get("nonlinear_cost"), calculated_cost,
                                   atol=1e-4, rtol=1e-8)):
                    failures.append(f"{context}: selected refinement cost differs from trajectory")
        if (np.max(np.abs(command)) > limit+1e-5 or
                np.max(np.abs(np.diff(np.r_[previous, command]))) > slew+1e-5):
            failures.append(f"{context}: logged future input violates bound")
        current_peak = float(np.max(prediction[:, 3]))
        max_logged_prediction = max(max_logged_prediction, current_peak)
        if current_peak > bound+2.1e-6:
            failures.append(f"{context}: nonlinear predicted P2 exceeds {bound} MPa")
        if not (_close(log.get("predicted_peak_p2_mpa"), current_peak)
                and _close(data["predicted_peak_p2_mpa"][index], current_peak)):
            failures.append(f"{context}: recorded predicted peak differs from nodes")
        prediction_violation = _finite_float(log.get("prediction_violation_mpa"))
        input_violation = _finite_float(log.get("input_violation_v"))
        if (not math.isfinite(prediction_violation)
                or not math.isfinite(input_violation)
                or prediction_violation > 2.1e-6 or input_violation > 1.1e-5):
            failures.append(f"{context}: nonlinear validation result missing or excessive")
        bounds = log.get("pressure_constraints") or []
        if not any(item.get("state") == 3 and _close(item.get("upper"), bound)
                   for item in bounds if isinstance(item, dict)) and not restored:
            failures.append(f"{context}: P2 hard constraint not present in QP metadata")
    audit.check("all applied commands have nonlinear validation or declared policy",
                not failures, case=case, detail=failures[:15] if failures else None)
    audit.check("restoration count matches logs", restoration_count ==
                int(np.sum(codes == 3)), case=case)
    audit.check("finite-grid refinement count matches logs", refinement_count ==
                int(np.sum(codes == 4)), case=case)
    return max_logged_prediction if math.isfinite(max_logged_prediction) else None


def _dense_sample_consistency(t, sampled, trace_t, trace_p):
    # Accepted Radau nodes, duplicate interval boundaries, and extrema roots
    # may share timestamps. Check any physical pressure at each control sample.
    largest = 0.
    for stamp, pressure in zip(t, sampled):
        i = int(np.searchsorted(trace_t, stamp))
        nearby = trace_p[max(0, i-3):min(len(trace_t), i+4)]
        nearby_t = trace_t[max(0, i-3):min(len(trace_t), i+4)]
        coincident = nearby[np.abs(nearby_t-stamp) <= 1e-9]
        if not len(coincident):
            return math.inf
        largest = max(largest, float(np.min(np.abs(coincident-pressure))))
    return largest


def _replay(audit, case, data, summary_cfg, safety_limit):
    strict_rtol = min(float(summary_cfg["rtol"])/10., 1e-8)
    plant = HydraulicPlant(config=summary_cfg["physical_parameters"],
                           origin_m=summary_cfg["origin_m"],
                           valve_tau=summary_cfg["valve_tau"],
                           makeup_area=summary_cfg["makeup_area"], rtol=strict_rtol)
    plant.twin.pressure_trace_subdivisions = 64
    plant.reset(summary_cfg["initial_state"])
    columns = [plant.state.copy()]
    commands = data["u_v"][:-1]
    try:
        for command in commands:
            plant.step(float(command), summary_cfg["ts"])
            columns.append(plant.state.copy())
    except Exception as exc:
        audit.check("tight-tolerance plant replay", False, case=case,
                    detail=f"integration failed: {exc}")
        return
    replay = np.asarray(columns)
    recorded = np.column_stack([data[key] for key in STATE_FIELDS])
    deviation = np.max(np.abs(replay-recorded), axis=0)
    max_p2 = float(np.max(plant.pressure_trace[:, 2])/1e6)
    audit.report["cases"][str(case)]["strict_replay"] = dict(
        relative_tolerance=strict_rtol, pressure_subdivisions=64,
        max_abs_state_deviation=dict(zip(STATE_FIELDS, deviation.tolist())),
        dense_peak_p2_mpa=max_p2,
        end_time_s=float(plant.t))
    thresholds = np.array([.05, .02, .05, .05, .01])
    audit.check("tight-tolerance replay agrees with recorded states",
                bool(np.all(deviation <= thresholds)), case=case,
                detail=dict(max_abs=deviation.tolist(), tolerance=thresholds.tolist()))
    audit.check("tight-tolerance replay dense P2 <= safety limit",
                max_p2 <= safety_limit+1e-6, case=case,
                detail={"peak_mpa": max_p2, "limit_mpa": safety_limit})


def validate_outputs(output_dir=None, strict=False):
    folder = Path(output_dir) if output_dir is not None else HERE/"outputs"
    audit = Audit(strict)
    report = audit.report
    # Read the frozen experiment parameters directly: importing the simulator
    # would require its optimizer and would make this audit depend on the code
    # that produced the data being checked.
    config = json.loads((HERE/"experiment_config.json").read_text(encoding="utf-8"))
    audit.check("seven requested prediction horizons", tuple(config["horizons"]) ==
                EXPECTED_HORIZONS, detail=config["horizons"])
    summary_path = folder/"summary.json"
    if not summary_path.is_file():
        audit.check("summary.json exists", False, detail=str(summary_path))
        return report
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        audit.check("summary.json readable", False, detail=str(exc))
        return report
    saved_cfg = summary.get("config", {})
    for name in config:
        audit.check(f"shared configuration: {name}", saved_cfg.get(name) == config[name],
                    detail={"saved": saved_cfg.get(name), "current": config[name]})
    rhs_path = HERE/"hydraulic_digital_twin.py"
    expected_hash = hashlib.sha256(rhs_path.read_bytes()).hexdigest()
    audit.check("same physical RHS source hash", saved_cfg.get("physical_rhs_sha256") ==
                expected_hash, detail=expected_hash)
    plant = HydraulicPlant(origin_m=config["origin_m"],
                           valve_tau=config["valve_tau"],
                           makeup_area=config["makeup_area"], rtol=config["rtol"])
    audit.check("shared physical parameters match current model",
                saved_cfg.get("physical_parameters") == plant.parameters)
    audit.check("6D augmented state convention", saved_cfg.get("n_augmented_states") == 6)
    results = summary.get("results", {})
    audit.check("summary contains all seven runs", set(results) ==
                {str(h) for h in EXPECTED_HORIZONS}, detail=sorted(results))
    prefix_record = None
    dense_limit = float(config["p_safe"])
    for case in EXPECTED_HORIZONS:
        key = str(case)
        case_report = report["cases"].setdefault(key, dict(checks=[]))
        stem = f"N{case:02d}"
        paths = (folder/f"{stem}.csv", folder/f"{stem}_pressure_trace.csv",
                 folder/f"{stem}_solver_logs.json")
        if not audit.check("CSV, dense pressure, and solver logs exist",
                           all(p.is_file() for p in paths), case=case,
                           detail=[str(p) for p in paths if not p.is_file()]):
            continue
        try:
            data, trace = _load_csv(paths[0]), _load_csv(paths[1])
            logs = json.loads(paths[2].read_text(encoding="utf-8"))
            metrics = results[key]
        except (ValueError, KeyError, OSError, TypeError) as exc:
            audit.check("case files parse", False, case=case, detail=str(exc))
            continue
        if not audit.check("required CSV columns present",
                           set(("t_s", "reference_mm", *STATE_FIELDS, "u_v",
                                "delta_u_v", "predicted_peak_p2_mpa", "status_code",
                                *TIME_FIELDS)).issubset(data) and
                           {"t_s", "p1_mpa", "p2_mpa"}.issubset(trace), case=case):
            continue
        t, p_t = data["t_s"], trace["t_s"]
        expected_n = round(config["duration"]/config["ts"])+1
        complete = (metrics.get("status") == "completed" and metrics.get("failure") is None
                    and len(t) == expected_n and _close(t[-1], config["duration"], atol=1e-10))
        audit.check("full closed-loop simulation completed", complete, case=case,
                    detail={"status": metrics.get("status"), "samples": len(t),
                            "expected_samples": expected_n, "last_t_s": float(t[-1])})
        audit.check("uniform 1 ms control clock", len(t) == expected_n and
                    bool(np.allclose(t, np.arange(len(t))*config["ts"], atol=1e-10)),
                    case=case)
        all_finite = all(np.all(np.isfinite(values)) for values in data.values())
        audit.check("all sampled fields finite", all_finite, case=case)
        audit.check("dense pressure finite and positive",
                    bool(np.all(np.isfinite(p_t)) and
                         np.all(np.isfinite(trace["p1_mpa"])) and
                         np.all(np.isfinite(trace["p2_mpa"])) and
                         np.min(trace["p1_mpa"]) > 0 and
                         np.min(trace["p2_mpa"]) > 0 and
                         np.all(np.diff(p_t) >= -1e-11)), case=case)
        audit.check("dense trace covers complete control interval",
                    _close(p_t[0], t[0], atol=1e-10) and
                    _close(p_t[-1], t[-1], atol=1e-10) and len(p_t) > len(t), case=case,
                    detail={"dense_points": len(p_t), "control_points": len(t)})
        max_sample_error = _dense_sample_consistency(
            t, data["p2_mpa"], p_t, trace["p2_mpa"])
        audit.check("sampled P2 agrees with dense trace", max_sample_error <= 1e-5,
                    case=case, detail={"max_error_mpa": max_sample_error})
        reference = np.where(t >= config["step_time"]-1e-12,
                             config["target_mm"], 0.)
        audit.check("identical prescribed step reference",
                    bool(np.allclose(data["reference_mm"], reference, atol=1e-10)),
                    case=case)
        initial = np.array(saved_cfg.get("initial_state", [math.nan]*5), dtype=float)
        first_state = np.array([data[name][0] for name in STATE_FIELDS])
        audit.check("shared initial physical state",
                    bool(initial.shape == (5,) and np.allclose(first_state, initial,
                                                               atol=1e-10)), case=case)
        idle = t < config["step_time"]-1e-12
        idle_state = np.column_stack([data[name][idle] for name in STATE_FIELDS])
        if prefix_record is None:
            prefix_record = idle_state
        else:
            audit.check("shared unforced initial trajectory",
                        idle_state.shape == prefix_record.shape and
                        bool(np.allclose(idle_state, prefix_record, atol=1e-10)),
                        case=case)
        limit = 10.
        valid_u = data["u_v"][np.isfinite(data["u_v"])]
        max_voltage = float(np.max(np.abs(valid_u))) if len(valid_u) else math.inf
        max_slew = float(np.max(np.abs(np.diff(np.r_[0., valid_u])))) if len(valid_u) else math.inf
        audit.check("applied valve voltage within ±10 V", max_voltage <= limit+1e-5,
                    case=case, detail={"max_abs_v": max_voltage})
        audit.check("applied voltage slew within configured limit",
                    max_slew <= config["max_du_step"]+1e-5, case=case,
                    detail={"max_abs_delta_v": max_slew,
                            "limit_v_per_sample": config["max_du_step"]})
        if all_finite:
            increment_error = float(np.max(np.abs(data["delta_u_v"]-
                                                 np.diff(np.r_[0., data["u_v"]]))))
            audit.check("recorded voltage increments match applied input",
                        increment_error <= 1e-7, case=case,
                        detail={"max_error_v": increment_error})
        peak = float(np.max(trace["p2_mpa"]))
        case_report["observed_dense_peak_p2_mpa"] = peak
        audit.check("true dense P2 <= 21 MPa", peak <= dense_limit+1e-6,
                    case=case, detail={"peak_mpa": peak, "limit_mpa": dense_limit})
        audit.check("sampled P2 does not exceed dense peak",
                    float(np.max(data["p2_mpa"])) <= peak+1e-6, case=case)
        if isinstance(logs, list):
            case_report["max_logged_predicted_p2_mpa"] = _validate_logs(
                audit, case, data, logs, config)
        else:
            audit.check("solver log is a list", False, case=case)
        if complete and all_finite:
            _validate_summary(audit, case, metrics, data, trace, config)
        if strict and case in (5, 20, 50) and complete and all_finite:
            _replay(audit, case, data, saved_cfg, dense_limit)
    report["passed"] = not report["errors"]
    report["checked_horizons"] = list(EXPECTED_HORIZONS)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE/"outputs")
    parser.add_argument("--strict", action="store_true",
                        help="重放 N=5,20,50 原始输入并检验更严格积分的真实峰压")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        report = validate_outputs(args.output_dir, args.strict)
    except Exception as exc:
        report = dict(passed=False, strict_replay=args.strict,
                      errors=[f"unexpected verifier error: {type(exc).__name__}: {exc}"])
    path = args.output_dir/"validation.json"
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2,
                               allow_nan=False),
                    encoding="utf-8")
    if report["passed"]:
        print(f"PASS: seven completed MPC runs verified; {path}")
    else:
        print(f"FAIL: {len(report['errors'])} audit findings; {path}", file=sys.stderr)
        for message in report["errors"][:30]:
            print("  " + message, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
