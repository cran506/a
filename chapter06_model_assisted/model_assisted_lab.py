"""《智能流体传动与控制》第六章：三策略闭环仿真、数据与四联图。

python model_assisted_lab.py                  # 四阶孪生（默认）
python model_assisted_lab.py --model classroom # 课堂等效模型
python model_assisted_lab.py --model all       # 两种模型
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter06-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "chapter06-cache"))

from model_assisted_controller import ModelAssistedController
from plotting import plot_comparison
from simulation import Experiment, simulate

ROOT = Path(__file__).resolve().parent
STRATEGIES = (("pid", False, False), ("nominal", True, False), ("full", True, True))


def save_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        data = result["data"]
        with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(data)
            writer.writerows(zip(*data.values()))
    summary = {key: {"config": value["config"], "metrics": value["metrics"]}
               for key, value in results.items()}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    checks = check_results(results)
    (output / "validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return checks


def check_results(results):
    """直接从积分状态检验物理/精度约束；不把参考目标改写成实测值。"""
    import numpy as np
    checks = {}
    base = results["full"]["data"]
    for key, result in results.items():
        d = result["data"]
        checks[f"{key}_finite"] = bool(all(np.all(np.isfinite(v)) for v in d.values()))
        checks[f"{key}_same_reference"] = bool(np.array_equal(d["reference_m"], base["reference_m"]))
        checks[f"{key}_voltage_within_10v"] = bool(np.max(abs(d["voltage_v"])) <= 10 + 1e-10)
        checks[f"{key}_positive_pressure"] = bool(min(d["p1_pa"].min(), d["p2_pa"].min()) > 0)
        checks[f"{key}_error_is_reference_minus_state"] = bool(np.allclose(
            d["error_m"], d["reference_m"] - d["position_m"], rtol=0, atol=1e-12))
    checks["full_max_error_below_0p05mm"] = bool(np.max(abs(base["error_m"])) * 1000 < .05)
    checks["full_better_than_nominal_at_load"] = bool(
        results["full"]["metrics"]["contact_max_abs_error_mm"] < results["nominal"]["metrics"]["contact_max_abs_error_mm"])
    checks["passed"] = all(checks.values())
    return checks


def run_twin(*, known_load=True, rtol=1e-6):
    results = {}
    for key, ff, schedule in STRATEGIES:
        result = simulate(Experiment(key, enable_ff=ff, enable_gain_sched=schedule,
                                     known_load=known_load, rtol=rtol))
        result["config"]["model"] = "four_state_hydraulic_twin"
        results[key] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("twin", "classroom", "all"), default="twin")
    parser.add_argument("--unknown-load", action="store_true", help="四阶模型：不向控制器提供已知负载计划")
    parser.add_argument("--rtol", type=float, default=1e-6, help="四阶模型 Radau 容差")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--save-data", action="store_true", help="兼容参数；CSV与指标默认始终保存")
    args = parser.parse_args()
    if not 0 < args.rtol < 1:
        parser.error("rtol 必须在 (0,1) 内")
    if args.unknown_load and args.model != "twin":
        parser.error("--unknown-load 适用于 --model twin")
    profiles = ("twin", "classroom") if args.model == "all" else (args.model,)
    failed = False
    for profile in profiles:
        if profile == "twin":
            results = run_twin(known_load=not args.unknown_load, rtol=args.rtol)
        else:
            from classroom_model import run_classroom
            results = run_classroom()
        folder = args.output / (profile + ("_unknown" if args.unknown_load else ""))
        checks = save_results(results, folder)
        failed = failed or not checks["passed"]
        path = plot_comparison(results, folder)
        for key, result in results.items():
            m = result["metrics"]
            print(f"{profile}/{key}: max|e|={m['max_abs_error_mm']:.6f} mm; "
                  f"起动={m['motion_start_ms']} ms; 负载 vmin={m['contact_min_velocity_m_s']:.6f} m/s", flush=True)
        print(f"图：{path}\n本次检查：{checks['passed']}", flush=True)
    if failed and not args.unknown_load:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
