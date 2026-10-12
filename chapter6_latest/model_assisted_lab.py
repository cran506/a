"""第六章课堂入口：相同物理系统上的三种模型辅助控制策略。"""
import argparse
import csv
import json
from pathlib import Path

from plotting import plot_comparison
from simulation import Experiment, simulate


ROOT = Path(__file__).resolve().parent
STRATEGIES = (("pid", False, False), ("nominal", True, False), ("full", True, True))


def run_experiments(*, known_load=True, rtol=1e-5):
    results = {}
    for name, enable_ff, enable_gain_sched in STRATEGIES:
        config = Experiment(name, enable_ff=enable_ff, enable_gain_sched=enable_gain_sched,
                            known_load=known_load, rtol=rtol)
        result = simulate(config)
        results[name] = result
        metrics = result["metrics"]
        print(f"{name}: 最大误差 {metrics['max_abs_error_mm']:.6f} mm；"
              f"接触最低速度 {metrics['contact_min_velocity_m_s']:.6f} m/s；"
              f"起动 {metrics['motion_start_ms']:.0f} ms", flush=True)
    return results


def save_data(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key, result in results.items():
        fields = list(result["data"])
        with (output / f"{key}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(zip(*(result["data"][field] for field in fields)))
    summary = {
        "scenario": "known_planned_process_load" if results["full"]["config"]["known_load"] else "unknown_load",
        "description": "Independent causal controllers; identical plant, PID, reference and sensor; actual Radau integration.",
        "units": "SI except metrics whose names explicitly state mm, ms, MPa, V or percent.",
        "strategies": {key: {"config": result["config"], "metrics": result["metrics"]}
                       for key, result in results.items()},
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-data", action="store_true", help="另存三组原始 CSV 和指标 JSON")
    parser.add_argument("--unknown-load", action="store_true", help="外载不告知前馈，用于检查适用条件")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Radau 相对容差，默认 1e-5")
    parser.add_argument("--output", type=Path, help="输出目录，默认当前代码旁的 outputs")
    args = parser.parse_args()
    if not 0 < args.rtol < 1:
        parser.error("rtol 必须在 (0,1) 之间")
    output = args.output or ROOT / ("outputs_unknown" if args.unknown_load else "outputs")
    results = run_experiments(known_load=not args.unknown_load, rtol=args.rtol)
    figure = plot_comparison(results, output)
    if args.save_data:
        save_data(results, output)
    print(f"结果图：{figure}", flush=True)


if __name__ == "__main__":
    main()
