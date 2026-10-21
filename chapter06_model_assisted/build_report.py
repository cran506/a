"""从本次仿真JSON生成中文实验报告，指标不手工抄入图片或结论。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    lines = ["# 第六章课堂编程实验报告", "", "本报告由 `build_report.py` 读取随包保存的原始指标自动生成；公式、参数及适用条件见 README。", ""]
    for profile, title in (("twin", "四阶液压数字孪生（主实验）"), ("classroom", "课堂等效模型（准静态压力）")):
        summary = json.loads((ROOT / "outputs" / profile / "summary.json").read_text(encoding="utf-8"))
        lines += [f"## {title}", "", f"![四联图]({profile}/model_assisted_comparison.png)", "",
                  "| 实测指标 | 纯 PID | 标称前馈 + PID | 完整模型辅助 |", "|---|---:|---:|---:|"]
        rows = [("全程最大绝对误差/mm", "max_abs_error_mm", 6), ("全程误差RMSE/mm", "rms_error_mm", 6),
                ("首控制电压/V", "startup_initial_voltage_v", 6), ("首次静摩擦释放/ms", "first_breakaway_ms", 3),
                ("速度达阈值时刻/ms", "motion_start_ms", 3), ("起动速度超调/(m/s)", "startup_velocity_overshoot_m_s", 6),
                ("负载窗口最低速度/(m/s)", "contact_min_velocity_m_s", 6),
                ("负载速度跌落/%", "contact_speed_drop_percent", 3),
                ("负载窗口最大绝对误差/mm", "contact_max_abs_error_mm", 6),
                ("轻载电压中位/V", "light_median_voltage_v", 6), ("重载电压中位/V", "heavy_median_voltage_v", 6),
                ("流量增益下降/%", "kq_decline_percent", 3), ("峰值绝对电压/V", "peak_abs_voltage_v", 4)]
        for label, field, places in rows:
            values = [summary[k]["metrics"].get(field) for k in ("pid", "nominal", "full")]
            formatted = [f"{v:.{places}f}" if v is not None else "—" for v in values]
            lines.append("| " + " | ".join([label] + formatted) + " |")
        full = summary["full"]["metrics"]
        lines += ["", f"完整组全程最大误差 **{full['max_abs_error_mm']:.6f} mm**，"
                  f"0.05 mm 验收：**{'通过' if full['max_abs_error_mm'] < .05 else '未通过'}**。", ""]
        if profile == "twin":
            cfg = summary["full"]["config"]
            lines += [f"主实验采用已知 {cfg['load_force']/1000:.1f} kN 工艺负载、"
                      f"{cfg['load_ramp']*1000:.0f} ms 加载和 {cfg['unload_ramp']*1000:.0f} ms 卸载；"
                      "压力连续积分。首个非零前馈避免 PID 的误差蓄能等待，实际运动仍受建压与阀动态限制。", ""]
        else:
            lines += ["该对象使用代数压力工作点和 60 kg 等效质量；不含独立压力储能和阀动态。"
                      "1.2、2.5、4.2 V 是公开标定值，速度跌落、起动延迟和误差是闭环计算结果。", ""]
    lines += ["## 结果解释", "", "纯 PID 等待误差积累后才跨过死区与静摩擦，随后出现速度超调。标称前馈消除了大部分起动等待，"
              "但重载时仍采用轻载增益，所需流量映射失配。完整控制在测得压力变化后提升有效阀开度，明显减小负载窗口速度和位置偏差。", "",
              "参考图中 50% 增益衰减与 2.5→4.2 V 并不同时满足含 0.4 V 死区的逆映射；本项目保留量纲与一致数值，未强制覆盖测量结果。"
              "主实验起动延迟不是课件所述 1 ms，加载也不是未知理想阶跃。", "",
              "数值复核、反事实无载实验与单元检查见 `numerical_checks.json` 和 `unit_tests.txt`。CSV 的时间、状态、参考和控制在同一采样时刻对齐。", ""]
    numerical = ROOT / "outputs" / "numerical_checks.json"
    if numerical.exists():
        checks = json.loads(numerical.read_text(encoding="utf-8"))
        reintegration = checks["reintegration"]
        lines += [f"主实验 rtol=1e-8 严格重积分：最大误差 **{reintegration['recomputed_max_abs_error_mm']:.9f} mm**，"
                  f"与已保存 CSV 的最大位置差 **{reintegration['max_position_difference_from_csv_mm']:.9f} mm**。"
                  "无载时原负载时刻没有人工电压阶跃；课堂等效模型的无载与负载移时检查见 `classroom/numerical_checks.json`。", ""]
    output = ROOT / "outputs" / "experiment_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
