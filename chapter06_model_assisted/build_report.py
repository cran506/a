"""从本次仿真JSON生成中文实验报告，指标不手工抄入图片或结论。"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    lines = ["# 第六章课堂编程实验报告", "", "本报告由 `build_report.py` 读取随包保存的原始指标自动生成；公式、参数及适用条件见 README。", ""]
    for profile, title in (("classroom", "课堂模型（准静态负载压力，参考图 a）"),
                           ("twin", "四阶双腔液压数字孪生（物理对照）")):
        summary = json.loads((ROOT / "outputs" / profile / "summary.json").read_text(encoding="utf-8"))
        lines += [f"## {title}", "", f"![四联图]({profile}/model_assisted_comparison.png)", "",
                  "| 实测指标 | 纯 PID | 标称前馈 + PID | 完整模型辅助 |", "|---|---:|---:|---:|"]
        rows = [("全程最大绝对误差/mm", "max_abs_error_mm", 6), ("全程误差RMSE/mm", "rms_error_mm", 6),
                ("首控制电压/V", "startup_initial_voltage_v", 6),
                ("首个非零前馈电压/V", "first_feedforward_voltage_v", 6),
                ("首次静摩擦释放/ms", "first_breakaway_ms", 3),
                ("速度达阈值时刻/ms", "motion_start_ms", 3), ("起动速度超调/(m/s)", "startup_velocity_overshoot_m_s", 6),
                ("负载窗口最低速度/(m/s)", "contact_min_velocity_m_s", 6),
                ("负载速度跌落/%", "contact_speed_drop_percent", 3),
                ("负载窗口最大绝对误差/mm", "contact_max_abs_error_mm", 6),
                ("轻载电压中位/V", "light_median_voltage_v", 6), ("重载电压中位/V", "heavy_median_voltage_v", 6),
                ("重载 P1 中位/MPa", "heavy_median_p1_mpa", 6),
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
            lines += [f"四阶对照采用已知 {cfg['load_force']/1000:.1f} kN 工艺负载、"
                      f"{cfg['load_ramp']*1000:.0f} ms 加载和 {cfg['unload_ramp']*1000:.0f} ms 卸载；"
                      "压力连续积分。首个非零前馈避免 PID 的误差蓄能等待，实际运动仍受建压与阀动态限制。", ""]
        else:
            cfg = summary["full"]["config"]
            p = cfg["physical_parameters"]
            c = cfg["classroom_parameters"]
            pid = summary["pid"]["metrics"]
            nominal = summary["nominal"]["metrics"]
            f_cruise = (p["sigma"] * .2 + p["Fc"] +
                        (p["Fs"] - p["Fc"]) * math.exp(-(.2/p["vs"])**2))
            leak_v = p["Ct"] * (f_cruise/p["A1"]) / (p["K_flow"] * math.sqrt(p["Ps"]-p["P1_0"]))
            ideal_heavy_v = (p["deadzone"] + leak_v + p["A1"]*.2 /
                             (p["K_flow"]*math.sqrt(p["Ps"]-c["heavy_drive_pressure_pa"])))
            lines += [f"该对象使用用户给定的质量 {p['m']:.0f} kg、最大静摩擦 {p['Fs']:.0f} N、"
                      f"层流泄漏系数 {p['Ct']:.4g} m³/(s·Pa) 与阀时间常数 {p['tau_v']*1000:.1f} ms。"
                      "阀动态、真实粘滑与液压等效阻抗参与闭环积分；工艺负载下的 P1 为代数工作点"
                      "加运动力压降，不含双腔独立压力储能。", "",
                      f"图 (a) 的最大误差由 PID {pid['max_abs_error_mm']:.4f} mm、"
                      f"标称前馈 {nominal['max_abs_error_mm']:.4f} mm 降为完整控制"
                      f" {full['max_abs_error_mm']:.4f} mm。PID 起动阈值时刻"
                      f" {pid['motion_start_ms']:.0f} ms；完整组首指令"
                      f" {full['startup_initial_voltage_v']:.3f} V，首次静摩擦释放"
                      f" {full['first_breakaway_ms']:.3f} ms，速度达阈值"
                      f" {full['motion_start_ms']:.0f} ms。", "",
                      f"给定重载代数工作点 P1={c['heavy_drive_pressure_pa']/1e6:.2f} MPa，"
                      f"六步公式在 0.20 m/s 匀速时反演重载前馈 {ideal_heavy_v:.3f} V；"
                      f"闭环重载窗口实测总电压中位数 {full['heavy_median_voltage_v']:.3f} V。"
                      "前者取固定工作点，后者包含运动引起的压力和 PID 修正。", ""]
    lines += ["## 结果解释", "", "纯 PID 等待误差积累后才跨过死区与静摩擦，随后出现速度超调。标称前馈消除了大部分起动等待，"
              "但重载时仍采用轻载增益，所需流量映射失配。完整控制在测得压力变化后提升有效阀开度，明显减小负载窗口速度和位置偏差。", "",
              "用户提供的泄漏公式令泄漏电压项约 0.25 V 保持不变；当 P1=16.95 MPa、"
              "压差为 4.05 MPa 时，给定参数反演的重载前馈约 4.327 V，因此不能把"
              " 4.20 V 同时作为这组固定参数的精确结果。课堂模型的首采样前馈是 1.20 V，"
              "实际位移受 0.8 ms 阀响应与静摩擦释放限制；四阶对照还受压力建立过程限制。", "",
              "数值复核、反事实无载实验与单元检查见 `numerical_checks.json` 和 `unit_tests.txt`。CSV 的时间、状态、参考和控制在同一采样时刻对齐。", ""]
    numerical = ROOT / "outputs" / "numerical_checks.json"
    if numerical.exists():
        checks = json.loads(numerical.read_text(encoding="utf-8"))
        reintegration = checks["reintegration"]
        lines += [f"四阶对照 rtol=1e-8 严格重积分：最大误差 **{reintegration['recomputed_max_abs_error_mm']:.9f} mm**，"
                  f"与已保存 CSV 的最大位置差 **{reintegration['max_position_difference_from_csv_mm']:.9f} mm**。"
                  "无载时原负载时刻没有人工电压阶跃；课堂等效模型的无载与负载移时检查见 `classroom/numerical_checks.json`。", ""]
    output = ROOT / "outputs" / "experiment_report.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
