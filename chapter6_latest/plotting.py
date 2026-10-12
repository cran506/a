"""第六章课堂三策略的真实数据四联图；只保存一个 PNG。"""
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter6-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "chapter6-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


STRATEGIES = (
    ("pid", "纯 PID 反馈", "#1261bb", (0, (5, 3))),
    ("nominal", "标称前馈 + PID", "#e38a13", (0, (5, 2, 1, 2))),
    ("full", "完整模型辅助", "#d42424", "-"),
)


def _number(value, digits=3):
    if value is None:
        return "未观测到"
    value = float(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else "无有效值"


def _font_settings():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if name in installed:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _validate(results):
    required = ("time_s", "reference_m", "reference_velocity_m_s", "error_m",
                "velocity_m_s", "voltage_v")
    for key, _, _, _ in STRATEGIES:
        if key not in results:
            raise ValueError(f"缺少 {key} 策略结果")
        data = results[key]["data"]
        t = np.asarray(data["time_s"])
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError(f"{key} 时间刻度必须严格递增")
        for field in required:
            values = np.asarray(data[field])
            if values.shape != t.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{key}.{field} 必须为与时间同步的有限数组")
        if t[0] > 0. or t[-1] < .62:
            raise ValueError("数据必须覆盖起动和负载观察窗口")
    baseline = results["pid"]["data"]
    for key in ("nominal", "full"):
        for field in ("time_s", "reference_m", "reference_velocity_m_s"):
            if not np.array_equal(baseline[field], results[key]["data"][field]):
                raise ValueError("三策略必须使用相同时间刻度与参考轨迹")


def _style_axes(axes):
    for ax in axes.flat:
        ax.grid(True, color="#bbc0c5", alpha=.55, ls=":", lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 仅由实际曲线确定范围，包含负速度和电压极值。
        ax.margins(y=.13)


def _curve_minimum(data, start, end, field):
    indices = np.flatnonzero((data["time_s"] >= start) & (data["time_s"] <= end))
    index = indices[np.argmin(np.asarray(data[field])[indices])]
    return float(data["time_s"][index]), float(data[field][index])


def _arrow(ax, text, point, location, color):
    ax.annotate(text, xy=point, xycoords="data", xytext=location,
                textcoords="axes fraction", fontsize=9, color=color,
                ha="left", va="top", annotation_clip=True,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.1},
                bbox={"boxstyle": "round,pad=.35", "facecolor": "white",
                      "edgecolor": color, "alpha": .94})


def _footer(results):
    full = results["full"]
    cfg, data = full["config"], full["data"]
    t = data["time_s"]
    rv = data["reference_velocity_m_s"]
    moving = np.flatnonzero(np.abs(rv) > 1e-12)
    motion_end = float(t[min(moving[-1] + 1, len(t)-1)]) if len(moving) else 0.
    distance_mm = float(data["reference_m"][-1]) * 1000
    ts_ms = float(cfg["ts"]) * 1000
    bits = int(cfg["adc_bits"])
    tau_ms = float(cfg["valve_tau"]) * 1000
    load_ramp_ms = float(cfg["load_ramp"]) * 1000
    first = (f"共同参考：{distance_mm:g} mm / {motion_end:g} s，"
             f"速度平台 {np.max(rv):g} m/s；Ts={ts_ms:g} ms；"
             f"{bits} bit ADC，LSB={float(cfg['sensor_lsb_mm']):.6f} mm。")
    known = bool(cfg.get("known_load", False))
    scenario = "前馈使用已知工艺计划，与加载模型匹配" if known else "未使用外载前馈"
    second = (f"{scenario}；载荷 {float(cfg['load_force'])/1000:g} kN，"
              f"{float(cfg['load_start']):g}–{float(cfg['load_end']):g} s，"
              f"加载/卸载过渡 {load_ramp_ms:g}/{float(cfg['unload_ramp'])*1000:g} ms；腔压、速度由真实 ODE 连续演化。")
    third = (f"阀 τ={tau_ms:g} ms；两类前馈共同采用一阶阀的 ZOH 逆模型；"
             "压力调度使用传感压力。所有注释来自本次计算，非指定目标值。")
    prediction = float(cfg.get("gain_prediction_s", 0.)) * 1000
    if prediction:
        third += f" 完整组含 {prediction:g} ms 工艺导数增益预测。"
    return "\n".join((first, second, third))


def plot_comparison(results, output):
    """保存 model_assisted_comparison.png，返回其 Path。

    results={pid/nominal/full: {data, metrics, config}}；output 为输出目录。
    本函数不运行仿真、不平滑数据，也不以材料中的理想指标替换测量结果。
    """
    _validate(results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"font.family": "sans-serif", "font.sans-serif": _font_settings(),
                "axes.unicode_minus": False, "font.size": 10,
                "figure.dpi": 120, "savefig.dpi": 180}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.4))
        error_ax, voltage_ax, start_ax, load_ax = axes.flat
        fig.suptitle("第六章 模型辅助复合控制：起动补偿与重载流量增益实时线性化",
                     fontsize=17, y=.978)
        base = results["pid"]["data"]
        time = base["time_s"]
        for ax in (start_ax, load_ax):
            lo, hi = ((0., .16) if ax is start_ax else (.48, .62))
            mask = (time >= lo) & (time <= hi)
            ax.plot(time[mask], base["reference_velocity_m_s"][mask],
                    color="#858585", ls=":", lw=2.4, label="参考速度 $v_d$")
        for key, label, color, dash in STRATEGIES:
            data = results[key]["data"]
            width = 1.8 if key == "full" else 1.45
            error_ax.plot(time, np.asarray(data["error_m"]) * 1000,
                          color=color, ls=dash, lw=width, label=label)
            voltage_ax.plot(time, data["voltage_v"], color=color, ls=dash,
                            lw=width, label=label)
            velocity_color = "#141414" if key == "full" else color
            for ax, lo, hi in ((start_ax, 0., .16), (load_ax, .48, .62)):
                mask = (time >= lo) & (time <= hi)
                ax.plot(time[mask], np.asarray(data["velocity_m_s"])[mask],
                        color=velocity_color, ls=dash, lw=width, label=label)
        load_start = float(results["full"]["config"]["load_start"])
        load_end = float(results["full"]["config"]["load_end"])
        for ax in (error_ax, voltage_ax):
            ax.axvspan(load_start, load_end, color="#d2d5d8", alpha=.25)
            ax.set_xlim(float(time[0]), float(time[-1]))
        load_ax.axvline(load_start, color="#888888", ls=":", lw=1.)
        start_ax.set_xlim(0., .16)
        start_ax.set_xticks(np.arange(0., .16001, .04))
        load_ax.set_xlim(.48, .62)
        load_ax.set_xticks(np.arange(.48, .62001, .02))
        error_ax.axhline(0., color="#888888", lw=.65, ls=":")
        error_ax.set_title("(a) 全程真实跟踪误差", loc="left", fontsize=12, pad=9)
        voltage_ax.set_title("(b) 全程控制电压", loc="left", fontsize=12, pad=9)
        start_ax.set_title("(c) 起动速度：0–0.16 s", loc="left", fontsize=12, pad=9)
        load_ax.set_title("(d) 负载切换速度：0.48–0.62 s", loc="left", fontsize=12, pad=9)
        error_ax.set_ylabel("真实误差 $e$ (mm)")
        voltage_ax.set_ylabel("总控制电压 $u$ (V)")
        start_ax.set_ylabel("实际速度 $v$ (m/s)")
        load_ax.set_ylabel("实际速度 $v$ (m/s)")
        for ax in axes.flat:
            ax.set_xlabel("仿真时间 $t$ (s)")
        _style_axes(axes)

        # 只读取仿真指标，不将 PPT/Word 的理想数字写成已测结果。
        metrics = {key: results[key]["metrics"] for key, _, _, _ in STRATEGIES}
        error_lines = ["全程最大真实误差 |e|："]
        for key, label, _, _ in STRATEGIES:
            error_lines.append(f"{label}  {_number(metrics[key]['max_abs_error_mm'])} mm")
        error_ax.text(.975, .97, "\n".join(error_lines), ha="right", va="top",
                      transform=error_ax.transAxes, fontsize=9,
                      bbox={"boxstyle": "round,pad=.4", "fc": "white",
                            "ec": "#b4b8bd", "alpha": .93})
        full_metrics = metrics["full"]
        first_ff = full_metrics.get("first_feedforward_ms")
        first_ff_voltage = full_metrics.get("first_feedforward_voltage_v")
        if first_ff is not None and first_ff_voltage is not None:
            data = results["full"]["data"]
            index = int(np.argmin(np.abs(time - first_ff / 1000)))
            _arrow(voltage_ax,
                   f"完整组首次前馈\n{_number(first_ff_voltage)} V @ {_number(first_ff, 1)} ms",
                   (float(time[index]), float(data["voltage_v"][index])),
                   (.48, .24), "#d42424")
        threshold = full_metrics["startup_velocity_threshold_m_s"]
        start_lines = [f"运动阈值 |v| > {threshold:g} m/s",
                       f"参考达阈值：{_number(full_metrics.get('reference_threshold_ms'), 1)} ms"]
        for key, label, _, _ in STRATEGIES:
            start_lines.append(f"{label}：{_number(metrics[key].get('motion_start_ms'), 1)} ms")
        start_ax.text(.035, .96, "\n".join(start_lines), transform=start_ax.transAxes,
                      ha="left", va="top", fontsize=8.5,
                      bbox={"boxstyle": "round,pad=.35", "fc": "white",
                            "ec": "#b4b8bd", "alpha": .93})
        for key, label, color, location in (
            ("nominal", "标称前馈", "#cf7705", (.06, .29)),
            ("full", "完整模型辅助", "#141414", (.54, .53)),
        ):
            data = results[key]["data"]
            # 指标窗口是 [0.50,0.62]；只从同一窗口找对应箭头点。
            point = _curve_minimum(data, .50, .62, "velocity_m_s")
            m = metrics[key]
            _arrow(load_ax,
                   f"{label}\n最低速度 {_number(m['contact_min_velocity_m_s'])} m/s\n"
                   f"相对参考跌落 {_number(m['contact_speed_drop_percent'], 1)}%",
                   point, location, color)
        error_ax.legend(loc="lower left", fontsize=9, framealpha=.93)
        voltage_ax.legend(loc="upper right", fontsize=9, framealpha=.93)
        start_ax.legend(loc="lower right", fontsize=8.5, framealpha=.93)
        load_ax.legend(loc="upper right", fontsize=8.5, framealpha=.93)
        fig.subplots_adjust(left=.075, right=.985, top=.905, bottom=.165,
                            hspace=.34, wspace=.23)
        fig.text(.075, .095, _footer(results), fontsize=8.6, color="#444444",
                 va="top", linespacing=1.8)
        path = output / "model_assisted_comparison.png"
        try:
            fig.savefig(path, facecolor="white")
        finally:
            plt.close(fig)
    return path
