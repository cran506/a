"""习题 5-1：由两组真实闭环数据生成唯一的横幅三联结果图。"""
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "homework5-1-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "homework5-1-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


RED, BLUE, ORANGE = "#d82c2c", "#0655d8", "#e59a24"


def _fmt(value, digits=3):
    if value is None:
        return "未观测到"
    value = float(value)
    return f"{value:.{digits}f}" if np.isfinite(value) else "无有效值"


def _font():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if candidate in names:
            return [candidate, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _validate(results):
    required = ("time_s", "reference_m", "position_m", "true_error_m",
                "voltage_v", "raw_voltage_v")
    for kind in ("positional", "incremental"):
        if kind not in results:
            raise ValueError(f"缺少 {kind} 结果")
        data = results[kind]["data"]
        time = np.asarray(data["time_s"])
        if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0):
            raise ValueError("仿真时间必须严格递增")
        for name in required:
            values = np.asarray(data[name])
            if values.shape != time.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{kind}.{name} 必须是与时间同步的有限数组")
    a, b = (results[kind]["data"] for kind in ("positional", "incremental"))
    for field in ("time_s", "reference_m"):
        if not np.array_equal(a[field], b[field]):
            raise ValueError("两组必须使用相同参考轨迹和控制时间刻度")


def _annotate(ax, text, point, location, color):
    ax.annotate(text, xy=point, xycoords="data", xytext=location,
                textcoords="axes fraction", ha="left", va="top", fontsize=12,
                color=color, annotation_clip=True,
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4},
                bbox={"boxstyle": "round,pad=.42", "fc": "white",
                      "ec": color, "alpha": .95})


def _footer(result):
    config = result.get("config", {})
    data = result["data"]
    period_ms = config.get("ts", float(data["time_s"][1] - data["time_s"][0])) * 1000
    limit = config.get("voltage_limit_v", 10.)
    mode = config.get("derivative_mode")
    if mode == "measurement":
        derivative = "两组共同 D-on-measurement（对测量微分），避免参考阶跃的微分 kick"
    elif mode == "error":
        derivative = "两组共同 D-on-error（教材误差微分），保留参考阶跃的微分 kick"
    else:
        derivative = "微分配置见实验参数；真实误差为参考减实际位移"
    gains = ""
    if all(name in config for name in ("kp", "ki", "kd", "alpha")):
        gains = (f"Kp={config['kp']:g} V/cm，Ki={config['ki']:g} V/(cm·s)，"
                 f"Kd={config['kd']:g} V·s/cm，alpha={config['alpha']:g}；")
    first = f"{gains} Ts={period_ms:g} ms；输出限幅 ±{limit:g} V；积分分离关闭；"
    if "adc_bits" in config:
        first += f"ADC {config['adc_bits']} bit"
        if "adc_lsb_mm" in config:
            first += f"，LSB={config['adc_lsb_mm']:.6f} mm"
        first += "。"
    physical = config.get("physical_parameters", {})
    friction = ""
    if "Fs" in physical:
        friction = f"；共同真实摩擦：Fs={physical['Fs']:g} N"
        if "Fc" in physical:
            friction += f"、Fc={physical['Fc']:g} N"
        friction += "，静止/释放由物理摩擦事件求解"
    second = derivative + friction + "；图线为本次实际数值积分数据。"
    return first + "\n" + second


def plot_comparison(results, output_path):
    """保存一张 PNG，返回 Path；不重算或平滑传入轨迹。

    results 包含 positional、incremental 的 data/config/metrics。
    横轴使用真实 time_s；纵轴由原始候选及实际信号共同自动确定，
    因此 D-on-error 的大幅微分冲击也不会被截去。
    """
    _validate(results)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    a, b = (results[kind]["data"] for kind in ("positional", "incremental"))
    ma, mb = (results[kind]["metrics"] for kind in ("positional", "incremental"))
    config = results["positional"].get("config", {})
    limit = float(config.get("voltage_limit_v", 10.))
    t = np.asarray(a["time_s"])
    reference = np.asarray(a["reference_m"]) * 1000
    pos_a, pos_b = (np.asarray(d["position_m"]) * 1000 for d in (a, b))
    error_a, error_b = (np.asarray(d["true_error_m"]) * 1000 for d in (a, b))
    raw_a = np.asarray(a["raw_voltage_v"])
    step_indices = np.flatnonzero(np.diff(reference) != 0) + 1
    step_time = float(t[step_indices[0]]) if len(step_indices) else float(t[0])
    amplitude = float(np.max(reference))
    settings = {"font.family": "sans-serif", "font.sans-serif": _font(),
                "font.size": 12, "axes.unicode_minus": False,
                "figure.dpi": 120, "savefig.dpi": 170}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(3, 1, figsize=(22, 12), sharex=True,
                                 gridspec_kw={"height_ratios": [1., 1.02, 1.]})
        position_ax, voltage_ax, error_ax = axes
        fig.suptitle("作业 5-1：大幅值阶跃下位置式与增量式 PID 的抗积分饱和（Anti-Windup）对比",
                     fontsize=21, y=.977)
        position_ax.step(t, reference, where="post", color="black", ls=":", lw=2.2,
                         label=f"阶跃参考 r(t)：{amplitude:g} mm @ {step_time:g} s")
        position_ax.plot(t, pos_a, color=RED, ls="--", lw=2,
                         label="实验 A：位置式 PID（仅最终输出限幅）")
        position_ax.plot(t, pos_b, color=BLUE, lw=2.1,
                         label="实验 B：增量式 PID（截断输出写回记忆）")
        voltage_ax.plot(t, raw_a, color=ORANGE, ls=(0, (6, 2, 1, 2)), lw=1.9,
                        label="实验 A：限幅前内部候选电压")
        voltage_ax.fill_between(t, limit, raw_a, where=raw_a > limit,
                                interpolate=True, color=ORANGE, alpha=.18)
        for bound in (-limit, limit):
            voltage_ax.axhline(bound, color="#888888", ls="--", lw=1.25,
                              label=f"执行器物理限幅 ±{limit:g} V" if bound > 0 else None)
        voltage_ax.plot(t, a["voltage_v"], color=RED, ls="--", lw=2,
                        label="实验 A：实际控制电压")
        voltage_ax.plot(t, b["voltage_v"], color=BLUE, lw=2,
                        label="实验 B：实际控制电压")
        error_ax.plot(t, error_a, color=RED, ls="--", lw=2,
                      label="实验 A：真实跟踪误差")
        error_ax.plot(t, error_b, color=BLUE, lw=2,
                      label="实验 B：真实跟踪误差")
        error_ax.axhline(0., color="black", ls=":", lw=1.3)
        crossing_a = ma.get("crossing_time_interpolated_s")
        crossing_b = mb.get("crossing_time_interpolated_s")
        if crossing_a is not None:
            error_ax.axvline(crossing_a, color=RED, ls="--", lw=1.3, alpha=.7,
                            label="A 首次真实误差过零（采样插值）")
        position_ax.set_ylabel("相对位移 $x$ (mm)")
        voltage_ax.set_ylabel("控制电压 $u$ (V)")
        error_ax.set_ylabel("真实误差 $e=r-x$ (mm)")
        error_ax.set_xlabel("仿真时间 $t$ (s)")
        for ax in axes:
            ax.set_xlim(float(t[0]), float(t[-1]))
            ax.margins(y=.17)
            ax.grid(True, ls=":", color="#b7bdc4", alpha=.55, lw=.8)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=11)
            ax.legend(loc="upper right", fontsize=11, framealpha=.95)

        # 1. 位移与超调使用当前记录的真实峰值。
        peak_a = int(np.argmax(pos_a))
        text = ("【看点 1：实际超调差异】\n"
                f"A：峰值 {_fmt(ma['peak_position_mm'], 2)} mm，超调 {_fmt(ma['overshoot_percent'], 2)}%\n"
                f"B：峰值 {_fmt(mb['peak_position_mm'], 2)} mm，超调 {_fmt(mb['overshoot_percent'], 2)}%")
        _annotate(position_ax, text, (float(t[peak_a]), float(pos_a[peak_a])),
                  (.43, .95), RED)
        position_ax.scatter(t[peak_a], pos_a[peak_a], color=RED, s=26, zorder=4)

        # 2. 候选输出与实际截断值分开显示，保留超过上限的存量。
        raw_peak = int(np.argmax(raw_a))
        delay = ma.get("desaturation_delay_after_crossing_ms")
        if delay is None:
            timing = "过零与退出时差：本时窗未观测到完整事件"
        elif delay >= 0:
            timing = f"A 过零后 {_fmt(delay, 2)} ms 首次退出正饱和"
        else:
            timing = f"A 首次退出早于过零 {_fmt(-delay, 2)} ms"
        text = ("【看点 2：积分存量与退饱和迟滞】\n"
                f"A 候选峰值 {_fmt(ma['max_internal_voltage_v'], 2)} V；"
                f"历史积分峰值 {_fmt(ma.get('max_explicit_integral_v'), 2)} V\n"
                f"A 首次退出 +{limit:g} V：{_fmt(ma.get('initial_positive_saturation_exit_s'))} s\n"
                + timing)
        _annotate(voltage_ax, text, (float(t[raw_peak]), float(raw_a[raw_peak])),
                  (.38, .95), "#a16b0d")

        # 3. 首次退饱和与误差反向是不同事件，保留提前退出和缺失事件。
        exit_b = mb.get("initial_positive_saturation_exit_s")
        relation = "B 的截断记忆避免跨周期保留越界候选输出"
        if exit_b is not None and crossing_b is not None:
            relation = ("B 首次退出早于误差过零，不能等同反向退饱和"
                        if exit_b < crossing_b else "B 退出与过零时刻按实际数据分别提取")
        text = ("【看点 3：误差反向与输出状态】\n"
                f"A 过零：{_fmt(crossing_a)} s；正饱和首次退出："
                f"{_fmt(ma.get('initial_positive_saturation_exit_s'))} s\n"
                f"B 过零：{_fmt(crossing_b)} s；正饱和首次退出：{_fmt(exit_b)} s\n"
                + relation)
        if crossing_a is not None:
            _annotate(error_ax, text, (float(crossing_a), 0.), (.35, .95), RED)
        else:
            error_ax.text(.35, .95, text, transform=error_ax.transAxes,
                          ha="left", va="top", color=RED, fontsize=12,
                          bbox={"boxstyle": "round,pad=.42", "fc": "white", "ec": RED})
        fig.subplots_adjust(left=.075, right=.985, top=.914, bottom=.15, hspace=.16)
        fig.text(.075, .065, _footer(results["positional"]), va="top",
                 fontsize=10.5, color="#444444", linespacing=1.8)
        try:
            fig.savefig(path, facecolor="white")
        finally:
            plt.close(fig)
    return path
