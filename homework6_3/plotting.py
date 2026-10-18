"""习题 6-3：由原始采样记录生成唯一的全周期/换向局部六面板图。"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "homework6-3-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "homework6-3-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


KINDS = ("under", "hard_over", "smooth")
COLORS = {"under": "#1264c0", "hard_over": "#d52e34", "smooth": "#168149"}
STYLES = {"under": "--", "hard_over": "-", "smooth": "-"}
NAMES = {"under": "欠补偿", "hard_over": "硬过补偿", "smooth": "tanh 光滑补偿"}


def _fonts():
    names = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if name in names:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _config(config, keys, default=None):
    for key in keys:
        if key in config:
            return config[key]
    return default


def _metric(metrics, keys):
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return None


def metric_caption(results):
    """只输出已计算的指标；实际字段由实验程序提供。"""
    lines = []
    fields = (("最长流量关闭平台", ("longest_flow_closed_ms", "zero_plateau_ms", "plateau_duration_ms", "deadzone_flat_duration_ms",
                           "flat_duration_ms", "plateau_ms"), ".2f", " ms"),
              ("控制颤振峰峰值", ("chatter_voltage_pp_v", "chatter_peak_to_peak_v", "voltage_chatter_pp_v", "chatter_pp_v",
                                   "control_chatter_p2p_v"), ".4f", " V"),
              ("速度 RMSE", ("velocity_rmse_mm_s", "rmse_velocity_mm_s", "speed_rmse_mm_s"), ".4f", " mm/s"))
    for kind in KINDS:
        metrics = results[kind].get("metrics", {})
        entries = []
        for label, keys, spec, unit in fields:
            value = _metric(metrics, keys)
            if value is not None:
                entries.append(f"{label} {format(value, spec)}{unit}")
        if entries:
            lines.append(f"{NAMES[kind]}：" + "；".join(entries))
    return "\n".join(lines)


def _validate(results):
    fields = ("reference_voltage_v", "reference_velocity_mm_s", "control_voltage_v",
              "valve_voltage_v", "effective_voltage_v", "velocity_mm_s")
    common = None
    for kind in KINDS:
        if kind not in results:
            raise ValueError(f"缺少策略 {kind}")
        data = results[kind]["data"]
        time = np.asarray(data["time_s"], dtype=float)
        if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
                or np.any(np.diff(time) <= 0)):
            raise ValueError("time_s 必须为严格递增的有限一维数组")
        for field in fields:
            values = np.asarray(data[field], dtype=float)
            if values.shape != time.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{kind}.{field} 必须与时间同步且为有限值")
        if common is None:
            common = data
        elif not all(np.array_equal(data[key], common[key]) for key in
                     ("time_s", "reference_voltage_v", "reference_velocity_mm_s")):
            raise ValueError("三策略必须使用相同的时间与参考指令")


def _limits(ax, arrays, window, time, *, include=()):
    mask = (time >= window[0]) & (time <= window[1])
    if not np.any(mask):
        raise ValueError(f"数据未覆盖绘图窗口 {window}")
    visible = [np.asarray(array)[mask] for array in arrays]
    if include:
        visible.append(np.asarray(include))
    lower = min(float(np.min(values)) for values in visible)
    upper = max(float(np.max(values)) for values in visible)
    span = max(upper-lower, max(abs(lower), abs(upper), .1)*.1)
    ax.set_ylim(lower-.12*span, upper+.22*span)


def plot_comparison(results, config, destination):
    """保存一张 3×2 PNG；只画传入原始记录，不平滑、不添加波纹。"""
    _validate(results)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    base = results["under"]["data"]
    time = np.asarray(base["time_s"])
    beta = float(_config(config, ("beta_default", "beta", "smooth_beta", "baseline_beta"), 18.))
    deadzone = float(_config(config, ("true_deadzone_v", "deadzone_v", "physical_deadzone_v"), .5))
    under_estimate = float(config.get("under_estimate_v", .4))
    over_estimate = float(config.get("over_estimate_v", .6))
    names = {"under": f"欠补偿（估计 {under_estimate:g} V）",
             "hard_over": f"硬过补偿（估计 {over_estimate:g} V）",
             "smooth": f"tanh 光滑补偿（估计 {over_estimate:g} V，β={beta:g} /V）"}
    settings = {"font.family": "sans-serif", "font.sans-serif": _fonts(),
                "font.size": 12, "axes.unicode_minus": False,
                "figure.dpi": 120, "savefig.dpi": 170}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(3, 2, figsize=(22, 13), sharex="col")
        fig.suptitle("作业 6-3：死区失配与光滑逆补偿——全周期响应及过零换向细节",
                     fontsize=22, y=.978)
        windows = ((0., 10.), (4.55, 5.45))
        fields = ("control_voltage_v", "effective_voltage_v", "velocity_mm_s")
        labels = ("控制电压 $u$ (V)", "阀芯死区后有效信号 $u_{eff}$ (V)", "真实速度 $v$ (mm/s)")
        for column, window in enumerate(windows):
            for row, field in enumerate(fields):
                ax = axes[row, column]
                values = []
                if column == 0:
                    ax.axvspan(*windows[1], color="#efce69", alpha=.15, zorder=0)
                if row == 0:
                    ax.plot(time, base["reference_voltage_v"], color="#222222", ls=":", lw=1.7,
                            label="线性电压参考 $u_{ref}$")
                    ax.axhspan(-deadzone, deadzone, color="#aab2bb", alpha=.12, zorder=0)
                    for boundary in (-deadzone, deadzone):
                        ax.axhline(boundary, color="#888888", ls=":", lw=1.,
                                   label=f"真实阀芯死区 ±{deadzone:g} V（阀动态后）" if boundary > 0 else None)
                    values.append(base["reference_voltage_v"])
                elif row == 1:
                    ax.plot(time, base["reference_voltage_v"], color="#222222", ls=":", lw=1.7,
                            label="线性有效信号参考 $u_{ref}$")
                    values.append(base["reference_voltage_v"])
                    ax.axhline(0., color="#666666", ls=":", lw=1.)
                else:
                    ax.plot(time, base["reference_velocity_mm_s"], color="#222222", ls=":", lw=1.7,
                            label="线性标度速度参考 $v_{ref}$")
                    values.append(base["reference_velocity_mm_s"])
                for kind in KINDS:
                    data = results[kind]["data"]
                    ax.plot(time, data[field], color=COLORS[kind], ls=STYLES[kind],
                            lw=1.65 if kind == "hard_over" else 2., label=names[kind])
                    values.append(data[field])
                ax.set_xlim(*window)
                _limits(ax, values, window, time,
                        include=(-deadzone, deadzone) if row == 0 else (0.,))
                ax.set_ylabel(labels[row])
                ax.grid(True, ls=":", lw=.75, color="#b4bac1", alpha=.55)
                ax.set_axisbelow(True)
                title = ("控制电压", "阀芯死区后的有效信号", "真实速度")[row]
                scope = "全周期 0–10 s" if column == 0 else "换向 4.55–5.45 s"
                ax.set_title(f"({chr(97+row)}{column+1}) {title} · {scope}",
                             loc="left", fontsize=13, pad=10)
                if row == 2:
                    ax.set_xlabel("仿真时间 $t$ (s)")
                ax.legend(loc="upper right", fontsize=9.8, framealpha=.94)
        intervals = results["under"].get("metrics", {}).get("flow_closed_intervals_s", [])
        if intervals:
            longest = max(intervals, key=lambda interval: interval[1]-interval[0])
            mid = .5 * (longest[0]+longest[1])
            duration = results["under"]["metrics"].get("longest_flow_closed_ms")
            if duration is not None:
                axes[1, 1].annotate(
                    f"欠补偿：最长关闭平台 {duration:.2f} ms", xy=(mid, 0.),
                    xytext=(.025, .94), textcoords="axes fraction", va="top",
                    color=COLORS["under"], fontsize=10,
                    bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": COLORS["under"], "alpha": .95},
                    arrowprops={"arrowstyle": "->", "color": COLORS["under"], "lw": 1.2})
        hm = results["hard_over"].get("metrics", {})
        if hm.get("velocity_turning_points") is not None:
            velocity = np.asarray(results["hard_over"]["data"]["velocity_mm_s"])
            indices = np.flatnonzero((time >= windows[1][0]) & (time <= windows[1][1]))
            differences = np.diff(velocity[indices])
            turns = np.flatnonzero(differences[:-1]*differences[1:] < 0.) + 1
            point = indices[turns[len(turns)//2]] if len(turns) else indices[len(indices)//2]
            text = f"硬过补偿：速度转折 {hm['velocity_turning_points']} 次"
            if hm.get("velocity_excess_variation_mm_s") is not None:
                text += f"\n超额总变差 {hm['velocity_excess_variation_mm_s']:.3f} mm/s"
            axes[2, 1].annotate(
                text, xy=(time[point], velocity[point]), xytext=(.025, .94),
                textcoords="axes fraction", va="top", color=COLORS["hard_over"], fontsize=10,
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": COLORS["hard_over"], "alpha": .95},
                arrowprops={"arrowstyle": "->", "color": COLORS["hard_over"], "lw": 1.2})
        caption = metric_caption(results)
        fig.subplots_adjust(left=.07, right=.984, top=.918, bottom=.225,
                            hspace=.18, wspace=.15)
        fig.text(.07, .162,
                 "死区作用于实际阀芯电压 u_v：u_eff=sign(u_v)·max(|u_v|−δ,0)，不是对控制指令 u 直接截断。"
                 "曲线为原始闭环采样数据，未绘图平滑或注入纹波。",
                 fontsize=10.5, color="#444444", va="top")
        if caption:
            fig.text(.07, .131, caption, fontsize=10.5, color="#333333", va="top", linespacing=1.7)
        fig.text(.07, .035,
                 f"平台按实际阀芯 |u_v|≤{deadzone:g} V 统计；颤振值为 {config.get('chatter_cutoff_hz', 20.):g} Hz "
                 "高通诊断峰峰值，滤波仅用于指标。局部阀–速度等效模型，有限 β 仍有残余平台；详见报告。",
                 fontsize=10.5, color="#555555", va="top")
        try:
            fig.savefig(destination, facecolor="white")
        finally:
            plt.close(fig)
    return destination
