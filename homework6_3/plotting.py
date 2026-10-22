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
    fields = (("有效信号零平台", ("longest_effective_zero_ms", "longest_flow_closed_ms", "plateau_duration_ms", "deadzone_flat_duration_ms",
                           "flat_duration_ms", "plateau_ms"), ".2f", " ms"),
              ("近零速度平台", ("longest_velocity_near_zero_ms",), ".2f", " ms"),
              ("速度高频峰峰值", ("chatter_velocity_pp_mm_s",), ".3f", " mm/s"))
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
    ax.set_ylim(lower-.08*span, upper+.08*span)


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
                "font.size": 10.5, "axes.unicode_minus": False,
                "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                "legend.fontsize": 8.8, "figure.dpi": 120, "savefig.dpi": 170}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(3, 2, figsize=(21, 11), sharex="col")
        fig.suptitle("作业 6-3：伺服阀死区逆补偿的鲁棒性边界与 tanh 光滑化",
                     fontsize=17, y=.976)
        windows = ((0., float(config["duration_s"])),
                   (float(config["zoom_start_s"]), float(config["zoom_end_s"])))
        fields = ("control_voltage_v", "effective_voltage_v", "velocity_mm_s")
        labels = ("控制电压 $u$ (V)", "物理死区输出 $u_{eff}=D(u)$ (V)", "活塞速度 $v_p$ (mm/s)")
        for column, window in enumerate(windows):
            for row, field in enumerate(fields):
                ax = axes[row, column]
                values = []
                if column == 0:
                    ax.axvspan(*windows[1], color="#ffe07a", alpha=.35, zorder=0)
                if row == 0:
                    ax.axhspan(-deadzone, deadzone, color="#aab2bb", alpha=.12, zorder=0)
                    for boundary in (-deadzone, deadzone):
                        ax.axhline(boundary, color="#888888", ls="--", lw=.9,
                                   label=f"物理输入死区 ±{deadzone:g} V" if boundary > 0 else None)
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
                ax.grid(True, ls=":", lw=.7, color="#b4bac1", alpha=.60)
                ax.set_axisbelow(True)
                title = ("控制电压 $u(t)$", "阀口有效信号 $u_{eff}(t)$", "活塞速度 $v_p(t)$")[row]
                scope = (f"全周期宏观响应（0–{windows[0][1]:g} s）" if column == 0
                         else f"过零换向局部放大（{window[0]:g}–{window[1]:g} s）")
                ax.set_title(f"({chr(97+column)}{row+1}) {title}  {scope}", pad=9)
                if row == 2:
                    ax.set_xlabel("仿真时间 $t$ (s)")
                ax.legend(loc="upper right", framealpha=.94)
        intervals = results["under"].get("metrics", {}).get("flow_closed_intervals_s", [])
        if intervals:
            longest = max(intervals, key=lambda interval: interval[1]-interval[0])
            mid = .5 * (longest[0]+longest[1])
            duration = _metric(results["under"]["metrics"],
                               ("longest_effective_zero_ms", "longest_flow_closed_ms"))
            if duration is not None:
                axes[1, 1].annotate(
                    f"欠补偿：有效信号零平台 {duration:.2f} ms", xy=(mid, 0.),
                    xytext=(.025, .10), textcoords="axes fraction", va="bottom",
                    color=COLORS["under"], fontsize=9.2,
                    bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": COLORS["under"], "alpha": .95},
                    arrowprops={"arrowstyle": "->", "color": COLORS["under"], "lw": 1.2})
        hm = results["hard_over"].get("metrics", {})
        if "linear_sign_reversals" in hm:
            linear = np.asarray(results["hard_over"]["data"]["linear_voltage_v"])
            control = np.asarray(results["hard_over"]["data"]["control_voltage_v"])
            indices = np.flatnonzero((time >= windows[1][0]) & (time <= windows[1][1]))
            crossings = np.flatnonzero(np.sign(linear[indices[:-1]]) !=
                                       np.sign(linear[indices[1:]]))
            point = indices[crossings[len(crossings)//2]] if len(crossings) else indices[len(indices)//2]
            axes[0, 1].annotate(
                f"硬过补偿：{hm['linear_sign_reversals']} 次符号切换\n"
                f"最大采样电压跳变 {hm['max_voltage_step_v']:.2f} V",
                xy=(time[point], control[point]), xytext=(.025, .08),
                textcoords="axes fraction", va="bottom", color=COLORS["hard_over"], fontsize=9.2,
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": COLORS["hard_over"], "alpha": .95},
                arrowprops={"arrowstyle": "->", "color": COLORS["hard_over"], "lw": 1.2})
        if "chatter_velocity_pp_mm_s" in hm:
            axes[2, 1].text(.025, .08,
                f"欠补偿近零速度平台 {results['under']['metrics']['longest_velocity_near_zero_ms']:.1f} ms\n"
                f"硬过补偿速度高频峰峰值 {hm['chatter_velocity_pp_mm_s']:.3f} mm/s\n"
                f"tanh 速度高频峰峰值 {results['smooth']['metrics']['chatter_velocity_pp_mm_s']:.3f} mm/s",
                transform=axes[2, 1].transAxes, va="bottom", color="#184b3a", fontsize=9.2,
                bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": COLORS["smooth"], "alpha": .95})
        caption = metric_caption(results)
        fig.subplots_adjust(left=.068, right=.986, top=.905, bottom=.205,
                            hspace=.32, wspace=.15)
        if caption:
            fig.text(.068, .14, caption, fontsize=9.2, color="#263548", va="top", linespacing=1.45,
                     bbox={"boxstyle": "round,pad=.4", "fc": "#f2f6fc", "ec": "#d4dce8", "lw": .7})
        fig.text(.986, .025,
                 f"共同对象：输入死区 → 阀动态 → 速度动态；二级 washout K={config['washout_gain']:g}、"
                 f"反馈延迟 {config['feedback_delay_samples']} 采样。有限 β 仍有残余零平台；高通仅用于指标。",
                 fontsize=8.2, color="#576372", ha="right", va="bottom")
        try:
            fig.savefig(destination, facecolor="white")
        finally:
            plt.close(fig)
    return destination
