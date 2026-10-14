"""第七章：绘制三种控制策略的真实位移、电压、压力与速度。"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter7-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "chapter7-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


STRATEGIES = ("pid", "input_mpc", "full_mpc")
LABELS = {"pid": "前馈 + PID（被动限幅）",
          "input_mpc": "MPC（仅输入约束）",
          "full_mpc": "MPC（输入 + P2 约束）"}
COLORS = {"pid": "#777777", "input_mpc": "#df8a22", "full_mpc": "#0759cc"}
STYLES = {"pid": "--", "input_mpc": "-.", "full_mpc": "-"}


def _fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if name in available:
            return [name, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _series(data, field, time):
    values = np.asarray(data[field], dtype=float)
    if values.shape != time.shape or not np.all(np.isfinite(values)):
        raise ValueError(f"{field} 必须是与时间同步的有限一维数组")
    return values


def _time(data, *, allow_repeated=False):
    time = np.asarray(data["time_s"], dtype=float)
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) < 0 if allow_repeated else np.diff(time) <= 0)):
        raise ValueError("time_s 必须是递增的有限一维数组")
    return time


def _pressure(result):
    trace = result.get("pressure_trace", result["data"].get("pressure_trace"))
    if trace is None:
        raise ValueError("缺少高密积分 pressure_trace；不能用控制采样掩盖采样间压力峰")
    # 混合事件与相邻积分段会重复记录边界；保留原记录，不重采样。
    time = _time(trace, allow_repeated=True)
    return time, _series(trace, "p2_mpa", time)


def _limits(ax, arrays, boundaries=()):
    values = [np.asarray(array, dtype=float).ravel() for array in arrays]
    if boundaries:
        values.append(np.asarray(boundaries, dtype=float))
    lower = min(float(np.min(value)) for value in values)
    upper = max(float(np.max(value)) for value in values)
    span = max(upper - lower, max(abs(lower), abs(upper), 1.) * .05)
    # 顶部空白供实测峰值图例使用；所有原始极值都处于显示范围内。
    ax.set_ylim(lower - .10 * span, upper + .30 * span)


def _peak_marker(ax, time, values, kind, *, absolute=False):
    index = int(np.argmax(np.abs(values) if absolute else values))
    ax.scatter(time[index], values[index], color=COLORS[kind], s=28,
               edgecolor="white", linewidth=.5, zorder=5)
    return float(values[index])


def plot_comparison(results, output_path):
    """仅从传入实际数据保存一张 2×2 PNG；不重算、不平滑或拼接曲线。"""
    missing = set(STRATEGIES) - set(results)
    if missing:
        raise ValueError(f"缺少控制策略：{sorted(missing)}")
    configs = [results[kind].get("config", {}) for kind in STRATEGIES]
    config = configs[-1]
    p_safe = float(config.get("p_safe", 21.))
    voltage_limit = float(config.get("voltage_limit_v", 10.))
    horizon = config.get("horizon", 20)
    ts = float(config.get("ts", .001))
    times, traces = {}, {}
    for kind in STRATEGIES:
        data = results[kind]["data"]
        time = _time(data)
        times[kind] = time
        traces[kind] = {field: _series(data, field, time) for field in
                        ("reference_mm", "position_mm", "control_v", "velocity_m_s")}
    first_time = times[STRATEGIES[0]]
    reference = traces[STRATEGIES[0]]["reference_mm"]
    for kind in STRATEGIES[1:]:
        if (not np.array_equal(first_time, times[kind]) or
                not np.array_equal(reference, traces[kind]["reference_mm"])):
            raise ValueError("三策略必须使用相同控制时间和参考指令")
    pressures = {kind: _pressure(results[kind]) for kind in STRATEGIES}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = {"font.family": "sans-serif", "font.sans-serif": _fonts(),
                "font.size": 12, "axes.unicode_minus": False,
                "figure.dpi": 120, "savefig.dpi": 170}
    with plt.rc_context(settings):
        fig, panels = plt.subplots(2, 2, figsize=(22, 12))
        ax_position, ax_voltage, ax_pressure, ax_velocity = panels.ravel()
        fig.suptitle("第七章：50 mm 极限定位下的输入约束与压力约束滚动优化",
                     fontsize=22, y=.981)
        ax_position.step(first_time, reference, where="post", color="black",
                         ls=":", lw=2, label="阶跃参考 $r(t)$")
        for kind in STRATEGIES:
            time, data = times[kind], traces[kind]
            style = {"color": COLORS[kind], "ls": STYLES[kind], "lw": 2.1}
            peak = float(np.max(data["position_mm"]))
            metric = results[kind].get("metrics", {})
            overshoot = metric.get("overshoot_percent")
            position_label = f"{LABELS[kind]}：峰值 {peak:.2f} mm"
            if overshoot is not None:
                position_label += f"，超调 {float(overshoot):.2f}%"
            ax_position.plot(time, data["position_mm"], label=position_label, **style)
            _peak_marker(ax_position, time, data["position_mm"], kind)
            maximum = float(np.max(np.abs(data["control_v"])))
            ax_voltage.plot(time, data["control_v"], **style,
                            label=f"{LABELS[kind]}：max|u|={maximum:.2f} V")
            _peak_marker(ax_voltage, time, data["control_v"], kind, absolute=True)
            ax_voltage.fill_between(time, voltage_limit, data["control_v"],
                                    where=data["control_v"] > voltage_limit,
                                    interpolate=True, color=COLORS[kind], alpha=.12)
            ax_voltage.fill_between(time, -voltage_limit, data["control_v"],
                                    where=data["control_v"] < -voltage_limit,
                                    interpolate=True, color=COLORS[kind], alpha=.12)
            pt, pressure = pressures[kind]
            pressure_peak = float(np.max(pressure))
            ax_pressure.plot(pt, pressure, **style,
                             label=f"{LABELS[kind]}：P2 峰值 {pressure_peak:.2f} MPa")
            _peak_marker(ax_pressure, pt, pressure, kind)
            ax_pressure.fill_between(pt, p_safe, pressure, where=pressure > p_safe,
                                     interpolate=True, color="#d43b36", alpha=.12)
            velocity = data["velocity_m_s"]
            ax_velocity.plot(time, velocity, **style,
                             label=(f"{LABELS[kind]}：最大 {np.max(velocity):.3f} / "
                                    f"最小 {np.min(velocity):.3f} m/s"))
            _peak_marker(ax_velocity, time, velocity, kind)
            minimum = int(np.argmin(velocity))
            ax_velocity.scatter(time[minimum], velocity[minimum], color=COLORS[kind],
                                s=23, marker="v", edgecolor="white", linewidth=.5, zorder=5)
        for bound in (-voltage_limit, voltage_limit):
            ax_voltage.axhline(bound, color="#888888", ls="--", lw=1.2,
                               label=f"执行器物理限幅 ±{voltage_limit:g} V" if bound > 0 else None)
        ax_pressure.axhline(p_safe, color="#d02c2c", ls="--", lw=1.6,
                            label=f"P2 硬上限 {p_safe:g} MPa（淡红为超压禁区）")
        ax_velocity.axhline(0., color="#888888", ls=":", lw=1)
        labels = ((ax_position, "(a) 相对位移：约束规划与超调", "相对位移 $x$ (mm)"),
                  (ax_voltage, "(b) 实际控制电压：执行器限幅", "实际电压 $u$ (V)"),
                  (ax_pressure, "(c) 有杆腔真实压力：采样间峰值与 P2 约束", "有杆腔压力 $P_2$ (MPa)"),
                  (ax_velocity, "(d) 实际速度：加速与制动", "速度 $v$ (m/s)"))
        for ax, title, ylabel in labels:
            ax.set_title(title, fontsize=15, loc="left", pad=10)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("仿真时间 $t$ (s)")
            ax.set_xlim(float(first_time[0]), float(first_time[-1]))
            ax.grid(True, ls=":", color="#b4bac2", alpha=.55, lw=.8)
            ax.set_axisbelow(True)
            ax.legend(loc="upper right", fontsize=10.2, framealpha=.96)
        _limits(ax_position, [reference, *[traces[k]["position_mm"] for k in STRATEGIES]])
        voltage_extent = max(15., 1.10 * max(float(np.max(np.abs(traces[k]["control_v"])))
                                          for k in STRATEGIES))
        ax_voltage.set_ylim(-voltage_extent, voltage_extent)
        ax_voltage.axhspan(voltage_limit, voltage_extent, color="#909090", alpha=.11, zorder=0)
        ax_voltage.axhspan(-voltage_extent, -voltage_limit, color="#909090", alpha=.11, zorder=0)
        _limits(ax_pressure, [pressures[k][1] for k in STRATEGIES], (p_safe,))
        ax_pressure.axhspan(p_safe, ax_pressure.get_ylim()[1], color="#d43b36", alpha=.055, zorder=0)
        _limits(ax_velocity, [traces[k]["velocity_m_s"] for k in STRATEGIES], (0.,))
        fig.subplots_adjust(left=.065, right=.984, bottom=.17, top=.918,
                            wspace=.16, hspace=.24)
        footer = (f"共同对象：5 个物理状态 + 上一步电压 u_prev = 6 维增广状态；"
                  f"局部 LTV 仿射 ZOH 预测，OSQP 滚动求解（N={horizon}，Ts={ts*1000:g} ms）。\n"
                  "压力取混合 ODE 的高密真实积分记录；无人工压力脉冲、无绘图平滑。"
                  "本实验完整组只施加 P2 上限，不代表 P1 已受保护；实际求解耗时另见 summary.json。")
        fig.text(.065, .091, footer, va="top", fontsize=11, color="#444444", linespacing=1.7)
        try:
            fig.savefig(output, facecolor="white")
        finally:
            plt.close(fig)
    return output
