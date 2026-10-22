"""习题 7-2：从闭环原始记录绘制预测时域与算力的四联对照图。"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "work72_matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "work72_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


COLORS = {5: "#e12b45", 20: "#2563eb", 50: "#177137"}
STYLES = {5: "--", 20: "-", 50: "-."}


def _font_family():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if candidate in installed:
            return [candidate, "DejaVu Sans"]
    return ["DejaVu Sans"]


def _config(config, *keys, default):
    for key in keys:
        if key in config:
            return float(config[key])
    return float(default)


def _series(record, field, time, allow_nan=False):
    values = np.asarray(record[field], dtype=float)
    valid = np.isfinite(values) | (np.isnan(values) if allow_nan else False)
    if values.ndim != 1 or values.shape != time.shape or not np.all(valid):
        raise ValueError(f"{field} 必须为与 t_s 对齐的有限一维序列")
    return values


def _record(result):
    """Keep dense P2 timestamps when supplied; never substitute sampled pressure."""
    t = np.asarray(result["t_s"], dtype=float)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("t_s 必须为严格递增的有限时刻")
    x = _series(result, "x_mm", t)
    # An infeasible terminal sample may have no applied command (NaN sentinel).
    # Matplotlib leaves that value blank instead of inventing a safe command.
    u = _series(result, "u_v", t, allow_nan=True)
    pressure_time = np.asarray(result.get("pressure_t_s", result.get("p2_t_s", t)), dtype=float)
    if (pressure_time.ndim != 1 or len(pressure_time) < 2 or
            not np.all(np.isfinite(pressure_time)) or np.any(np.diff(pressure_time) < 0)):
        raise ValueError("pressure_t_s 必须为非降的有限时刻")
    p2 = _series(result, "p2_mpa", pressure_time)
    failure = result.get("failure_time_s")
    if failure is not None and not np.isfinite(float(failure)):
        raise ValueError("failure_time_s 必须为有限数值")
    return t, x, u, pressure_time, p2, (None if failure is None else float(failure))


def _number(value, digits=2):
    numeric = _as_float(value)
    return "未记录" if not np.isfinite(numeric) else f"{numeric:.{digits}f}"


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def plot_results(results, summary_rows, config, output_path):
    """Plot actual N=5/20/50 traces and all-horizon QP-time/pressure metrics.

    ``results`` maps horizon to ``t_s``, ``x_mm``, ``u_v``, ``p2_mpa`` and
    optionally ``pressure_t_s`` (dense pressure times) and ``failure_time_s``.
    A failed run remains a partial trace with an explicit terminal marker.
    ``summary_rows`` contain ``horizon``, ``peak_pressure_mpa``,
    ``solver_mean_ms``, ``overshoot_pct`` and ``status``.
    """
    rows = {int(item.get("horizon", item.get("N"))): item for item in summary_rows}
    if not rows:
        raise ValueError("需要至少一行预测时域指标")
    p_safe = _config(config, "p_safe", "p_safe_mpa", default=21.)
    voltage_limit = _config(config, "voltage_limit_v", "u_max", default=10.)
    ts_ms = 1000. * _config(config, "ts", "sample_time_s", default=.001)
    step_time = _config(config, "step_time", "step_time_s", default=.01)
    target = _config(config, "target_mm", default=50.)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": _font_family(),
                         "axes.unicode_minus": False, "font.size": 11,
                         "figure.dpi": 120, "savefig.dpi": 180}):
        fig, panels = plt.subplots(2, 2, figsize=(20, 11.2))
        ax_x, ax_u, ax_p, ax_trade = panels.ravel()
        fig.suptitle("作业 7-2：预测时域 $N$ 对定位、压力与计算开销的影响", fontsize=20, y=.985)
        max_time = _config(config, "duration", "duration_s", default=.15)
        ax_x.plot([0., step_time, step_time, max_time], [0., 0., target, target],
                  color="#333333", ls=":", lw=1.8, label=f"参考阶跃 {target:g} mm")
        for horizon in (5, 20, 50):
            if horizon not in results:
                continue
            row = rows.get(horizon, {})
            t, x, u, pt, p2, failure = _record(results[horizon])
            max_time = max(max_time, float(t[-1]))
            color, style = COLORS[horizon], STYLES[horizon]
            status = str(row.get("status", "completed")).lower()
            failed = failure is not None or status not in ("completed", "ok", "optimal", "success")
            if failed and failure is None:
                failure = float(t[-1])
            suffix = "，未完成" if failed else ""
            over = _number(row.get("overshoot_pct"), 2)
            ax_x.plot(t, x, color=color, ls=style, lw=2.,
                      label=f"N={horizon}，超调 {over}%{suffix}")
            ax_u.plot(t, u, color=color, ls=style, lw=2., label=f"N={horizon}{suffix}")
            ax_p.plot(pt, p2, color=color, ls=style, lw=2.,
                      label=f"N={horizon}，峰值 {np.max(p2):.2f} MPa{suffix}")
            if failure is not None:
                ax_x.scatter(t[-1], x[-1], color=color, marker="x", s=55, zorder=5)
                valid_u = np.flatnonzero(np.isfinite(u))
                if len(valid_u):
                    ax_u.scatter(t[valid_u[-1]], u[valid_u[-1]],
                                 color=color, marker="x", s=55, zorder=5)
                ax_p.scatter(pt[-1], p2[-1], color=color, marker="x", s=55, zorder=5)
                for axis in (ax_x, ax_u, ax_p):
                    axis.axvline(failure, color=color, lw=.9, ls=":", alpha=.55)
        ax_u.axhspan(voltage_limit, 15., color="#aaaaaa", alpha=.14, zorder=0)
        ax_u.axhspan(-15., -voltage_limit, color="#aaaaaa", alpha=.14, zorder=0)
        for voltage in (-voltage_limit, voltage_limit):
            ax_u.axhline(voltage, color="#999999", ls="--", lw=1.,
                         label=f"物理边界 ±{voltage_limit:g} V" if voltage > 0 else None)
        ax_p.axhline(p_safe, color="#d92d40", ls="--", lw=1.6,
                     label=f"安全上限 {p_safe:g} MPa")
        ax_p.axhspan(p_safe, max(p_safe + 1., ax_p.get_ylim()[1]),
                     color="#e94e63", alpha=.09, zorder=0)

        horizons = np.array(sorted(rows), dtype=float)
        solver = np.array([_as_float(rows[int(n)].get("solver_mean_ms")) for n in horizons])
        pressure = np.array([_as_float(rows[int(n)].get("peak_pressure_mpa")) for n in horizons])
        finite_solver, finite_pressure = np.isfinite(solver), np.isfinite(pressure)
        if finite_solver.any():
            ax_trade.plot(horizons[finite_solver], solver[finite_solver], color="#f07800",
                          marker="s", markersize=5, lw=2., label="QP 平均求解时间")
            positive = solver[finite_solver & (solver > 0.)]
            if len(positive) and np.max(positive)/np.min(positive) > 20.:
                ax_trade.set_yscale("log")
                ax_trade.set_ylim(max(0.1, np.min(positive)/2.), np.max(positive)*1.7)
        ax_trade.axhline(ts_ms, color="#f07800", ls="--", lw=1.2,
                         label=f"{ts_ms:g} ms 采样周期")
        timing_unit = "ms，对数轴" if ax_trade.get_yscale() == "log" else "ms"
        ax_trade.set_ylabel(f"QP 平均求解时间 ({timing_unit})", color="#d66a00")
        ax_trade.tick_params(axis="y", labelcolor="#d66a00")
        pressure_axis = ax_trade.twinx()
        if finite_pressure.any():
            pressure_axis.plot(horizons[finite_pressure], pressure[finite_pressure],
                               color="#2864cd", marker="o", markersize=5, lw=2.,
                               label="实际高密 P2 峰值")
        failed_horizons = np.array([int(n) for n in horizons
                                    if str(rows[int(n)].get("status", "completed")).lower()
                                    not in ("completed", "ok", "optimal", "success")], dtype=int)
        if len(failed_horizons):
            finite_failed = np.array([n for n in failed_horizons
                                      if np.isfinite(pressure[list(horizons).index(n)])], dtype=int)
            if len(finite_failed):
                failure_pressure = np.array([pressure[list(horizons).index(n)]
                                             for n in finite_failed])
                pressure_axis.scatter(finite_failed, failure_pressure, marker="x", s=75,
                                      linewidths=2., color="#b42338", zorder=6,
                                      label="闭环提前终止：仅有部分数据")
        pressure_axis.axhline(p_safe, color="#d92d40", ls=":", lw=1.35,
                              label=f"{p_safe:g} MPa 安全线")
        pressure_axis.set_ylabel("有杆腔 P2 峰值 (MPa)", color="#2864cd")
        pressure_axis.tick_params(axis="y", labelcolor="#2864cd")
        ax_trade.set_xticks(horizons)
        ax_trade.set_xlim(horizons[0] - 2., horizons[-1] + 2.)
        handles, names = ax_trade.get_legend_handles_labels()
        extra_handles, extra_names = pressure_axis.get_legend_handles_labels()
        ax_trade.legend(handles + extra_handles, names + extra_names,
                        loc="upper left", fontsize=9., framealpha=.94)

        for axis, title, ylabel in ((ax_x, "(a) 活塞位移：预测视野与超调", "相对位移 $x$ (mm)"),
                                    (ax_u, "(b) 实际伺服阀控制电压", "控制电压 $u$ (V)"),
                                    (ax_p, "(c) 有杆腔真实压力：采样间极值", "有杆腔压力 $P_2$ (MPa)")):
            axis.set_title(title, loc="left", fontsize=13, pad=8)
            axis.set_xlim(0., max_time)
            axis.set_ylabel(ylabel)
            axis.set_xlabel("仿真时间 $t$ (s)")
            axis.grid(True, ls=":", lw=.75, color="#b6bcc5", alpha=.6)
            axis.set_axisbelow(True)
            axis.legend(loc="upper right", fontsize=8.7, framealpha=.95)
        ax_x.set_ylim(bottom=min(-2., ax_x.get_ylim()[0]))
        ax_u.set_ylim(-15., 15.)
        ax_trade.set_title("(d) 预测时域、真实峰压与 QP 耗时", loc="left", fontsize=13, pad=8)
        ax_trade.set_xlabel("预测时域步数 $N$")
        ax_trade.grid(True, ls=":", lw=.75, color="#b6bcc5", alpha=.6)
        ax_trade.set_axisbelow(True)
        fig.subplots_adjust(left=.073, right=.93, top=.925, bottom=.145, hspace=.28, wspace=.2)
        fig.text(.073, .065,
                 "电压、位移直接取闭环记录；压力使用积分器高密轨迹。× 表示该时域闭环提前终止。\n"
                 "QP 均值仅计求解步骤；CVXPY 构图、线性化、非线性可行性验证与整体控制耗时须另看指标表。",
                 va="top", fontsize=9.2, color="#4a5563", linespacing=1.5)
        try:
            fig.savefig(output, facecolor="white")
        finally:
            plt.close(fig)
    return output
