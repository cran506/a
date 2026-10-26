"""从保存的训练日志和独立闭环记录绘制课堂四联图。

只读取真实 CSV；不插入参考响应，不对误差、电压或训练奖励作拟合。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter8_reference_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np


POLICIES = ("pid", "nominal_ff", "sac_fp32", "edge_int8")
LABELS = {"pid": "传统 PID", "nominal_ff": "标称模型前馈", "sac_fp32": "SAC · FP32",
          "edge_int8": "70% 剪枝 · INT8 权重 / INT16 激活"}
COLORS = {"pid": "#444444", "nominal_ff": "#ed8b00", "sac_fp32": "#3567ee", "edge_int8": "#15963e"}
STYLES = {"pid": "--", "nominal_ff": ":", "sac_fp32": "-", "edge_int8": "-."}


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"缺少真实运行数据：{path}")
    return rows


def _column(rows, name, *, allow_nan=False):
    try:
        result = np.asarray([float(row[name]) for row in rows], dtype=float)
    except (KeyError, ValueError) as error:
        raise ValueError(f"CSV 字段 {name} 缺失或无效") from error
    if not allow_nan and not np.isfinite(result).all():
        raise ValueError(f"CSV 字段 {name} 含非有限数值")
    return result


def _font():
    available = {item.name for item in font_manager.fontManager.ttflist}
    candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei", "Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams.update({"font.family": "sans-serif",
        "font.sans-serif": [name for name in candidates if name in available],
        "axes.unicode_minus": False, "font.size": 9.5, "axes.titlesize": 11.5,
        "axes.labelsize": 10, "legend.fontsize": 8, "savefig.dpi": 180,
        "pdf.fonttype": 42, "ps.fonttype": 42})


def _policy_data(rows):
    data = {}
    for policy in POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        if not group:
            raise ValueError(f"缺少独立闭环记录：{policy}")
        columns = {key: _column(group, key) for key in
                   ("time_s", "velocity_error_mms", "control_v")}
        if np.any(np.diff(columns["time_s"]) <= 0):
            raise ValueError(f"{policy} 的时刻必须严格递增")
        data[policy] = columns
    reference_time = data[POLICIES[0]]["time_s"]
    for policy in POLICIES[1:]:
        if not np.array_equal(data[policy]["time_s"], reference_time):
            raise ValueError("四策略必须使用同一个评估时钟")
    return data


def _recovery_time(t, error, fault_time=.5):
    post = np.flatnonzero(t >= fault_time - 1e-10)
    peak = post[np.argmax(np.abs(error[post]))]
    for index in range(peak, len(t)):
        if t[-1] - t[index] < .05 - 1e-10:
            break
        if np.all(np.abs(error[index:]) <= 1.):
            return float(t[index] - fault_time)
    return None


def generate_figures(root=None):
    """Save PNG/PDF from raw outputs; return paths to the generated figures."""
    root = Path(root or Path(__file__).resolve().parent)
    out = root / "outputs"
    training = _read_csv(out / "training.csv")
    data = _policy_data(_read_csv(out / "evaluation.csv"))
    safety = _read_csv(out / "safety.csv")
    summary = json.loads((out / "safety_summary.json").read_text(encoding="utf-8"))
    metadata_path = out / "metrics.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    fault_time = float(metadata.get("evaluation", {}).get("fault_time_s", .5))
    _font()
    fig, axes = plt.subplots(2, 2, figsize=(17.8, 9.7))
    fig.subplots_adjust(left=.065, right=.935, top=.885, bottom=.125, wspace=.28, hspace=.35)
    fig.suptitle("第八章 深度强化学习 SAC：五维参考接口、HOCBF 安全滤波与复合故障验证", fontsize=15.5, y=.965)
    for ax in axes.flat:
        ax.grid(True, linestyle=":", alpha=.4)
        ax.set_axisbelow(True)

    ax = axes[0, 0]
    steps = _column(training, "timesteps")
    reward = _column(training, "reward_mean")
    entropy = _column(training, "voltage_entropy")
    if np.any(np.diff(steps) <= 0):
        raise ValueError("训练步数必须严格递增")
    last_step = float(steps[-1]) / 1000.
    phases = [(0., min(50., last_step), "#eeeeee", "0–50k：课件阶段 I"),
              (50., min(400., last_step), "#e8edfc", "50k–400k：课件阶段 II"),
              (400., last_step, "#e7f5e9", ">400k：课件阶段 III")]
    phase_handles = []
    for left, right, color, label in phases:
        if right > left:
            ax.axvspan(left, right, color=color, zorder=0)
            phase_handles.append(Patch(facecolor=color, label=label))
    line_reward, = ax.plot(steps / 1000., reward, color=COLORS["sac_fp32"], lw=1.7, label="实际平均每步奖励")
    if np.max(np.abs(reward)) > 10000.:
        ax.set_yscale("symlog", linthresh=10.)
        ax.set_ylabel("原始平均每步奖励（对称对数轴）", color=COLORS["sac_fp32"])
    else:
        ax.set_ylabel("原始平均每步奖励", color=COLORS["sac_fp32"])
    ax.tick_params(axis="y", colors=COLORS["sac_fp32"])
    entropy_ax = ax.twinx()
    line_entropy, = entropy_ax.plot(steps / 1000., entropy, color="#333333", ls="--", lw=1.4,
                                   label="实际电压动作熵 H(u)")
    entropy_ax.set_ylabel("H(u) / nat（tanh 后，u 以 V 计）")
    ax.set_xlim(0., last_step)
    ax.set_xlabel("实际训练交互步数 / 千步")
    ax.set_title(f"(a) SAC 真实训练日志（{int(steps[-1]):,} 步）", pad=12)
    ax.text(.02, .03, "机理教师预热 + SAC；色带仅标记课件的步数区间", transform=ax.transAxes,
            fontsize=7.2, color="#444444", bbox=dict(fc="white", ec="none", alpha=.8))
    ax.legend(handles=[*phase_handles, line_reward, line_entropy], loc="best", framealpha=.95, fontsize=7.2)

    ax = axes[0, 1]
    t = _column(safety, "time_s")
    xs = _column(safety, "x_safe_mm")
    xu = _column(safety, "x_unsafe_mm", allow_nan=True)
    us = _column(safety, "u_safe_v")
    shown = t <= .25 + 1e-12
    ax.axhspan(100., 116., color="#f9dce2", alpha=.85)
    ax.axhline(100., color="#df365c", ls="--", lw=1.2)
    line_unsafe, = ax.plot(t, xu, color="#e43c61", ls=":", lw=1.8, label="未滤波：首次触边结束")
    line_safe, = ax.plot(t, xs, color=COLORS["sac_fp32"], lw=1.8, label="HOCBF-QP：受保护位移")
    ax.set(xlim=(0., .25), ylim=(15., 115.), xlabel="时间 t / s", ylabel="活塞位移 / mm")
    ax.yaxis.label.set_color(COLORS["sac_fp32"])
    voltage_ax = ax.twinx()
    line_voltage, = voltage_ax.plot(t, us, color="#208c38", ls="-.", lw=1.5, label="最小干预安全电压")
    voltage_ax.set_ylabel("安全电压 / V", color="#208c38")
    voltage_ax.tick_params(axis="y", colors="#208c38")
    voltage_ax.set_ylim(min(-4., float(us[shown].min()) - .5), max(10., float(us[shown].max()) + .5))
    ax.set_title("(b) 二阶约化模型：HOCBF-QP 行程防撞", pad=12)
    ax.legend(handles=[line_unsafe, line_safe, line_voltage], loc="lower right", framealpha=.95)
    intervention = summary.get("first_intervention_s")
    detail = (f"首次干预 {intervention:.3f} s\n" if intervention is not None else "未触发干预\n")
    detail += (f"最低安全电压 {summary['minimum_safe_voltage_v']:.3f} V\n"
               f"{summary['duration_s']:g} s 位置 {summary['safe_final_position_mm']:.3f} mm")
    ax.text(.53, .57, detail, transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec=COLORS["sac_fp32"], alpha=.95))

    for ax, column, ylabel, title in (
        (axes[1, 0], "velocity_error_mms", "速度跟踪误差 v − v_ref / (mm/s)",
         "(c) 5000 N 阶跃负载 + 体积模量减半：速度误差"),
        (axes[1, 1], "control_v", "实际阀控制电压 / V",
         "(d) 四策略控制电压与混合精度定点闭环验证"),
    ):
        times = data[POLICIES[0]]["time_s"]
        ax.axvspan(fault_time, times[-1], color="#eeeeee", zorder=0)
        ax.axvline(fault_time, color="#777777", ls="--", lw=.9)
        ax.axhline(0. if column == "velocity_error_mms" else 10., color="#777777", ls=":", lw=.8)
        for policy in POLICIES:
            ax.plot(data[policy]["time_s"], data[policy][column], color=COLORS[policy],
                    ls=STYLES[policy], lw=1.8 if policy == "sac_fp32" else 1.5, label=LABELS[policy])
        ax.set_xlim(float(times[0]), float(times[-1]))
        ax.set_xlabel("全局时间 t / s")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=12)
        ax.legend(loc="best", framealpha=.95)

    ax = axes[1, 0]
    fp = data["sac_fp32"]
    post = fp["time_s"] >= fault_time - 1e-12
    peak_error = float(np.abs(fp["velocity_error_mms"][post]).max())
    recovery = _recovery_time(fp["time_s"], fp["velocity_error_mms"], fault_time)
    text = f"SAC 故障后最大绝对误差 {peak_error:.2f} mm/s\n"
    text += f"持续恢复至 ±1 mm/s：{recovery:.3f} s" if recovery is not None else "本窗口未持续恢复至 ±1 mm/s"
    ax.text(.45, .47, text, transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=.3", fc="white", ec=COLORS["sac_fp32"], alpha=.95))
    fig.text(.065, .065, "(a) 为实际 SAC 日志；部署均值另经离线雅可比精修。(c)(d) 四策略共用物理对象，FP32/INT8 分别运行自身闭环。", fontsize=8.5, color="#444444")
    fig.text(.065, .04, f"(b) 为独立教学约化模型 ẋ=v，v̇={summary['k_gain_mm_s2_per_v']:g}u−{summary['damping_per_s']:g}v；其二阶证书不能直接转移到四状态液压对象。", fontsize=8.5, color="#444444")
    paths = [out / "chapter8_comparison.png", out / "chapter8_comparison.pdf"]
    for path in paths:
        fig.savefig(path, facecolor="white")
    plt.close(fig)
    return [str(path) for path in paths]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for generated in generate_figures(args.root):
        print(generated)
