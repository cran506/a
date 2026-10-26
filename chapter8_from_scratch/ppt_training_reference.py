"""绘制 PPT 第 12 页的三阶段教学目标示意，绝不作为训练日志。

本脚本不读取模型或训练 CSV。端点/阶段取自课件，明确列出的中间
锚点仅用于表达 S 形教学关系。实际 SAC 曲线由 reporting.py 另行绘制。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter8_ppt_schematic_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "chapter8_ppt_schematic_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import numpy as np
from scipy.interpolate import PchipInterpolator


# The intermediate points are illustration design choices, not measured data.
STEPS_K = np.array([0., 50., 100., 150., 200., 300., 400., 1000.])
REWARD_TARGET = np.array([-180., -170., -145., -105., -65., -15., -2., -2.])
ENTROPY_TARGET = np.array([2.8, 2.75, 2.4, 1.9, 1.15, .5, .35, .35])


def draw_ppt_panel(ax):
    """Embed a clearly labelled teaching schematic in the classroom main figure."""
    x=np.linspace(0.,1000.,1001)
    handles=[]
    for lo,hi,color,label in ((0.,50.,'#eeeeee','阶段 I：随机探索（0–50k）'),
                              (50.,400.,'#e9edfb','阶段 II：策略成型（50k–400k）'),
                              (400.,1000.,'#e8f5e9','阶段 III：稳态微调（>400k）')):
        ax.axvspan(lo,hi,color=color,zorder=0);handles.append(Patch(facecolor=color,label=label))
    reward,=ax.plot(x,PchipInterpolator(STEPS_K,REWARD_TARGET)(x),lw=1.8,color='#3c67ee',label='奖励目标形态（左轴，示意）')
    ax.set(xlim=(0.,1000.),ylim=(-200.,20.),xlabel='训练交互步数（千步 / k Timesteps）',ylabel='平均奖励目标（教学示意）')
    ax.tick_params(axis='y',colors='#3c67ee');ax.yaxis.label.set_color('#3c67ee')
    right=ax.twinx()
    entropy,=right.plot(x,PchipInterpolator(STEPS_K,ENTROPY_TARGET)(x),'--',lw=1.4,color='#24282e',label='策略熵目标形态（右轴，示意）')
    right.set(ylim=(0.,3.2),ylabel='H(u) / nat（教学示意）')
    ax.set_title('(a) SAC 三阶段收敛目标：PPT 教学示意（非训练日志）',pad=12)
    ax.legend(handles=[*handles,reward,entropy],loc='center right',fontsize=6.8,framealpha=.96)
    right.annotate('稳态目标 H ≈ 0.35\n示意曲线不作为实测证据',xy=(805.,.35),xytext=(430.,.95),fontsize=7,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#888888'),arrowprops=dict(arrowstyle='->',lw=1.1))
    return right


def generate_ppt_reference(root=None):
    """Return paths to an explicitly labelled target schematic and its source JSON."""
    root = Path(root or Path(__file__).resolve().parent)
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    available = {item.name for item in font_manager.fontManager.ttflist}
    fonts = [name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Micro Hei",
                              "Microsoft YaHei", "SimHei", "DejaVu Sans") if name in available]
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": fonts,
                         "axes.unicode_minus": False, "font.size": 11,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        steps = np.linspace(0., 1000., 1001)
        rewards = PchipInterpolator(STEPS_K, REWARD_TARGET)(steps)
        entropy = PchipInterpolator(STEPS_K, ENTROPY_TARGET)(steps)
        if np.any(np.diff(rewards) < -1e-12) or np.any(np.diff(entropy) > 1e-12):
            raise AssertionError("Teaching-target interpolation must preserve monotonicity")

        fig, ax = plt.subplots(figsize=(12.8, 6.45))
        fig.subplots_adjust(left=.085, right=.915, top=.82, bottom=.18)
        fig.suptitle("PPT 三阶段目标示意（非训练日志）", fontsize=19, y=.965, weight="medium")
        fig.text(.5, .892, "第八章 8.5.1 节 · 图 8.7 的教学目标形态；实际训练记录另行保存",
                 ha="center", fontsize=10.5, color="#535b66")

        phases = [(0., 50., "#eeeeee", "阶段 I：随机探索（0–50k）"),
                  (50., 400., "#e9edfb", "阶段 II：策略成型（50k–400k）"),
                  (400., 1000., "#e8f5e9", "阶段 III：稳态微调（>400k）")]
        handles = []
        for left, right, color, label in phases:
            ax.axvspan(left, right, color=color, zorder=0)
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        ax.axvline(50., color="#a6adb8", lw=.6, alpha=.6)
        ax.axvline(400., color="#a6adb8", lw=.6, alpha=.6)
        reward_line, = ax.plot(steps, rewards, color="#3c67ee", lw=2.2,
                              label="平均奖励的教学目标（左轴）")
        ax.set(xlim=(0., 1000.), ylim=(-200., 20.),
               xlabel="训练交互步数（千步 / k Timesteps）", ylabel="平均奖励目标（示意）")
        ax.set_xticks(np.arange(0., 1001., 200.))
        ax.set_yticks(np.arange(-200., 1., 25.))
        ax.yaxis.label.set_color("#3c67ee")
        ax.tick_params(axis="y", labelcolor="#3c67ee")
        ax.grid(True, ls=":", lw=.7, alpha=.45)
        ax.set_axisbelow(True)

        right = ax.twinx()
        entropy_line, = right.plot(steps, entropy, color="#24282e", ls="--", lw=1.8,
                                   label="策略熵的教学目标（右轴）")
        right.set(ylim=(0., 3.2), ylabel="策略熵目标 H(u) / nat（示意）")
        right.set_yticks(np.arange(0., 3.01, .5))
        ax.legend(handles=[*handles, reward_line, entropy_line], loc="center right",
                  bbox_to_anchor=(.985, .61), fontsize=9, framealpha=.96, borderpad=.8)
        right.annotate("课件阶段 III 的目标：H ≈ 0.35\n此线不是智能体的实测策略熵",
                       xy=(800., .35), xytext=(445., .98), fontsize=9,
                       bbox=dict(boxstyle="round,pad=.4", fc="white", ec="#a0a8b4", alpha=.98),
                       arrowprops=dict(arrowstyle="->", lw=1.3, color="#323a45"))
        fig.text(.085, .087,
                 "来源：第八章 PPT 第 12 页；中间锚点以单调 PCHIP 插值表达阶段关系，未添加随机抖动。",
                 fontsize=9, color="#535b66")
        fig.text(.085, .045,
                 "本图不含 SAC 采样数据，不作为训练收敛、控制性能或安全性的验证证据。",
                 fontsize=9.4, color="#9a3c3c")
        paths = [out / "ppt_training_target.png", out / "ppt_training_target.pdf"]
        for path in paths:
            fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)

    ppt = root / "materials/chap08-ppt.pdf"
    metadata = {
        "artifact_type": "clearly_labelled_teaching_target_schematic",
        "contains_measured_training_data": False,
        "source": {"file": "materials/chap08-ppt.pdf", "page": 12, "section": "8.5.1",
                   "sha256": hashlib.sha256(ppt.read_bytes()).hexdigest() if ppt.is_file() else None},
        "source_explicit_targets": {
            "stage_I_steps": [0, 50000], "stage_I_entropy_approx": 2.8,
            "stage_II_steps": [50000, 400000], "stage_II_reward_description": "-180 to -15, then the steady platform",
            "stage_III_start_step": 400000, "stage_III_reward_approx": -2.,
            "stage_III_entropy_approx": .35},
        "interpolation": "shape-preserving monotone piecewise cubic Hermite (SciPy PchipInterpolator)",
        "anchors_steps_k": STEPS_K.tolist(),
        "anchors_reward_target": REWARD_TARGET.tolist(),
        "anchors_entropy_target": ENTROPY_TARGET.tolist(),
        "intermediate_anchor_scope": "Illustration design choices, not values extracted from an original training log",
        "random_jitter_added": False,
        "warning": "Not model output, not a training CSV, not convergence/performance/safety evidence",
    }
    meta_path = out / "ppt_training_target.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [str(path) for path in [*paths, meta_path]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    for result in generate_ppt_reference(args.root):
        print(result)
