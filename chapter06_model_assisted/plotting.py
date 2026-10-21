"""第六章三策略四联图：直接绘制数值积分记录，注释均由本次数据计算。"""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "chapter06-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "chapter06-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


STYLES = (
    ("pid", "纯 PID 反馈", "#2254d9", (0, (5, 2.4))),
    ("nominal", "标称前馈 + PID", "#ee8c18", (0, (5, 2, 1, 2))),
    ("full", "完整模型辅助", "#db2427", "-"),
)


def _settings():
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        chinese = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        names = {font.name for font in font_manager.fontManager.ttflist}
        chinese = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP") if name in names), "DejaVu Sans")
    return {
        "font.family": "sans-serif", "font.sans-serif": [chinese, "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10.5,
        "axes.labelsize": 10.5, "axes.titlesize": 11.5,
        "legend.fontsize": 9.1, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "figure.dpi": 110, "savefig.dpi": 300,
    }


def _validate(results):
    required = ("time_s", "error_m", "voltage_v", "velocity_m_s", "reference_velocity_m_s")
    base = None
    for key, *_ in STYLES:
        if key not in results:
            raise ValueError(f"缺少策略结果：{key}")
        data = results[key]["data"]
        t = np.asarray(data["time_s"])
        if t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0):
            raise ValueError(f"{key} 的时间必须为严格递增的一维数组")
        for name in required:
            a = np.asarray(data[name])
            if a.shape != t.shape or not np.all(np.isfinite(a)):
                raise ValueError(f"{key}.{name} 必须是与时间同形的有限数组")
        if t[0] > 0 or t[-1] < .62:
            raise ValueError("数据需要覆盖 0–0.62 s 观察窗口")
        if base is not None:
            for name in ("time_s", "reference_velocity_m_s"):
                if not np.array_equal(base[name], data[name]):
                    raise ValueError("三种策略必须使用相同时间与参考速度")
        base = data


def _num(value, places=3):
    if value is None or not np.isfinite(float(value)):
        return "未观测到"
    return f"{float(value):.{places}f}"


def _motion_start(data):
    moving = np.flatnonzero(np.abs(data["velocity_m_s"]) > 1e-4)
    return float(data["time_s"][moving[0]]) * 1000 if moving.size else None


def _stats(data):
    t = np.asarray(data["time_s"])
    window = (t >= .5) & (t <= .62)
    e = np.asarray(data["error_m"])
    v = np.asarray(data["velocity_m_s"])
    return {
        "error": float(np.max(np.abs(e))) * 1000,
        "start": _motion_start(data),
        "vmin": float(np.min(v[window])),
    }


def _limits(ax, values, padding=.13):
    values = np.concatenate([np.ravel(value) for value in values])
    lo, hi = float(np.min(values)), float(np.max(values))
    span = hi - lo
    if span < 1e-10:
        span = max(abs(hi), 1.) * .08
    ax.set_ylim(lo - span * padding, hi + span * padding)


def _note(ax, text, xy, pos, color):
    ax.annotate(text, xy=xy, xycoords="data", xytext=pos, textcoords="axes fraction",
                color=color, fontsize=8.9, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=.32", fc="white", ec=color, lw=.85, alpha=.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.1),
                annotation_clip=True)


def plot_comparison(results, output):
    """输出 PNG 与可缩放 PDF，返回 PNG 的 Path。

    results 的 pid/nominal/full 各含 data 字典；本函数不平滑、不裁剪数据。
    起动指标定义为 |v| > 10⁻⁴ m/s 的首个采样时刻，不等同于静摩擦释放事件。
    """
    _validate(results)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = {key: results[key]["data"] for key, *_ in STYLES}
    t = np.asarray(data["full"]["time_s"])
    statistics = {key: _stats(value) for key, value in data.items()}
    cfg = results["full"].get("config", {})
    load_start, load_end = float(cfg.get("load_start", .5)), float(cfg.get("load_end", .92))
    with plt.rc_context(_settings()):
        fig, axes = plt.subplots(2, 2, figsize=(16.8, 9.0))
        ax_e, ax_u, ax_s, ax_l = axes.flat
        model = cfg.get("model", "four_state_hydraulic_twin")
        profile = "课堂等效模型（准静态压力）" if model != "four_state_hydraulic_twin" else "四阶液压孪生（有限建压与阀动态）"
        fig.suptitle("第六章  模型辅助复合控制：起动死区补偿与突变负载抗扰动", fontsize=17, y=.974)
        fig.text(.525, .924, profile, ha="center", fontsize=10.5, color="#526072")
        fig.subplots_adjust(left=.065, right=.985, top=.885, bottom=.16, hspace=.39, wspace=.17)
        for ax in axes.flat:
            ax.grid(True, color="#b2bac3", ls=":", alpha=.63, lw=.65)
            ax.set_axisbelow(True)
            ax.spines["top"].set_color("#9da5ad")
            ax.spines["right"].set_color("#9da5ad")
            ax.tick_params(length=3.2)
        for ax in (ax_e, ax_u):
            ax.axvspan(load_start, load_end, color="#e1e4e8", alpha=.65, zorder=0)
            ax.set_xlim(float(t[0]), float(t[-1]))
            ax.set_xlabel("时间 $t$（s）")
        ax_e.axhline(0, color="#707880", lw=.8, ls=":")
        ax_e.set_title("(a) 全行程位置跟踪误差  $e(t)=r(t)-x_p(t)$", pad=12)
        ax_e.set_ylabel("位置跟踪误差（mm）")
        ax_u.set_title("(b) 全行程控制电压：起动前馈与重载增益补偿", pad=12)
        ax_u.set_ylabel("控制电压 $u$（V）")
        ax_s.set_title("(c) 起动瞬态放大（0–0.16 s，对应教材图 6.4）", pad=12)
        ax_l.set_title("(d) 负载切换放大（0.48–0.62 s，对应教材图 6.5）", pad=12)
        for ax, lo, hi in ((ax_s, 0., .16), (ax_l, .48, .62)):
            m = (t >= lo - 1e-12) & (t <= hi + 1e-12)
            ax.plot(t[m], np.asarray(data["full"]["reference_velocity_m_s"])[m],
                    color="#969ca4", lw=2.7, label="参考速度 $v_d$", zorder=2)
            ax.set_xlim(lo, hi)
            ax.set_xlabel("起动时间 $t$（s）" if ax is ax_s else "负载切换时间 $t$（s）")
            ax.set_ylabel("活塞速度 $v_p$（m/s）")
        for key, label, color, dash in STYLES:
            width = 1.9 if key == "full" else 1.55
            ax_e.plot(t, np.asarray(data[key]["error_m"]) * 1000,
                      color=color, ls=dash, lw=width, label=label)
            ax_u.plot(t, data[key]["voltage_v"], color=color, ls=dash, lw=width, label=label)
            for ax, lo, hi in ((ax_s, 0., .16), (ax_l, .48, .62)):
                m = (t >= lo - 1e-12) & (t <= hi + 1e-12)
                ax.plot(t[m], np.asarray(data[key]["velocity_m_s"])[m],
                        color="#16191d" if key == "full" else color,
                        ls=dash, lw=width, label=label, zorder=3)
        for ax, field, lo, hi in ((ax_s, "velocity_m_s", 0., .16), (ax_l, "velocity_m_s", .48, .62)):
            m = (t >= lo - 1e-12) & (t <= hi + 1e-12)
            values = [np.asarray(data[key][field])[m] for key, *_ in STYLES]
            values.append(np.asarray(data["full"]["reference_velocity_m_s"])[m])
            _limits(ax, values, padding=.17)
        _limits(ax_e, [np.asarray(value["error_m"]) * 1000 for value in data.values()], padding=.16)
        _limits(ax_u, [np.asarray(value["voltage_v"]) for value in data.values()], padding=.16)
        ax_e.legend(loc="upper right", framealpha=.96, edgecolor="#d3d8de")
        ax_u.legend(loc="upper right", framealpha=.96, edgecolor="#d3d8de")
        ax_s.legend(loc="upper left", framealpha=.96, edgecolor="#d3d8de")
        ax_l.legend(loc="lower right", framealpha=.96, edgecolor="#d3d8de")
        # 全部标注均取自输入记录，未将课件中的目标数字作为实测值。
        pid_error = np.asarray(data["pid"]["error_m"]) * 1000
        peak = int(np.argmax(np.abs(pid_error)))
        _note(ax_e, f"全程最大绝对误差\nPID：{statistics['pid']['error']:.3f} mm\n完整控制：{statistics['full']['error']:.3f} mm",
              (t[peak], pid_error[peak]), (.035, .94), "#2254b5")
        light = (t >= .42) & (t < .49)
        heavy = (t >= .56) & (t <= .60)
        if np.any(light) and np.any(heavy):
            volts = np.asarray(data["full"]["voltage_v"])
            _note(ax_u, f"完整控制实测\n起动首指令 {volts[0]:.3f} V\n轻载/重载中位 {np.median(volts[light]):.2f} / {np.median(volts[heavy]):.2f} V",
                  (t[np.flatnonzero(heavy)[0]], volts[np.flatnonzero(heavy)[0]]), (.035, .93), "#c72d30")
        load_mask = np.flatnonzero((t >= .5) & (t <= .62))
        v_pid = np.asarray(data["pid"]["velocity_m_s"])
        trough = int(load_mask[np.argmin(v_pid[load_mask])])
        _note(ax_l, f"窗口最低速度\nPID {statistics['pid']['vmin']:.4f} m/s\n完整控制 {statistics['full']['vmin']:.4f} m/s",
              (t[trough], v_pid[trough]), (.035, .93), "#c72d30")
        lines = []
        for key, label, *_ in STYLES:
            s = statistics[key]
            lines.append(f"{label}：全程 |e|max = {s['error']:.4f} mm；起动速度达阈值 = {_num(s['start'], 1)} ms；负载窗口 vmin = {s['vmin']:.4f} m/s。")
        fig.text(.065, .062, "\n".join(lines), fontsize=9.6, color="#24334b", va="center", linespacing=1.65,
                 bbox=dict(boxstyle="round,pad=.5", fc="#f2f5fa", ec="#d2dae6", lw=.8))
        fig.text(.985, .015, r"起动判据：$|v| > 10^{-4}$ m/s；灰色区域表示工艺负载窗口。图线与注释均来自本次仿真。",
                 fontsize=8.5, color="#626e7c", ha="right", va="bottom")
        path = output / "model_assisted_comparison.png"
        fig.savefig(path, dpi=300, facecolor="white")
        fig.savefig(output / "model_assisted_comparison.pdf", facecolor="white")
        plt.close(fig)
    return path
