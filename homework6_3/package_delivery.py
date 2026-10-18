"""用作业6-3已有结果生成 Word 报告、SHA256 清单和完整 ZIP，不执行仿真。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parent
REPORT_NAME = "第六章作业6-3_死区补偿报告.docx"
ARCHIVE_NAME = "第六章作业6-3_完整交付.zip"
KINDS = ("under", "hard_over", "smooth")
LABELS = {"under": "A 欠补偿", "hard_over": "B 硬过补偿", "smooth": "C tanh 光滑补偿"}
EXCLUDED_DIRS = {"__pycache__", ".cache", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value, places=4):
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "通过" if value else "未通过"
    number = float(value)
    if round(number, places) == 0:
        number = 0.
    return f"{number:.{places}f}"


def _table(document, headers):
    table = document.add_table(rows=1, cols=len(headers))
    table.autofit = True
    for cell, text in zip(table.rows[0].cells, headers):
        cell.text = text
        for run in cell.paragraphs[0].runs:
            run.bold = True
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "left", "right", "insideH", "insideV"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "single" if edge in ("top", "bottom") else "nil")
        element.set(qn("w:sz"), "8")
        element.set(qn("w:color"), "555555")
        borders.append(element)
    table._tbl.tblPr.append(borders)
    for cell in table.rows[0].cells:
        properties = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        element = OxmlElement("w:bottom")
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "6")
        borders.append(element)
        properties.append(borders)
    return table


def _status(validation):
    for key in ("passed", "all_passed", "all_checks_passed"):
        if isinstance(validation.get(key), bool):
            return "全部通过" if validation[key] else "未全部通过"
    return "以 validation.json 各项记录为准（未记录统一结论）"


def _figure(requested=None):
    if requested:
        path = Path(requested)
        path = path if path.is_absolute() else ROOT / path
    else:
        candidates = sorted((ROOT / "outputs").glob("*.png"))
        if len(candidates) != 1:
            raise RuntimeError("outputs 中须有唯一六面板 PNG，或用 --figure 指定")
        path = candidates[0]
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise RuntimeError("请先生成位于本作业目录内的结果图")
    return path


def _small(value):
    if value is None:
        return "未记录"
    value = float(value)
    return f"{value:.3e}" if 0 < abs(value) < 1e-4 else f"{value:.6f}"


def _robustness_section(document):
    path = ROOT / "outputs/robustness.json"
    if not path.is_file():
        return
    result = _json(path)
    document.add_heading("补充：测量噪声与零参考保持", 2)
    noise = result.get("noise_sensitivity", {})
    runs = noise.get("beta_runs", [])
    if runs:
        document.add_paragraph(
            f"补充噪声试验采用标准差 {_number(noise.get('std_mm_s'), 3)} mm/s、"
            f"seed={noise.get('seed', '未记录')} 的共同加性测量噪声；"
            f"重建噪声序列一致性检查：{_number(noise.get('shared_additive_noise_verified'))}。"
            "这些结果独立保存，未改变无噪声主图或平滑实际速度。")
        table = _table(document, ("β / V⁻¹", "最长关闭 / ms", "高通峰峰值 / V", "全程 RMSE / mm/s"))
        for item in sorted(runs, key=lambda item: item["beta"]):
            metrics = item["metrics"]
            values = (item["beta"], metrics.get("longest_flow_closed_ms"),
                      metrics.get("chatter_voltage_pp_v"), metrics.get("velocity_rmse_mm_s"))
            for cell, value, places in zip(table.add_row().cells, values, (0, 3, 5, 5)):
                cell.text = _number(value, places)
    hold = result.get("zero_reference_hold", {})
    normal, counter = hold.get("nominal_feedback_runs", {}), hold.get("kp0_counterfactual_runs", {})
    if normal and counter:
        disabled = hold.get("all_error_feedback_disabled_when_kp0", False)
        counter_label = "关闭误差反馈" if disabled else "只关闭 P"
        document.add_paragraph(
            f"保持试验从各策略正式轨迹 {hold.get('start_s', '未记录')} s 的实际阀位与速度继续，"
            f"保持零速度参考 {hold.get('duration_s', '未记录')} s，统计末 "
            f"{float(hold.get('tail_s', .1))*1000:g} ms。"
            "控制器历史由正式测量重建，不添加速度种子或测量噪声。"
            f"同一策略的反事实沿用相同真实起点，{counter_label}，检验反复翻转是否由反馈维持。")
        table = _table(document, ("策略", "原反馈速度\n峰峰值 mm/s", "原反馈电压\n峰峰值 V",
                                  f"{counter_label}速度\n峰峰值 mm/s", "有效估计频率\n/ Hz"))
        for kind in KINDS:
            if kind not in normal or kind not in counter:
                continue
            metrics, counter_metrics = normal[kind]["metrics"], counter[kind]["metrics"]
            frequency = metrics.get("frequency_from_median_upcrossing_period_hz")
            frequency_text = (_number(frequency, 2) if metrics.get("frequency_estimate_meaningful")
                              and frequency is not None else "不报告")
            values = (LABELS[kind], _small(metrics.get("tail_velocity_pp_mm_s")),
                      _small(metrics.get("tail_voltage_pp_v")),
                      _small(counter_metrics.get("tail_velocity_pp_mm_s")), frequency_text)
            for cell, text in zip(table.add_row().cells, values):
                cell.text = text
        hard = normal.get("hard_over", {}).get("metrics", {})
        if hard:
            if hard.get("preceding_tail_velocity_pp_mm_s") is not None:
                document.add_paragraph(
                    f"硬过补偿前一等长窗口速度峰峰值为 {_small(hard['preceding_tail_velocity_pp_mm_s'])} mm/s，"
                    f"末段/前段幅值比为 {_number(hard.get('tail_to_preceding_pp_ratio'), 4)}。"
                    "以相邻窗口幅值比较区分持续小幅运动与单次换向后逐渐衰减的振铃。")
            document.add_paragraph(
                f"硬过补偿末段记录速度过零 {hard.get('tail_speed_zero_crossings', '未记录')} 次、"
                f"完整同向上穿周期 {hard.get('tail_complete_upcrossing_cycles', '未记录')} 个。"
                "频率仅在速度幅度有诊断意义且至少存在两次同向上穿时估计；"
                "采用同向插值过零周期的中位数，而不是把单次脉冲频谱当振荡频率。"
                "该保持试验与无反馈反事实共同检验本采样等效模型的小幅极限环，"
                "不能推断任意液压对象的固有模态或全局稳定性。")
    if any(key in result for key in ("passed", "all_passed", "all_checks_passed")):
        document.add_paragraph("补充试验检查状态：" + _status(result) + "；条件及原始记录见 outputs/robustness.json。")


def write_report(summary, validation, figure):
    config, strategies = summary["config"], summary["strategies"]
    if not set(KINDS) <= set(strategies):
        raise RuntimeError("summary.json 必须包含 under、hard_over、smooth 三策略")
    required = ("sample_time_s", "duration_s", "amplitude_v", "frequency_hz", "physical_deadzone_v",
                "under_estimate_v", "over_estimate_v", "beta_default", "gain_m_s_v", "tau_valve_s",
                "tau_velocity_s", "kp", "ki", "kd", "zoom_start_s", "zoom_end_s",
                "velocity_flat_threshold_mm_s", "chatter_cutoff_hz")
    if not set(required) <= set(config):
        raise RuntimeError(f"报告参数缺失：{sorted(set(required)-set(config))}")
    c = config
    document = Document()
    for section in document.sections:
        section.top_margin = section.bottom_margin = Inches(.65)
        section.left_margin = section.right_margin = Inches(.65)
    for name in ("Normal", "Title", "Heading 1", "Heading 2"):
        style = document.styles[name]
        style.font.name = "Noto Sans CJK SC"
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Noto Sans CJK SC")
    document.styles["Normal"].font.size = Pt(10)
    document.add_heading("《智能流体传动与控制》作业 6-3", 0)
    document.add_paragraph("死区逆补偿失配的鲁棒性边界与 tanh 光滑化")
    document.add_heading("1. 任务与共同实验条件", 1)
    document.add_paragraph(
        "按第六章 PPT 课后习题 6-3，比较欠补偿、硬过补偿和连续 tanh 逆补偿，"
        "提取全周期及第二次过零附近的控制电压、实际阀口有效信号和速度，"
        "并用 β 扫描量化平台、颤振和跟踪误差之间的取舍。原材料保存在 materials/。")
    parameters = _table(document, ("参数", "本次设置"))
    rows = (("参考指令", f"u_ref={c['amplitude_v']:g} sin(2π·{c['frequency_hz']:g}t) V"),
            ("控制采样 / 仿真时长", f"{c['sample_time_s']*1000:g} ms / {c['duration_s']:g} s"),
            ("真实阀芯死区", f"±{c['physical_deadzone_v']:g} V"),
            ("欠补偿 / 过补偿估计", f"{c['under_estimate_v']:g} / {c['over_estimate_v']:g} V"),
            ("基准 β / 扫描", f"{c['beta_default']:g} V⁻¹ / {c.get('beta_values', [])}"),
            ("等效速度增益 Kv", f"{c['gain_m_s_v']:g} m/(s·V)"),
            ("阀 / 速度时间常数", f"τv={c['tau_valve_s']*1000:g} ms；τm={c['tau_velocity_s']*1000:g} ms"),
            ("反馈参数", f"Kp={c['kp']:g}，Ki={c['ki']:g}，Kd={c['kd']:g}"),
            ("测量噪声标准差", f"{c.get('measurement_noise_std_mm_s', 0.):g} mm/s"),
            ("换向统计窗口", f"{c['zoom_start_s']:g}–{c['zoom_end_s']:g} s"))
    for label, value in rows:
        cells = parameters.add_row().cells
        cells[0].text, cells[1].text = label, value
    document.add_paragraph(
        f"速度参考为 v_d=Kv·u_ref，幅值 {c['gain_m_s_v']*c['amplitude_v']*1000:g} mm/s。"
        "Kv 用于参考图的速度标度，τm、τv 为参考所用的等效时间尺度；这些参数不是设备辨识结果。"
        "三组从共同零阀位、零速度初态出发，共用参考、反馈和采样时钟，只改变死区估计或逆映射。"
        "材料中的模块三原代码作为参考保留，本题实际入口使用局部等效模型接口。")
    document.add_heading("2. 模型、控制结构与数学机理", 1)
    document.add_paragraph(
        "本题采用局部低频阀–速度等效通道，不是完整双腔液压数字孪生：\n"
        "τv·du_v/dt=u−u_v；u_eff=sign(u_v)·max(|u_v|−δ,0)；\n"
        "τm·dv/dt+v=Kv·u_eff。\n"
        "其中 u_v 是实际阀芯的等效电压，u_eff 仍是电压信号，不是阀芯位移。"
        "真实死区作用在阀动态之后，不能直接用 deadzone(u) 替代 deadzone(u_v)。"
        "每个采样区间保持 u，求出阀芯穿越死区边界的精确时刻，按固定分支解析积分速度，"
        "未用 Euler 制造振荡，也未注入纹波或裁剪模型输出。")
    document.add_paragraph(
        "共同控制器为等效线性逆前馈加速度残差反馈，再整体通过死区逆映射：\n"
        "u_lin=(v_d+τm·a_d)/Kv+PID((v_d−v_meas)/Kv)；\n"
        "硬补偿：u=u_lin+δ_hat·sign(u_lin)，零位保护区输出为零；\n"
        "光滑补偿：u=u_lin+δ_hat·tanh(βu_lin)。\n"
        "实际执行电压最终限制在 ±10 V。反馈先参与 u_lin 合成，再进入逆映射；"
        "返回的补偿分量包含非线性映射作用，不能直接当作原始线性 PID 值。")
    if c["ki"] == 0 and c["kd"] == 0:
        document.add_paragraph(
            "本次只启用比例反馈，以隔离死区失配，不混入积分累积或微分噪声；"
            "代码保留数字 PID 接口。反馈误差以 Kv 归一化为等效电压。")
    document.add_paragraph(
        "忽略动态与零位保护时，欠补偿的复合静态通道留下 δ−δ_hat 的输入阈值。"
        "硬过补偿在零点两侧形成反向跳变，但 sign 本身不自动产生持续颤振："
        "只有反馈、采样及阀/速度动态使 u_lin 反复跨零，才形成闭环高频翻转。"
        "因此本例颤振幅值与频率依赖采样、反馈增益及等效对象参数，不能解释为固有液压模态。")
    document.add_paragraph(
        "有限 β 的 tanh 映射连续，却仍可能小于真实硬死区阈值。正残余阈值 z0 满足 "
        "z0+δ_hat·tanh(βz0)=δ；只要 z0>0，静态复合映射就仍有平台。"
        "β 增大通常缩小该静态平台，同时把逆映射零点斜率提高到 1+δ_hat·β，趋近硬补偿。"
        "离散闭环可能出现更多翻转、极限环甚至失稳风险；是否出现以及表现多大必须看本次数据，"
        "不能预设 β=18 在所有指标上最好。")
    document.add_heading("3. 指标口径与三策略实测结果", 1)
    document.add_paragraph(
        f"流量关闭平台按换向窗口内 |u_v|≤{c['physical_deadzone_v']:g} V 统计，"
        "对相邻采样作分段线性阈值重建，报告最长连续时长及累计时长；不是连续时间精确事件长度。"
        f"速度近零另用 |v|≤{c['velocity_flat_threshold_mm_s']:g} mm/s 判据。"
        "阀口关闭时速度仍可因惯性衰减，所以两个平台指标不等价，也不能把近零速度称为严格静止。")
    document.add_paragraph(
        f"颤振峰峰值取完整控制电压经 4 阶 {c['chatter_cutoff_hz']:g} Hz Butterworth "
        "高通的零相位 sosfiltfilt 分量，再在换向窗口内计算。该离线过滤只定义诊断指标，"
        "不进入反馈或结果曲线；并列相邻采样最大跳变和 u_lin 符号翻转数，避免把低频正弦跨度误称颤振。"
        "单次快速换向也包含高频频谱，不能只凭高通峰峰值断言持续极限环。"
        "速度转折次数和超额总变差（总变差减首尾净变差）用于补充识别真实模型速度的反复变化。"
        "速度 RMSE 相对线性标度参考计算，分别列全周期与换向窗口值。")
    table = _table(document, ("指标", *[LABELS[k] for k in KINDS]))
    metric_rows = (("longest_flow_closed_ms", "最长流量关闭平台 / ms", 3),
                   ("total_flow_closed_ms", "累计流量关闭时长 / ms", 3),
                   ("longest_velocity_near_zero_ms", "最长速度近零时长 / ms", 3),
                   ("chatter_voltage_pp_v", "高通颤振峰峰值 / V", 5),
                   ("chatter_voltage_rms_v", "高通颤振 RMS / V", 5),
                   ("max_voltage_step_v", "窗口最大采样电压跳变 / V", 5),
                   ("linear_sign_reversals", "窗口 u_lin 符号翻转次数", 0),
                   ("velocity_turning_points", "窗口真实速度转折次数", 0),
                   ("velocity_excess_variation_mm_s", "窗口速度超额总变差 / mm/s", 5),
                   ("velocity_rmse_mm_s", "全周期速度 RMSE / mm/s", 5),
                   ("zoom_velocity_rmse_mm_s", "窗口速度 RMSE / mm/s", 5),
                   ("max_abs_voltage_v", "全周期最大 |u| / V", 4),
                   ("sampled_saturation_count", "采样电压饱和点数", 0))
    for key, label, places in metric_rows:
        cells = table.add_row().cells
        cells[0].text = label
        for cell, kind in zip(cells[1:], KINDS):
            cell.text = _number(strategies[kind]["metrics"].get(key), places)
    section = document.sections[0]
    document.add_picture(str(figure), width=section.page_width-section.left_margin-section.right_margin)
    document.add_paragraph("图 1　原始采样记录的控制电压、实际阀芯死区后有效信号及真实速度；左列全周期，右列换向窗口。")
    document.add_heading("4. β 扫描与残余平台", 1)
    scan = sorted(summary.get("beta_scan", []), key=lambda item: item["beta"])
    if not scan:
        raise RuntimeError("summary.json 缺少 β 扫描结果")
    table = _table(document, ("β / V⁻¹", "最长关闭\n/ms", "颤振峰峰值\n/V", "全程 RMSE\n/mm/s",
                              "局部 RMSE\n/mm/s", "符号翻转\n/次"))
    for item in scan:
        m = item["metrics"]
        values = (item["beta"], m.get("longest_flow_closed_ms"), m.get("chatter_voltage_pp_v"),
                  m.get("velocity_rmse_mm_s"), m.get("zoom_velocity_rmse_mm_s"), m.get("linear_sign_reversals"))
        for cell, value, places in zip(table.add_row().cells, values, (0, 3, 5, 5, 5, 0)):
            cell.text = _number(value, places)
    static = summary.get("static_reference", {})
    if static:
        document.add_paragraph(
            f"无动态、无反馈的欠补偿静态平台参照为 {_number(static.get('under_static_plateau_ms'), 3)} ms。"
            "下表把有限 β 的静态正阈值与闭环实测平台分开列示；"
            "静态平台由参考正弦穿越 ±z0 的时长推得，不等于反馈、前馈相位与阀动态共同作用后的平台。")
        table = _table(document, ("β / V⁻¹", "静态阈值 z0 / V", "静态平台 / ms", "逆映射零点斜率"))
        for item in static.get("smooth", []):
            values = (item["beta"], item["linear_threshold_v"], item["static_plateau_ms"], item["inverse_zero_slope"])
            for cell, value, places in zip(table.add_row().cells, values, (0, 6, 3, 3)):
                cell.text = _number(value, places)
    comparisons = []
    for key, label in (("velocity_rmse_mm_s", "全周期 RMSE"),
                       ("longest_flow_closed_ms", "最长关闭平台"), ("chatter_voltage_pp_v", "颤振峰峰值")):
        candidates = [item for item in scan if item["metrics"].get(key) is not None]
        if candidates:
            best = min(candidates, key=lambda item: item["metrics"][key])
            comparisons.append(f"{label}最小对应 β={best['beta']:g}")
    if comparisons:
        document.add_paragraph("本次离散扫描：" + "；".join(comparisons)
                               + "。这些仅是已计算候选点上的单项比较，不是全局最优或所有采样条件下的保证。"
                               "符号频繁翻转可证明本次换向抖动，不能单凭该计数宣称状态发散。")
    if all(item["metrics"].get("velocity_turning_points") == 0 for item in scan):
        document.add_paragraph(
            "本次所扫 β 在换向窗口内的速度转折数均为 0；"
            "不能把较大 β 的高通峰峰值增加单独写成已发生持续机械极限环或状态发散。"
            "这是指定采样、反馈和模型下的数值实验，不对任意硬过补偿作 Lyapunov 稳定性断言。")
    document.add_heading("5. 验证、局限与复现", 1)
    document.add_paragraph("验证状态：" + _status(validation) + "。实际检查条件、误差及测试记录见 outputs/validation.json。")
    for key in ("test_count", "tests_run", "unit_test_count"):
        value = validation.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            document.add_paragraph(f"记录的回归测试数量：{int(value)}。")
            break
    checks = validation.get("checks")
    if isinstance(checks, dict):
        document.add_paragraph(f"验证文件含 {len(checks)} 项具名检查，未执行项不视为通过。")
        failed = [name for name, result in checks.items() if result is False or
                  (isinstance(result, dict) and result.get("passed") is False)]
        if failed:
            document.add_paragraph("未通过项：" + "、".join(failed) + "。")
    _robustness_section(document)
    document.add_paragraph(
        "本模型隔离研究死区和离散反馈，不包含双腔压力、真实摩擦、负载、压差流量增益变化或传感器全动态。"
        "不能据本次速度改善推断完整液压装置的稳定性或安全性。有限 β 保留残余平台，"
        "采样与增益也会改变颤振；参数不是设备辨识值。噪声若启用，各策略共用同一序列，"
        "默认设置及 seed 均随配置保存。")
    document.add_paragraph(
        "解压后进入 homework6_3：python -m pip install -r requirements.txt；python work_6_3.py；"
        "python -m unittest -v；其他独立复核命令见 README.md；python package_delivery.py。"
        "CSV 保留 N+1 个采样时刻，状态记录在本周期指令施加前，电压在随后一个采样区间保持。"
        "包内包含全部源码、配置、材料、三策略与 β 扫描 CSV、summary、validation、唯一六面板图和报告。"
        "manifest.json 记录相对路径、字节数与 SHA256；ZIP 与 manifest 自身不参与循环哈希。")
    path = ROOT / "deliverables" / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(path)
    return path


def included_files():
    return sorted((path for path in ROOT.rglob("*") if path.is_file()
                   and not EXCLUDED_DIRS.intersection(path.relative_to(ROOT).parts)
                   and path.name != "manifest.json"
                   and path.suffix.lower() not in {".zip", ".pyc", ".pyo", ".tmp"}),
                  key=lambda path: path.relative_to(ROOT).as_posix())


def package(figure=None):
    required = {"README.md", "requirements.txt", "experiment_config.json", "work_6_3.py", "simulation.py",
                "deadzone_model.py", "model_assisted_controller.py", "plotting.py", "package_delivery.py",
                "materials/chap06-ppt.pdf", "materials/module3_reference.py", "outputs/summary.json",
                "outputs/validation.json", "outputs/under.csv", "outputs/hard_over.csv", "outputs/smooth.csv",
                "outputs/beta_5.csv", "outputs/beta_18.csv", "outputs/beta_50.csv"}
    available = {path.relative_to(ROOT).as_posix() for path in included_files()}
    if not required <= available:
        raise RuntimeError(f"交付文件尚未齐全：{sorted(required-available)}")
    figure = _figure(figure)
    write_report(_json(ROOT / "outputs/summary.json"), _json(ROOT / "outputs/validation.json"), figure)
    files = included_files()
    manifest = {"project": "homework6_3", "files": {
        path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size,
                                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in files}}
    manifest_path = ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    destination = ROOT / "deliverables" / ARCHIVE_NAME
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, (Path(ROOT.name) / path.relative_to(ROOT)).as_posix())
    with ZipFile(destination) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise RuntimeError(f"ZIP 完整性检查失败：{broken}")
        for relative, entry in manifest["files"].items():
            payload = archive.read(f"{ROOT.name}/{relative}")
            if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"ZIP 文件与清单不符：{relative}")
    print(f"报告：{ROOT / 'deliverables' / REPORT_NAME}")
    print(f"ZIP：{destination}（{len(files)+1} 个文件，{destination.stat().st_size} 字节）")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", help="指定本作业目录内已生成的六面板 PNG")
    args = parser.parse_args()
    package(args.figure)


if __name__ == "__main__":
    main()
