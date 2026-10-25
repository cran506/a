# 第七章习题 7-2：预测时域与制动限压（重写版）

本作业依据新上传的 `chap07-ppt.pdf` 第 15 页重写，使用四个物理状态 `[位移, 速度, P1, P2]` 和上一周期电压构成五维增广状态。固定同一套教学参数、目标权重与约束，只改变预测时域 `N=[5,10,15,20,30,40,50]`，每个采样周期通过 CVXPY/OSQP 求解一次 QP，仅执行序列首步。

[实验报告](REPORT.md)包含模型推导、课件对应关系、七组结果、最小制动距离讨论和实时性判断；[四联图](outputs/horizon_comparison.png)分别显示位移、电压、连续时间压力和“峰压—单次 QP 耗时”双轴权衡。[制动窗口放大图](outputs/pressure_braking_detail.png)将压力与对应电压放在共同时间轴上，便于比较短时域持续限压和长时域提前降压。交付文件位于 `deliverables/`。

## 运行

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python work_7_2.py
.venv/bin/python verify.py
.venv/bin/python -m unittest -v
```

Windows 使用 `.venv\Scripts\python.exe`，可省略命令前的线程环境变量。脚本本身还会通过 `threadpoolctl` 将计算线程数限制为 1。单组调试示例：

```bash
python work_7_2.py --horizon 20 --output outputs_N20
```

完整扫描保存在 `outputs/`。`--config` 可指定另一份 JSON 参数文件；改变任何模型、权重、约束或求解器参数后，应重新运行完整扫描与独立验证。CPU、系统负载和库版本会改变耗时，不能沿用此次记录的毫秒数。

需要只重绘图时运行 `python plotting.py`。安装 [Pandoc](https://pandoc.org/installing.html) 后，运行 `python build_delivery.py` 可从已验证结果更新报告表格，生成带公式的 Word 报告、ZIP 和 SHA256 清单；改变模型后仍须重新审查报告分析文字。

## 文件与检查范围

| 文件 | 作用 |
| --- | --- |
| `config.json` | 全部物理、控制、采样参数；各时域共用 |
| `model.py` | 力学与流量连续方程、四状态局部灰盒、精确 ZOH、采样间压力极值 |
| `mpc.py` | 五维增广、F/Phi/H、稀疏与凝聚 QP；主程序选用固定矩阵的凝聚形式 |
| `work_7_2.py` | 七组闭环、控制台表与原始数据生成 |
| `plotting.py` | 从原始记录生成 PNG/PDF 四联图和独立制动窗口放大图 |
| `pressure_metrics.py` | 从高密压力记录线性插值阈值交点，统计所有 N 共用的 20 MPa 暴露时间 |
| `verify.py` | 用独立 SI 方程和 DOP853 重放所有控制输入，核对连续压力及结果一致性 |
| `test_model.py`、`test_mpc.py`、`test_pressure_experiment.py` | 单位、符号、ZOH、QP 等价性，以及压力分离、稳定收敛和连续安全的八项测试 |
| `outputs/metrics.csv`、`summary.json` | 七组定量指标及环境信息 |
| `outputs/Nxx.csv`、`Nxx_qp.json` | 实际施加电压、测量状态、每次 QP 完整计划及诊断 |
| `outputs/Nxx_pressure_trace.csv` | 连续时间轨迹上的压力采样与极值根 |
| `outputs/textbook_matrices_N20.npz` | 可直接检查的五维增广与凝聚矩阵 |
| `outputs/validation.json` | 独立验证结论 |
| `materials/` | 本次上传课件与提取文本 |
| `pressure_revision_comparison/` | 修订前参数、图、结果与三组原始压力记录，以及可复核的新旧差异指标 |

## 本次主要结果与适用范围

本次修订针对“三组压力几乎重合”的问题，统一采用 `m=50 kg`、`σ=2500 N·s/m`、`V1=8×10⁻⁴ m³`，共同速度权重为 `qv=0.5`。无杆腔局部压力释放时间常数由原来的 1 ms 增至 8 ms，超过 N=5 的预测视野；所有 N 仍使用同一套参数和权重。改动作用于对象和控制器预测模型，图中曲线取同次闭环记录。

N=5 的位移超调为 **1.391%**，N=20 为 **0.454%**，N=50 为 **0.345%**。在 **t=0.080 s**，三组 P2 分别为 **20.35、10.04、9.30 MPa**；P2≥20 MPa 的持续时间为 **5.07、3.73、3.33 ms**。三组峰压都接近同一安全边界，差异体现在何时开始制动、何时退出高压段及后续压力释放，而非峰值高度。

七组连续时间 P2 峰值约 **20.84–20.86 MPa**，两腔均处于 0–21 MPa。底层单次 QP 平均耗时分别为 **0.119、1.261、8.617 ms**（N=5/20/50），完整控制步骤均值为 **1.333、2.614、10.425 ms**。本机实现未通过 1 ms 整周期实时验收。

[新旧量化对照](pressure_revision_comparison/comparison.json)保留了原工况约 0.395 ms 的 N=5/20 高压持续时间差与本次约 1.345 ms 的差异；原始记录可逐项核对。持续时间采用高密轨迹上的线性阈值交点插值，曲线共用原始坐标。

控制量是零偏置等效模型电压，假设标称值为 0 V、可用范围为 ±10 V；如实机标称电压非零，须按实际总电压平移约束。该局部净流量模型不辨识真实阀芯的标称开度或结构。

这是课件线性化框架的**固定工作点 LTI 特例**：冻结 Kq、Kc 和容积，便于隔离 N 的影响，未声称实现了工况时变的全孔口非线性对象。初始压力为 P1=5 MPa、P2=2 MPa，4000 N 恒定预载荷使其静力平衡。常数 Kc 表示工作点附近的等效压力阻尼；模型压力回归平衡点不等于真实闭中位阀自动卸压。

图像与参考图保留同类比较关系，但数值不完全重合：本模型在 0.04 s 的 P2 约 0.35 MPa，接近恒速时约 5.1 MPa，约 0.078 s 首次到达目标；峰压留有数值裕量。课件未提供唯一的完整物理参数或指定处理器。详细差异和工程整定结论见报告。
