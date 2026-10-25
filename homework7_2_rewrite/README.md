# 第七章习题 7-2：预测时域与制动限压（重写版）

本作业依据新上传的 `chap07-ppt.pdf` 第 15 页重写，使用四个物理状态 `[位移, 速度, P1, P2]` 和上一周期电压构成五维增广状态。固定同一套教学参数、目标权重与约束，只改变预测时域 `N=[5,10,15,20,30,40,50]`，每个采样周期通过 CVXPY/OSQP 求解一次 QP，仅执行序列首步。

[实验报告](REPORT.md)包含模型推导、课件对应关系、七组结果、最小制动距离讨论和实时性判断；[四联图](outputs/horizon_comparison.png)分别显示位移、电压、连续时间压力和“峰压—单次 QP 耗时”双轴权衡。交付文件位于 `deliverables/`。

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
| `plotting.py` | 从原始记录生成 PNG/PDF 四联图 |
| `verify.py` | 用独立 SI 方程和 DOP853 重放所有控制输入，核对连续压力及结果一致性 |
| `test_model.py`、`test_mpc.py` | 单位、符号、ZOH、预测矩阵、QP 等价性与约束测试 |
| `outputs/metrics.csv`、`summary.json` | 七组定量指标及环境信息 |
| `outputs/Nxx.csv`、`Nxx_qp.json` | 实际施加电压、测量状态、每次 QP 完整计划及诊断 |
| `outputs/Nxx_pressure_trace.csv` | 连续时间轨迹上的压力采样与极值根 |
| `outputs/textbook_matrices_N20.npz` | 可直接检查的五维增广与凝聚矩阵 |
| `outputs/validation.json` | 独立验证结论 |
| `materials/` | 本次上传课件与提取文本 |

## 本次主要结果与适用范围

N=5 的位移超调为 **0.866%**，N=20 为 **0.271%**，N=50 为 **0.325%**。七组连续时间 P2 峰值约 **20.83 MPa**，两腔均处于 0–21 MPa。底层单次 QP 平均耗时分别为 **0.105、1.148、5.862 ms**（N=5/20/50），完整控制步骤均值为 **1.317、2.443、7.487 ms**。本机实现未通过 1 ms 整周期实时验收。

控制量是零偏置等效模型电压，假设标称值为 0 V、可用范围为 ±10 V；如实机标称电压非零，须按实际总电压平移约束。该局部净流量模型不辨识真实阀芯的标称开度或结构。

这是课件线性化框架的**固定工作点 LTI 特例**：冻结 Kq、Kc 和容积，便于隔离 N 的影响，未声称实现了工况时变的全孔口非线性对象。初始压力为 P1=5 MPa、P2=2 MPa，4000 N 恒定预载荷使其静力平衡。常数 Kc 表示工作点附近的等效压力阻尼；模型压力回归平衡点不等于真实闭中位阀自动卸压。

图像与参考图保留同类比较关系，但数值不完全重合：本模型巡航 P2 约 7.75 MPa，约 0.072 s 首次到达目标，峰压留有数值裕量；课件未提供唯一的完整物理参数或指定处理器。详细差异和工程整定结论见报告。
