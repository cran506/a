# 第五章数字 PID 闭环实验

基于用户提供的第3/4/5章PPT、已有 `HydraulicTwin` 与 Word 中的 PID、S 型轨迹两段代码完成。包括三组课堂实验及第五章课后习题5-1，所有图表来自数值积分，没有预设误差、增益衰减或振荡曲线。

## 运行

需要 Python 3.10 或更高版本。在解压后的本目录执行：

```bash
python -m pip install -r requirements.txt
python digital_pid_lab.py
python -m unittest -v test_chapter5
```

程序没有交互弹窗，默认输出到 `outputs/`。在云环境中使用现有检出目录，无需新建 Git worktree。

- `outputs/实验报告.docx`：带参数、实测指标和图表的可编辑报告。
- `outputs/实验报告.md`：同内容文本报告。
- 五组 PNG/PDF 图：课堂总览、液压机理、噪声隔离，以及课后原式/滤波对比。
- 八份 CSV 与 `summary.json`：完整时间同步日志及定量指标。

常见参数调整：

```bash
python digital_pid_lab.py --output outputs_other --load-force 180000 --valve-tau 0.015
```

`--load-force` 为课堂阻力 N，`--valve-tau` 为阀滞后 s（0表示理想阀），`--classroom-ki`、`--classroom-kd` 仅调整课堂增益，`--homework-duration` 必须为1ms的整数倍且不小于1.2s。修改参数后应重新评估性能，不能将附图的理想表现视为保证。

## 代码对应

| 文件 | 内容 |
|---|---|
| `hydraulic_digital_twin.py` | 上传的原始 twin，逐字节保留；其原有开环演示不在本实验中执行 |
| `physical_config.json` | 原 twin 演示的23个物理参数，不冒充第4章辨识结果 |
| `twin_adapter.py` | 电压归一化、时窗外载、阀动态、压差回流及低压气穴近似 |
| `controllers.py` | Word 的增量式 PID 与习题位置式 PID |
| `trajectory.py` | 七段式急动度有界的解析 S 曲线 |
| `simulation.py` | 1kHz采样、量化、ZOH、Radau及指标提取 |
| `digital_pid_lab.py` | 一次运行七个闭环实验、隔离噪声实验、图表与报告 |
| `test_chapter5.py` | 运动学、PID公式、ADC和孪生接口的有意义验证 |

## 参数与结果的边界

PID 输入位置单位为 m，内部误差换算为 cm，增益单位分别为 V/cm、V/(cm·s)、V·s/cm。实际电压限幅±10V后除10传入 twin。传感器跨度300mm、12位，LSB=0.073242mm。

课堂 `Kp=3.8/16/3.8`、`alpha=.85/.85/0` 使用 Word 积分分离（误差≥1.5mm时积分为5%）；Ki=22、Kd=.06为本程序公开补充的默认值，Word未给课堂Ki/Kd。习题固定 `Kp=4.2, Ki=22, Kd=.06`，两组关闭积分分离，分别以α=0复现教材原式，再以α=.85做共同滤波补充。

加载采用[0.35,0.70)s内180kN外部阻力，直接进入力平衡，Kq由实测腔压计算。该负载幅值未在上传材料中给出，是可修改的实验假设。四阶液压状态保留；额外阀芯动态沿用原示例15ms，形成4+1状态。原点放在约50mm并按ADC代码校准，以免把后端盖碰撞与位置跟踪混在一起。

全时 `kq_m2_s` 表示按当前P1计算的伸出支路潜在增益；阀反向时它不是当前反向支路的局部增益。默认轻/重载统计窗均为正向伸出，因此增益衰减统计有效；Ps≤P1时CSV留NaN而不强制构造正向增益。

原模型对反压差缺少流向符号，且低压残余模量不能阻止负绝对压力。适配层依据第3章恢复signed-square-root回流，令低压有效体积模量在2kPa蒸气压处趋零，不裁剪结果压力；这是等效气穴近似，未包含完整多相流、补油阀和溢流阀。

未提供第4章测量数据，因此没有重新辨识参数。示例图的65%、17.4mm、保守组无振荡和0.05mm精度不能视为已验证值。报告逐项记录当前参数下的真实结果，并将采样误差过零与真实位置过零分开。

课后阶跃的零历史状态产生微分kick；限幅后的增量输出可能在误差过零前先退出正饱和。因此首次退出减过零时间可以为负，硬限幅不能保证所有误差反号时立刻退饱和，也不能保证全工况稳定。

已使用 NumPy 2.3.5、SciPy 1.17.0、Matplotlib 3.10.8、python-docx 1.2.0 执行验证。
