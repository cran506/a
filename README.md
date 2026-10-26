# 智能流体传动与控制课堂编程

## 第八章：SAC、HOCBF 与剪枝 INT8

第八章独立项目提供真实百万步 SAC 训练、液压四状态故障对比、可配置二阶 HOCBF 安全实验和独立整数网络推理。新版补充经验 Z-score 等价坐标变换，采用 70% 固定掩码剪枝与逐通道 INT8 导出；原始日志、策略文件与实测指标随代码一起保存。

- [完整项目与运行说明](chapter8_latest/README.md)
- [第八章运行入口](chapter8_latest/drl_hydraulic_lab.py)
- [四联结果图](chapter8_latest/outputs/chapter8_comparison.png)
- [Word 实验报告](chapter8_latest/outputs/第八章实验报告.docx)
- [完整 ZIP](chapter8_latest/deliverables/第八章_SAC安全控制完整交付.zip)

```bash
cd chapter8_latest
python drl_hydraulic_lab.py --mode all --reuse-model
```

需要先按项目说明安装 CPU PyTorch、Gymnasium 和 Stable-Baselines3。`--reuse-model` 使用随交付保存的真实训练策略；省略该选项会重新训练。

新版安全演示实际在 0.104891 s 记录未保护触边，HOCBF 最低制动电压约 −3.092 V，受保护轨迹趋于 98.8 mm；这是单独声明的二阶模型验证。5000 N 与体积模量减半故障下，FP32/INT8 的严格恢复时间分别约 0.068/0.050 s，显示窗口速度 RMS 差约 0.612 mm/s。四状态对象的真实速度谷值仍比课件示意更深，完整指标及实际推理耗时见项目报告。

## 第六章作业 6-3：死区逆补偿鲁棒性与 tanh 光滑化

独立作业比较真实 ±0.50 V 死区下的 0.40 V 欠补偿、0.60 V 硬过补偿与 tanh 补偿，给出全周期/换向窗口六联图、β=5/18/50 扫描和定量报告。修订版采用输入死区、阀及速度串联模型和明确公开的二级动态残差压力测试，恢复明显的欠补偿近零平台与硬过补偿速度颤振；有限 β 的残余平台据实保留。

- [作业说明与参考代码](homework6_3/README.md)
- [六联结果图](homework6_3/outputs/deadzone_comparison.png)
- [模块三控制器](homework6_3/model_assisted_controller.py)
- [完整 ZIP](homework6_3/deliverables/第六章作业6-3_完整交付.zip)
- [Word 报告](homework6_3/deliverables/第六章作业6-3_死区补偿报告.docx)

```bash
cd homework6_3
python -m pip install -r requirements.txt
python work_6_3.py
```

## 第七章：MPC 约束控制与滚动优化

50 mm 极限定位课堂实验，在共同液压孪生上比较传统前馈 + PID、仅输入约束 MPC 和输入及 P2 压力约束 MPC。代码补齐真实雅可比、仿射预测、阀动态及求解检查，输出一张位移、电压、压力和速度四联图。

- [第七章完整项目与说明](chapter7_latest/README.md)
- [模块二 MPC 核心](chapter7_latest/mpc_core.py)
- [四联结果图](chapter7_latest/outputs/mpc_comparison.png)
- [第七章完整 ZIP](chapter7_latest/deliverables/第七章_MPC约束控制完整交付.zip)
- [Word 实验报告](chapter7_latest/deliverables/第七章_MPC约束控制实验报告.docx)

```bash
cd chapter7_latest
python -m pip install -r requirements.txt
python mpc_hydraulic_lab.py
```

压力峰值使用包含采样间极值的真实混合积分记录，未裁剪到 21 MPa；求解耗时及与 PPT 的数值差异单独披露。Ts=1 ms 是仿真采样周期，当前 Python 实现尚未达到 1 ms 硬实时要求。

## 第七章作业 7-2：预测时域与压力—算力权衡

重新依据第七章课件第 15 页编写的作业，使用四状态局部灰盒、五维增广、固定物理参数和权重，仅改变 N=5、10、15、20、30、40、50。默认采用教材凝聚形式，每采样周期求解一次 CVXPY/OSQP QP；记录位移、电压、连续时间压力极值与真实计算耗时。

- [重写版说明与代码](homework7_2_rewrite/README.md)
- [完整作业报告](homework7_2_rewrite/REPORT.md)
- [四联对比图](homework7_2_rewrite/outputs/horizon_comparison.png)
- [七组定量结果](homework7_2_rewrite/outputs/metrics.csv)
- [完整 ZIP](homework7_2_rewrite/deliverables/第七章_习题7-2_重写版完整作业.zip)
- [Word 报告](homework7_2_rewrite/deliverables/第七章_习题7-2_重写版报告.docx)

```bash
cd homework7_2_rewrite
python -m pip install -r requirements.txt
python work_7_2.py
python verify.py
```

重写版改用压力释放时间常数为 8 ms 的共同工况，N=5/20/50 超调约 1.39%/0.45%/0.35%；在 0.080 s 的 P2 分别约为 20.35/10.04/9.30 MPa，展示短视预测的延迟卸压差异。[制动局部对照图](homework7_2_rewrite/outputs/pressure_braking_detail.png) 与高压持续时间均由实际轨迹计算，七组连续峰压低于 21 MPa。固定工作点模型用于隔离 N 的作用，不能当成全孔口非线性设备验证；本机完整控制步骤未达到 1 ms 实时要求。旧版保留在 [homework7_2](homework7_2/) 供比较。

## 第五章作业 5-1：PID 抗积分饱和对比

新项目在相同液压对象、PID 增益和 ±10 V 限幅下，对比位置式的积分累积与增量式的截断输出记忆。独立目录包含 `work_5_1.py`、一张三联图、两种微分模式的 CSV、Word 报告、验证记录和完整 ZIP。

- [完整项目与运行说明](homework5_1/README.md)
- [三联结果图](homework5_1/outputs/anti_windup_comparison.png)
- [作业 5-1 完整 ZIP](homework5_1/deliverables/作业5-1_PID抗积分饱和完整交付.zip)
- [Word 实验报告](homework5_1/deliverables/作业5-1_PID抗积分饱和实验报告.docx)

```bash
cd homework5_1
python -m pip install -r requirements.txt
python work_5_1.py
```

默认两组共同对测量值微分，避免阶跃参考的微分冲击；原误差微分公式可切换复核。本次位置式超调约 30.77%，增量式约 0.55%，位置式在真实误差反向后仍有约 13 ms 的退饱和迟滞。精确数据与参考图差异见项目说明。

## 第六章模型辅助复合控制

在同一液压数字孪生上对比纯 PID、标称前馈 + PID 和完整模型辅助控制，输出一张四联图：全程误差、控制电压、起动速度和负载切换速度。模块三、独立运行入口、原始参考材料、CSV 和验证记录统一放在新文件夹中。

- [第六章完整交付文件夹](chapter6_latest/)
- [模块三控制器代码](chapter6_latest/model_assisted_controller.py)
- [四联结果图](chapter6_latest/outputs/model_assisted_comparison.png)
- [第六章交付 ZIP](chapter6_latest/deliverables/第六章_模型辅助控制交付.zip)
- [运行方式、结果与适用条件](chapter6_latest/README.md)

```bash
cd chapter6_latest
python -m pip install -r requirements.txt
python model_assisted_lab.py --save-data
```

新版采用柔和接近的解析 S 轨迹、公开含气油模型和真实静摩擦粘滑模型。净驱动力位于 `[-Fs,Fs]` 时活塞严格静止，压力与阀继续积分；超过 Fs=800 N 后由采样间事件释放。滑动时采用有向 Stribeck 函数，速度过零后按净力重新粘住或反向释放。Fs、Fc、vs、σ 及已有液压、阀参数保持不变，配置中的 gamma 保留但不再用于 tanh 摩擦；上传 Twin 源文件仍按字节保留。

两个模型辅助组按规划速度、加速度、跃度的首个非零方向提出起动静摩擦前馈。首样本计划力为 800 N，规划与力误差反馈共同给出 69000 N/s 的建压需求，完整组 G/H 反演与 0.40 V 死区计算出约 **0.564599 V** 首目标。默认 `enable_startup_regularization=True` 允许冷启动阀误差按 `e_v[k]=a^k e_v[0]` 自然衰减，前馈记忆及真实阀状态仍从零开始；API 设为 `False` 可复核硬 ZOH 逆对照。首值由逆动力学计算，不强设为 PPT 示例的 1.20 V。

本版完整组静摩擦在 20.93 ms、净驱动力约 800 N 时释放；速度达阈值为 23 ms，首电压约 0.565 V，全程最大误差约 0.044 mm。38 项测试与严格积分复算通过；旧参考起动时限等 5 项指标未达，详见验证记录。

两个模型辅助组共享已知工艺计划、净液压力误差反馈、Kv=60 s⁻¹ 速度误差阻尼、静止力锥持位规则及近零流量死区逆，完整组额外使用实时孔口增益调度。控制器仅使用计划和当前传感器值，不读取对象真实外载或强微分压力。全部曲线来自混合 ODE 积分；真实释放事件与速度达阈值时刻分别记录。当前数值、严格复算与未知外载边界以 [第六章说明](chapter6_latest/README.md)、`outputs/summary.json` 和 `outputs/validation.json` 为准。

## 第五章数字 PID 单图实验

结合第 3、4、5 章 PPT、上传的液压数字孪生和 Word 中的 PID、S 型轨迹代码，生成一张三联结果图：位移跟踪、真实跟踪误差、控制电压。三组闭环仿真分别展示保守增益、激进增益和未滤波微分的表现。

- [最新完整交付文件夹](chapter5_latest/)
- [三联结果图](chapter5_latest/chapter5/outputs/classroom_comparison.png)
- [单图交付 ZIP](chapter5_latest/deliverables/第五章_PID单图交付.zip)
- [代码运行与模型说明](chapter5_latest/chapter5/README.md)

从项目根目录运行：

```bash
cd chapter5_latest/chapter5
python -m pip install -r requirements.txt
python digital_pid_lab.py
```

默认只输出 `chapter5_latest/chapter5/outputs/classroom_comparison.png`。需要复核原始数据时运行 `python digital_pid_lab.py --save-data`，另外保存三组 CSV 和 `summary.json`，用于诊断。

本次将实际接触负载设为 40 ms 加载、20 ms 卸载的余弦过渡，保留激进组卸载后的振荡。激进组加载电压峰值由约 9.21 V 降为 7.31 V，超出重载平台的部分减少约 90%；PID、阀和其余物理参数保持相同。保守组末端真实误差由约 +0.009 mm 增至 +0.019 mm，最后 0.1 s 仍处于 ±0.05 mm 内。

从项目根目录运行验证：

```bash
python -m unittest discover -s chapter5_latest/chapter5 -v
```

26 项公式和接口测试已通过。

上传的 `hydraulic_digital_twin.py` 按字节保留，原示例参数也保存在 `physical_config.json`。课堂阀配置、阀动态、补油单向阀及工艺外载均公开说明；这些是教学仿真假设，未经设备参数辨识。图中误差和流量增益由实际数值积分计算，重载增益下降约 64%，没有将 65% 硬赋给模型，也不保证与参考图片逐像素重合。
