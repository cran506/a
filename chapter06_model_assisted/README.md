# 第六章课堂编程：模型辅助复合控制

`model_assisted_lab.py` 比较纯 PID、标称前馈 + PID、完整模型辅助控制。针对参考图 (a) 的课堂实验使用用户提供的六步控制器、500 kg 对象参数和显式粘滑事件；另保留双腔压力连续积分的四阶液压孪生作为物理对照。两种模型的图表、原始数据和结论分别保存。

课堂模型图 (a) 的纯 PID 全程最大误差为 **1.4824 mm**，标称前馈为 **1.4829 mm**，完整控制为 **0.0432 mm**。PID 在 **88 ms** 速度达阈值，两个前馈策略在 **4 ms** 达阈值；完整控制首指令 **1.200 V**，轻/重载电压实测中位数 **2.50→4.31 V**。负载窗口最低速度为纯 PID **0.1342 m/s**、完整控制 **0.1915 m/s**。四阶孪生完整组最大误差为 **0.03662 mm**，但其有限建压令速度达阈值时间为 **13 ms**。

![第六章课堂模型四联图](outputs/classroom/model_assisted_comparison.png)

## 运行

需要 Python 3.10+。在本文件所在目录运行：

```bash
python -m pip install -r requirements.txt
python model_assisted_lab.py --model all
python -m unittest discover -v
python verify_numerics.py
python verify_classroom.py
python build_report.py
python package_delivery.py
```

默认运行参考图 (a) 的课堂对象：`python model_assisted_lab.py`。单独运行四阶孪生：`python model_assisted_lab.py --model twin`。CSV 默认保存，兼容 `--save-data` 参数。绘图使用无界面后端；中文依次查找 Noto CJK、微软雅黑、黑体，建议安装其中一种。

- [课堂模型四联图 PNG](outputs/classroom/model_assisted_comparison.png) / [PDF](outputs/classroom/model_assisted_comparison.pdf)
- [四阶液压孪生 PNG](outputs/twin/model_assisted_comparison.png) / [PDF](outputs/twin/model_assisted_comparison.pdf)
- [实验报告与实测对照](outputs/experiment_report.md)
- [数值复核](outputs/numerical_checks.json)
- [课堂等效模型数值与移时反事实检查](outputs/classroom/numerical_checks.json)
- [完整 ZIP](deliverables/chapter06_model_assisted.zip)
- [课件页码、公式及一致性核对](docs/course_notes.md)

## 四阶压力动力学对照

被控对象保留 `x, v, P1, P2` 四个积分状态，并增加一阶阀芯动态。上传的 [hydraulic_digital_twin.py](hydraulic_digital_twin.py) 按原始字节保留；闭环适配位于 `twin_adapter.py`，通过继承扩展，不把计算后的位移、速度或压力裁剪成目标曲线。

1. **策略 1**：`enable_ff=False`，仅公共 PID 依靠误差起动。
2. **策略 2**：`enable_ff=True, enable_gain_sched=False`，加入逆动力学、连续性及泄漏、死区逆补偿，孔口增益采用标称空载工作点。
3. **策略 3**：两开关都打开，使用当前双腔压力实时反演孔口增益。

三组分别重新初始化相同对象、阀状态、传感器和 PID。位置参考与解析速度、加速度、跃度完全相同。PID 误差内部用 cm，增益为 `Kp=4 V/cm, Ki=450 V/(cm·s), Kd=0.10 V·s/cm`，微分滤波系数 0.85；输出 ±10 V。控制周期 1 ms；ADC 16 bit、0.3 m 量程，LSB 约 0.00458 mm。误差以真实积分位移计算，不用量化后的误差替代。

原始示例参数在 `physical_config.json`，四阶对象的教学整定覆盖项在 `classroom_config.json`。几何面积 `A1=0.01, A2=0.005 m²`、质量 500 kg；供油 21 MPa、回油 0.1 MPa。四阶对象采用 0.3 m 行程、0.1 L 死区容积、油液体积模量 1.8 GPa、回油压力下含气率 1.8%、最大阀芯行程 3.73 mm/归一化输入、死区 ±0.40 V、阀时间常数 2.5 ms。摩擦 `Fs=800 N, Fc=500 N, vs=0.05 m/s, sigma=5000 N·s/m`。这些是公开教学标定，未冒称第 4 章的实测辨识值。下文的课堂约化模型改用用户本次给出的另一套参数，两者不可混同。

对原始模型的三项必要扩展：

- `tanh` 平滑摩擦改为真实静摩擦集合值条件：零速且净力在 `[-Fs, Fs]` 内时保持静止，但继续积分压力与阀；越阈值由采样间事件释放，滑动摩擦采用 Stribeck 分支。原 `tanh(0)=0` 不能产生严格静止平台。
- 孔口采用有向压差，补油单向阀处理低压腔；绝对压力中扣除外界大气对差动面积的作用。压力由连续性方程积分，未用时间表强制赋值。
- 使用公开含气油柔度与一阶阀动态，检查 Radau 返回状态。原上传脚本中演示用的空间接触工件在本实验停用，改用所有策略共同的工艺负载台架。

负载计划在 `t=0.50 s` 开始，**25 ms 余弦加载到 142 kN**，`0.92 s` 卸载完成，卸载过渡 30 ms。这是有已知工艺负载及其解析导数的实验，不是未知理想阶跃。加载坡度使有限流量可完成真实建压；图中保留该过程。

## 公式到代码

课件第 3–4 页的架构为 `u = u_ff + u_fb`。第 5 页式 (6.1) 的纯 Stribeck 项与粘性项分开计算，避免把 `sigma*v` 加两遍。第 8 页式 (6.2) 实现时包含阀芯电压比例：

```text
K_flow = Cd*w*(xv_max/10)*sqrt(2/rho)
Q = K_flow*u_eff*sqrt(DeltaP)
u_eff = sign(u)*max(abs(u)-0.40, 0)
```

这保证反演结果单位为 V，而不是阀芯位移 m。最小反演压差为 0.5 MPa，当前无噪压力测量不额外低通（`pressure_alpha=0`）；滤波接口可配置。

四阶对象中，只有 `Q=A*v+C_ip*DeltaP` 不能瞬时建立所需驱动力。因此 `ModelAssistedController` 进一步从双腔连续性推导 `F_hyd_dot=G*u_lin+H`，其中 `H` 包含运动、内外泄漏和补油，`G` 包含真实双孔口压差。由 `m*a_d+F_f(v_d)+F_ext` 及其解析导数构造目标，再叠加力误差修正和速度残差阻尼。两种前馈组共享这些增强，仅实时孔口增益的选择不同。

控制器还包含已知阀一阶动态逆、近零流量死区偏置限速和一次起动增益调度。首次运动前建压力误差增益 385 s⁻¹；实测 `|v|>1e-4 m/s` 后锁存切换到 80 s⁻¹，速度残差阻尼 60 s⁻¹。逻辑仅依赖当前传感器，未使用预设起动时间。每步前馈、反馈、起动模式及实际力环增益均导出到 CSV。它们属于工程增强；上传 PDF 第 11 页只给出架构解算链，没有逐行代码清单，故本项目实现公式与架构，不声称逐字复现未提供的清单。

`--unknown-load` 可检查移除负载先验后的表现。内力反馈仍以空载力为目标，会抵制额外负载所需的驱动力，因此该模式不是默认工况的精度保证，也不能用它否定纯压力流量线性化的抗扰能力。

## 课堂约化模型：修正参考图 (a)

[textbook_controller.py](textbook_controller.py) 实现本次提供的六步链路：逆动力学受力、理论建压与层流泄漏、当前压力下的阀口流量逆映射、死区逆补偿，以及公共 PID 残差反馈。控制器在缩回时使用有杆腔面积及压力，三策略只切换前馈与实测压力增益调度。解析 S 轨迹见 [textbook_trajectory.py](textbook_trajectory.py)，参数见 [textbook_config.json](textbook_config.json)。上传代码片段末尾混入参数字典而无法执行，仓库保留其公式并整理为可运行 Python。

[classroom_model.py](classroom_model.py) 以 `m=500 kg`、`Fs=3100 N`、`Fc=400 N`、`Ct=2.7027e-9 m³/(s·Pa)`、`K_flow=2.7027e-7 m³/(s·V·√Pa)` 和 0.8 ms 阀时间常数分别闭环积分三组对象。保留物理输入死区、层流泄漏等效阻抗、真实零速粘住和 Stribeck 切换；使用 ODE 事件定位静摩擦释放。连续性推得的被动速度时间常数 `Ct·m/A1²=13.5 ms` 已包含惯性，因此配置中的参考 `tau_m=12 ms` 没有再串联一次。`gamma` 平滑摩擦系数由严格粘滑事件替代。压力 `P1` 是准静态工艺负载工作点加运动力造成的压力，`P2` 为基准回油腔压力；两者不是独立储能状态。`t=0.50 s` 的压力变化是对象的代数负载近似，不应外推为四阶对象可瞬时建压。课堂计算得到静摩擦释放约 **2.986 ms**、速度达 `1e-4 m/s` 阈值 **4 ms**，因此“零延迟”仅指首采样给出前馈命令。

课堂对象的物理阀口增益由代数负载工作点压力计算；控制器读取的 `P1` 还包含运动力造成的约 0.10 MPa 压力增量，因此压力调度保留一小部分模型残差。CSV 中 `kq_ratio` 表示工作点增益比，`p1_pa` 表示交给控制器的测量压力。

该六步代码令 `Q_leak=Ct·P_req·√(ΔP/ΔP_0)`，所以泄漏电压项约 **0.25 V** 随压差缩窄仍保持不变。给定 `P1=16.95 MPa`，阀口压差从 16.0 MPa 降至 4.05 MPa，运动流量电压约从 **1.85 V** 增至 **3.677 V**，再加泄漏 **0.25 V** 和死区 **0.40 V**，理论重载前馈为 **4.327 V**。要得到恰好 **4.20 V**，固定其他参数时 `P1` 应约为 **16.655 MPa**；图和报告显示闭环计算值，不把不相容目标硬写进结果。完整组全程 **0.0432 mm** 满足用户的 `<0.05 mm` 目标。

## 文件和验收

`trajectory.py` 与 `textbook_trajectory.py` 分别给四阶/课堂模型提供解析参考；`controllers.py` 提供公共数字 PID。`friction_model.py` 与 `media_model.py` 分别处理四阶对象的粘滑和含气柔度。`simulation.py` 记录三组四阶实验，`plotting.py` 直接绘制两组实验记录；`verify_numerics.py` 和 `verify_classroom.py` 做严格积分及反事实复算。`test_textbook_reference.py` 独立核算首指令、恒速/重载逆映射与有杆腔反向分支。`build_report.py` 从 JSON 重建实测表；`package_delivery.py` 生成 SHA-256 清单和完整 ZIP。

`validation.json` 检查电压限幅、有限正压力、共同参考、真实误差恒等式、完整组全程 0.05 mm 精度及负载误差优势；未达到的指标按实际保留。起动同时报告静摩擦释放事件和速度达 `1e-4 m/s` 的采样时刻，不把首个非零前馈等同于真实运动。
