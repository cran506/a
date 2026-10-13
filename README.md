# 智能流体传动与控制课堂编程

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
