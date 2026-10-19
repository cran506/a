# 第六章习题6-3：死区逆补偿鲁棒性（修订版）

本版修正了上一版“欠补偿平台过窄、硬过补偿速度颤振过弱”的问题。入口仍是 `work_6_3.py`，主结果仍为唯一一张3×2图：左列0–10s，右列4.55–5.45s，三行依次是控制电压、静态死区输出、模型速度。

- [六联图](outputs/deadzone_comparison.png)
- [模块三参考实现](model_assisted_controller.py)
- [定量表](outputs/metrics_table.txt) / [表格CSV](outputs/metrics_table.csv)
- [完整ZIP](deliverables/第六章作业6-3_完整交付.zip)
- [Word报告](deliverables/第六章作业6-3_死区补偿报告.docx)

```bash
cd homework6_3
python -m pip install -r requirements.txt
python work_6_3.py
python -m unittest -v test_work_6_3.py
python package_delivery.py
```

可用 `--config 自定义.json` 改参数、`--output-dir 目录` 另存、`--no-plot` 只算数据、`--skip-robustness` 跳过独立补充试验。

## 这次改了什么

上一版加入Kp=4的直接速度反馈，会补回慢变死区失配，压缩了欠补偿平台；又把动态阀芯后的信号作为题图静态有效信号，使红色颤振被削弱。这次采用以下**共同的课堂压力测试结构**，三策略只改变补偿阈值或函数：

```text
u_eff = D_0.5(u) = sign(u) * max(abs(u)-0.5, 0)
tau_v * dz/dt = u_eff - z
tau_m * dv/dt = K_v * z - v
```

这里的 `u_eff` 是输入死区的输出，`z` 是动态有效阀口的等效电压，两者不相同。`u_eff=0` 不等于动态阀口立刻零流量。`input_deadzone_model.py` 对每个零阶保持输入解析积分两级动态；原 `deadzone_model.py` 中的旧顺序保留作回归对照，不再是主实验对象。

参考指令 `u_ref=5*sin(2*pi*0.1*t)` V，`vd=Kv*u_ref`，加速度解析求导。共同参数为 `Kv=0.01 m/(s·V)`、`tau_v=0.8 ms`、`tau_m=12 ms`、`Ts=1 ms`，速度标度为±50mm/s。真实输入死区±0.50V，欠补偿估计0.40V，硬过补偿及tanh估计0.60V，默认beta=18V^-1。这些幅值和时间常数保持此前设定。

**新增且明确公开**的动态残差支路为二级washout（不补偿DC误差）：

```text
e[k] = (vd[k]-v_meas[k])/Kv
ed = e[k-6]                         # 整个残差缓存延迟；非只延迟测量
alpha = exp(-Ts/0.002)
L1[k] = alpha*L1[k-1] + (1-alpha)*ed
h1 = ed-L1[k]
L2[k] = alpha*L2[k-1] + (1-alpha)*h1
h2 = h1-L2[k]
u_lin = (vd+tau_m*ad)/Kv + 17*h2
```

普通PID默认为Kp=Ki=Kd=0；接口保留，可另行实验。该二级动态残差和6ms缓存是新加的压力测试假设，并非原PPT已给定或设备辨识得到的参数。它避免直接P反馈补平慢变失配，同时让硬逆函数在有延迟的反馈环中暴露颤振。不能将它冒称为原模块三的完整双腔逆动力学PID。

最后三组分别计算 `u=u_lin+0.4*sign(u_lin)`、`u=u_lin+0.6*sign(u_lin)`、`u=u_lin+0.6*tanh(beta*u_lin)`，执行电压限幅±10V。硬映射在 `|u_lin|<=1e-10 V` 输出0；三组主图均无测量噪声。没有为某个策略单独加纹波、裁速度或平滑曲线。

## 修订后的结果

| 策略 | 有效信号零平台/ms | 近零速度平台/ms | 速度高频峰峰值/mm/s | 换向窗口符号反转 |
| --- | ---: | ---: | ---: | ---: |
| 欠补偿0.40V | 63.620 | 95.424 | 0.04094 | 1 |
| 硬过补偿0.60V | 0.831 | 7.117 | 1.28979 | 13 |
| tanh，beta18 | 28.621 | 35.402 | 0.28500 | 1 |

近零速度采用 `|v|<=0.5mm/s`；有效信号零平台由 `|u|<=0.5V` 的采样线性阈值重建估计。速度高频峰峰值取4阶20Hz零相位高通后的窗口分量，**滤波只用于指标，不用于绘图或实际输入**。静态信号平台、近零速度平台、物理零流量不能混称。未强行把63.62ms标成参考示意图的110ms。

普通比例反馈关闭后，过补偿/欠补偿的低频残差不再被补平；宏观速度峰值约为51/49mm/s，这是0.10V估计偏差乘Kv的结果，不是绘图误差。全周期RMSE约1mm/s；主图形状与参考更接近，但不承诺示意图全部峰值或周期。

## beta敏感性与持续振荡检验

beta=5/18/50分别独立仿真，全部原始数据为 `outputs/beta_*.csv`。有限beta必然存在正根 `q0+0.6*tanh(beta*q0)=0.5`，因此光滑并不意味着严格消除死区。beta增大通常缩短残余平台，同时提高逆映射零点斜率 `1+0.6*beta`；不能预设所有tanh参数均稳定。

`robustness_checks.py` 做两类独立补充：同种子0.02mm/s测量噪声下的beta扫描；以及从正式5s真实状态继续、保持零参考1s的试验。保持前逐样本回放正式测量，恢复**完整误差缓存和两个低通记忆**，不能只保留速度/阀状态然后重置控制器。反事实仅在保持段设置 `feedback_enabled=False`，停用包括washout在内的所有误差反馈；仅设置Kp=0在本版不是停反馈。

硬过补偿保持末100ms的速度峰峰值约1.886mm/s，说明存在持续小幅极限环；欠补偿和beta18衰减收敛，关闭全部反馈后硬过补偿也衰减。beta50保持试验另列实际结果，不能从主换向只有一次过零就推断其所有工况无振荡。

## 验证和文件

13项单元测试包括旧模型回归、新输入死区对象的独立高精度ODE对照和半步等价、残差缓存因果性、washout零DC增益及完整停反馈。默认主实验51项检查包含有效信号定义、输入边界、因果阀状态更新，以及欠补偿平台和硬过补偿速度振荡可见性要求；另有6项噪声/保持检查。详见 `outputs/validation.json`、`outputs/robustness.json`。

CSV的 `effective_voltage_v` 是D(command)，`valve_effective_voltage_v`及兼容字段`valve_voltage_v`为滤波后的有效开口状态。`longest_flow_closed_ms`是旧字段名兼容别名，语义已变为静态有效信号零平台，优先读 `longest_effective_zero_ms`。状态记录在当次指令作用前，u及D(u)施加于随后一个Ts区间。

`materials/`保留PPT、原模块三与参数说明。本作业仍是降阶课堂模型，不计算双腔压力、层流泄漏、静摩擦和负载力；不能用此图证明真实液压缸全局稳定或压力安全。图、Word、CSV、源码和配置同时打包，`manifest.json`列SHA256。
