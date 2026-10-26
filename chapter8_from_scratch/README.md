# 第八章：真实从零 SAC 训练附录

本目录保存重新核对 PPT 第 4、5、7、12–14 页后完成的真实百万步训练。主展示图按用户选择采用明确标注的 **PPT 教学示意**，位于相邻 [主项目](../chapter8_reference/README.md)；本目录的实测数据与示意曲线分别保存。

- [真实完整训练曲线](outputs/sac_training_actual.png) / [原始训练 CSV](outputs/training.csv)
- [固定重载工况验证奖励与策略熵](outputs/training_validation.png) / [验证逐步记录](outputs/training_validation_trace.csv)
- [PPT 三阶段目标示意（非训练日志）](outputs/ppt_training_target.png)
- [同一新模型的控制诊断](outputs/chapter8_comparison.png) / [全部性能指标](outputs/metrics.json)
- [训练参数和模型来源](models/training_metadata.json) / [完整包](deliverables/第八章_从零训练修订版.zip)

## 训练流程修正

上一版先做 6000 步教师预热，并把初始 `log_std` 设为 −3，跳过了课件的高熵探索；日志首点还使用了不同的回报平均口径。本版取消教师预训练、教师损失和训练后 FP32 精修，保留最终 SAC 检查点的原始权重。

新网络为 5–64–64–1，随机隐层、零均值输出头、初始 `log_std=-0.1`；初始电压策略熵约 **2.985 nat**。前 50000 步均匀探索、不更新网络。随后 4 个环境每 16 次交互执行两次梯度更新，累计 **1000000 次交互、118750 次梯度更新**。源码和模型 SHA256 随元数据保存。

五维观察、标准化和奖励保留课件定义：

```text
obs = [e_mm, de_mm_s, dde_mm_s2, PL_MPa, Ps_MPa]
mean = [0, 0, 0, 10, 21]; std = [10, 50, 500, 8, 2]
r_base = -100 e² - 0.1 u² - (u-u_prev)² + 2 I(|e|<0.2)
```

基础奖励始终用最近 10000 个真实步骤平均，失败终止成本 −100000 单列；图中不把优化器的 0.001 缩放奖励当作原始奖励。首个不满窗口仍在 CSV，完整奖励图从首个满窗口开始绘制。质量 ±20%、体积模量 600–1400 MPa、摩擦 ±30% 均进入真实四状态动力学。

熵含 tanh 雅可比修正，并使用 `H(u)=H(a)+ln(10)`；目标电压熵 0.35 对应 `target_entropy=0.35-ln(10)`。主训练图使用 `replay_voltage_entropy`，固定故障探针熵另列。后 10 万步 replay 熵均值约 **0.341 nat**，最后一个采样值约 0.290。

## 三种图的口径

1. 完整训练回报保留全部早期负值：前 50k 步约 −1.1 万至 −1.3 万，最终 10000 步窗口约 −6.29。使用对称对数轴及线性局部放大，未修改数据或插值成 S 形。
2. 固定工况验证逐个实际执行保存检查点：16 个带种子随机化对象，每回合 50 ms；初始位置 50 mm、速度 50 mm/s，负载 5000 N、体积模量倍率 0.5，压力与上一电压为各对象稳态平衡。平均每步奖励从 **−272.69 改善至 −2.79**。8800 个额外验证步骤未加入训练 replay，也未计入百万步。这是短窗局部性能，不能替代 2000 步完整回合或突加负载恢复试验。
3. PPT 示意使用公开目标锚点和单调 PCHIP 插值：奖励 −180→−2、熵 2.8→0.35。标题、文件名和 JSON 都标注“非训练日志”，不作为收敛或性能证据。

PPT 没有提供完全相同的种子、网络和初态。本次原生 SAC 的熵在 50k 后下降较快，不能把示意曲线当作实测结果。图 (a) 的主展示方式已按用户选择设为教学示意，真实记录完整附在这里。

## 控制诊断的实际限制

本次主要修正训练过程。最终 FP32 的故障速度谷值约 −16.13 mm/s、72 ms 恢复，但故障前速度 RMS 仍约 3.49 mm/s，存在电压振荡；70% 剪枝后的整数策略在该对象上退化。因此本目录不替换此前较稳定的机理引导部署对照模型，也不宣称新模型达到课件全部性能。

`outputs/validation.json` 分别记录数据/方法校验与性能目标。代码和数据校验通过不表示部署性能全部通过；控制诊断曲线未被隐藏、滤波或替换为旧模型数据。

## 运行

需要 Python 3.11+、g++ 和中文字体；先安装 CPU PyTorch，再安装依赖。

```bash
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python drl_hydraulic_lab.py --mode all --reuse-model
```

重新训练、单独查看图或测试：

```bash
python drl_hydraulic_lab.py --mode all --timesteps 1000000 --seed 2026
python training_plot.py
python training_validation.py
python drl_hydraulic_lab.py --mode ppt-reference
python -m unittest discover -p 'test_*.py' -q
```
