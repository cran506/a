# 智能流体传动与控制课堂编程

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

新版采用柔和接近的解析 S 轨迹、公开含气油模型和共同液压力误差反馈。两个模型辅助组默认启用 `enable_startup_regularization=True`：首指令由逆动力学 G/H 反演、0.40 V 死区和阀初值相容化计算，完整组约为 **0.411927 V**，之后随逆模型需求平缓变化。前馈阀记忆和真实阀状态均从零开始，初值误差分量按 `e_v[k]=a^k e_v[0]` 自然衰减；通过解析修正项避免硬 ZOH 逆的首样本放大，没有伪置阀初态或抬高图中电压。API 设为 `False` 可运行同参数的硬 ZOH 逆对照。

完整组前 20 ms 的电压峰值约为 0.451255 V，最大单步下降约为 0.033605 V；速度达阈值为 11 ms，参考达同一阈值为 5 ms。纯 PID 在 81 ms 起动，首个速度峰值约 0.060 m/s，重载误差峰值约 1.385 mm；完整组全程最大误差约 0.043206 mm。所有曲线来自真实积分。保留当前物理参数时，不将 PPT 示例的 1.20 V 强设为持续起动电压；此处采用包含力、压力和油液压缩的逆动力学，具体首值由模型算出。

两个模型辅助组共享同一已知工艺负载计划，完整组额外使用实时孔口增益调度。与参考图的数值对照、量化 PID 的精度复核及未知外载限制见第六章说明和 `outputs/reference_comparison.json`。

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
