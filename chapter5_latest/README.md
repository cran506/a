# 第五章数字 PID 单图实验

结合第 3、4、5 章 PPT、上传的液压数字孪生和 Word 中的 PID、S 型轨迹代码，生成一张三联结果图：位移跟踪、真实跟踪误差、控制电压。三组闭环仿真分别展示保守增益、激进增益和未滤波微分的表现。

- [三联结果图](chapter5/outputs/classroom_comparison.png)
- [单图交付 ZIP](deliverables/第五章_PID单图交付.zip)
- [代码运行与模型说明](chapter5/README.md)

从项目根目录运行：

```bash
cd chapter5
python -m pip install -r requirements.txt
python digital_pid_lab.py
```

默认只输出 `chapter5/outputs/classroom_comparison.png`。需要复核原始数据时运行 `python digital_pid_lab.py --save-data`，另外保存三组 CSV 和 `summary.json`，用于诊断。

本次将实际接触负载设为 40 ms 加载、20 ms 卸载的余弦过渡，保留激进组卸载后的振荡。激进组加载电压峰值由约 9.21 V 降为 7.31 V，超出重载平台的部分减少约 90%；PID、阀和其余物理参数保持相同。保守组末端真实误差由约 +0.009 mm 增至 +0.019 mm，最后 0.1 s 仍处于 ±0.05 mm 内。

从项目根目录运行验证：

```bash
python -m unittest discover -s chapter5 -v
```

26 项公式和接口测试已通过。

上传的 `hydraulic_digital_twin.py` 按字节保留，原示例参数也保存在 `physical_config.json`。课堂阀配置、阀动态、补油单向阀及工艺外载均公开说明；这些是教学仿真假设，未经设备参数辨识。图中误差和流量增益由实际数值积分计算，重载增益下降约 64%，没有将 65% 硬赋给模型，也不保证与参考图片逐像素重合。
