# a

第五章数字 PID 闭环控制编程实验。已整合液压数字孪生及 Word 中的 PID、S 型轨迹代码，包含三组课堂实验与课后抗积分饱和对比。

- [完整 ZIP 交付包](deliverables/第五章_PID编程完整交付.zip)
- [Word 实验报告](chapter5/outputs/实验报告.docx)
- [实验报告文本与图表](chapter5/outputs/实验报告.md)
- [代码及运行说明](chapter5/README.md)
- [验证记录](chapter5/outputs/验证说明.md)
- [CSV、JSON 与全部仿真图表](chapter5/outputs/)
- [上传课件与 Word 的文本摘录](chapter5_materials/)

在 `chapter5` 目录运行：

```bash
python -m pip install -r requirements.txt
python digital_pid_lab.py
python -m unittest -v test_chapter5
```

已实际完成七组闭环仿真和20项代码验证。当前默认参数下，课堂实验未达到1秒时真实位置误差小于0.05mm的目标；报告保留了与示意图不同的真实结果，并注明模型假设和适用范围。
