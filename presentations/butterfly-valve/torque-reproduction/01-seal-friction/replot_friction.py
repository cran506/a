"""Replot published Fig.2.6 using its stated linearity and endpoints; NOT independent integral-model validation."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from openpyxl import Workbook
from openpyxl.chart import ScatterChart,Series,Reference
from openpyxl.styles import Font,PatternFill,Alignment
O=Path(__file__).resolve().parent
KO=840.92;KC=174.64
p=np.linspace(0,1,101);mo=KO*p;mc=KC*p
assert mo[0]==mc[0]==0 and abs(mo[-1]-840.92)<1e-10 and abs(mc[-1]-174.64)<1e-10
assert abs(KO*.45-378.414)<1e-10 and abs(KC*.45-78.588)<1e-10
with (O/'图2_6_线性复绘数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['pressure_MPa','opening_friction_Nm','closing_friction_Nm']);w.writerows(zip(p,mo,mc))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False
fig,ax=plt.subplots(figsize=(7.2,5.2));ax.plot(p,mo,color='#db2828',lw=1.8,label=r'$M_{f1}$ 开启摩擦力矩');ax.plot(p,mc,color='#289b31',ls='-.',lw=1.8,label=r'$M_{f2}$ 关闭摩擦力矩')
ax.set_xlim(0,1.12);ax.set_ylim(0,1000);ax.set_xticks(np.arange(0,1.01,.2));ax.set_xlabel('介质压力 p / MPa',fontproperties=font);ax.set_ylabel('密封面摩擦力矩 / (N·m)',fontproperties=font);ax.set_title('图2.6复绘：阀座与蝶板密封面的摩擦力矩',fontproperties=font,fontsize=13)
ax.annotate('(1, 840.92)',(1,KO),xytext=(.62,880),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=10);ax.annotate('(1, 174.64)',(1,KC),xytext=(.62,310),arrowprops=dict(arrowstyle='->',color='#555'),fontsize=10);ax.annotate('(0, 0)',(0,0),xytext=(.04,70),fontsize=10);ax.legend(prop=font,loc='upper left');ax.grid(alpha=.15)
fig.text(.5,.01,'依据论文线性关系及端点复绘；不是从接触积分公式独立验算',ha='center',fontproperties=font,fontsize=9,color='#555');fig.tight_layout(rect=(0,.04,1,1));fig.savefig(O/'图2_6_密封摩擦力矩_复绘.png',dpi=220);fig.savefig(O/'图2_6_密封摩擦力矩_复绘.pdf')
w=Workbook();s=w.active;s.title='说明与输入';rows=[['图2.6密封面摩擦力矩：论文结果复绘','数值或说明'],['方法','依据论文明确给出的线性关系和1 MPa端点，不是积分模型独立计算'],['开启斜率 / (N·m/MPa)',KO],['关闭斜率 / (N·m/MPa)',KC],['查询压力 / MPa',.45],['开启摩擦力矩 / N·m',"=B3*B5"],['关闭摩擦力矩 / N·m',"=B4*B5"],['来源','用户提供2.pdf，印刷页20–21：图2.6及其后文字'],['适用范围','仅复绘论文给定结构的0至1 MPa；不外推至其他阀门'],['零压假设','论文忽略预紧等引起的残余摩擦；不代表真实阀门零压扭矩为零'],['压力含义','沿用论文“介质压力”定义，暂不替换为任意工况压差'],['未完成项','µ_M等参数未明确，尚未独立数值积分验证公式2.7至2.23'],['几何说明','此表不使用此前补充假设的2 mm阀座模型，不证明该模型与论文等价']]
for r in rows:s.append(r)
s.column_dimensions['A'].width=34;s.column_dimensions['B'].width=95
for row in s:
 for cell in row:cell.alignment=Alignment(wrap_text=True,vertical='top')
s['B5'].fill=PatternFill('solid',fgColor='FFF2CC');s['B3'].fill=s['B4'].fill=PatternFill('solid',fgColor='DDEBF7')
d=w.create_sheet('曲线数据');d.append(['压力 / MPa','开启摩擦力矩 / N·m','关闭摩擦力矩 / N·m'])
for i,x in enumerate(p,2):d.append([float(x),f"=A{i}*'说明与输入'!$B$3",f"=A{i}*'说明与输入'!$B$4"])
for col in ['A','B','C']:d.column_dimensions[col].width=26
chart=ScatterChart();chart.title='图2.6：密封面摩擦力矩（结果复绘）';chart.x_axis.title='介质压力 / MPa';chart.y_axis.title='摩擦力矩 / N·m';chart.width=22;chart.height=14
for j,color in [(2,'DB2828'),(3,'289B31')]:
 series=Series(Reference(d,min_col=j,min_row=2,max_row=102),Reference(d,min_col=1,min_row=2,max_row=102),title_from_data=False,title=d.cell(1,j).value);series.graphicalProperties.line.solidFill=color
 if j==3:series.graphicalProperties.line.prstDash='dashDot'
 chart.series.append(series)
d.add_chart(chart,'E2')
for sheet in w:
 sheet.freeze_panes='A2'
 for cell in sheet[1]:cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476A')
w.save(O/'01_密封摩擦力矩_结果复绘.xlsx')
(O/'检查结果.json').write_text(json.dumps({'method':'published-endpoint linear replot, not independent equation verification','endpoint_1_MPa_Nm':{'opening':KO,'closing':KC},'at_0_45_MPa_Nm':{'opening':KO*.45,'closing':KC*.45},'origin_both_zero':True},ensure_ascii=False,indent=2))
print('created chart, CSV, Excel; endpoint and 0.45 MPa arithmetic checks passed')
