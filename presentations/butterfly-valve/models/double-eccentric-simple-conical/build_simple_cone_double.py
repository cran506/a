from pathlib import Path
import os,math,csv,json,zipfile,shutil
os.environ['MPLCONFIGDIR']='/workspace/work/valve_model/mplcache'
import cadquery as cq
import numpy as np
O=Path('/workspace/output/双偏心锥形蝶阀_简化装配');O.mkdir(parents=True,exist_ok=True)
R=117.5;c=30.;e=15.;b=15.;theta=10.;k=math.tan(math.radians(theta))
H1=(c,e,-180);H2=(c,e,180)
dp=[(-b/2,0),(-b/2,R-k*b/2),(b/2,R+k*b/2),(b/2,0)]
sp=[(-b/2,140),(b/2,140),(b/2,R+k*b/2),(-b/2,R-k*b/2)]
disc=cq.Workplane('XY').polyline(dp).close().revolve(360,(0,0,0),(1,0,0)).val()
seat=cq.Workplane('XY').polyline(sp).close().revolve(360,(0,0,0),(1,0,0)).val()
axis=cq.Solid.makeCylinder(5,300,cq.Vector(c,e,-150),cq.Vector(0,0,1))
for s in [disc,seat,axis]:assert s.isValid() and len(s.Solids())==1
for name,s in [('01_蝶板.step',disc),('02_阀座.step',seat),('03_H轴参考圆柱.step',axis)]:cq.exporters.export(s,str(O/name))
poses=O/'指定角度位置';poses.mkdir(exist_ok=True)
for angle in [0,.1,1,5,10,30,90]:
 a=cq.Assembly(name=f'DoubleEccentric_Simple_{angle}')
 a.add(seat,name='Seat_fixed',color=cq.Color(.22,.48,.7));a.add(disc.rotate(H1,H2,angle),name='Disc',color=cq.Color(.93,.60,.2));a.add(axis,name='H_reference_fixed',color=cq.Color(.22,.27,.32))
 a.export(str(poses/f'装配位置_{angle:g}度.step'))
 if angle==0:a.export(str(O/'04_简化完整装配_关闭.step'))
rows=[]
for angle in [0,.01,.025,.05,.1,.25,.5,1,2,3,5,10,15,20,30,45,60,75,90]:
 q=disc.rotate(H1,H2,angle).intersect(seat,tol=1e-6);assert q.isValid();vol=q.Volume() if q.Solids() else 0;assert vol>=-1e-6
 rows.append({'angle_deg':angle,'disc_seat_intersection_mm3':max(0,vol)})
 if angle==.1 and vol>.001:cq.exporters.export(q,str(O/'05_开启0.1度_交叠区域.step'))
with (O/'06_各角度干涉检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
readme='''# 双偏心锥形蝶阀：按用户截图形式制作的简化装配

## 本次模型
仅有蝶板、阀座、H轴参考圆柱，不加阀体、压板、软密封、凸台或驱动机构。
蝶板是完整厚度的锥台，阀座内锥面与蝶板外锥面来自同一圆锥，关闭名义零间隙、零过盈。没有把锥面改成球面或软密封唇。
这一版刻意不添加边缘退让，以直接观察整段锥面旋转是否干涉；不是可制造或已验证能正常启闭的设计。

## 参数及坐标
单位mm、度。x为流道方向，蝶板中面x=0，厚15（x=±7.5）。
H轴通过(x,y)=(30,15)，沿z方向；c=30是到蝶板中面的轴向偏心，e=15是相对圆盘中心的径向偏心。
锥轴与流道同轴，没有第三角偏心，α=0。
锥半角θ=10°，指母线与流道轴的夹角；r(x)=117.5+x·tan10°。
117.5是本示例x=0处的参考半径；原表R0的具体底面位置缺失，所以仅沿用数值，不声称定义完全一致。A0/B0未采用。
阀座外半径140、轴向厚15，H轴参考圆柱半径5、长300为示例尺寸。
H轴圆柱是固定参考件，不代表真实阀杆与蝶板连接结构，也不应纳入密封副干涉检查。

## 打开与使用
04是关闭位置三件装配；01/02/03为可另存SLDPRT的独立零件，共用坐标。
“指定角度位置”文件夹直接提供0、0.1、1、5、10、30、90°位置，不用先制作动画。角度按+z右手方向，从关闭位置直接计算，非累加。
STEP不包含SolidWorks原生配合、特征树和动画。按原坐标插入后，固定阀座及参考圆柱，蝶板浮动；蝶板内部建立通过(30,15,0)且平行z的基准轴，配合参考圆柱轴线，然后用轴向定位与角度配合控制状态。

## 干涉说明
这个参数组合和完整15 mm锥面接触段会在初开出现刚体交叠。动画/姿态能够显示穿过去，不意味着机械上能穿过。
06给出完整三维实体求交体积；05是0.1°的实际交叠区域。可单独着红，用于观察发生位置。
在SolidWorks中用“评估→干涉检查”，只选蝶板、阀座，并在初开小角度加密检查。不要只看一个截面，也不要把参考圆柱当成碰撞零件。
关闭处零交叠只代表名义贴合，不证明开启避让。有限角度检查也不是连续全行程、公差和变形条件下的严格证明。
若要设计成能正常启闭，需要再确定接触带宽、边缘退让、锥角、偏心方向和参数，不能以软件允许转动代替几何与接触校核。
'''
(O/'00_模型说明.md').write_text(readme,encoding='utf8')
(O/'07_参数.json').write_text(json.dumps({'R_at_midplane':R,'c':c,'e':e,'b':b,'theta_cone_halfangle_deg':theta,'alpha_third_offset_deg':0,'shaft_direction':[0,0,1],'shaft_point':[c,e,0],'seat_outer_radius':140,'nominal_interference':0,'full_thickness_conical_contact':True},ensure_ascii=False,indent=2),encoding='utf8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,axes=plt.subplots(1,3,figsize=(12,7),facecolor='#f4f7fb')
diskpoly=np.array([[-7.5,-R+k*7.5],[-7.5,R-k*7.5],[7.5,R+k*7.5],[7.5,-R-k*7.5]])
for ax,angle in zip(axes,[0,.1,10]):
 rad=math.radians(angle);rot=np.array([[math.cos(rad),-math.sin(rad)],[math.sin(rad),math.cos(rad)]])
 poly=(diskpoly-np.array([c,e]))@rot.T+np.array([c,e])
 for sign in [1,-1]:
  ss=np.array([[-7.5,sign*140],[7.5,sign*140],[7.5,sign*(R+k*7.5)],[-7.5,sign*(R-k*7.5)]])
  ax.add_patch(Polygon(ss,facecolor='#397aad',alpha=.6))
 ax.add_patch(Polygon(poly,facecolor='#e99a36',alpha=.9,edgecolor='#9c641c'))
 ax.add_patch(Circle((c,e),5,color='#273c50'));ax.text(c+7,e,'H',fontsize=12)
 ax.axhline(0,color='#b7c1cc',lw=.7,ls='--');ax.set(xlim=(-40,75),ylim=(-150,150),aspect='equal');ax.set_title(f'开启{angle:g}°',fontproperties=font);ax.set_xlabel('轴向x / mm',fontproperties=font)
axes[0].set_ylabel('径向y / mm',fontproperties=font)
fig.suptitle('简化双偏心锥形蝶阀：阀板、阀座与H轴',fontproperties=font,fontsize=19)
fig.text(.05,.025,'θ=10°，c=30，e=15，b=15 mm；初开有刚体干涉，细节以三维交叠检查为准。',fontproperties=font,fontsize=12)
fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(O/'08_侧视位置预览.png',dpi=140);plt.close(fig)
shutil.copy2(__file__,O/'build_simple_cone_double.py')
with zipfile.ZipFile('/workspace/output/双偏心锥形蝶阀_简化装配.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in O.rglob('*'):
  if f.is_file():z.write(f,'双偏心锥形蝶阀_简化装配/'+str(f.relative_to(O)))
print('DONE',json.dumps(rows[:5]),flush=True)
