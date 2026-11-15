from pathlib import Path
import math
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Arc
from matplotlib.font_manager import FontProperties
O=Path(__file__).resolve().parent
fp=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=fp.get_name();plt.rcParams['axes.unicode_minus']=False
fig=plt.figure(figsize=(15,9));gs=fig.add_gridspec(2,2,height_ratios=[1.1,1],width_ratios=[1.5,1],hspace=.38,wspace=.24)
a=fig.add_subplot(gs[0,:]);d=fig.add_subplot(gs[1,0]);t=fig.add_subplot(gs[1,1]);t.axis('off')
th=math.radians(10);ga=math.radians(-5);xx=-650;yy=56;xe=40
m1=math.tan(ga+th);m2=math.tan(ga-th)
up=lambda x: yy+(x-xx)*m1
lo=lambda x: yy+(x-xx)*m2
poly=[(xx,yy),(xe,up(xe)),(xe,lo(xe))]
a.add_patch(Polygon(poly,facecolor='#d8e9fa',alpha=.45,edgecolor='#3370a0',lw=1.7))
a.plot([xx,xe],[yy,yy+(xe-xx)*math.tan(ga)],'--',c='#3370a0',lw=1.2)
a.plot([xx,xx+175],[yy,yy],':',c='#777',lw=1)
a.scatter(xx,yy,color='black',s=22);a.text(xx-8,yy+10,'O  锥顶',ha='right',fontsize=11)
slab=[(-7.5,lo(-7.5)),(7.5,lo(7.5)),(7.5,up(7.5)),(-7.5,up(-7.5))]
a.add_patch(Polygon(slab,facecolor='#eba94f',edgecolor='#a76a1e',hatch='////',lw=1.2))
a.plot([0,0],[lo(0)-15,up(0)+15],'--',color='#ad761c',lw=1)
a.scatter(30,15,s=60,c='#c93838',zorder=5);a.text(43,18,'H  转轴在本剖面中的交点',fontsize=10,c='#ad2525')
a.add_patch(Arc((xx,yy),200,200,theta1=-5,theta2=5,color='#3370a0'));a.text(xx+113,yy-1,r'$\theta$',color='#3370a0',fontsize=15)
a.text(xx+192,yy-13,r'$\alpha$：圆锥轴线相对管道轴向的倾角',fontsize=10,c='#3370a0')
a.annotate('管道轴向 X',xy=(-230,yy),xytext=(-440,yy),arrowprops=dict(arrowstyle='->',color='#666'),fontsize=10,color='#666',va='bottom')
a.text(-130,up(-130)+17,'圆锥母线（本剖面的边界）',fontsize=10,c='#3370a0')
a.annotate('蝶板：用两平行平面截取圆锥',xy=(0,85),xytext=(-310,150),arrowprops=dict(arrowstyle='->'),fontsize=11)
a.set_xlim(-720,180);a.set_ylim(-160,200);a.set_aspect('equal');a.axis('off');a.set_title('A  圆锥与蝶板的关系（沿假定转轴 Z 方向观察）',loc='left',fontsize=13)
# Enlarged local section: positions are illustrative, not a geometrical reconstruction.
d.add_patch(Polygon([(-7.5,lo(-7.5)),(7.5,lo(7.5)),(7.5,up(7.5)),(-7.5,up(-7.5))],facecolor='#f2c176',edgecolor='#a76a1e',hatch='///',lw=1.3))
xs=np.array([-22,55]);d.plot(xs,up(xs),c='#3370a0',lw=1.5);d.plot(xs,lo(xs),c='#3370a0',lw=1.5)
d.axvline(0,c='#ad761c',ls='--',lw=1);d.axhline(0,c='#888',ls='--',lw=.8)
d.scatter([0,30],[0,15],c=['#333','#c93838'],s=[25,60],zorder=5);d.text(-3,-10,'P',fontsize=11);d.text(33,18,'H（转轴沿纸面法向）',fontsize=10,c='#ad2525')
d.plot([30,30],[0,40],':',c='#c93838');d.plot([0,30],[15,15],':',c='#c93838')
d.annotate('',xy=(0,40),xytext=(30,40),arrowprops=dict(arrowstyle='<->',color='#c93838'));d.text(15,45,'c：轴向偏置',fontsize=11,ha='center',color='#ad2525')
d.annotate('',xy=(45,0),xytext=(45,15),arrowprops=dict(arrowstyle='<->',color='#c93838'));d.text(48,6,'e',fontsize=13,color='#ad2525')
d.annotate('',xy=(-7.5,-150),xytext=(7.5,-150),arrowprops=dict(arrowstyle='<->',color='#a76a1e'));d.plot([-7.5,-7.5],[-130,-155],':',color='#a76a1e');d.plot([7.5,7.5],[-130,-155],':',color='#a76a1e');d.text(0,-164,'b：蝶板厚度',ha='center',fontsize=11,color='#84531a')
d.text(-20,105,'两侧斜边是锥面的剖切线',fontsize=10,c='#3370a0')
d.annotate('中性面 x=0',xy=(0,-65),xytext=(20,-90),arrowprops=dict(arrowstyle='->'),fontsize=10)
d.set_xlim(-28,95);d.set_ylim(-178,140);d.set_aspect(.35);d.axis('off');d.set_title('B  局部放大：仅示意厚度与轴位（非等比例）',loc='left',fontsize=12)
t.text(0,1,'阅读这张推测图',fontsize=14,weight='bold',va='top')
notes=[('蓝色','密封圆锥的轴向剖面，非完整阀体。'),('橙色','蝶板实体在此平面中的剖面。'),('P 点','假定蝶板中性面参考中心。'),('红色 H 点','假定阀杆轴线穿过本图平面的位置。'),('c、e','分别用轴向与径向坐标差表示。'),('α 与 θ','α 是角偏心；θ 是圆锥半角，二者不同。')]
y=.87
for k,v in notes:t.text(0,y,k+'：'+v,fontsize=11,va='top',wrap=True);y-=.12
t.text(0,.08,'R0、A0、B0及原图字母对应未确认，\n本图不为它们指定未经核实的位置。',fontsize=11,color='#a12525',va='top')
fig.suptitle('图2.2 的一种可能几何解释——根据文字推测，不是原图复原',fontsize=18,y=.99)
fig.text(.03,.015,'约定：X 为管道轴向，Y 为径向，Z 为转轴方向；偏心正负、参考中心与剖切位置均待核实。此图不能证明原图截面定义，也不用于确定尺寸或验证力矩公式。',fontsize=10,color='#8d2525')
fig.savefig(O/'图2_2_推测示意.png',dpi=180,bbox_inches='tight');fig.savefig(O/'图2_2_推测示意.pdf',bbox_inches='tight')
