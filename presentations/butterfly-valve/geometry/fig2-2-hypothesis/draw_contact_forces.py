"""Conditional sliding friction on disc in the hypothetical cone section.
Arrows exist only while contact persists; not an actual contact solution.
"""
from pathlib import Path
import numpy as np, math
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Patch
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
O=Path(__file__).resolve().parent
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
H=np.array([30.,15.]);up=lambda x:56+(x+650)*math.tan(math.radians(5));lo=lambda x:56+(x+650)*math.tan(math.radians(-15))
fig,axs=plt.subplots(1,2,figsize=(13,10));fig.subplots_adjust(top=.87,bottom=.23,wspace=.16)
for ax,sgn,title in zip(axs,[1,-1],['开启方向示例：绕 H 逆时针','关闭方向示例：绕 H 顺时针']):
 ax.add_patch(Polygon([(-7.5,lo(-7.5)),(7.5,lo(7.5)),(7.5,up(7.5)),(-7.5,up(-7.5))],facecolor='#f3c47e',hatch='///',edgecolor='#9a6b25',lw=1.2))
 xx=np.array([-75,75]);ax.plot(xx,up(xx),c='#777',lw=1);ax.plot(xx,lo(xx),c='#777',lw=1)
 ax.axvline(0,c='#b38b53',ls='--',lw=.6);ax.axhline(0,c='#bbb',ls=':',lw=.7)
 for name,x,y,dx,dy in [('A',-7.5,up(-7.5),-10,14),('M',7.5,up(7.5),10,14),('C',-7.5,lo(-7.5),-10,-17),('N',7.5,lo(7.5),10,-17)]:
  ax.scatter(x,y,s=24,c='#85328c',zorder=6);ax.text(x+dx,y+dy,name,c='#85328c',fontsize=13,ha='center')
 ax.scatter(*H,c='#333',s=45,zorder=6);ax.text(H[0]+4,H[1]-5,'H（转轴）',fontsize=11)
 ax.text(-69,30,'蝶板',fontsize=12,c='#875b1b');ax.annotate('',xy=(-3,15),xytext=(-47,27),arrowprops=dict(arrowstyle='->',color='#875b1b'))
 for i,Q,tangent,inward in [(1,np.array([0.,up(0)]),np.array([math.cos(math.radians(5)),math.sin(math.radians(5))]),np.array([math.sin(math.radians(5)),-math.cos(math.radians(5))])),(2,np.array([0.,lo(0)]),np.array([math.cos(math.radians(-15)),math.sin(math.radians(-15))]),np.array([math.sin(math.radians(15)),math.cos(math.radians(15))]))]:
  r=Q-H;v=sgn*np.array([-r[1],r[0]]);vt=np.dot(v,tangent)*tangent;st=vt/np.linalg.norm(vt);fr=-st
  assert np.dot(fr,vt)<0 and abs(np.dot(fr,inward))<1e-10
  ax.scatter(*Q,c='black',s=38,zorder=8)
  ax.text(Q[0]+5,Q[1]+(-16 if i==1 else 12),f'Q{i}',fontsize=12,weight='bold')
  def arrow(direction,length,color,label,offset):
   end=Q+direction*length;ax.annotate('',xy=end,xytext=Q,arrowprops=dict(arrowstyle='-|>',color=color,lw=2.3,mutation_scale=15),zorder=7)
   ax.text(end[0]+offset[0],end[1]+offset[1],label,color=color,fontsize=12,ha='center')
  arrow(st,50,'#268c43',r'$v_t$',(0,-13 if i==1 else -12))
  arrow(fr,50,'#d12b30',r'$F_f$',(0,9 if i==1 else 9))
  arrow(inward,38,'#226eb6',r'$N$',(11 if i==1 else 7,0))
 ax.set_xlim(-85,95);ax.set_ylim(-166,160);ax.set_aspect('equal');ax.axis('off');ax.set_title(title+'\n仅在接触保持且发生滑动时',fontsize=13)
fig.suptitle('推测剖面：作用在蝶板上的法向力与滑动摩擦力',fontsize=18,y=.97)
handles=[Line2D([0],[0],color='#d12b30',lw=2,label='红色 Ff：蝶板所受摩擦力'),Line2D([0],[0],color='#268c43',lw=2,label='绿色 vt：相对滑动速度'),Line2D([0],[0],color='#226eb6',lw=2,label='蓝色 N：阀座对蝶板的法向力')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.16),ncol=3,fontsize=11,frameon=False)
fig.text(.07,.13,'Q1、Q2：任选的上下接触位置，用于说明局部方向；A、M、C、N 是剖面角点，不代表四个集中载荷。',fontsize=11)
fig.text(.07,.092,'滑动时：Ff = −μN · vt/|vt|。反转滑动方向，摩擦方向随之反转；阀座所受摩擦力与图示反向。',fontsize=11)
fig.text(.07,.053,'重要：开启后若已经脱离，N=0、Ff=0；未滑动时应按静摩擦条件判断。图中箭头仅表示方向，不表示大小。',fontsize=11,color='#a42e2e')
fig.text(.07,.015,'沿用前图的假设坐标、母线斜率和转轴位置，未恢复原论文图2.2，也没有求解真实接触区域或接触压力。',fontsize=10,color='#8b3333')
fig.savefig(O/'摩擦力方向与作用点_推测.png',dpi=180,bbox_inches='tight');fig.savefig(O/'摩擦力方向与作用点_推测.pdf',bbox_inches='tight')
