from pathlib import Path
import os,math,json,csv,shutil,zipfile
os.environ['MPLCONFIGDIR']='/workspace/work/valve_model/mplcache'
import cadquery as cq
import numpy as np
OUT=Path('/workspace/output/双偏心锥形密封_修正版');OUT.mkdir(parents=True,exist_ok=True)
OLD=Path('/workspace/output/双偏心蝶阀_完整参考模型')
R=117.5;c=30.;e=15.;b=15.;h=.5
H1=(c,e,-220);H2=(c,e,220)
body=cq.importers.importStep(str(OLD/'04_阀体.step')).val()
shaft=cq.importers.importStep(str(OLD/'05_阀杆.step')).val()
colors={'Body':(.46,.57,.68),'Seat':(.14,.53,.74),'Disc':(.94,.60,.18),'Shaft':(.22,.27,.33)}
rear=cq.Workplane('XY').box(600,300,600).translate((0,150,0)).val()
angles=[0,.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5,.6,.75,1,1.25,1.5,2,3,5,7.5,10,15,20,30,45,60,75,90]
versions=[]
def overlap(a,b):
 q=a.intersect(b,tol=1e-6);assert q.isValid();v=q.Volume() if q.Solids() else 0;assert v>=-1e-6,v
 return q,max(0,v)
for theta in [10,15]:
 k=math.tan(math.radians(theta));folder=OUT/f'锥半角{theta}度';folder.mkdir(exist_ok=True)
 # Exact conical sealing land, plus explicit relief outside it.
 profile=[(-7.5,0),(-7.5,112),(-2,112),(-h,R-k*h),(h,R+k*h),(2,112),(7.5,112),(7.5,0)]
 disc=cq.Workplane('XY').polyline(profile).close().revolve(360,(0,0,0),(1,0,0)).val()
 boss=cq.Workplane('XY').box(30,22,190).translate((21,15,0)).val()
 disc=disc.fuse(boss).clean().cut(cq.Solid.makeCylinder(8.15,400,cq.Vector(c,e,-200),cq.Vector(0,0,1))).clean()
 seatprofile=[(-12,125),(12,125),(12,124),(3,121),(h,R+k*h),(-h,R-k*h),(-3,121),(-12,124)]
 seat=cq.Workplane('XY').polyline(seatprofile).close().revolve(360,(0,0,0),(1,0,0)).val().clean()
 assert disc.isValid() and seat.isValid()
 parts={'Body':body,'Seat':seat,'Disc':disc,'Shaft':shaft}
 for key,name in [('Disc','02_阀板.step'),('Seat','03_阀座圈_锥形刚体.step')]:cq.exporters.export(parts[key],str(folder/name))
 for a,name in [(0,'01_完整装配_关闭.step'),(90,'04_完整装配_开启90度.step')]:
  ass=cq.Assembly(name=f'Double_Eccentric_Cone_{theta}')
  for namepart,shape in parts.items():ass.add(shape.rotate(H1,H2,a) if namepart in ['Disc','Shaft'] else shape,name=namepart,color=cq.Color(*colors[namepart]))
  ass.export(str(folder/name))
 rows=[]
 for angle in angles:
  p=disc.rotate(H1,H2,angle)
  common,sv=overlap(p,seat);_,mv=overlap(p,body)
  r={'angle_deg':angle,'disc_seat_overlap_mm3':sv,'disc_body_overlap_mm3':mv,'seat_interference_above_0.001mm3':sv>.001}
  rows.append(r)
  if angle in [.1,.5] and sv>.001:cq.exporters.export(common,str(folder/f'05_干涉交叠实体_{angle}度.step'))
 with (folder/'06_全周三维干涉检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 meshes={}
 for key,shape in parts.items():
  if key in ['Body','Seat']:shape=shape.intersect(rear)
  vs,fs=shape.tessellate(.6,.2)
  meshes[key]={'vertices':[[round(v.x,6),round(v.y,6),round(v.z,6)] for v in vs],'faces':[list(f) for f in fs],'color':list(colors[key]),'rotating':key in ['Disc','Shaft']}
 # These red meshes are shown only at their exactly evaluated angle.
 red={}
 for angle in [.1,.5]:
  q,vol=overlap(disc.rotate(H1,H2,angle),seat)
  if vol>.001:
   vs,fs=q.tessellate(.05,.15);red[str(angle)]={'vertices':[[round(v.x,6),round(v.y,6),round(v.z,6)] for v in vs],'faces':[list(f) for f in fs],'color':[.9,.12,.12],'rotating':False}
 margin=k*e-abs(k*R-c)-(1+k*k)*h
 versions.append({'theta':theta,'meshes':meshes,'rows':rows,'red':red,'margin_mm_per_rad':margin})
 print(theta,'margin',margin,'interference_angles',[r['angle_deg'] for r in rows if r['seat_interference_above_0.001mm3']],flush=True)

for fname in ['04_阀体.step','05_阀杆.step']:shutil.copy2(OLD/fname,OUT/fname)
# Reuse the tested depth-buffered renderer; add a parameter selector and initial-opening focus mode.
oldhtml=(OLD/'10_完整模型_交互启闭动画.html').read_text()
script=oldhtml.split('<script>')[1].split('</script>')[0]
tail=script[script.index(',C=document'):]
tail='const '+tail[1:]
tail=tail.replace("angle.toFixed(1)+'°'","angle.toFixed(3)+'°'")
tail=tail.replace('let angle=0,playing=true','let maxAngle=90,angle=0,playing=true')
tail=tail.replace('for(let[k,m]of Object.entries(meshes))','for(let[k,m]of Object.entries(activeMeshes()))')
tail=tail.replace('gl.drawArrays(gl.TRIANGLES,0,data.length/6);','gl.drawArrays(gl.TRIANGLES,0,data.length/6);updateStatus();')
tail=tail.replace('*.018','*(maxAngle===2?.00035:.018)').replace('if(angle>=90){angle=90;','if(angle>=maxAngle){angle=maxAngle;')
head='const versions='+json.dumps(versions,separators=(',',':'))+';let version=versions[0],meshes=version.meshes;'
extra='''
function activeMeshes(){let m={...meshes};for(let[a,r]of Object.entries(version.red))if(Math.abs(angle-Number(a))<.00001)m.Interference=r;return m}
function updateStatus(){let row=version.rows.reduce((best,r)=>Math.abs(r.angle_deg-angle)<Math.abs(best.angle_deg-angle)?r:best,version.rows[0]);document.querySelector('#status').textContent='锥半角 '+version.theta+'°；最近CAD检查角度 '+row.angle_deg+'°，阀板/阀座交叠 '+row.disc_seat_overlap_mm3.toFixed(6)+' mm³。此数值仅对应该采样角度，不是当前角度实时求解。'}
document.querySelector('#theta').onchange=e=>{version=versions[Number(e.target.value)];meshes=version.meshes;angle=0;draw()};
document.querySelector('#focus').onclick=()=>{maxAngle=maxAngle===90?2:90;slider.max=maxAngle;angle=0;dir=1;playing=true;document.querySelector('#play').textContent='暂停';document.querySelector('#focus').textContent=maxAngle===2?'返回全行程':'观察初开0～2°';draw()};
for(let[a,id]of [[.1,'at01'],[.5,'at05']])document.querySelector('#'+id).onclick=()=>{angle=a;playing=false;document.querySelector('#play').textContent='播放';draw()};
'''
# Function declarations are hoisted; assign UI handlers before the initial draw/loop.
tail=tail.replace('draw();requestAnimationFrame(loop);',extra+'draw();requestAnimationFrame(loop);')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>双偏心锥形密封：锥角与干涉</title><style>body{font:17px system-ui;background:#f4f7fb;color:#17324d;max-width:1350px;margin:auto;padding:20px}canvas{width:100%;background:white;border-radius:12px;touch-action:none}button,select{padding:9px 12px;font-size:16px;margin:5px}input[type=range]{width:35%}p{line-height:1.7}.bar{display:flex;align-items:center;flex-wrap:wrap}#status{padding:12px;background:#fff0dd;border-radius:8px}</style><h1>双偏心锥形密封：锥角与初开干涉</h1><p>本次修正为刚性锥形密封副，无软阀座过盈、无第三角偏心。阀板锥面与阀座锥面同轴于流道，转轴偏置c=30、e=15 mm。先看10°原参数，再切到15°比较。</p><div class="bar"><select id="theta"><option value="0">锥半角10°：沿用原值</option><option value="1">锥半角15°：对比方案</option></select><button id="play">暂停</button><button id="reverse">反向</button><button id="reset">关闭</button><button id="focus">观察初开0～2°</button><button id="at01">停在0.1°</button><button id="at05">停在0.5°</button><input id="angle" type="range" min="0" max="90" step="0.005" value="0"><b id="num"></b></div><p id="status"></p><canvas width="1200" height="700"></canvas><p>拖动画面旋转视角。橙：阀板；蓝：锥形阀座圈；灰蓝：阀体；深灰：阀杆。只为观察而将阀体/阀座半剖，检查使用完整三维实体。10°方案停在0.1°和0.5°时，红色为该角度真实CAD求交得到的交叠区域，尺寸很小，不一定在当前视角可见。</p><p><b>含义：</b>关闭位置锥面贴合不代表转动轨迹必然避让。10°方案初开出现刚体交叠；15°方案在随附采样点未检出交叠，不构成连续运动、加工公差和受载变形下的无干涉保证。本模型没有计算接触应力或扭矩。</p><p><b>几何假设：</b>R=117.5 mm解释为密封带中面的参考半径，阀板主体厚15 mm，锥形接触带轴向宽1 mm，其余外缘做退让。θ为圆锥母线与流道轴线的夹角。阀座圈、退让、阀体及法兰尺寸为教学示例；不声称复原来图的制造结构。</p><script>'''+head+tail+'''</script></html>'''
(OUT/'07_完整模型_锥角与启闭交互演示.html').write_text(page,encoding='utf8')

# Scientific contact plot: genuine sampled intersection volumes, not invented torque curves.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig,ax=plt.subplots(figsize=(9,5))
for version in versions:
 rows=[r for r in version['rows'] if r['angle_deg']<=2]
 ax.plot([r['angle_deg'] for r in rows],[r['disc_seat_overlap_mm3'] for r in rows],'-o',label=f"{version['theta']}°")
ax.set_xlabel('开启转角 / °',fontproperties=font);ax.set_ylabel('阀板与阀座刚体交叠体积 / mm³',fontproperties=font);ax.set_title('相同两项偏心下的锥角对比（初开段）',fontproperties=font);ax.legend(title='Cone half-angle');ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(OUT/'08_锥角与初开干涉对比.png',dpi=150);plt.close(fig)

fig=plt.figure(figsize=(12,7),facecolor='#f4f7fb');axes=[];collections=[];arrays=[]
for j,version in enumerate(versions):
 ax=fig.add_subplot(1,2,j+1,projection='3d');ax.set_facecolor('#f4f7fb');cc={};aa={}
 for key,m in version['meshes'].items():
  v=np.array(m['vertices']);fs=np.array(m['faces']);aa[key]=(v,fs);coll=Poly3DCollection(v[fs],facecolors=m['color'],edgecolors='none');ax.add_collection3d(coll);cc[key]=coll
 ax.set(xlim=(-170,190),ylim=(-170,170),zlim=(-190,190));ax.set_box_aspect((360,340,380));ax.view_init(elev=20,azim=-35);ax.set_title('锥半角'+str(version['theta'])+'°',fontproperties=font,fontsize=16)
 axes.append(ax);collections.append(cc);arrays.append(aa)
title=fig.suptitle('',fontproperties=font,fontsize=18)
fig.text(.05,.055,'同为刚性锥形密封；θ=10°方案初开有交叠，15°方案采样未检出交叠。',fontproperties=font,fontsize=12)
fig.text(.05,.025,'完整模型来自同一CAD几何。图中半剖仅为展示；是否干涉请看CSV，不能只凭动画目测。',fontproperties=font,fontsize=11)
timeline=[0]*5+list(range(0,91,5))+[90]*5+list(range(85,-1,-5))+[0]*5
frames=[]
for angle in timeline:
 rad=math.radians(angle);rot=np.array([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]])
 for j in range(2):
  for key in ['Disc','Shaft']:
   v,fs=arrays[j][key];vv=(v-np.array([c,e,0]))@rot.T+np.array([c,e,0]);collections[j][key].set_verts(vv[fs])
 title.set_text(f'双偏心锥形密封启闭对比｜转角 {angle}°')
 fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(im.quantize(colors=128))
 if angle==30:im.save(OUT/'10_完整模型预览.png')
frames[0].save(OUT/'09_完整模型启闭对比.gif',save_all=True,append_images=frames[1:],duration=140,loop=0,optimize=True);plt.close(fig)

low=math.degrees(math.atan(c/(R+e)));high=math.degrees(math.atan(c/(R-e)))
readme=f'''# 双偏心锥形密封：纠正前版结构，直接检查锥角干涉

前版双偏心模型采用弹性密封唇，不适合回答本次锥角问题。本次使用刚性、同轴圆锥密封副，只有c和e两项位置偏心，没有第三角偏心，不用橡胶变形解释或掩盖穿透。

## 文件
- “锥半角10度”：沿用原半角10°，完整装配、独立阀板、独立锥形阀座圈、全开装配、0.1°/0.5°交叠实体及CSV。
- “锥半角15度”：只将半角改为15°，用于对比。并非原图参数复原。
- 根目录阀体、阀杆为两方案共用。完整几何装配含阀体/阀座圈/阀板/阀杆，省略压板、紧固件、轴承、填料、实际传扭连接和驱动装置，不是制造装配。
- 07交互HTML将模型与启闭动画集成在同一离线文件。下载后浏览器打开，可比较半角、慢看初开、暂停和旋转视角。
- 08为真实CAD求交结果图，09为模型启闭GIF，10为预览。
- STEP不含SolidWorks原生特征树、配合和马达，不存在本交付将动画嵌入STEP的说法。

## 参数与几何定义
x是流道轴向，z是阀杆方向，y是径向偏置方向。阀板中面x=0，主体厚度b=15。H轴为(x,y)=(30,15)、平行z轴；正转按+z右手规则。
沿用c=30、e=15、b=15 mm。117.5重新解释为x=0处密封圆锥半径R，不声称与原文缺失图中的圆锥底面位置相同。A0、B0未使用。
圆锥半径函数r(x)=R+x·tanθ，锥轴与流道同轴。θ为母线与流道轴的夹角（不是母线与阀板平面的夹角）；α=0，因此是双偏心而不是三偏心。
阀板和阀座共用同一圆锥的x∈[-0.5,0.5]段作为刚性密封面，名义零间隙、零过盈。接触带轴向宽1 mm属于假设，图中未给实际密封带宽。
主体蝶板在接触带外退让至半径112，x=±2至±7.5为退让外缘，x=±0.5至±2为过渡斜面。阀座圈在带外退让至x=±3处半径121、x=±12处124。这样不会把整块15 mm厚蝶板都当成同一个锥面接触带。退让尺寸是研究假设，不是图纸数据。
阀座外半径125；其他阀体、法兰和轴孔沿用前教学示例。没有强度、接触变形或压力等级校核。

## 为什么10°会碰，而15°可能避让：一阶几何判断
这段推导仅适用于以上坐标、同轴圆锥、正转方向和圆周接触带，不是所有双偏心蝶阀的通用公式。
取密封带中面圆周点x=0、y=R cosφ、z=R sinφ，小转角δ（弧度）绕H轴运动：
dx/dδ=-(y-e)，dy/dδ=x-c。
定义径向间隙g=R+x tanθ-sqrt(y²+z²)，g>0表示阀板在锥孔内侧；从贴合开始，dg/dδ<0表示局部进入阀座。
在x=0处：dg/dδ=e tanθ+(c-R tanθ)cosφ。
全周最小值为e tanθ-|c-R tanθ|。因此薄接触线的一阶无穿入必要条件是：
|c-R tanθ|≤e tanθ。
等价于c/(R+e)≤tanθ≤c/(R-e)，本例约{low:.3f}°≤θ≤{high:.3f}°。
对轴向半宽h=0.5的完整锥面带，最小一阶间隙变化率进一步为：
g'_min=e tanθ-|c-R tanθ|-(1+tan²θ)h。
10°时约{versions[0]['margin_mm_per_rad']:.6f} mm/rad，为负，说明初始就有局部穿入；15°时约{versions[1]['margin_mm_per_rad']:.6f} mm/rad，为正，初始全周离座。
一阶条件只描述刚开始的运动，不足以证明0～90°都无干涉；边界等号还需要高阶检查。有限厚度、退让、轴位、方向、公差和受载变形都可能改变结论。

## 实际CAD检查
每种方案按{len(angles)}个角度检查完整阀板对完整阀座和完整阀体的交叠。关闭名义贴合，未给软密封过盈。
10°方案检出初开段交叠；15°方案在这些采样角度未检出交叠。该结果不等于已经完成连续全行程或制造公差下的无干涉证明。
检查阈值0.001 mm³仅过滤数值噪声，不能当作允许穿透量或工程验收标准。交叠体积不等于力、应力、扭矩或压缩量。
随附10°的交叠STEP是求交得到的真实区域，可在SolidWorks中单独着红显示。原尺寸下很小，不能因为动画看不见就忽略。

## SolidWorks中判断方法
1. 打开方案完整STEP，确认单位mm；固定阀体/阀座，使阀板与阀杆绕轴H同步转动。
2. 用“评估→干涉检查”，分组检查阀板—阀座、阀板—阀体、阀杆—阀体，记录交叠位置与体积。
3. 先在0、0.01、0.025、0.05、0.075、0.1、0.15、0.2、0.25、0.3、0.4、0.5、0.6、0.75、1、1.25、1.5、2°加密检查，再到90°。这些位置为本次CSV角度，可直接对照。
4. 在非接触区还需量最小间隙，并叠加加工/装配/轴承间隙、热膨胀及受压变形的不利影响。几何间隙大于0不等于工程余量足够。
5. 剖面帮助解释，但必须检查完整三维密封带全周。不要用仅半剖展示的网格代替完整实体检查。
6. CAD动画能够转过去不代表零件能够实际穿过。刚体有交叠时，需调整角度、轴位、带宽或退让几何；若设计依靠弹性接触，必须另做接触有限元与试验，不能凭“硬密封也会变形”免除校核。

## 使用建议
要复现你问的“为什么锥角会碰”，先看10°方案停在0.1°，再看08曲线及交叠实体；要比较避让几何，再切15°。此次模型是为了直接比较锥角，不能替代来图的精确结构复原。
'''
(OUT/'00_锥角干涉原理与检查说明.md').write_text(readme,encoding='utf8')
params={'coordinate':'x axial, y radial, z shaft','H':[30,15,0],'R_reference_at_x0':R,'c':c,'e':e,'disc_thickness':b,'sealing_land_axial_width':2*h,'cone_half_angles_deg':[10,15],'third_angular_eccentricity_deg':0,'sealing':'rigid cone-cone; no nominal overlap at closure','sampling_angles_deg':angles,'threshold_mm3':.001,'thin_line_initial_angle_range_deg':[low,high],'initial_clearance_rate_mm_per_rad':{str(v['theta']):v['margin_mm_per_rad'] for v in versions}}
(OUT/'11_参数与坐标.json').write_text(json.dumps(params,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copy2(__file__,OUT/'build_conical_double.py')
with zipfile.ZipFile('/workspace/output/双偏心锥形密封_完整模型与动画.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.rglob('*'):
  if f.is_file():z.write(f,'双偏心锥形密封_修正版/'+str(f.relative_to(OUT)))
print('DONE',flush=True)
