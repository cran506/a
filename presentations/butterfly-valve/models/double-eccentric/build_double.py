from pathlib import Path
import os,math,json,csv,shutil,zipfile
os.environ['MPLCONFIGDIR']='/workspace/work/valve_model/mplcache'
import cadquery as cq
import numpy as np
OUT=Path('/workspace/output/双偏心蝶阀_完整参考模型');OUT.mkdir(parents=True,exist_ok=True)
R=117.5;b=15;c=30;e=15
H1=(c,e,-220);H2=(c,e,220)
vx=cq.Vector(1,0,0);vz=cq.Vector(0,0,1)
def cyl(r,l,x,y=0,z=0,d=vx):return cq.Solid.makeCylinder(r,l,cq.Vector(x,y,z),d)
def box(dx,dy,dz,x,y,z):return cq.Workplane('XY').box(dx,dy,dz).translate((x,y,z)).val()
# Two offsets are c (axial) and e (radial); no tilted conical sealing axis.
disc_profile=[(-7.5,0),(-7.5,116.5),(-3,117.5),(3,117.5),(7.5,116.5),(7.5,0)]
disc=cq.Workplane('XY').polyline(disc_profile).close().revolve(360,(0,0,0),(1,0,0)).val()
boss=box(30,22,190,21,15,0)
disc=disc.fuse(boss).clean().cut(cyl(8.15,400,c,e,-200,vz)).clean()
shaft=cyl(8,370,c,e,-185,vz)
body=cyl(145,90,-45).cut(cyl(125,110,-55))
for xx in [-55,45]:
 flange=cyl(162,10,xx).cut(cyl(125,10,xx))
 for i in range(8):
  a=2*math.pi*i/8+math.pi/8;flange=flange.cut(cyl(6.5,12,xx-1,150*math.cos(a),150*math.sin(a)))
 body=body.fuse(flange)
body=body.cut(cyl(8.6,400,c,e,-200,vz)).clean()
# Soft-seat lip is compressed at closure. It is intentionally not treated as rigid clearance.
prof=[(-12,125),(12,125),(12,124),(3,121),(.7,117.2),(-.7,117.2),(-3,121),(-12,124)]
seat=cq.Workplane('XY').polyline(prof).close().revolve(360,(0,0,0),(1,0,0)).val().clean()
parts={'Body':body,'Seat':seat,'Disc':disc,'Shaft':shaft}
for k,s in parts.items():assert s.isValid() and len(s.Solids())==1,(k,len(s.Solids()))
colors={'Body':(.46,.57,.68),'Seat':(.14,.53,.74),'Disc':(.94,.60,.18),'Shaft':(.22,.27,.33)}
def assemble(deg,name):
 a=cq.Assembly(name='DoubleEccentric_Reference')
 for k,s in parts.items():
  if k in ['Disc','Shaft']:s=s.rotate(H1,H2,deg)
  a.add(s,name=k,color=cq.Color(*colors[k]))
 a.export(str(OUT/name))
for k,n in [('Disc','02_阀板.step'),('Seat','03_阀座_弹性密封圈.step'),('Body','04_阀体.step'),('Shaft','05_阀杆.step')]:cq.exporters.export(parts[k],str(OUT/n))
assemble(0,'01_完整装配_关闭位置.step');assemble(90,'06_完整装配_开启90度.step')

# Cutaways are for display only. STEP and volume checks use the complete solids.
rear=box(600,300,600,0,150,0)
display={'Body':body.intersect(rear),'Seat':seat.intersect(rear),'Disc':disc,'Shaft':shaft}
meshes={}
for k,s in display.items():
 vs,fs=s.tessellate(.7,.25)
 meshes[k]={'vertices':[[round(v.x,5),round(v.y,5),round(v.z,5)] for v in vs],'faces':[list(f) for f in fs],'color':list(colors[k]),'rotating':k in ['Disc','Shaft']}
(OUT/'geometry.json').write_text(json.dumps(meshes,separators=(',',':')),encoding='utf8')

def volume(a,b):
 s=a.intersect(b,tol=1e-6)
 assert s.isValid()
 vol=s.Volume() if s.Solids() else 0.0
 assert vol>=-1e-6, ('Invalid signed intersection volume',vol)
 return max(0,vol)
rows=[]
for angle in [0,.05,.1,.25,.5,1,2,3,5,7.5,10,15,20,30,45,60,75,90]:
 p=disc.rotate(H1,H2,angle)
 r={'angle_deg':angle,'disc_body_overlap_mm3':volume(p,body),'disc_soft_seat_overlap_mm3':volume(p,seat),'shaft_body_overlap_mm3':volume(shaft,body)}
 rows.append(r);print(r,flush=True)
with (OUT/'08_全周刚体交叠检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metadata={'units':'mm / degree','type':'double_eccentric_resilient_seat_reference','reused':{'c_axial_offset':30,'e_radial_offset':15,'b_disc_thickness':15},'reinterpreted':{'R0_as_disc_max_radius':117.5},'not_reused':['theta=10 deg','alpha=5 deg','A0','B0'],'coordinates':{'x':'flow/axial; disc midplane x=0','y':'radial offset direction','z':'shaft axis direction'},'axis_point':[30,15,0],'axis_direction':[0,0,1],'assumptions':{'seat_lip_inner_radius':117.2,'nominal_radial_interference_at_midplane':.3,'seat_outer_radius':125,'seat_axial_length':24,'body_inner_radius':125,'body_outer_radius':145,'body_length_without_flanges':90,'flange_outer_radius':162,'flange_thickness':10,'shaft_radius':8,'shaft_bore_in_disc_radius':8.15,'shaft_bore_in_body_radius':8.6},'scope':'conceptual full valve geometry; no actuator, bearings, stem packing, keys, fasteners, pressure-rating or strength validation','tolerance_note':'0.001 mm3 is numerical reporting threshold, not a design acceptance tolerance','contact_note':'soft-seat geometric overlap is intentional precompression; it does not verify allowable rubber strain, torque or sealing.'}
(OUT/'09_参数与假设.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')

# Static preview and GIF from genuine tessellations of these CAD parts.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.font_manager import FontProperties
from PIL import Image
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
fig=plt.figure(figsize=(10,7),facecolor='#f4f7fb');ax=fig.add_subplot(111,projection='3d');ax.set_facecolor('#f4f7fb')
collections={};arrays={}
for k,m in meshes.items():
 v=np.array(m['vertices']);f=np.array(m['faces']);arrays[k]=(v,f)
 coll=Poly3DCollection(v[f],facecolors=m['color'],edgecolors='none',alpha=.8 if k=='Body' else 1)
 ax.add_collection3d(coll);collections[k]=coll
ax.set(xlim=(-170,190),ylim=(-170,170),zlim=(-190,190),xlabel='X / mm',ylabel='Y / mm',zlabel='Z / mm');ax.set_box_aspect((360,340,380));ax.view_init(elev=20,azim=-35)
title=fig.suptitle('',fontproperties=font,fontsize=18)
fig.text(.08,.065,'双偏心参考模型：阀体/阀座剖开展示，阀板绕H=(30,15)转动。',fontproperties=font,fontsize=12)
fig.text(.08,.035,'软阀座关闭预压缩量0.3 mm；动画不是弹性接触或载荷求解。',fontproperties=font,fontsize=12,color='#a14b27')
timeline=[0]*4+list(range(0,91,3))+[90]*5+list(range(87,-1,-3))+[0]*4
frames=[]
for angle in timeline:
 rad=math.radians(angle);rot=np.array([[math.cos(rad),-math.sin(rad),0],[math.sin(rad),math.cos(rad),0],[0,0,1]])
 for k in ['Disc','Shaft']:
  v,f=arrays[k];vv=(v-np.array([c,e,0]))@rot.T+np.array([c,e,0]);collections[k].set_verts(vv[f])
 title.set_text(f'双偏心蝶阀启闭｜当前转角 {angle}°')
 fig.canvas.draw();im=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(im.quantize(colors=128))
 if angle==30:im.save(OUT/'11_完整模型剖视预览.png')
frames[0].save(OUT/'07_完整模型启闭动画.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=True)
plt.close(fig)

template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>双偏心蝶阀三维启闭演示</title><style>body{font:17px system-ui;background:#f4f7fb;color:#17324d;max-width:1300px;margin:auto;padding:20px}canvas{width:100%;background:white;border-radius:12px;touch-action:none}button{padding:9px 15px;font-size:17px;margin:5px}input[type=range]{width:45%}p{line-height:1.7}.bar{display:flex;align-items:center;flex-wrap:wrap}</style><h1>双偏心蝶阀完整参考模型 · 启闭动画</h1><div class="bar"><button id="play">暂停</button><button id="reverse">反向</button><button id="reset">回到关闭</button><input id="angle" type="range" min="0" max="90" step="0.1" value="0"><b id="num">0°</b></div><canvas width="1200" height="700"></canvas><p>鼠标拖动画面旋转视角。阀体与阀座显示为半剖，完整STEP未剖切。灰蓝：阀体；蓝：弹性阀座；橙：阀板；深灰：阀杆。动画在同一几何模型上施加刚体转动，未计算软阀座变形。</p><p><b>参数：</b>轴向偏心c=30 mm，径向偏心e=15 mm，阀板主体厚度b=15 mm；将前表R0=117.5 mm重新解释为本示例阀板最大半径。原圆锥半角10°、第三偏心5°及A0/B0未沿用。</p><p><b>接触判断：</b>软阀座中部内半径117.2 mm，关闭时存在名义径向过盈0.3 mm；这不是已验证的橡胶压缩许可值。必须区分金属碰撞、软密封设计过盈与纯几何贴合。检查结果见随附CSV，不能仅凭动画判断无干涉。</p><script>
const meshes=__MESH__,C=document.querySelector('canvas'),gl=C.getContext('webgl',{antialias:true}),slider=document.querySelector('#angle');let angle=0,playing=true,dir=1,last=0,yaw=-.7,pitch=.35,down=false,px=0,py=0;
if(!gl)throw Error('此浏览器需要启用WebGL以查看三维模型');
function shader(type,s){let q=gl.createShader(type);gl.shaderSource(q,s);gl.compileShader(q);if(!gl.getShaderParameter(q,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(q));return q}
let program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec3 pos;attribute vec3 col;varying vec3 color;void main(){gl_Position=vec4(pos,1.0);color=col;}'));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 color;void main(){gl_FragColor=vec4(color,1.0);}'));gl.linkProgram(program);gl.useProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));
let buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);for(let[name,offset]of [['pos',0],['col',12]]){let l=gl.getAttribLocation(program,name);gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,3,gl.FLOAT,false,24,offset)}gl.enable(gl.DEPTH_TEST);gl.clearColor(1,1,1,1);
function transform(p,rot){let[x,y,z]=p;if(rot){x-=30;y-=15;let a=angle*Math.PI/180;[x,y]=[x*Math.cos(a)-y*Math.sin(a),x*Math.sin(a)+y*Math.cos(a)];x+=30;y+=15}let u=x*Math.cos(yaw)-y*Math.sin(yaw),v=x*Math.sin(yaw)+y*Math.cos(yaw);return [u,z*Math.cos(pitch)-v*Math.sin(pitch),v*Math.cos(pitch)+z*Math.sin(pitch)]}
function draw(){let data=[];for(let[k,m]of Object.entries(meshes)){let v=m.vertices.map(p=>transform(p,m.rotating));for(let f of m.faces){let a=v[f[0]],b=v[f[1]],c=v[f[2]],u=b.map((x,i)=>x-a[i]),w=c.map((x,i)=>x-a[i]),n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]],len=Math.hypot(...n)||1;let shade=.55+.45*Math.abs((.3*n[0]+.6*n[1]-.74*n[2])/len),col=m.color.map(x=>x*shade);for(let q of [a,b,c])data.push(q[0]/375,q[1]/218.75,q[2]/600,...col)}}gl.viewport(0,0,C.width,C.height);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);gl.drawArrays(gl.TRIANGLES,0,data.length/6);slider.value=angle;document.querySelector('#num').textContent=angle.toFixed(1)+'°'}
slider.oninput=()=>{angle=+slider.value;playing=false;document.querySelector('#play').textContent='播放';draw()};document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'暂停':'播放'};document.querySelector('#reverse').onclick=()=>dir=-dir;document.querySelector('#reset').onclick=()=>{angle=0;dir=1;draw()};C.onpointerdown=e=>{down=true;px=e.clientX;py=e.clientY;C.setPointerCapture(e.pointerId)};C.onpointerup=()=>down=false;C.onpointermove=e=>{if(down){yaw+=(e.clientX-px)*.008;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-py)*.008));px=e.clientX;py=e.clientY;draw()}};
function loop(t){if(last&&playing){angle+=dir*Math.min(50,t-last)*.018;if(angle>=90){angle=90;dir=-1}if(angle<=0){angle=0;dir=1}draw()}last=t;requestAnimationFrame(loop)}draw();requestAnimationFrame(loop);
</script></html>'''
(OUT/'10_完整模型_交互启闭动画.html').write_text(template.replace('__MESH__',json.dumps(meshes,separators=(',',':'))),encoding='utf8')

bad=[r for r in rows if r['disc_body_overlap_mm3']>.001]
contact=[r for r in rows if r['disc_soft_seat_overlap_mm3']>.001]
readme=f'''# 双偏心蝶阀完整参考模型与启闭动画

## 交付与范围
01为完整几何装配（阀体、弹性阀座、阀板、阀杆），02/03为单独阀板与阀座，04/05为单独阀体与阀杆。06为全开位置。STEP保留实体几何和装配层级/颜色，但不含SolidWorks原生配合、特征树或马达。07为完整模型启闭GIF，10为自包含HTML三维演示（离线打开，可暂停、拖动角度及旋转视角）。动画与STEP使用同一CAD几何，阀体和阀座只在动画中半剖。
本模型是双偏心软密封教学参考，未设计驱动装置、轴承、填料、键连接、紧固件、泄漏等级或承压额定值，不能直接制造。

## 参数
沿用c=30、e=15、b=15 mm。坐标x为流道方向，阀板中面x=0；转轴平行z并通过H=(30,15,0)。轴向偏心明确相对阀板中面测量，径向偏心相对圆盘中心测量。
前表R0=117.5原指圆锥底面半径，本模型将117.5重新用作阀板最大半径，属于明确的尺寸假设，不声称其定义与原表相同。
原θ=10°、α=5°、A0、B0不沿用。双偏心并不需要第三偏心；本模型密封外缘是短圆柱密封带与两侧倒角，阀座为带导入斜面的弹性密封唇，不是三偏心倾斜锥面。
主体蝶板：厚15，外缘在x=±3之间为半径117.5的短圆柱密封带，两侧倒角退至x=±7.5处半径116.5，加简化连接凸台和阀杆孔。这些密封带与倒角尺寸为演示假设。凸台外伸部分不属于b=15的主体厚度。阀杆半径8，阀板孔8.15；阀体轴孔8.6 mm。传扭连接省略，动画规定阀杆与阀板同步转动。
阀座外半径125，轴向宽24；中心密封唇内半径117.2、轴向宽1.4，外侧逐渐退让至124 mm。名义径向过盈0.3 mm；软阀座材料未标定，不能据此判断应力或压缩量是否可接受。
阀体流道半径125、外半径145、主体轴向长90；端部法兰外半径162、厚10、每端8孔。这些尺寸均为演示假设，不对应标准法兰。

## 直接检查结果
完整金属阀板（含凸台）与阀体的采样交叠超过0.001 mm³的角度数：{len(bad)}。
阀板与软阀座有交叠的采样角度数：{len(contact)}；关闭位置交叠体积：{rows[0]['disc_soft_seat_overlap_mm3']:.6f} mm³。
逐角度结果见08 CSV。0.001 mm³只是数值过滤阈值，不是验收允许值。正角度按+z右手规则转动。本结果是18个角度的离散检查，不是连续运动、全制造公差下的无干涉证明。

## 干涉判断方法（SolidWorks）
1. 打开01完整装配，确认长度单位mm。若导入为多实体零件，可按实体拆分或单独导入02～05；后者共用坐标，插入时保留原位置。
2. 使用“评估→干涉检查”。第一组只选阀板与金属阀体，第二组检查阀杆与阀体；不要把软密封过盈混进金属碰撞判断。
3. 对阀板与阀座单独检查，记录交叠体积和位置。关闭时设计预压缩会产生几何重叠，不能直接按刚体碰撞判定失败；也不能仅凭“橡胶”就认定任何重叠均合理。
4. 从关闭状态逐步检查0、0.05、0.1、0.25、0.5、1、2、3、5、7.5、10、15、20、30、45、60、75、90°。发现突变区间再加密角度，正反开启方向分开检查。必要时使用可用的动态碰撞检测。
5. 在未接触区同时测量最小间隙，结合加工公差、装配偏差、轴承间隙、温度和受压变形评估余量；间隙大于0不等于工程上足够安全。
6. 剖面用于理解位置，真正检查要覆盖完整三维圆周。动画半剖模型不能用于干涉判断，实际检查必须使用完整STEP。
7. 若要判断软阀座变形、接触压力和启闭扭矩，需补充橡胶本构、摩擦、实际预紧及压差，进行非线性接触有限元分析，并以试验验证。几何交叠体积不是力、应力或载荷。

## 在SolidWorks中重建启闭动画
固定阀体，阀座保持原位，阀板与阀杆保持刚性同步关系；让阀杆绕自身轴线转动，限制其他自由度。阀杆圆柱侧面可用于马达位置与方向选择，参考零件为固定阀体。
本模型已有真实参考阀杆，不需要再建悬空的辅助圆柱。马达只规定运动，不会自动计算摩擦或证明无碰撞。
建议动画0秒0°、3秒90°、6秒0°。优先先试0～5°，检查方向。原生动画需要在SolidWorks中设置，STEP没有嵌入动画的标准。

## 再生成
build_double.py需要Python、CadQuery、NumPy、Matplotlib、Pillow；按需要修改输出路径后运行。所有假设另存于09 JSON。
'''
(OUT/'00_模型说明与干涉判断方法.md').write_text(readme,encoding='utf8')
shutil.copy2(__file__,OUT/'build_double.py')
with zipfile.ZipFile('/workspace/output/双偏心蝶阀_完整模型与动画.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in OUT.iterdir():z.write(f,'双偏心蝶阀_完整参考模型/'+f.name)
print('DONE: metal collisions',len(bad),'soft seat contact samples',len(contact),flush=True)
