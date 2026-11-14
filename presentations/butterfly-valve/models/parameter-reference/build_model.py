"""Parametric reference geometry, not a reconstruction of the missing Fig. 2.2."""
from pathlib import Path
import math,json,csv,shutil,zipfile
import cadquery as cq
import numpy as np

OUT=Path('/workspace/output/三偏心蝶阀_参数化参考模型');OUT.mkdir(parents=True,exist_ok=True)
theta=math.radians(10);alpha=math.radians(-5);B=113.05;b=15.0;c=30.0;e=15.0;R0=117.5
a=np.array([math.cos(alpha),math.sin(alpha),0.0]);Q=np.outer(a,a)-math.cos(theta)**2*np.eye(3)
L=B*math.cos(theta)/math.sqrt(Q[0,0]-Q[0,1]**2/Q[1,1])
apex=np.array([-L,Q[0,1]*L/Q[1,1],0.0])
Acalc=B*math.cos(theta)/math.sqrt(-Q[1,1])
v=lambda v:cq.Vector(*map(float,v))
cone_length=800.0
cone=cq.Solid.makeCone(0,cone_length*math.tan(theta),cone_length,v(apex),v(a))
slab=cq.Workplane('XY').box(b,320,320).val()
plate=cone.intersect(slab).clean()
outer=cq.Solid.makeCylinder(140,b,cq.Vector(-b/2,0,0),cq.Vector(1,0,0))
seat=outer.cut(cone).clean()
assert plate.isValid() and seat.isValid()
assert len(plate.Solids())==1 and len(seat.Solids())==1
H1=(c,e,-180);H2=(c,e,180)
marker=cq.Solid.makeCylinder(5,300,cq.Vector(c,e,-150),cq.Vector(0,0,1))
for name,obj in [('01_蝶板_关闭位置.step',plate),('02_阀座_关闭位置.step',seat),('03_H轴参考圆柱_非结构件.step',marker)]:cq.exporters.export(obj,str(OUT/name))
# This optional multi-solid part lets a motor select a cylindrical face belonging to the plate part.
cq.exporters.export(cq.Compound.makeCompound([plate,marker]),str(OUT/'04_蝶板带马达定位圆柱_仅动画用.step'))
def assembly(deg,name,axis=False):
 p=plate.rotate(H1,H2,deg)
 assy=cq.Assembly(name='Reference_Valve')
 assy.add(seat,name='Seat',color=cq.Color(.20,.46,.68))
 assy.add(p,name='Disc',color=cq.Color(.92,.60,.21))
 if axis:assy.add(marker,name='H_axis_REFERENCE_ONLY',color=cq.Color(.15,.22,.31))
 assy.export(str(OUT/name))
assembly(0,'05_关闭装配_仅蝶板与阀座.step')
assembly(0,'06_关闭装配_含H轴参考件.step',True)
assembly(10,'07_转动10度_用于观察.step',True)
assembly(90,'08_转动90度_用于观察.step',True)

rows=[]
for deg in [0,.1,.5,1,2,5,10,20,30,45,60,75,90,-.1,-.5,-1,-2,-5,-10,-20,-30,-45,-60,-75,-90]:
 moved=plate.rotate(H1,H2,deg)
 common=seat.intersect(moved)
 volume=common.Volume() if common.Solids() else 0.0
 rows.append({'angle_deg':deg,'overlap_volume_mm3':max(0,volume),'interference_above_0.001_mm3':volume>.001})
 print('angle',deg,'overlap_mm3',round(volume,6),flush=True)
with (OUT/'09_刚体干涉检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
ref_center=apex+a*(R0/math.tan(theta))
manifest={
 'units':'mm, deg','model_status':'assumption_based_reference_not_paper_reconstruction',
 'coordinates':{'x':'pipe axial direction','y':'radial offset direction','z':'shaft direction'},
 'parameters':{'R0_mm':R0,'theta_half_angle_deg':10,'c_from_plate_midplane_mm':c,'e_from_midplane_ellipse_center_mm':e,'alpha_magnitude_deg':5,'assumed_alpha_signed_deg':-5,'B0_midplane_ellipse_semiaxis_z_mm':B,'A0_reported_mm':113.49,'A0_computed_mm':Acalc,'b_axial_thickness_mm':b,'seat_outer_radius_assumed_mm':140,'seat_axial_thickness_assumed_mm':15,'reference_shaft_radius_assumed_mm':5},
 'cone':{'apex_xyz_mm':apex.tolist(),'axis_unit_vector':a.tolist(),'R0_reference_section_center_xyz_mm':ref_center.tolist(),'reference_section_plane_normal':a.tolist()},
 'shaft':{'point_xyz_mm':[c,e,0],'direction':[0,0,1]},
 'checked_angles_deg':[r['angle_deg'] for r in rows],
 'interference_note':'Rigid solid overlap for plate and seat only. Reference shaft excluded. Positive rotation follows right-hand rule about +z. No elastic contact, no load analysis.'
}
(OUT/'10_参数与坐标定义.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
positive=[r for r in rows if r['angle_deg']>0 and r['interference_above_0.001_mm3']]
negative=[r for r in rows if r['angle_deg']<0 and r['interference_above_0.001_mm3']]
txt=f'''# 三偏心蝶阀参数化参考模型

## 使用性质
这是根据用户提供的表2-2和图2.3建立的假设性简化模型，不是缺失图2.2的精确复原，不是生产图，也未证明三维全行程无干涉。单位为毫米、角度。

## 采用的参数及不能唯一确定的部分
- 密封圆锥半角 θ=10°；角偏心幅值 α=5°。假设圆锥轴线位于x-y平面，由+x向-y倾斜5°。
- 蝶板中面定义为x=0；厚度b=15，采用x=±7.5两个平面截取圆锥实体。
- 假设B0=113.05是蝶板中面椭圆沿z方向的半轴。按精确圆锥求得另一半轴A0={Acalc:.8f}，与表中113.49相差{Acalc-113.49:.8f} mm。保留精确圆锥，不强行同时拟合两个四舍五入后的半轴。原文未明确A0、B0定义，因此此处只是合理解释。
- R0=117.5被解释为密封圆锥某个法向参考截面的半径。该参考截面由锥顶距离R0/tanθ确定；不将它直接当成蝶板最大外半径。参考截面是否就是原文所指底面，仍待核实。
- 转轴为平行z轴、通过H=(30,15,0)的直线。c=30暂按到蝶板中面的轴向偏置解释，e=15按中面椭圆中心的径向偏置解释。
- 阀座内锥面与蝶板来自同一个圆锥；阀座外半径140、轴向宽度15为示例值，不是原文数据。
- 未建立实际阀体、轴承、填料、密封环压圈、连接结构及预紧过盈。

## 模型文件
01：干净蝶板实体；02：阀座实体。两者均按关闭位置导出，共用坐标。
03：H轴参考圆柱，不是实际阀杆设计。
04：蝶板与定位圆柱放在同一STEP文件内，含两个分离实体；仅用于动画马达选择。不要把它用于强度、流场或质量计算。
05：关闭装配，仅蝶板和阀座。
06：关闭装配，增加H轴参考件。
07/08：分别绕+z轴正转10°/90°的位置示例。文件名不表示这些角度已经通过干涉检查。
09：蝶板与阀座的刚体交叠体积，单位mm³；数值阈值0.001 mm³仅用于过滤计算噪声，不是制造或验收公差。
10：完整参数与坐标定义。

## 干涉检查结果与解释
关闭位置交叠体积：{rows[0]['overlap_volume_mm3']:.8f} mm³。
正角度样本中发生交叠：{len(positive)}个；负角度样本中发生交叠：{len(negative)}个。
完整结果见CSV。若存在交叠，不应强制宣称模型能无干涉开启；它提示当前几何解释、轴位、开启方向或阀座接触范围需要进一步确定。
检查使用有限角度采样，不是连续全行程的严格证明。CAD中的刚体重叠不等于有限元中的可接受弹性过盈；模型也没有计算关闭扭矩或颗粒夹持载荷。

## SolidWorks打开与动画
优先打开05查看配合关系。STEP不含原生旋转、配合或马达特征。
若需独立零件，分别打开01和02另存为SLDPRT，再按原坐标插入新装配体。固定阀座，阀板浮动。
如需选择马达圆柱面，可将04作为一个多实体零件导入，代替01（不是同时插入01和04）。定位圆柱中心轴就是H轴；马达位置与方向选圆柱侧面，相对零件选阀座。
从0.1°、0.5°、1°等小角度检查运动方向和全周干涉，不能直接因为动画能够播放就认定无碰撞。

## 可重建性
build_model.py使用Python、CadQuery 2.6+、NumPy建立几何并执行检查。输出路径可按需要调整。
'''
(OUT/'00_建模假设与使用说明.md').write_text(txt,encoding='utf8')
shutil.copy2(__file__,OUT/'build_model.py')
with zipfile.ZipFile('/workspace/output/三偏心蝶阀_参数化参考模型.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():z.write(p,'三偏心蝶阀_参数化参考模型/'+p.name)
print('DONE',json.dumps({'A0_computed':Acalc,'positive_interference_samples':len(positive),'negative_interference_samples':len(negative)}),flush=True)
