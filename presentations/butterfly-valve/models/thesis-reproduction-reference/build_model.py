"""Assumption-explicit cone/disc/seat reference; not exact thesis CAD or CFD-ready geometry."""
from pathlib import Path
import math,json,csv
import cadquery as cq
import numpy as np
OUT=Path(__file__).resolve().parent
th=math.radians(10);al=math.radians(-5);B=113.05;b=15;c=30;e=15
axis=np.array([math.cos(al),math.sin(al),0.]);Q=np.outer(axis,axis)-math.cos(th)**2*np.eye(3)
L=B*math.cos(th)/math.sqrt(Q[0,0]-Q[0,1]**2/Q[1,1]);apex=np.array([-L,Q[0,1]*L/Q[1,1],0.]);A=B*math.cos(th)/math.sqrt(-Q[1,1])
v=lambda x:cq.Vector(*map(float,x))
cone=cq.Solid.makeCone(0,800*math.tan(th),800,v(apex),v(axis))
disc=cone.intersect(cq.Workplane('XY').box(b,320,320).val()).clean()
seat=cq.Solid.makeCylinder(140,2,cq.Vector(5.5,0,0),cq.Vector(1,0,0)).cut(cone).clean()
shaft=cq.Solid.makeCylinder(5,300,cq.Vector(c,e,-150),cq.Vector(0,0,1))
H1=(c,e,-150);H2=(c,e,150)
for name,obj in [('01_Disc_closed.step',disc),('02_Seat_closed.step',seat),('03_H_axis_reference_NOT_structural_shaft.step',shaft)]:
 assert obj.isValid() and len(obj.Solids())==1,name
 cq.exporters.export(obj,str(OUT/name))

def assy(angle,path):
 a=cq.Assembly(name='Thesis_reference_assumptions');a.add(seat,name='Seat',color=cq.Color(.25,.5,.8));a.add(disc.rotate(H1,H2,angle),name='Disc',color=cq.Color(.95,.65,.2));a.add(shaft,name='H_axis_reference',color=cq.Color(.25,.25,.28));a.export(str(path))
assy(0,OUT/'04_Assembly_closed.step')
poses=OUT/'开度位置';poses.mkdir(exist_ok=True)
for a in [15,30,45,60,75,90]:assy(a,poses/f'Assembly_{a:02d}deg.step')
rows=[]
for a in [0,.1,.25,.5,1,2,3,4,5,7.5,10,12.5,15,20,30,45,60,75,90]:
 moved=disc.rotate(H1,H2,a);z=seat.intersect(moved);vol=max(0,z.Volume()) if z.Solids() else 0.
 rows.append(dict(angle_deg=a,disc_seat_overlap_mm3=vol,overlap_above_0_001_mm3=vol>.001));print(a,vol,flush=True)
with (OUT/'刚体交叠检查.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
params={'status':'assumption-based reference, not exact thesis reconstruction; not CFD-ready','units':'mm, deg','paper_table':{'R0':117.5,'theta_half_angle':10,'c':30,'e':15,'alpha_magnitude':5,'B0':113.05,'A0':113.49,'b':15},'coordinates':{'x':'axial; disc midplane x=0','y':'radial offset direction','z':'shaft direction'},'interpretations':{'midplane_ellipse_semiaxis_z':B,'computed_midplane_ellipse_semiaxis_y':A,'signed_cone_axis_tilt_deg':-5,'c_reference':'disc midplane','e_reference':'midplane ellipse centre'},'assumed_dimensions':{'seat_outer_radius':140,'seat_axial_width':2,'seat_axial_range':[5.5,7.5],'reference_axis_cylinder_radius':5,'reference_axis_cylinder_length':300},'cone':{'apex':apex.tolist(),'axis':axis.tolist(),'R0_reference_section_center':(apex+axis*(117.5/math.tan(th))).tolist()},'shaft_reference':{'point':[c,e,0],'direction':[0,0,1]},'solid_validity':{'disc':disc.isValid(),'seat':seat.isValid(),'axis_reference':shaft.isValid()},'closed_overlap_mm3':rows[0]['disc_seat_overlap_mm3'],'step_note':'STEP assemblies carry placement, no SolidWorks mates/motors/features'}
(OUT/'参数与坐标.json').write_text(json.dumps(params,ensure_ascii=False,indent=2))
# Round-trip independent exported STEP solid check.
checks={}
for f in OUT.glob('*.step'):
 s=cq.importers.importStep(str(f)).val();checks[f.name]={'valid':s.isValid(),'solid_count':len(s.Solids()),'volume_mm3':s.Volume()};assert s.isValid()
(OUT/'STEP回读检查.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
# 3D overview: show 45-degree disc and a cutaway seat for visibility only.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig=plt.figure(figsize=(11,8));ax=fig.add_subplot(111,projection='3d')
cut=cq.Workplane('XY').box(500,500,250).translate((0,0,-125)).val()
for obj,color,opacity in [(seat.cut(cut),'#4d8dcc',.6),(disc.rotate(H1,H2,45),'#e8a13c',1),(shaft,'#555566',1)]:
 verts,tris=obj.tessellate(.7);polys=[[(verts[i].x,verts[i].y,verts[i].z) for i in t] for t in tris];ax.add_collection3d(Poly3DCollection(polys,facecolor=color,edgecolor='none',alpha=opacity))
ax.set(xlim=(-150,150),ylim=(-150,150),zlim=(-150,150),xlabel='X axial / mm',ylabel='Y radial / mm',zlabel='Z shaft / mm');ax.set_box_aspect((1,1,1));ax.view_init(22,-48);ax.set_title('45 deg reference pose; seat cut away for display\nGrey cylinder marks H axis only; not a connected structural shaft')
fig.savefig(OUT/'模型预览_45度_阀座剖切.png',dpi=150,bbox_inches='tight')
print('DONE',A,flush=True)
