"""Independent numerical checks of MATLAB equations using SciPy quad and split trapezoids.
Not a MATLAB execution test. Demo mu is an assumption, not thesis calibration.
"""
from pathlib import Path
import json,csv
import numpy as np
from scipy.integrate import quad,trapezoid
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
base=dict(A_mm=113.49,B_mm=113.05,R0_mm=117.5,b_mm=15,c_mm=30,e_mm=15,theta_half_deg=10,tilt_deg=5,mu=.05,normal_projection='local')
def calculate(p,trap=False):
 A,B,R,b,c,e=[p[k]/1000 for k in ['A_mm','B_mm','R0_mm','b_mm','c_mm','e_mm']];t,g=np.deg2rad([p['theta_half_deg'],p['tilt_deg']]);mu=p['mu'];C=np.sqrt(A*A-B*B);h=R*np.cos(t+g)/np.sin(t)-b/2
 rho=lambda z:B*B/(A-C*np.cos(z));t1=np.arccos((C-e)*A/(B*B+C*(C-e)))
 K=lambda z:rho(z)*np.cos(z)-A-C+h*np.tan(t+g)
 L=lambda z:np.hypot(K(z),R*np.sin(z))
 ca=lambda z:h/np.hypot(h,L(z));sa=lambda z:L(z)/np.hypot(h,L(z))
 if trap:
  integ=lambda f,a,b:trapezoid(f(np.linspace(a,b,20001)),np.linspace(a,b,20001))
 else:integ=lambda f,a,b:quad(f,a,b,epsabs=1e-13,epsrel=1e-9)[0]
 k1=integ(rho,0,np.pi);k2=integ(rho,0,t1);D=2*((np.sin(t)-mu*np.cos(t))*k1+2*mu*np.cos(t)*k2);assert D>0
 projection=ca if p['normal_projection']=='local' else lambda z:np.cos(g)+np.zeros_like(z)
 q=lambda z:np.pi*A*B*projection(z)/D
 fs=[lambda z:2*mu*q(z)*ca(0)*(rho(z)*np.cos(z)-(C-e))*rho(z),lambda z:2*mu*q(z)*ca(0)*((C-e)-rho(z)*np.cos(z))*rho(z),lambda z:2*mu*q(z)*sa(0)*ca(z)*c*rho(z)]
 M=np.array([integ(fs[0],0,t1),integ(fs[1],t1,np.pi),integ(fs[2],0,t1),integ(fs[2],t1,np.pi)])*1e6
 return M, M@np.array([[1,1],[1,-1],[1,-1],[-1,1]]),dict(C_mm=C*1000,h_mm=h*1000,theta1_deg=np.rad2deg(t1),alpha1_deg=np.rad2deg(np.arctan2(sa(0),ca(0))))
M,slopes,geo=calculate(base);Mt,st,_=calculate(base,True)
assert np.max(abs(M-Mt))<1e-5
p=base.copy();p['mu']=0;assert np.allclose(calculate(p)[0],0)
p=base.copy()
for k in ['A_mm','B_mm','R0_mm','b_mm','c_mm','e_mm']:p[k]*=2
assert np.allclose(calculate(p)[1],8*slopes,rtol=1e-9)
p=base.copy();p['c_mm']=0;assert np.allclose(calculate(p)[0][2:],0)
assert np.allclose(slopes*0,0)
checks={'status':'SciPy numerical verification only; MATLAB unavailable','demo_parameters':base,'geometry':geo,'components_at_1_MPa_Nm':M.tolist(),'slopes_Nm_per_MPa':slopes.tolist(),'quad_trapezoid_max_component_difference_Nm':float(max(abs(M-Mt))),'checks':{'zero_mu_zero_torque':True,'all_lengths_doubled_torque_times_8':True,'zero_c_zero_M3_M4':True,'zero_pressure_zero_torque':True}}
(O/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2))
press=np.linspace(0,1,101)
with (O/'示例结果_假设mu0_05.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['p_MPa','M1_Nm','M2_Nm','M3_Nm','M4_Nm','opening_Nm','closing_Nm'])
 for z in press:w.writerow([z,*(M*z),*(slopes*z)])
fig,ax=plt.subplots(figsize=(8,5));ax.plot(press,press*slopes[0],'r-',label='Opening (computed integral)');ax.plot(press,press*slopes[1],'g-.',label='Closing (computed integral)');ax.set(xlabel='Medium pressure p / MPa',ylabel='Seal friction torque / N m',title='Parameterized model example: mu=0.05 (assumed), local projection');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(O/'model_example.png',dpi=180)
print(json.dumps(checks,ensure_ascii=False,indent=2))
