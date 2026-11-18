"""第二章：SI制四状态阀控非对称缸，独立统一课程实验对象。"""
from dataclasses import dataclass,asdict,replace
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    def njit(*args,**kwargs):
        return lambda f:f


@dataclass(frozen=True)
class Parameters:
    A1:float=.002
    A2:float=.001
    L:float=.30
    V01:float=.00015
    V02:float=.00015
    mass:float=500.
    Ps:float=21e6
    Pr:float=1e5
    beta:float=1.10e9
    Ct:float=2.4e-12
    Cd:float=.62
    width:float=.02
    xvmax:float=.001
    rho:float=870.
    deadzone:float=.5
    Fc:float=500.
    Fs:float=800.
    vs:float=.05
    viscous:float=1000.
    gamma:float=300.
    makeup_area:float=.0001
    fault_beta_ratio:float=1.
    def vector(self):return np.array(list(asdict(self).values()),dtype=np.float64)


@njit(cache=True)
def friction(v,p):
    return (p[15]+(p[16]-p[15])*math.exp(-(v/p[17])**2))*math.tanh(p[19]*v)+p[18]*v


@njit(cache=True)
def friction_slope(v,p):
    ex=math.exp(-(v/p[17])**2);th=math.tanh(p[19]*v)
    return -2*v/p[17]**2*(p[16]-p[15])*ex*th+(p[15]+(p[16]-p[15])*ex)*p[19]*(1-th*th)+p[18]


@njit(cache=True)
def effective_voltage(u,delta=.5):return math.copysign(max(abs(u)-delta,0.),u)


def inverse_deadzone(w,delta=.5):
    return 0. if abs(w)<1e-7 else float(w+math.copysign(delta,w))


@njit(cache=True)
def root(dp):return dp/(dp*dp+1000.**2)**.25


@njit(cache=True)
def rhs_effective(y,w,load,p):
    x,v,p1,p2=y
    gain=p[10]*p[11]*p[12]/10*math.sqrt(2/p[13])
    if w>=0:q1=gain*w*root(p[6]-p1);q2=-gain*w*root(p2-p[7])
    else:q1=gain*w*root(p1-p[7]);q2=-gain*w*root(p[6]-p2)
    # 补油单向阀是连续方程中的流量，不裁剪积分后的压力。
    q1+=p[10]*p[20]*math.sqrt(2/p[13]*max(p[7]-p1,0.))
    q2+=p[10]*p[20]*math.sqrt(2/p[13]*max(p[7]-p2,0.))
    # 固定0.1%含气量的局部Wood柔度：低压时beta趋向P/alpha，避免非物理穿过真空。
    beta1=p[21]/(.999/p[8]+.001/max(p1,1.))
    beta2=p[21]/(.999/p[8]+.001/max(p2,1.))
    leak=p[9]*(p1-p2)
    return np.array([v,(p[0]*(p1-p[7])-p[1]*(p2-p[7])-friction(v,p)-load)/p[5],
                     beta1/(p[3]+p[0]*x)*(q1-p[0]*v-leak),
                     beta2/(p[4]+p[1]*(p[2]-x))*(q2+p[1]*v+leak)])


@njit(cache=True)
def advance_effective(y,w,load,p,dt=.001,substeps=64):
    z=y.copy();h=dt/substeps
    extremes=np.array([z[0],z[0],z[2],z[2],z[3],z[3]])
    for _ in range(substeps):
        a=rhs_effective(z,w,load,p);b=rhs_effective(z+h*a/2,w,load,p)
        c=rhs_effective(z+h*b/2,w,load,p);d=rhs_effective(z+h*c,w,load,p)
        z=z+h*(a+2*b+2*c+d)/6
        for j,index in enumerate([0,2,3]):
            extremes[2*j]=min(extremes[2*j],z[index]);extremes[2*j+1]=max(extremes[2*j+1],z[index])
    return z,extremes


@njit(cache=True)
def simulate_inputs(u,loads,p,initial,dt=.001,substeps=64):
    states=np.empty((len(u)+1,4));states[0]=initial
    for k in range(len(u)):
        states[k+1],_=advance_effective(states[k],effective_voltage(u[k],p[14]),loads[k],p,dt,substeps)
        if not np.isfinite(states[k+1]).all():
            states[k+1:]=np.nan
            return states
    return states


class HydraulicTwin:
    def __init__(self,parameters=None,substeps=64):
        self.parameters=parameters or Parameters();self.p=self.parameters.vector();self.substeps=int(substeps)
        self.state=np.array([.04,0.,1e5,1e5]);self.t=0.
    def equilibrium(self,velocity,position=.08):
        from scipy.optimize import brentq
        p=self.parameters;fric=friction(velocity,self.p)
        gain=p.Cd*p.width*p.xvmax/10*math.sqrt(2/p.rho)
        def ratio(p2):
            p1=p.Pr+(fric+p.A2*(p2-p.Pr))/p.A1
            leak=p.Ct*(p1-p2)
            return (p.A1*velocity+leak)/(gain*root(p.Ps-p1))-(p.A2*velocity+leak)/(gain*root(p2-p.Pr))
        p2=brentq(ratio,p.Pr+1.,p.Ps*.999)
        p1=p.Pr+(fric+p.A2*(p2-p.Pr))/p.A1
        w=(p.A1*velocity+p.Ct*(p1-p2))/(gain*root(p.Ps-p1))
        return np.array([position,velocity,p1,p2]),float(w+math.copysign(p.deadzone,w))
    def reset(self,state):self.state=np.array(state,dtype=float);self.t=0.
    def set_beta(self,beta):self.parameters=replace(self.parameters,beta=float(beta));self.p=self.parameters.vector()
    def set_beta_ratio(self,ratio):self.parameters=replace(self.parameters,fault_beta_ratio=float(ratio));self.p=self.parameters.vector()
    def step(self,u,load=0.,dt=.001):
        if not np.isfinite(u) or abs(u)>10+1e-8:raise ValueError('输入必须是±10 V内的有限值')
        y,ext=advance_effective(self.state,effective_voltage(u,self.p[14]),load,self.p,dt,self.substeps)
        if not np.isfinite(y).all() or min(ext[2],ext[4])<=0 or ext[0]<0 or ext[1]>self.p[2]:
            raise RuntimeError('液压积分失败或机械/绝对压力越界，状态未裁剪')
        self.state=y;self.t+=dt;return y.copy(),ext


def trajectory(t,distance=.08,duration=.8,origin=.04):
    """五次多项式S轨迹，位置/速度/加速度连续，提供解析跃度。"""
    scalar=np.ndim(t)==0;t=np.atleast_1d(t).astype(float);s=np.clip(t/duration,0.,1.)
    x=origin+distance*(10*s**3-15*s**4+6*s**5)
    v=distance/duration*(30*s**2-60*s**3+30*s**4)
    a=distance/duration**2*(60*s-180*s**2+120*s**3)
    j=distance/duration**3*(60-360*s+360*s**2)
    outside=(t>=duration)|(t<0);v[outside]=0;a[outside]=0;j[outside]=0
    result=np.stack([x,v,a,j],axis=-1)
    return result[0] if scalar else result
