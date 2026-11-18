"""第三章：SI增量PID、G/H逆模型二自由度控制与双腔限压LTV-MPC。"""
import time,math,contextlib,io
import numpy as np
from scipy.linalg import expm
from scipy import sparse
import osqp
from physics import rhs_effective,advance_effective,friction,friction_slope,effective_voltage,inverse_deadzone,trajectory


class IncrementalPID:
    """微分先滤波后作增量，饱和后输出作为下次记忆；误差单位m。"""
    def __init__(self,kp=250.,ki=150.,kd=1.,dt=.001,alpha=.85):
        self.kp=kp;self.ki=ki;self.kd=kd;self.dt=dt;self.alpha=alpha
        self.e0=0.;self.d0=0.;self.u0=0.;self.last={}
    def update(self,reference,state,t=0.):
        e=float(reference[0]-state[0]);d=self.alpha*self.d0+(1-self.alpha)*self.kd*(e-self.e0)/self.dt
        du=self.kp*(e-self.e0)+self.ki*e*self.dt+d-self.d0
        requested=self.u0+du;u=float(np.clip(requested,-10.,10.))
        self.last=dict(requested_v=requested,saturation=abs(requested)>10,estimated_load_n=0.)
        self.e0=e;self.d0=d;self.u0=u;return u


class LoadObserver:
    """仅由本控制器当前压力和历史速度估计未知外载；不读取仿真负载标签。"""
    def __init__(self,p,dt=.001,tau=.008):self.p=p;self.dt=dt;self.a=math.exp(-dt/tau);self.previous_v=0.;self.estimate=0.;self.initialized=False
    def update(self,y):
        p=self.p
        acceleration=(y[1]-self.previous_v)/self.dt if self.initialized else 0.
        self.previous_v=y[1];self.initialized=True
        force=p[0]*(y[2]-p[7])-p[1]*(y[3]-p[7])-friction(y[1],p)-p[5]*acceleration
        self.estimate=float(np.clip(self.a*self.estimate+(1-self.a)*force,-15000,15000))
        return self.estimate


class ModelAssistedController:
    """位置-速度外环、逆动力学+液压力G/H反演、实时压差增益与死区逆。

    相比仓库第六章，统一为四状态即时阀；保留已知物理因果链，
    外载由独立因果扰动估计得到，不把真实5000N事件作为前馈真值。
    """
    def __init__(self,parameters,dt=.001):
        self.parameters=parameters;self.p=parameters.vector();self.observer=LoadObserver(self.p,dt)
        self.integral=0.;self.dt=dt;self.last={}
    def update(self,reference,state,t=0.):
        p=self.p;x,v,p1,p2=state;xd,vd,ad,jd=reference
        load=self.observer.update(state);e=xd-x;self.integral=float(np.clip(self.integral+e*self.dt,-.01,.01))
        desired_acc=ad+70*(vd-v)+4000*e+2000*self.integral
        desired_force=p[5]*desired_acc+friction(vd,p)+load
        actual_force=p[0]*(p1-p[7])-p[1]*(p2-p[7])
        target_rate=p[5]*jd+friction_slope(vd,p)*ad+80*(desired_force-actual_force)
        f0=rhs_effective(state,0.,load,p)
        h=p[0]*f0[2]-p[1]*f0[3]
        direction=1. if target_rate-h>=0 else -1.
        f1=rhs_effective(state,direction,load,p)
        gain=direction*(p[0]*(f1[2]-f0[2])-p[1]*(f1[3]-f0[3]))
        if gain<=0:raise RuntimeError('逆模型局部流量增益非正')
        w=float(np.clip((target_rate-h)/gain,-9.5,9.5));u=inverse_deadzone(w,p[14])
        self.last=dict(estimated_load_n=load,inverse_gain_n_s_v=gain,desired_force_n=desired_force,
                       requested_v=u,saturation=abs(w)>=9.5-1e-9,feedforward_rate_n_s=target_rate)
        return u


def scaled_rhs(z,w,load,p):
    units=np.array([1000.,1.,1e-6,1e-6])
    return rhs_effective(np.asarray(z)/units,float(w),float(load),p)*units


def linearize(y,w,load,p,dt=.001,direction=1.):
    """x+=Ad*x+Bd*w+c；精确ZOH离散化当前局部仿射模型，保留c项。"""
    units=np.array([1000.,1.,1e-6,1e-6]);z=np.asarray(y)*units
    f=scaled_rhs(z,w,load,p);a=np.zeros((4,4))
    increments=[1e-3,1e-5,1e-3,1e-3]
    for j,eps in enumerate(increments):
        d=np.zeros(4);d[j]=eps
        a[:,j]=(scaled_rhs(z+d,w,load,p)-scaled_rhs(z-d,w,load,p))/(2*eps)
    sign=np.sign(w) if abs(w)>1e-6 else direction
    zero=scaled_rhs(z,0.,load,p)
    b=sign*(scaled_rhs(z,sign,load,p)-zero)
    c=f-a@z-b*w
    matrix=np.zeros((6,6));matrix[:4,:4]=a;matrix[:4,4]=b;matrix[:4,5]=c
    disc=expm(matrix*dt)
    return disc[:4,:4],disc[:4,4],disc[:4,5]


def prediction_matrices(a,b,c,horizon):
    n=len(c);f=np.empty((n*horizon,n));g=np.zeros((n*horizon,horizon));offset=np.empty(n*horizon)
    fk=np.eye(n);gk=np.zeros((n,horizon));ck=np.zeros(n)
    for i in range(horizon):
        fk=a@fk;gk=a@gk;gk[:,i]+=b;ck=a@ck+c
        f[i*n:(i+1)*n]=fk;g[i*n:(i+1)*n]=gk;offset[i*n:(i+1)*n]=ck
    return f,g,offset


def sequence_matrices(initial,nominal,load,p,dt):
    n=4;horizon=len(nominal);f=[];g=[];offset=[]
    fk=np.eye(n);gk=np.zeros((n,horizon));ck=np.zeros(n);state=initial.copy()
    models=[]
    for i,w in enumerate(nominal):
        aa,bb,cc=linearize(state,w,load,p,dt,direction=1.)
        fk=aa@fk;gk=aa@gk;gk[:,i]+=bb;ck=aa@ck+cc
        f.append(fk.copy());g.append(gk.copy());offset.append(ck.copy());models.append((aa,bb,cc))
        state,_=advance_effective(state,w,load,p,dt,32)
        if not np.isfinite(state).all():raise RuntimeError('MPC标称轨迹积分非有限')
    return np.vstack(f),np.vstack(g),np.concatenate(offset),models


class LTVMPC:
    def __init__(self,parameters,dt=.001,horizon=30,pressure=True,move_duration=.8,distance=.08,origin=.04):
        self.parameters=parameters;self.p=parameters.vector();self.dt=dt;self.N=int(horizon)
        self.pressure=pressure;self.duration=move_duration;self.distance=distance;self.origin=origin
        self.previous_w=0.;self.observer=LoadObserver(self.p,dt);self.last={};self.last_matrices=None
        self.q=np.tile([100.,100.,0.,0.],self.N);self.move_weight=.5
        self.S=np.tril(np.ones((self.N,self.N)));self.nominal=None
    def update(self,reference,state,t=0.):
        start=time.perf_counter();load=self.observer.update(state);n=self.N
        nominal=np.full(n,self.previous_w) if self.nominal is None else np.r_[self.nominal[1:],self.nominal[-1]]
        refs=trajectory(t+self.dt*np.arange(1,n+1),self.distance,self.duration,self.origin)
        target=np.zeros((n,4));target[:,0]=refs[:,0]*1000;target[:,1]=refs[:,1];target=target.ravel()
        z=state*np.array([1000.,1.,1e-6,1e-6]);pressure_indices=np.array([4*i+j for i in range(n) for j in [2,3]])
        total_qp=0.;warm=None
        for iteration in range(3):
            f,g,offset,models=sequence_matrices(state,nominal,load,self.p,self.dt);phi=g@self.S
            free=f@z+g@np.full(n,self.previous_w)+offset
            h=2*(phi.T@(phi*self.q[:,None])+self.move_weight*np.eye(n)+.001*self.S.T@self.S)/100.
            q=2*(phi.T@(self.q*(free-target))+.001*self.S.T@np.full(n,self.previous_w))/100.
            blocks=[np.eye(n),self.S,self.S];low=[np.full(n,-1.),np.full(n,-9.5-self.previous_w),nominal-2.-self.previous_w]
            high=[np.full(n,1.),np.full(n,9.5-self.previous_w),nominal+2.-self.previous_w]
            if self.pressure:
                blocks.append(phi[pressure_indices]);low.append(np.full(2*n,-np.inf));high.append(19.5-free[pressure_indices])
            amat=np.vstack(blocks);l=np.concatenate(low);upper=np.concatenate(high)
            solver=osqp.OSQP();solver.setup(P=sparse.csc_matrix(np.triu(h)),q=q,A=sparse.csc_matrix(amat),
                                           l=l,u=upper,verbose=False,eps_abs=1e-5,eps_rel=1e-5,
                                           max_iter=30000,polishing=True,warm_starting=True)
            if warm is not None:solver.warm_start(x=warm)
            with contextlib.redirect_stdout(io.StringIO()):result=solver.solve(raise_error=False)
            total_qp+=float(result.info.solve_time)
            if result.x is None or result.info.status_val not in [1,2,7]:raise RuntimeError('MPC QP无可行解: '+result.info.status)
            if result.info.status_val==7:
                residual=max(np.max(l-amat@result.x),np.max(amat@result.x-upper),0.)
                if residual>2e-4:raise RuntimeError(f'MPC迭代上限且约束残差{residual:.3g}')
            violation=float(max(np.max(l-amat@result.x),np.max(amat@result.x-upper),0.))
            if violation>2e-3:raise RuntimeError(f'MPC QP约束残差过大: {violation:g}')
            nominal=np.clip(self.previous_w+self.S@result.x,-9.5,9.5);warm=result.x.copy()
        self.nominal=nominal;w=float(nominal[0]);u=inverse_deadzone(w,self.p[14])
        prediction=(free+phi@result.x).reshape(n,4)
        self.last=dict(estimated_load_n=load,qp_status=result.info.status,qp_iterations=int(result.info.iter),
                       qp_residual=violation,qp_dual_residual=float(result.info.dual_res),qp_seconds=total_qp,control_seconds=time.perf_counter()-start,
                       predicted_min_pressure_mpa=float(prediction[:,2:].min()),predicted_max_pressure_mpa=float(prediction[:,2:].max()),
                       requested_v=u,saturation=abs(w)>=9.5-1e-8,sequential_qps=3)
        self.last_matrices=dict(Ad=models[0][0],Bd=models[0][1],affine_c=models[0][2],F=f,G=g,Phi=phi,H=h,q=q,A=amat,l=l,upper=upper,
                                initial_scaled=z,free=free,solution=result.x,predicted=prediction)
        self.previous_w=w;return u
