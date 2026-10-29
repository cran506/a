#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""《智能流体传动与控制》第八章课堂编程实践——独立重新设计版。

8.1 四状态液压机理 → 五维 Gym 观测、Z-score、复合奖励。
8.2 SB3 SAC：双 Q、tanh 高斯策略、自动温度；真实训练日志。
8.3 相对阶3的行程 HOCBF + 相对阶1的两腔压力屏障；死区分支 QP。
8.4 每回合质量±20%、模量600~1400 MPa、库伦摩擦±30%。
8.5 实际 ODE 闭环对比，0.50 s 同时施加5000 N和模量减半。
8.6 70%隐藏通道结构剪枝、校准、INT8仿射权重/激活、INT32累加。

python drl_hydraulic_lab.py --mode all --steps 1000000
python drl_hydraulic_lab.py --mode all --steps 5000 --run-dir quick_run
python drl_hydraulic_lab.py --mode safety
说明：短训练是连通性验证，不代表收敛；不内置教材响应曲线或旧模型。
"""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass, asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from collections import deque
import numpy as np
from scipy.optimize import brentq

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hydraulic_fresh_mpl')
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def decorate(function): return function
        return decorate
try:
    import gymnasium as gym
except ImportError:
    gym = None


@dataclass(frozen=True)
class Physics:
    """SI单位，保留上传数字孪生文件的几何、摩擦和油源标称参数。

    电压±10 V映射上传文件的归一化阀指令±1，死区0.05→0.5 V。
    x_limit=.1 是试验软件安全行程，原液压缸物理总行程仍为1m。
    """
    A1: float = .01
    A2: float = .005
    L: float = 1.
    V01: float = .001
    V02: float = .001
    mass: float = 500.
    Ps: float = 21e6
    Pr: float = 1e5
    beta: float = 1e9
    rho: float = 870.
    Cd: float = .62
    width: float = .02
    xv_max: float = .005
    deadzone: float = .5
    leak: float = 1e-12
    Fc: float = 500.
    Fs: float = 800.
    vs: float = .05
    viscous: float = 1000.
    gamma: float = 500.

    def vector(self):
        return np.array([getattr(self, k) for k in self.__dataclass_fields__], dtype=np.float64)


@njit(cache=True)
def friction(v, p):
    base = p[15] + (p[16]-p[15])*math.exp(-(v/p[17])**2)
    return base*math.tanh(p[19]*v)+p[18]*v


@njit(cache=True)
def friction_gradient(v, p):
    ex=math.exp(-(v/p[17])**2);th=math.tanh(p[19]*v)
    base=p[15]+(p[16]-p[15])*ex
    return -2*v/(p[17]**2)*(p[16]-p[15])*ex*th+base*p[19]*(1-th*th)+p[18]


@njit(cache=True)
def signed_root(dp):
    # 有向流量；1kPa层流正则化使零压差处可微，避免sqrt导数奇异。
    return dp / ((dp*dp+1000.**2)**.25)


@njit(cache=True)
def hydraulic_rhs(y, u, load, beta_ratio, p):
    x,v,p1,p2=y
    w=math.copysign(max(abs(u)-p[13],0.),u)
    c=p[10]*p[11]*p[12]/10.*math.sqrt(2./p[9])
    if w>=0:
        q1=c*w*signed_root(p[6]-p1)
        q2=-c*w*signed_root(p2-p[7])
    else:
        q1=c*w*signed_root(p1-p[7])
        q2=-c*w*signed_root(p[6]-p2)
    b1=p[8]*beta_ratio*(.001+.999*.5*(math.tanh((p1-1e5)/5e4)+1))
    b2=p[8]*beta_ratio*(.001+.999*.5*(math.tanh((p2-1e5)/5e4)+1))
    leakage=p[14]*(p1-p2)
    a=(p[0]*p1-p[1]*p2-friction(v,p)-load)/p[5]
    return np.array([v,a,b1/(p[3]+p[0]*x)*(q1-p[0]*v-leakage),
                     b2/(p[4]+p[1]*(p[2]-x))*(q2+p[1]*v+leakage)])


@njit(cache=True)
def integrate_rk4(y,u,load,beta_ratio,p,dt,substeps):
    """固定子步RK4；返回子步节点极值，不裁剪位置、压力或速度。"""
    z=y.copy();h=dt/substeps
    extrema=np.array([z[0],z[0],min(z[2],z[3]),max(z[2],z[3])])
    for _ in range(substeps):
        k1=hydraulic_rhs(z,u,load,beta_ratio,p)
        k2=hydraulic_rhs(z+.5*h*k1,u,load,beta_ratio,p)
        k3=hydraulic_rhs(z+.5*h*k2,u,load,beta_ratio,p)
        k4=hydraulic_rhs(z+h*k3,u,load,beta_ratio,p)
        z=z+h/6.*(k1+2*k2+2*k3+k4)
        extrema[0]=min(extrema[0],z[0]);extrema[1]=max(extrema[1],z[0])
        extrema[2]=min(extrema[2],z[2],z[3]);extrema[3]=max(extrema[3],z[2],z[3])
    return z,extrema


class HydraulicTwin:
    def __init__(self,physics=None,substeps=8,integrator="rk4"):
        self.physics=physics or Physics();self.p=self.physics.vector()
        self.substeps=int(substeps);self.integrator=integrator
        if self.substeps<1: raise ValueError('substeps必须为正整数')
        self.state=np.array([.02,0.,1e5,2e5]);self.t=0.

    def rhs(self,y,u,load=0.,beta_ratio=1.):
        return hydraulic_rhs(np.asarray(y,dtype=float),float(u),float(load),float(beta_ratio),self.p)

    def equilibrium(self,velocity=.05,load=0.,position=.02):
        """恒速时受力+两腔连续性联立求根；不借用教材工作点电压。"""
        p=self.physics;force=friction(velocity,self.p)+load
        if velocity<=0: raise ValueError('本课堂平衡初始化用于正向恒速参考')
        gain=p.Cd*p.width*p.xv_max/10*math.sqrt(2/p.rho)
        def residual(p2):
            p1=(p.A2*p2+force)/p.A1
            leak=p.leak*(p1-p2)
            # 正向阀口两路需要相同有效电压。
            u1=(p.A1*velocity+leak)/(gain*signed_root(max(p.Ps-p1,1.)))
            u2=(p.A2*velocity+leak)/(gain*signed_root(max(p2-p.Pr,1.)))
            return u1-u2
        upper=min(p.Ps*.999,(p.A1*p.Ps-force)/p.A2*.999)
        p2=brentq(residual,p.Pr+1.,upper,xtol=1e-5)
        p1=(p.A2*p2+force)/p.A1
        w=(p.A1*velocity+p.leak*(p1-p2))/(gain*signed_root(p.Ps-p1))
        return np.array([position,velocity,p1,p2]),float(w+p.deadzone)

    def reset(self,state):
        self.state=np.asarray(state,dtype=float).copy();self.t=0.

    def step(self,u,dt=.001,load=0.,beta_ratio=1.):
        if not np.isfinite(u) or abs(u)>10+1e-9: raise ValueError('电压须在±10V内')
        if self.integrator=='radau':
            from scipy.integrate import solve_ivp
            sol=solve_ivp(lambda t,y:self.rhs(y,u,load,beta_ratio),(0.,dt),self.state,
                          method='Radau',rtol=1e-8,atol=np.array([1e-10,1e-9,.01,.01]),max_step=dt/8)
            if not sol.success:raise FloatingPointError(sol.message)
            z=sol.y[:,-1];ext=np.array([sol.y[0].min(),sol.y[0].max(),sol.y[2:].min(),sol.y[2:].max()])
        else:
            z,ext=integrate_rk4(self.state,float(u),float(load),float(beta_ratio),self.p,float(dt),self.substeps)
        if not np.isfinite(z).all() or ext[2]<=0 or ext[0]<0 or ext[1]>self.physics.L:
            raise FloatingPointError('物理积分越界或出现非正绝对压力，未裁剪状态')
        self.state=z;self.t+=dt
        return z.copy(),ext


class InfeasibleSafety(RuntimeError): pass


class HOCBFFilter:
    """同一四阶液压对象：行程相对阶3，压力相对阶1。

    h+=x_max-margin-x；h-=x-x_min-margin；k1=k2=k3=k。
    h''' + 3k h'' + 3k² h' + k³ h >=0。
    每个阀死区分支上，约束对物理电压u是仿射的。分别求区间QP后
    选全局最小|u-u_rl|。分支合并整体并非单个凸QP。
    此约束在采样时刻求解；子步监测不等于连续时间/失配鲁棒证明。
    """
    def __init__(self,k=25.,margin=.0012,x_min=0.,x_max=.1,p_min=5e4,p_max=25e6,sample_reserve=0.,sample_refinement=True):
        self.k=float(k);self.margin=float(margin);self.x_min=x_min;self.x_max=x_max
        self.p_min=p_min;self.p_max=p_max;self.sample_reserve=float(sample_reserve);self.sample_refinement=sample_refinement;self.dt=.001
        self.last={}

    def diagnostics(self,twin,load=0.):
        x,v,p1,p2=twin.state;k=self.k
        a=twin.rhs(twin.state,0.,load)[1]
        hs=np.array([x-self.x_min-self.margin,self.x_max-self.margin-x])
        hd=np.array([v,-v]);hdd=np.array([a,-a])
        return hs,hd+k*hs,hdd+2*k*hd+k*k*hs

    def project(self,twin,u_rl,load=0.,beta_ratio=1.):
        if not np.isfinite(u_rl): raise ValueError('非有限动作')
        x,v,p1,p2=twin.state;p=twin.physics;k=self.k
        f0=twin.rhs(twin.state,0.,load,beta_ratio);acc=f0[1]
        j0=(p.A1*f0[2]-p.A2*f0[3]-friction_gradient(v,twin.p)*acc)/p.mass
        hu=self.x_max-self.margin-x;hl=x-self.x_min-self.margin
        if min(hu,hl)<-1e-8: raise InfeasibleSafety('当前位置已在安全内边界以外')
        candidates=[]
        # branch: voltage interval, input offset, d(rhs)/d effective voltage
        for left,right,offset,probe in [(-10.,-p.deadzone,p.deadzone,-p.deadzone-1.),
                                       (-p.deadzone,p.deadzone,0.,0.),
                                       (p.deadzone,10.,-p.deadzone,p.deadzone+1.)]:
            if probe==0.: g=np.zeros(4)
            elif probe>0.: g=twin.rhs(twin.state,probe,load,beta_ratio)-f0
            else: g=f0-twin.rhs(twin.state,probe,load,beta_ratio)
            jg=(p.A1*g[2]-p.A2*g[3])/p.mass
            # (a,b) denotes a*u+b >= 0, with w=u+offset.
            constraints=[(-jg,-j0-jg*offset-3*k*acc-3*k*k*v+k**3*hu-self.sample_reserve),
                         (jg,j0+jg*offset+3*k*acc+3*k*k*v+k**3*hl-self.sample_reserve)]
            for pi,di,gi in [(p1,f0[2],g[2]),(p2,f0[3],g[3])]:
                constraints.extend([(gi/1e6,(di+gi*offset+40*(pi-self.p_min))/1e6),
                                    (-gi/1e6,(-di-gi*offset+40*(self.p_max-pi))/1e6)])
            lo,hi=left,right
            for aa,bb in constraints:
                if abs(aa)<1e-12:
                    if bb < -1e-7: lo,hi=1.,0.;break
                elif aa>0:lo=max(lo,-bb/aa)
                else:hi=min(hi,-bb/aa)
            if lo<=hi+1e-10:
                if lo>hi:lo=hi
                if self.sample_refinement:
                    # 额外离散预测收紧：h、psi1、psi2在下一采样点均非负。
                    # 用同一非线性ODE求根，不将连续约束直接冒充采样保证。
                    # 多点划分寻找预测可行分量；有限网格不是解析全局证明。
                    cache={}
                    def future_slack(voltage):
                        key=float(voltage)
                        if key not in cache:
                            z,_=integrate_rk4(twin.state,key,float(load),float(beta_ratio),twin.p,self.dt,twin.substeps)
                            az=hydraulic_rhs(z,key,float(load),float(beta_ratio),twin.p)[1]
                            zz=np.array([az+2*k*z[1]+k*k*(z[0]-self.x_min-self.margin),
                                         -az-2*k*z[1]+k*k*(self.x_max-self.margin-z[0])])
                            hs=np.array([z[0]-self.x_min-self.margin,self.x_max-self.margin-z[0]])
                            ps1=np.array([z[1],-z[1]])+k*hs
                            # 保持递推安全集，不强加过严的psi2指数等式收缩。
                            cache[key]=np.r_[hs,ps1,zz]
                        return cache[key]
                    # Stribeck曲率可能使预测约束非单调，不能只看端点。
                    proposed=float(np.clip(u_rl,lo,hi))
                    def margin_at(voltage):return float(np.min(future_slack(voltage)))
                    if margin_at(proposed)<-1e-8:
                        nodes=np.linspace(lo,hi,9);cuts=list(nodes)
                        for lnode,rnode in zip(nodes[:-1],nodes[1:]):
                            for side in range(6):
                                fl=future_slack(lnode)[side];fr=future_slack(rnode)[side]
                                if fl*fr<0:
                                    cuts.append(brentq(lambda voltage:float(future_slack(voltage)[side]),lnode,rnode,xtol=1e-12))
                        cuts=sorted(set(cuts));valid=[]
                        for left_cut,right_cut in zip(cuts[:-1],cuts[1:]):
                            if margin_at((left_cut+right_cut)/2)>=-1e-8:
                                guess=float(np.clip(u_rl,left_cut,right_cut))
                                if margin_at(guess)>=-1e-7:valid.append(guess)
                        valid.extend(float(node) for node in cuts if margin_at(node)>=-1e-8)
                        if not valid:continue
                        # 连续QP可行域中的额外预测可行分量，选择最小改动。
                        proposed=min(valid,key=lambda voltage:abs(voltage-u_rl))
                    lo=hi=proposed
                action=float(np.clip(u_rl,lo,hi))
                candidates.append((abs(action-u_rl),action,lo,hi))
        if not candidates: raise InfeasibleSafety('行程与压力屏障在±10V内无共同可行解')
        _,action,lo,hi=min(candidates)
        h,psi1,psi2=self.diagnostics(twin,load)
        self.last=dict(proposed_v=float(u_rl),applied_v=action,lower_v=lo,upper_v=hi,
                       h_min=float(h.min()),psi1_min=float(psi1.min()),psi2_min=float(psi2.min()),
                       intervened=bool(abs(action-u_rl)>1e-6))
        return action


OBS_MEAN=np.array([0.,0.,0.,.1,21.],dtype=np.float64)
OBS_STD=np.array([5.,100.,10000.,10.,1.],dtype=np.float64)


class HydraulicGymEnv(gym.Env if gym else object):
    """课件五维观测；固定参考速度，PPT接口是部分观测近似。

    未含u_prev、绝对x/两腔压力、随机化参数，不能严格宣称五维完全MDP。
    奖励e用mm，u用V，其他内部量用SI；终止成本与式8.1分开记录。
    """
    metadata={'render_modes':[]}
    def __init__(self,domain_randomization=True,safety=True,horizon=1000):
        if gym is None: raise ImportError('请安装gymnasium和stable-baselines3')
        super().__init__();self.action_space=gym.spaces.Box(-1.,1.,(1,),np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(5,),np.float32)
        self.randomize=domain_randomization;self.safety=safety;self.horizon=int(horizon)
        self.dt=.001;self.filter=HOCBFFilter();self.base=Physics()

    def raw_obs(self):
        x,v,p1,p2=self.twin.state
        acc=self.twin.rhs(self.twin.state,self.last_u,self.load,self.beta_ratio)[1]
        return np.array([(self.x0+self.v_ref*self.n*self.dt-x)*1000,
                         (self.v_ref-v)*1000,-acc*1000,
                         (p1-self.twin.physics.A2/self.twin.physics.A1*p2)/1e6,
                         self.twin.physics.Ps/1e6])

    def observe(self):return ((self.raw_obs()-OBS_MEAN)/OBS_STD).astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);o=options or {};rng=self.np_random
        p=self.base
        if self.randomize:
            p=replace(p,mass=p.mass*rng.uniform(.8,1.2),beta=rng.uniform(.6e9,1.4e9),
                      Fc=p.Fc*rng.uniform(.7,1.3))
        self.twin=HydraulicTwin(p);self.v_ref=float(o.get('v_ref',.05))
        self.x0=float(o.get('x0',rng.uniform(.015,.035) if self.randomize else .02))
        y,u=self.twin.equilibrium(self.v_ref,position=self.x0);self.twin.reset(y)
        self.n=0;self.last_u=u;self.load=0.;self.beta_ratio=1.
        # 故障增强是对域随机化的独立补充，不把标签放入策略观测。
        self.fault_step=int(o.get('fault_step',rng.integers(100,800)))
        self.fault_load=float(o.get('fault_load',rng.uniform(0,5000) if self.randomize else 5000))
        self.fault_beta=float(o.get('fault_beta',rng.uniform(.5,1.) if self.randomize else .5))
        self.failed=False
        return self.observe(),dict(physics=asdict(p),initial_voltage_v=u)

    def step(self,action):
        value=np.asarray(action,dtype=float)
        if value.shape!=(1,) or not np.isfinite(value).all(): raise ValueError('动作必须为一个有限标量')
        proposed=float(np.clip(value[0],-1,1))*10.;failure='';old=self.last_u
        if self.n>=self.fault_step:self.load=self.fault_load;self.beta_ratio=self.fault_beta
        try:
            applied=self.filter.project(self.twin,proposed,self.load,self.beta_ratio) if self.safety else proposed
            _,ext=self.twin.step(applied,load=self.load,beta_ratio=self.beta_ratio)
            if ext[0]<0 or ext[1]>.1 or ext[2]<5e4 or ext[3]>25e6:failure='state_constraint'
        except (InfeasibleSafety,FloatingPointError) as error:
            applied=old;failure=type(error).__name__
        self.last_u=applied;self.n+=1;obs=self.observe();raw=self.raw_obs();e=raw[0]
        components=np.array([-100*e*e,-.1*applied*applied,-(applied-old)**2,2. if abs(e)<.2 else 0.])
        base=float(components.sum());penalty=-10000. if failure else 0.
        self.failed=bool(failure)
        info=dict(base_reward=base,terminal_penalty=penalty,raw_reward=base+penalty,
                  proposed_v=proposed,applied_v=applied,error_mm=float(e),
                  intervention=abs(applied-proposed)>1e-6,failure=failure,
                  reward_error=float(components[0]),reward_energy=float(components[1]),
                  reward_smooth=float(components[2]),reward_bonus=float(components[3]))
        # 只缩放优化器奖励，记录不缩放的教学奖励。
        return obs,(base+penalty)*.01,bool(failure),self.n>=self.horizon,info


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:raise ValueError('没有可保存的数据')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def train(run_dir,steps=1_000_000,seed=2026,warmup=5000):
    """真正初始化并训练SAC；不加载外部模型、教师数据或既有日志。"""
    import torch
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    torch.set_num_threads(1);torch.manual_seed(seed);np.random.seed(seed)
    root=Path(run_dir);root.mkdir(parents=True,exist_ok=True)
    if (root/'sac.zip').exists() or (root/'training.csv').exists():
        raise FileExistsError('训练目录已有结果，请使用新的--run-dir，防止混合实验')
    env=Monitor(HydraulicGymEnv())
    warmup=min(int(warmup),max(100,int(steps)//5))
    model=sb3.SAC('MlpPolicy',env,seed=seed,learning_starts=warmup,buffer_size=100000,
                  batch_size=128,learning_rate=3e-4,train_freq=4,gradient_steps=1,
                  gamma=.99,tau=.005,ent_coef='auto_0.1',target_entropy=.35-math.log(10),
                  policy_kwargs=dict(net_arch=[64,64]),device='cpu',verbose=0)
    class Log(BaseCallback):
        def __init__(self):
            super().__init__();self.rewards=deque(maxlen=1000);self.penalties=deque(maxlen=1000)
            self.interventions=deque(maxlen=1000);self.rows=[];self.started=time.perf_counter()
        def _on_step(self):
            for item in self.locals['infos']:
                self.rewards.append(item['base_reward']);self.penalties.append(item['terminal_penalty'])
                self.interventions.append(item['intervention'])
            if self.num_timesteps==1 or self.num_timesteps%max(100,min(1000,steps//20))==0:self.record()
            return True
        def record(self):
            # 独立随机数上下文保证记录频率不会改变训练采样序列。
            obs=np.asarray(self.locals['new_obs'],dtype=np.float32)
            with torch.random.fork_rng(devices=[]),torch.no_grad():
                torch.manual_seed(seed+self.num_timesteps)
                tensor=torch.as_tensor(np.repeat(obs,256,axis=0))
                _,logp=model.actor.action_log_prob(tensor)
                entropy=float(-logp.mean())+math.log(10.)
            row=dict(steps=int(self.num_timesteps),mean_base_reward=float(np.mean(self.rewards)),
                     mean_terminal_penalty=float(np.mean(self.penalties)),voltage_entropy_nat=entropy,
                     intervention_fraction=float(np.mean(self.interventions)),
                     alpha=float(model.log_ent_coef.exp().detach().cpu()) if model.log_ent_coef is not None else .1,
                     gradient_updates=int(model._n_updates),elapsed_s=time.perf_counter()-self.started)
            self.rows.append(row);write_csv(root/'training.csv',self.rows)
            if self.num_timesteps%10000==0:print(json.dumps(row),flush=True)
    callback=Log();t0=time.perf_counter()
    model.learn(total_timesteps=int(steps),callback=callback,progress_bar=False)
    if callback.rows[-1]['steps']!=model.num_timesteps:callback.record()
    model.save(root/'sac.zip')
    # 导出确定性均值，量化环节不包含探索标准差头。
    linears=[module for module in model.actor.latent_pi if isinstance(module,torch.nn.Linear)]+[model.actor.mu]
    arrays={}
    for i,layer in enumerate(linears):
        arrays[f'W{i}']=layer.weight.detach().numpy().astype(np.float32)
        arrays[f'b{i}']=layer.bias.detach().numpy().astype(np.float32)
    np.savez(root/'actor_fp32.npz',**arrays)
    metadata=dict(seed=seed,requested_steps=int(steps),actual_steps=int(model.num_timesteps),
                  updates=int(model._n_updates),warmup_steps=warmup,wall_seconds=time.perf_counter()-t0,
                  torch=torch.__version__,sb3=sb3.__version__,python=platform.python_version(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  physics=asdict(Physics()),normalization_mean=OBS_MEAN.tolist(),normalization_std=OBS_STD.tolist(),
                  domain_randomization=dict(mass_ratio=[.8,1.2],beta_pa=[.6e9,1.4e9],Fc_ratio=[.7,1.3]),
                  safety_filter_in_training=True,optimizer_reward_scale=.01,
                  observation='PPT five features; partially observed when parameters randomized',
                  entropy='H(u)=H(a)+ln(10), local-state Monte Carlo; no forced 0.35 plateau',
                  convergence_claim=False,origin='fresh random initialization, no external checkpoints or data')
    save_json(root/'training_metadata.json',metadata);env.close()
    return model


class FloatActor:
    def __init__(self,arrays):
        self.weights=[np.asarray(arrays[f'W{i}'],dtype=np.float32) for i in range(3)]
        self.biases=[np.asarray(arrays[f'b{i}'],dtype=np.float32) for i in range(3)]
    def __call__(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            x=x@w.T+b
            if i<2:x=np.maximum(x,0)
        return np.tanh(x).astype(np.float32)
    def arrays(self):
        return {key:val for i,(w,b) in enumerate(zip(self.weights,self.biases)) for key,val in [(f'W{i}',w),(f'b{i}',b)]}


def qparams(values):
    lo=min(float(np.min(values)),0.);hi=max(float(np.max(values)),0.)
    scale=max((hi-lo)/255,1e-9);zero=int(np.clip(np.rint(-128-lo/scale),-128,127))
    return scale,zero


def quantize(values,scale,zero):
    if not np.isfinite(scale) or scale<=0 or not -128<=zero<=127:raise ValueError('非法量化参数')
    return np.clip(np.rint(np.asarray(values)/scale)+zero,-128,127).astype(np.int8)


class Int8Actor:
    """INT8权重+INT8激活；居中后INT32矩阵乘和偏置累加。

    输入仿射量化、层间重标定和输出tanh保留浮点；不宣称纯整数MCU实现。
    这里验证真实量化误差，CPU NumPy计时不能代表嵌入式实时性。
    """
    def __init__(self,data):self.data={k:np.asarray(v) for k,v in data.items()}
    def __call__(self,obs):
        x=np.asarray(obs,dtype=np.float32)
        for i in range(3):
            d=self.data;s=float(d[f'sx{i}']);z=int(d[f'zx{i}'])
            qx=quantize(x,s,z).astype(np.int32)-z
            qw=d[f'QW{i}'].astype(np.int32)-int(d[f'zw{i}'])
            acc=qx@qw.T+d[f'qb{i}'].astype(np.int32)
            x=acc.astype(np.float32)*(s*float(d[f'sw{i}']))
            if i<2:x=np.maximum(x,0)
        return np.tanh(x).astype(np.float32)


def compress_actor(actor,calibration,keep_ratio=.3):
    """结构剪枝：每隐藏层保留约30%的完整神经元（64→19）。

    70%通道移除不等于70%连接移除，不用随机权重或预设信号替代策略。
    按校准激活*下游权重范数排序，并对保留的末层做小规模岭回归拟合
    原actor的pre-tanh输出（仅无标签校准，不使用负载真值或理想电压）。
    """
    xs=np.asarray(calibration,dtype=np.float32)
    w0,w1,w2=actor.weights;b0,b1,b2=actor.biases
    n0=max(1,int(round(w0.shape[0]*keep_ratio)));n1=max(1,int(round(w1.shape[0]*keep_ratio)))
    h0=np.maximum(xs@w0.T+b0,0)
    score0=np.sqrt(np.mean(h0*h0,axis=0))*np.linalg.norm(w1,axis=0)
    i0=np.sort(np.argsort(score0)[-n0:])
    h1=np.maximum(h0@w1.T+b1,0)
    score1=np.sqrt(np.mean(h1*h1,axis=0))*np.linalg.norm(w2,axis=0)
    i1=np.sort(np.argsort(score1)[-n1:])
    arrays=dict(W0=w0[i0],b0=b0[i0],W1=w1[np.ix_(i1,i0)],b1=b1[i1],W2=w2[:,i1],b2=b2.copy())
    pruned=FloatActor(arrays)
    features=np.maximum(np.maximum(xs@pruned.weights[0].T+pruned.biases[0],0)@pruned.weights[1].T+pruned.biases[1],0)
    design=np.c_[features,np.ones(len(xs),dtype=np.float32)].astype(np.float64)
    target=h1@w2.T+b2
    ridge=np.eye(design.shape[1])*1e-4;ridge[-1,-1]=0.
    coef=np.linalg.solve(design.T@design+ridge,design.T@target)
    pruned.weights[2]=coef[:-1].T.astype(np.float32);pruned.biases[2]=coef[-1].astype(np.float32)
    data={};x=xs
    for i,(w,b) in enumerate(zip(pruned.weights,pruned.biases)):
        sx,zx=qparams(x);sw,zw=qparams(w)
        data.update({f'QW{i}':quantize(w,sw,zw),f'sw{i}':np.array(sw),f'zw{i}':np.array(zw),
                     f'sx{i}':np.array(sx),f'zx{i}':np.array(zx)})
        biasq=np.rint(b/(sx*sw))
        if np.max(np.abs(biasq))+w.shape[1]*65025>np.iinfo(np.int32).max:raise OverflowError('INT32累加器可能溢出')
        data[f'qb{i}']=biasq.astype(np.int32)
        # 后一层校准使用当前量化层的真实输出，保持转换链一致。
        qx=quantize(x,sx,zx).astype(np.int32)-zx;qw=data[f'QW{i}'].astype(np.int32)-zw
        x=(qx@qw.T+data[f'qb{i}']).astype(np.float32)*(sx*sw)
        if i<2:x=np.maximum(x,0)
    report=dict(hidden_widths_before=[w0.shape[0],w1.shape[0]],hidden_widths_after=[n0,n1],
                removed_hidden_channel_fraction=1-(n0+n1)/(w0.shape[0]+w1.shape[0]),
                calibration_count=len(xs),calibration='separate nominal and 600/1400MPa trajectories, no test fault trace',
                calibration_head_refit=True,
                weight_connections_before=sum(w.size for w in actor.weights),
                weight_connections_after=sum(w.size for w in pruned.weights))
    edge=Int8Actor(data)
    report['calibration_action_mae_v']=float(np.mean(abs(actor(xs)-edge(xs)))*10)
    return pruned,edge,report


def export_c(edge,path):
    """导出静态工作区的C99推理；没有malloc，量化/重标定/tanh保留浮点。"""
    code=['/* Generated by the fresh Chapter 8 lab. Input: 5 Z-score features. */',
          '#include <stdint.h>', '#include <math.h>',
          'static int32_t quant(float x, float s, int z) {',
          '  float q=nearbyintf(x/s)+(float)z;',
          '  return (int32_t)fminf(127.0f,fmaxf(-128.0f,q));', '}',
          'float hydraulic_actor(const float input[5]) {',
          '  float a[64]={0}, b[64]={0};',
          '  for(int j=0;j<5;++j) a[j]=input[j];']
    for i in range(3):
        w=edge.data[f'QW{i}'];bias=edge.data[f'qb{i}'];nout,nin=w.shape
        sx=float(edge.data[f'sx{i}']);sw=float(edge.data[f'sw{i}'])
        zx=int(edge.data[f'zx{i}']);zw=int(edge.data[f'zw{i}'])
        wtext=','.join(str(int(v)) for v in w.ravel());btext=','.join(str(int(v)) for v in bias)
        code += [f'  static const int8_t w{i}[{w.size}]={{'+wtext+'};',
                 f'  static const int32_t bias{i}[{bias.size}]={{'+btext+'};',
                 f'  for(int o=0;o<{nout};++o) {{',f'    int32_t sum=bias{i}[o];',
                 f'    for(int j=0;j<{nin};++j) sum+=(quant(a[j],{sx:.12e}f,{zx})-({zx}))*((int32_t)w{i}[o*{nin}+j]-({zw}));',
                 f'    b[o]=(float)sum*{sx*sw:.12e}f;', '  }']
        if i<2:code += [f'  for(int j=0;j<{nout};++j) a[j]=fmaxf(0.0f,b[j]);']
        else:code += ['  return tanhf(b[0]);']
    code += ['}', '/* Link with -lm. Output is normalized action; voltage = 10*output. */']
    Path(path).write_text('\n'.join(code)+'\n',encoding='utf-8')


def safety_experiment(run_dir):
    root=Path(run_dir);root.mkdir(parents=True,exist_ok=True)
    protected=HydraulicTwin();unprotected=HydraulicTwin(integrator="radau")
    # 静止、零加速度、双腔预压的可行初始状态，避免初始压力失衡。
    initial=np.array([.02,0.,5e6,10e6]);protected.reset(initial);unprotected.reset(initial)
    barrier=HOCBFFilter();rows=[];unsafe_alive=True;collision=None;failure=None;unsafe_failure=None;unsafe_failure_time=None
    minima=[np.inf,np.inf,np.inf,np.inf]
    for i in range(1001):
        t=i*.001
        try:u=barrier.project(protected,8.5)
        except InfeasibleSafety as error:failure=str(error);break
        hs,ps1,ps2=barrier.diagnostics(protected)
        rows.append(dict(time_s=t,safe_x_mm=protected.state[0]*1000,safe_v_mms=protected.state[1]*1000,
                         u_rl_v=8.5,u_safe_v=u,unsafe_x_mm=unprotected.state[0]*1000 if unsafe_alive else '',
                         physical_h=(.1-protected.state[0])*protected.state[0]/.1**2,
                         psi1_min=float(ps1.min()),psi2_min=float(ps2.min()),
                         p1_mpa=protected.state[2]/1e6,p2_mpa=protected.state[3]/1e6))
        if i==1000:break
        try:_,ext=protected.step(u)
        except FloatingPointError as error:failure=str(error);break
        minima=[min(minima[0],ext[0]),max(0 if not np.isfinite(minima[1]) else minima[1],ext[1]),
                min(minima[2],ext[2]),max(0 if not np.isfinite(minima[3]) else minima[3],ext[3])]
        if unsafe_alive:
            try:
                _,extu=unprotected.step(8.5)
                if extu[1]>=.1:
                    collision=(i+1)*.001;unsafe_alive=False
                elif extu[2]<5e4 or extu[3]>25e6:
                    unsafe_failure='pressure_constraint_before_stroke_limit'
                    unsafe_failure_time=(i+1)*.001;unsafe_alive=False
            except FloatingPointError as error:
                unsafe_failure=str(error);unsafe_failure_time=(i+1)*.001;unsafe_alive=False
    write_csv(root/'safety.csv',rows)
    summary=dict(model='same four-state asymmetric hydraulic ODE as training',relative_degree=3,
                 protected_duration_s=rows[-1]['time_s'],failure=failure,
                 software_limit_mm=100.,physical_cylinder_length_mm=1000.,
                 inner_margin_mm=1.2,initial_state_si=initial.tolist(),
                 unprotected_first_detected_limit_time_s=collision,
                 unprotected_other_failure=unsafe_failure,unprotected_other_failure_time_s=unsafe_failure_time,
                 minimum_safe_voltage_v=min(r['u_safe_v'] for r in rows),
                 final_safe_position_mm=rows[-1]['safe_x_mm'],
                 observed_substep_min_x_mm=minima[0]*1000,observed_substep_max_x_mm=minima[1]*1000,
                 min_pressure_mpa=minima[2]/1e6,max_pressure_mpa=minima[3]/1e6,
                 empirical_node_safety_pass=bool(failure is None and minima[0]>=0 and minima[1]<=.1 and minima[2]>=5e4 and minima[3]<=25e6),
                 certificate_scope='sample-time continuous HOCBF plus grid/root one-step viability refinement; substep-node monitoring is not a continuous-time or Sim2Real proof')
    save_json(root/'safety_summary.json',summary);return summary


def observation(twin,time_s,initial_position=.02,velocity=.05,u=0.,load=0.,beta_ratio=1.):
    x,v,p1,p2=twin.state;p=twin.physics
    raw=np.array([(initial_position+velocity*time_s-x)*1000,(velocity-v)*1000,
                  -twin.rhs(twin.state,u,load,beta_ratio)[1]*1000,
                  (p1-p.A2/p.A1*p2)/1e6,p.Ps/1e6])
    return ((raw-OBS_MEAN)/OBS_STD).astype(np.float32)


def calibration_rollouts(actor,seed=2026):
    rng=np.random.default_rng(seed+8);samples=[]
    for beta in [.6e9,1e9,1.4e9]:
        plant=HydraulicTwin(replace(Physics(),beta=beta));y,u=plant.equilibrium();plant.reset(y);cbf=HOCBFFilter()
        for i in range(500):
            load=2500. if i>=200 else 0.
            obs=observation(plant,i*.001,u=u,load=load)
            samples.append(obs.copy())
            # 加少量探索用于校准覆盖；固定独立种子，不使用测试轨迹。
            proposal=float(np.clip(float(actor(obs).item())*10+rng.normal(0,.1),-10,10))
            try:u=cbf.project(plant,proposal,load);plant.step(u,load=load)
            except (InfeasibleSafety,FloatingPointError):break
    if len(samples)<30:raise RuntimeError('校准样本不足')
    return np.asarray(samples,dtype=np.float32)


class PID:
    """位置PID：巡航偏置、不完全速度微分、条件积分抗饱和。"""
    def __init__(self,bias):self.bias=bias;self.integral=0.;self.filtered_de=0.
    def __call__(self,e,de,dt=.001):
        self.filtered_de=.85*self.filtered_de+.15*de
        # e单位m，de单位m/s；非教材响应反拟合的固定课堂整定。
        tentative=self.integral+e*dt
        candidate=self.bias+120.*e+60.*tentative+1.5*self.filtered_de
        if abs(candidate)<=10 or candidate*e<0:self.integral=tentative
        return float(np.clip(self.bias+120*e+60*self.integral+1.5*self.filtered_de,-10,10))


def evaluate(run_dir):
    root=Path(run_dir)
    with np.load(root/'actor_fp32.npz') as data:fp32=FloatActor(data)
    cal=calibration_rollouts(fp32);np.save(root/'calibration_observations.npy',cal)
    pruned,edge,compression=compress_actor(fp32,cal)
    np.savez(root/'actor_pruned_fp32.npz',**pruned.arrays());np.savez(root/'actor_int8.npz',**edge.data)
    export_c(edge,root/'hydraulic_actor.c')
    rows=[];metrics={};initial_voltage=None
    for policy in ['PID','nominal_feedforward','SAC_FP32','pruned_INT8']:
        twin=HydraulicTwin();state,bias=twin.equilibrium();initial_voltage=bias;twin.reset(state)
        controller=PID(bias);cbf=HOCBFFilter();u=bias;failure=None;group=[]
        # 所有策略使用相同模型/初态/故障/屏障，分别推进自身闭环状态。
        for i in range(1001):
            t=i*.001;load=5000. if i>=500 else 0.;beta=.5 if i>=500 else 1.
            obs=observation(twin,t,u=u,load=load,beta_ratio=beta)
            x,v,p1,p2=twin.state
            if policy=='PID':proposal=controller(.02+.05*t-x,.05-v)
            elif policy=='nominal_feedforward':proposal=bias # 标称空载恒速逆解，无实际负载标签
            elif policy=='SAC_FP32':proposal=float(fp32(obs).item())*10
            else:proposal=float(edge(obs).item())*10
            try:applied=cbf.project(twin,proposal,load,beta)
            except InfeasibleSafety as error:failure=str(error);break
            row=dict(policy=policy,time_s=t,x_m=x,v_m_s=v,p1_pa=p1,p2_pa=p2,
                     error_mm=(.02+.05*t-x)*1000,velocity_error_mms=(v-.05)*1000,
                     proposed_v=proposal,applied_v=applied,intervention=int(abs(applied-proposal)>1e-6),
                     load_n=load,beta_ratio=beta)
            group.append(row);rows.append(row)
            if i==1000:break
            try:_,ext=twin.step(applied,load=load,beta_ratio=beta)
            except FloatingPointError as error:failure=str(error);break
            u=applied
            if ext[0]<0 or ext[1]>.1 or ext[2]<5e4 or ext[3]>25e6:
                failure='physical_constraint_exceeded_at_substep';break
        if not group:raise RuntimeError(f'{policy}首步已不可行: {failure}')
        post=[r for r in group if r['time_s']>=.5]
        values=np.array([r['velocity_error_mms'] for r in post])
        recovery=None
        if post:
            peak=int(np.argmax(abs(values)))
            for j in range(peak,len(post)):
                if post[-1]['time_s']-post[j]['time_s']>=.05-1e-9 and np.all(abs(values[j:])<=1.):
                    recovery=post[j]['time_s']-.5;break
        metrics[policy]=dict(completed=group[-1]['time_s']>=1.-1e-9 and failure is None,failure=failure,
                             samples=len(group),last_time_s=group[-1]['time_s'],
                             max_speed_drop_mms=float(values.min()) if post else None,
                             post_fault_velocity_rmse_mms=float(np.sqrt(np.mean(values**2))) if post else None,
                             recovery_to_1mms_s=recovery,
                             steady_velocity_error_mms=float(np.mean(values[-100:])) if len(values)>=100 else None,
                             intervention_fraction=float(np.mean([r['intervention'] for r in group])))
    write_csv(root/'evaluation.csv',rows)
    compression['fp32_npz_bytes']=(root/'actor_fp32.npz').stat().st_size
    compression['int8_npz_bytes']=(root/'actor_int8.npz').stat().st_size
    compression['file_reduction_fraction']=1-compression['int8_npz_bytes']/compression['fp32_npz_bytes']
    # 真正单样本在线推理计时，模型加载与编译在计时之外。
    timings={}
    for name,actor in [('FP32_numpy',fp32),('INT8_numpy',edge)]:
        for _ in range(30):actor(cal[0])
        samples=[]
        for _ in range(300):
            start=time.perf_counter_ns();actor(cal[0]);samples.append((time.perf_counter_ns()-start)/1e6)
        timings[name]=dict(median_ms=float(np.median(samples)),p95_ms=float(np.quantile(samples,.95)),samples=300)
    summary=dict(metrics=metrics,compression=compression,cpu_numpy_inference=timings,
                 initial_nominal_voltage_v=initial_voltage,
                 all_policies_use_identical_safety_filter=True,
                 timestep_s=.001,fault_time_s=.5,fault_load_n=5000.,fault_beta_ratio=.5,
                 deployment_scope='Python CPU reference; no claimed MCU timing, no real-world data or fine-tuning')
    save_json(root/'evaluation_summary.json',summary);return summary


def plot_results(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available={font.name for font in font_manager.fontManager.ttflist}
    fonts=[n for n in ['Noto Sans CJK SC','Noto Sans CJK JP','Microsoft YaHei','SimHei','DejaVu Sans'] if n in available]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':fonts,'axes.unicode_minus':False,'font.size':10})
    root=Path(run_dir);training=read_csv(root/'training.csv');safety=read_csv(root/'safety.csv');evaluation=read_csv(root/'evaluation.csv')
    meta=json.loads((root/'training_metadata.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(17,10));fig.subplots_adjust(left=.075,right=.92,top=.86,bottom=.13,wspace=.28,hspace=.4)
    fig.suptitle('第八章：SAC 真实训练、四阶液压 HOCBF 与结构剪枝量化',fontsize=17,y=.975)
    fig.text(.5,.92,f"本次从零训练 {meta['actual_steps']:,} 步；曲线来自保存的训练日志与独立物理闭环，无预设教材响应",ha='center',fontsize=11)
    def col(rows,name):return np.array([float(r[name]) for r in rows])
    ax=axes[0,0];steps=col(training,'steps');reward=col(training,'mean_base_reward')
    l1,=ax.plot(steps/1000,reward,color='royalblue',label='最近1000步平均基础奖励')
    # 保留巨大早期损失，不裁掉日志或把奖励改造成S曲线。
    ax.set_yscale('symlog',linthresh=10)
    right=ax.twinx();l2,=right.plot(steps/1000,col(training,'voltage_entropy_nat'),'k--',label='电压策略熵 H(u)')
    ax.set(xlabel='实际交互步数 / 千步',ylabel='式8.1平均奖励（对称对数轴）',title='(a) 从随机初始化开始的 SAC 训练日志')
    right.set_ylabel('电压微分熵 / nat');ax.legend(handles=[l1,l2],fontsize=8,loc='best')
    ax=axes[0,1];ts=col(safety,'time_s');xs=col(safety,'safe_x_mm')
    a,=ax.plot(ts,xs,color='royalblue',label='同一四阶模型：受保护位移')
    valid=[r for r in safety if r['unsafe_x_mm']!='']
    b,=ax.plot(col(valid,'time_s'),col(valid,'unsafe_x_mm'),':',color='crimson',label='未保护（越界/超压/气穴后终止）')
    ax.axhline(100,color='crimson',ls='--');ax.axhspan(100,max(105,float(xs.max())+5),color='#f8dfe3')
    right=ax.twinx();c,=right.plot(ts,col(safety,'u_safe_v'),'-.',color='forestgreen',label='实际 HOCBF 安全电压')
    ax.set(xlabel='时间 / s',ylabel='活塞位移 / mm',title='(b) 三阶行程 + 双腔压力屏障（数值验证）')
    right.set_ylabel('电压 / V');ax.legend(handles=[a,b,c],loc='best',fontsize=8)
    styles={'PID':('#444444','--'),'nominal_feedforward':('#e58a00',':'),'SAC_FP32':('royalblue','-'),'pruned_INT8':('forestgreen','-.')}
    labels={'PID':'固定参数 PID','nominal_feedforward':'标称空载前馈','SAC_FP32':'本次 SAC · FP32','pruned_INT8':'70% 通道剪枝 + INT8'}
    for ax,key,title,ylabel in [(axes[1,0],'velocity_error_mms','(c) 5000 N + 模量减半：真实闭环速度误差','速度误差 / (mm/s)'),
                                 (axes[1,1],'applied_v','(d) 实际执行电压与独立 INT8 策略闭环','阀执行电压 / V')]:
        ax.axvspan(.5,1.,color='#eeeeee');ax.axvline(.5,color='gray',ls='--')
        for policy,(color,style) in styles.items():
            group=[r for r in evaluation if r['policy']==policy]
            ax.plot(col(group,'time_s'),col(group,key),color=color,ls=style,label=labels[policy])
        ax.set(xlim=(.4,1.),xlabel='时间 / s',ylabel=ylabel,title=title);ax.legend(fontsize=8,loc='best')
    for ax in axes.flat:ax.grid(ls=':',alpha=.4)
    fig.text(.065,.04,'实际模型沿用上传文件的标称参数；100 mm为软件安全行程。所有跟踪策略使用相同安全滤波。\n短训练不代表收敛；剪枝后的退化如实保留。HOCBF采样/子步数值检验不等于真机连续安全证明。',fontsize=10)
    for suffix in ['png','pdf']:fig.savefig(root/f'chapter8_results.{suffix}',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=['all','train','evaluate','safety','plot'],default='all')
    parser.add_argument('--steps',type=int,default=1_000_000)
    parser.add_argument('--warmup',type=int,default=5000)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent/'runs'/'seed2026')
    args=parser.parse_args()
    if args.steps<1:parser.error('--steps必须为正整数')
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.mode in ['all','train']:train(args.run_dir,args.steps,args.seed,args.warmup)
    if args.mode in ['all','safety']:print(json.dumps(safety_experiment(args.run_dir),ensure_ascii=False,indent=2))
    if args.mode in ['all','evaluate']:print(json.dumps(evaluate(args.run_dir),ensure_ascii=False,indent=2))
    if args.mode in ['all','plot']:plot_results(args.run_dir)


if __name__=='__main__':main()
