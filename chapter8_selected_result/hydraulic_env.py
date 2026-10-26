"""参考代码五维接口，内部由真实四状态受力/流量方程推进。

观测为 [e, de, dde, PL, Ps]，其中 e=x-r（mm）、de=v-v_ref
（mm/s）。质量、体积模量和摩擦随机量均实际作用于物理内核。
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hydraulic_twin import HydraulicTwinCore


class HydraulicGymEnv(gym.Env):
    metadata={"render_modes": []}
    observation_names=["position_error_mm", "velocity_error_mm_s",
                       "acceleration_error_mm_s2", "load_pressure_MPa", "supply_pressure_MPa"]
    obs_mean=np.array([0.,0.,0.,10.,21.],dtype=np.float32)
    obs_std=np.array([10.,50.,500.,8.,2.],dtype=np.float32)

    def __init__(self,twin_config=None,domain_randomization=True,
                 fault_augmentation=True,max_steps=2000):
        super().__init__()
        self.twin=HydraulicTwinCore(config=twin_config)
        self.default_m=float(self.twin.p['m'])
        self.default_beta=float(self.twin.p['beta_e'])
        self.default_Fc=float(self.twin.p['F_c'])
        self.base_physics=dict(m=self.default_m,beta_e=self.default_beta,F_c=self.default_Fc)
        self.domain_randomization=bool(domain_randomization)
        self.fault_augmentation=bool(fault_augmentation)
        self.max_steps=int(max_steps)
        if self.max_steps<1:raise ValueError('max_steps must be positive')
        self.dt=.001
        self.action_space=spaces.Box(-1.,1.,shape=(1,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(5,),dtype=np.float32)
        self.step_count=0;self.last_u=0.

    @property
    def time(self):return self.step_count*self.dt

    @property
    def physical_state(self):return self.twin.state.copy()

    def normalize_obs(self,obs):
        return ((np.asarray(obs,dtype=np.float64)-self.obs_mean)/
                (self.obs_std+np.float32(1e-8))).astype(np.float32)

    def current_fault(self):
        return ((self.load_after,self.beta_after) if self.time>=self.fault_time-1e-12
                else (0.,1.))

    def raw_observation(self):
        x,v,p1,p2=self.twin.state
        load,beta=self.current_fault()
        acceleration=self.twin.rhs(self.twin.state,self.last_u,load,beta)[1]
        reference=self.initial_x+self.v_ref*self.time
        p=self.twin.p
        pl=(p1-p['A2']/p['A1']*p2)/1e6
        return np.array([(x-reference)*1000.,(v-self.v_ref)*1000.,
                         acceleration*1000.,pl,p['Ps']/1e6],dtype=np.float64)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        options=options or {};rng=self.np_random
        dr=self.domain_randomization and not options.get('evaluation',False)
        pars=(dict(m=self.default_m*rng.uniform(.8,1.2),
                   beta_e=self.default_beta*rng.uniform(.6,1.4),
                   F_c=self.default_Fc*rng.uniform(.7,1.3)) if dr else self.base_physics.copy())
        self.twin.update_physics(pars)
        # The requested five features contain velocity error, not the requested
        # velocity itself.  Use the declared 50 mm/s classroom task so that a
        # hidden reference speed does not make identical observations ambiguous.
        self.v_ref=float(options.get('velocity',.05))
        self.initial_x=float(options.get('position',rng.uniform(.0001,.005) if dr else
                                         self.twin.p['initial_position_m']))
        state,voltage=self.twin.equilibrium_for_velocity(self.v_ref,position=self.initial_x)
        self.twin.reset(x0=state)
        self.step_count=0;self.last_u=0.
        use_fault=self.fault_augmentation and rng.random()<.8
        self.fault_time=float(options.get('fault_time',rng.uniform(.05,.8) if use_fault else np.inf))
        self.load_after=float(options.get('load_n',rng.uniform(0.,5500.) if use_fault else 0.))
        self.beta_after=float(options.get('beta_multiplier',rng.uniform(.5,1.) if use_fault else 1.))
        obs=self.normalize_obs(self.raw_observation());self._last_finite_obs=obs.copy()
        return obs,dict(**pars,fault_time=self.fault_time,load_n=self.load_after,
                        beta_multiplier=self.beta_after,initial_voltage_v=voltage,
                        initial_position_m=self.initial_x,reference_velocity_m_s=self.v_ref)

    def step(self,action_norm):
        action=np.asarray(action_norm,dtype=float)
        if action.shape!=(1,) or not np.isfinite(action).all():
            raise ValueError('Action must contain one finite normalized voltage')
        voltage=float(np.clip(action[0],-1.,1.)*10.)
        load,beta=self.current_fault();old_u=self.last_u;failure=None
        try:
            state,info=self.twin.step(voltage,dt=self.dt,load_n=load,beta_multiplier=beta)
        except (RuntimeError,ValueError) as error:
            failure=str(error);state=self.twin.state.copy();info=dict(integration_failure=failure)
        self.last_u=voltage;self.step_count+=1
        try:raw=self.raw_observation()
        except (RuntimeError,ValueError):raw=np.full(5,np.nan)
        if not np.isfinite(state).all() or not np.isfinite(raw).all():
            failure=failure or 'nonfinite_physical_state'
        elif not 0.<state[0]<self.twin.p['L']:
            failure=failure or 'mechanical_stroke_exit'
        elif min(state[2:])<=0. or max(state[2:])>40e6:
            failure=failure or 'pressure_outside_training_envelope'
        elif abs(raw[0])>20. or abs(state[1])>1.:
            failure=failure or 'tracking_outside_training_envelope'
        error=float(raw[0]) if np.isfinite(raw[0]) else 20.
        r_error=-100.*error**2;r_energy=-.1*voltage**2
        r_smooth=-(voltage-old_u)**2;r_bonus=2. if abs(error)<.2 else 0.
        base_reward=r_error+r_energy+r_smooth+r_bonus
        # Terminal failure must not offer a cheap escape from future tracking
        # costs.  The textbook base reward above remains separately available.
        penalty=-100000. if failure else 0.
        reward=base_reward+penalty
        obs=self.normalize_obs(raw) if np.isfinite(raw).all() else self._last_finite_obs.copy()
        self._last_finite_obs=obs.copy()
        info.update(raw_reward=float(reward),base_reward=float(base_reward),failure_penalty=penalty,
                    reward_error=r_error,reward_energy=r_energy,reward_smooth=r_smooth,reward_bonus=r_bonus,
                    position_error_mm=error,velocity_error_mms=float(raw[1]),control_v=voltage,
                    physical_failure=failure,load_n=load,beta_multiplier=beta)
        return obs,float(reward),bool(failure),self.step_count>=self.max_steps,info
