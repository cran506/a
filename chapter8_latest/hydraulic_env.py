"""Gymnasium training environment. All observations and rewards have explicit units.

This ten-dimensional observation expands the textbook five-variable sketch:
pressures, chamber-volume position, reference and previous action are needed by
the physical model/reward. Unknown randomized parameters remain hidden context.
"""
from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hydraulic_twin import HydraulicTwinCore


class HydraulicGymEnv(gym.Env):
    metadata = {"render_modes": []}
    observation_names = ["position_error_mm", "velocity_error_mm_s",
        "acceleration_error_mm_s2", "P1_MPa", "P2_MPa", "position_mm",
        "reference_velocity_mm_s", "reference_acceleration_mm_s2", "previous_u_V", "Ps_MPa"]
    obs_mean = np.array([0, 0, 0, 8, 6, 20, 50, 0, 3.5, 21], np.float32)
    obs_std = np.array([1, 50, 20000, 8, 8, 40, 25, 500, 5, 2], np.float32)

    def __init__(self, twin_config=None, domain_randomization=True,
                 fault_augmentation=True, max_steps=2000):
        super().__init__()
        self.twin = HydraulicTwinCore(config=twin_config)
        self.base_physics={name:self.twin.p[name] for name in ['m','beta_e','F_c']}
        self.domain_randomization = domain_randomization
        self.fault_augmentation = fault_augmentation
        self.max_steps = int(max_steps)
        self.dt = .001
        self.action_space = spaces.Box(-1., 1., shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self.step_count = 0
        self.last_u = 0.

    def normalize_obs(self, obs):
        return ((np.asarray(obs)-self.obs_mean)/(self.obs_std+1e-8)).astype(np.float32)

    @property
    def time(self):
        return self.step_count * self.dt

    def current_fault(self):
        active = self.time >= self.fault_time - 1e-12
        return (self.load_after, self.beta_after) if active else (0., 1.)

    def raw_observation(self):
        x, v, p1, p2 = self.twin.state
        load, bm = self.current_fault()
        acceleration = self.twin.rhs(self.twin.state, self.last_u, load, bm)[1]
        r = self.initial_x + self.v_ref*self.time
        return np.array([(r-x)*1000, (self.v_ref-v)*1000, -acceleration*1000,
            p1/1e6, p2/1e6, x*1000, self.v_ref*1000, 0., self.last_u,
            self.twin.p['Ps']/1e6], dtype=np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        dr = self.domain_randomization and not options.get('evaluation', False)
        pars = dict(m=self.base_physics['m']*rng.uniform(.8,1.2),
                    beta_e=self.base_physics['beta_e']*rng.uniform(.6,1.4),
                    F_c=self.base_physics['F_c']*rng.uniform(.7,1.3)) if dr else self.base_physics.copy()
        self.twin.update_physics(pars)
        self.v_ref = float(options.get('velocity', rng.uniform(.035,.065) if dr else .05))
        self.initial_x = float(options.get('position', rng.uniform(.001,.01) if dr else .001))
        initial, steady_u = self.twin.equilibrium_for_velocity(self.v_ref, position=self.initial_x)
        self.twin.reset(x0=initial)
        self.step_count = 0
        self.last_u = float(steady_u)
        use_fault = self.fault_augmentation and rng.random() < .8
        self.fault_time = float(options.get('fault_time', rng.uniform(.04,.8) if use_fault else np.inf))
        self.load_after = float(options.get('load_n', rng.uniform(1000.,5500.) if use_fault else 0.))
        self.beta_after = float(options.get('beta_multiplier', rng.uniform(.5,1.) if use_fault else 1.))
        obs = self.normalize_obs(self.raw_observation())
        self._last_finite_obs = obs.copy()
        return obs, dict(**pars, fault_time=self.fault_time, load_n=self.load_after,
                         beta_multiplier=self.beta_after, initial_voltage_v=steady_u)

    def step(self, action_norm):
        action = np.asarray(action_norm, dtype=float)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError('Action must be one finite normalized voltage')
        u = float(np.clip(action[0], -1., 1.)*10.)
        load, bm = self.current_fault()
        old_u = self.last_u
        failure = None
        try:
            state, info = self.twin.step(u, dt=self.dt, load_n=load, beta_multiplier=bm)
        except RuntimeError as error:
            failure = str(error)
            state = self.twin.state.copy()
            info = dict(integration_failure=failure)
        self.last_u = u
        self.step_count += 1
        raw = self.raw_observation()
        e = float(raw[0])
        reward = -100.*e**2 - .1*u**2 - (u-old_u)**2 + (2. if abs(e)<.2 else 0.)
        if not np.isfinite(state).all() or not np.isfinite(raw).all():
            failure = failure or 'nonfinite_physical_state'
        elif not 0. < state[0] < self.twin.p['L']:
            failure = failure or 'mechanical_stroke_exit'
        elif min(state[2:]) <= 0. or max(state[2:]) > 40e6:
            failure = failure or 'pressure_outside_experiment_envelope'
        elif abs(e)>20. or abs(state[1])>1.:
            failure = failure or 'tracking_outside_training_envelope'
        terminated = failure is not None
        if terminated:
            reward -= 500.
        obs = self.normalize_obs(raw) if np.isfinite(raw).all() else self._last_finite_obs.copy()
        self._last_finite_obs = obs.copy()
        info.update(position_error_mm=e, velocity_error_mms=(self.v_ref-state[1])*1000,
            control_v=u, physical_failure=failure, load_n=load, beta_multiplier=bm)
        return obs, float(reward), terminated, self.step_count>=self.max_steps, info
