"""Five-feature textbook interface, causal rewards and physical randomization."""
import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from hydraulic_env import HydraulicGymEnv
from hydraulic_twin import HydraulicTwinCore


class EnvironmentTests(unittest.TestCase):
    def test_sb3_gym_contract(self):
        env = HydraulicGymEnv()
        try:
            check_env(env, warn=True)
        finally:
            env.close()

    def test_five_observations_and_fixed_normalization(self):
        env = HydraulicGymEnv(domain_randomization=False, fault_augmentation=False)
        obs, _ = env.reset(seed=3)
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.observation_space.shape, (5,))
        self.assertEqual(obs.shape, (5,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(env.obs_mean, [0, 0, 0, 10, 21])
        np.testing.assert_array_equal(env.obs_std, [10, 50, 500, 8, 2])
        reconstructed = obs.astype(float) * env.obs_std + env.obs_mean
        np.testing.assert_allclose(reconstructed, env.raw_observation(), atol=1e-6)
        x, v, p1, p2 = env.physical_state
        self.assertAlmostEqual(env.raw_observation()[0], 0., places=10)
        self.assertAlmostEqual(env.raw_observation()[1], 0., places=10)
        self.assertAlmostEqual(env.raw_observation()[3], (p1-env.twin.p['A2']/env.twin.p['A1']*p2)/1e6)
        self.assertEqual(env.raw_observation()[4], 21.)

    def test_seed_and_fault_replay(self):
        a, b = HydraulicGymEnv(), HydraulicGymEnv()
        oa, ia = a.reset(seed=881)
        ob, ib = b.reset(seed=881)
        self.assertEqual(ia, ib)
        np.testing.assert_array_equal(oa, ob)
        rng = np.random.default_rng(220)
        for _ in range(100):
            action = rng.uniform(-1, 1, 1)
            oa, ra, da, ta, _ = a.step(action)
            ob, rb, db, tb, _ = b.step(action)
            np.testing.assert_array_equal(oa, ob)
            self.assertEqual((ra, da, ta), (rb, db, tb))
            if da or ta:
                break

    def test_raw_reward_uses_causal_previous_action(self):
        env = HydraulicGymEnv(domain_randomization=False, fault_augmentation=False)
        env.reset(seed=3)
        self.assertEqual(env.last_u, 0.)
        previous = 0.
        for voltage in (4., 3.5):
            _, reward, terminated, _, info = env.step(np.array([voltage / 10.]))
            error = info['position_error_mm']
            expected = (-100*error**2 - .1*voltage**2 - (voltage-previous)**2
                        + (2. if abs(error) < .2 else 0.))
            self.assertFalse(terminated)
            self.assertAlmostEqual(info['base_reward'], expected, places=10)
            self.assertAlmostEqual(reward, expected + info['failure_penalty'], places=10)
            self.assertEqual(env.last_u, voltage)
            previous = voltage

    def test_fault_boundary_preserves_state_and_changes_acceleration(self):
        env = HydraulicGymEnv(domain_randomization=False, fault_augmentation=False)
        _, info = env.reset(seed=8, options=dict(evaluation=True, fault_time=.1,
                                                load_n=5000., beta_multiplier=.5))
        initial = env.physical_state
        # Keep the actual equilibrium input, rather than the reset memory u_prev=0.
        action = np.array([info['initial_voltage_v'] / 10.])
        for _ in range(100):
            _, _, done, _, _ = env.step(action)
            self.assertFalse(done)
        self.assertEqual(env.current_fault(), (5000., .5))
        expected = initial.copy()
        expected[0] += .05 * .1
        np.testing.assert_allclose(env.physical_state, expected, rtol=1e-12, atol=1e-8)
        self.assertAlmostEqual(env.raw_observation()[2], -5000./250.*1000., places=4)
        # The jump changes force/derivatives, not stored pressure or velocity.
        self.assertAlmostEqual(env.physical_state[1], .05, places=11)

    def test_randomized_parameters_are_within_envelope_and_applied(self):
        env = HydraulicGymEnv()
        for seed in range(12):
            _, info = env.reset(seed=seed)
            for name, lower, upper in (('m', .8*250., 1.2*250.),
                                       ('beta_e', .6e9, 1.4e9),
                                       ('F_c', .7*1200., 1.3*1200.)):
                self.assertGreaterEqual(info[name], lower)
                self.assertLessEqual(info[name], upper)
                self.assertEqual(env.twin.p[name], info[name])
        twin = HydraulicTwinCore()
        state, voltage = twin.equilibrium_for_velocity(.05)
        baseline = twin.rhs(state, voltage, load_n=5000.)
        twin.update_physics({'m': 300.})
        heavier = twin.rhs(state, voltage, load_n=5000.)
        self.assertAlmostEqual(heavier[1], baseline[1]*250./300., places=10)
        twin.update_physics({'m': 250., 'F_c': 1500.})
        friction = twin.rhs(state, voltage, load_n=5000.)
        self.assertLess(friction[1], baseline[1])
        twin.update_physics({'F_c': 1200., 'beta_e': .5e9})
        pressure = twin.rhs(state, voltage + 1., load_n=5000.)
        twin.update_physics({'beta_e': 1e9})
        pressure_stiff = twin.rhs(state, voltage + 1., load_n=5000.)
        self.assertLess(abs(pressure[2]), abs(pressure_stiff[2]))

    def test_finite_action_contract_and_explicit_voltage_clip(self):
        env = HydraulicGymEnv(domain_randomization=False, fault_augmentation=False)
        env.reset(seed=4)
        before = env.physical_state
        for action in (np.array([np.nan]), np.array([1., 2.]), np.array(1.)):
            with self.assertRaises(ValueError):
                env.step(action)
            np.testing.assert_array_equal(env.physical_state, before)
        _, _, _, _, info = env.step(np.array([2.]))
        self.assertEqual(info['control_v'], 10.)


if __name__ == '__main__':
    unittest.main()
