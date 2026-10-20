"""Causal reward, deterministic reset and real Gym integration checks."""
import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from hydraulic_env import HydraulicGymEnv


class EnvironmentTests(unittest.TestCase):
    def test_sb3_gym_contract(self):
        check_env(HydraulicGymEnv(),warn=True)

    def test_seed_and_fault_replay(self):
        a,b=HydraulicGymEnv(),HydraulicGymEnv()
        oa,ia=a.reset(seed=881);ob,ib=b.reset(seed=881)
        self.assertEqual(ia,ib)
        np.testing.assert_array_equal(oa,ob)
        rng=np.random.default_rng(220)
        for _ in range(100):
            action=rng.uniform(-1,1,1)
            oa,ra,da,ta,_=a.step(action);ob,rb,db,tb,_=b.step(action)
            np.testing.assert_array_equal(oa,ob)
            self.assertEqual((ra,da,ta),(rb,db,tb))
            if da or ta:break

    def test_raw_reward_and_previous_action(self):
        env=HydraulicGymEnv(domain_randomization=False,fault_augmentation=False)
        _,info=env.reset(seed=3)
        old=info['initial_voltage_v']
        obs,reward,done,_,info=env.step(np.array([.4]))
        e=info['position_error_mm']
        expected=-100*e*e-.1*4**2-(4-old)**2+(2 if abs(e)<.2 else 0)
        self.assertFalse(done)
        self.assertAlmostEqual(reward,expected,places=10)
        reconstructed=obs*env.obs_std+env.obs_mean
        self.assertAlmostEqual(float(reconstructed[8]),4.,places=5)

    def test_fault_boundary_does_not_reset_physical_state(self):
        env=HydraulicGymEnv(domain_randomization=False,fault_augmentation=False)
        env.reset(seed=8,options=dict(evaluation=True,fault_time=.1,load_n=5000,beta_multiplier=.5))
        for _ in range(100):
            env.step(np.array([env.last_u/10]))
        self.assertEqual(env.current_fault(),(5000.,.5))
        self.assertLess(abs(env.twin.state[1]-.05),1e-8)
        # At the instant of fault the state is continuous, its acceleration jumps.
        self.assertAlmostEqual(env.raw_observation()[2],20000.,places=3)


if __name__=='__main__':unittest.main()
