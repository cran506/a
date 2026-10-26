"""Configured HOCBF demonstration must propagate and verify the same model."""
import json
from pathlib import Path
import tempfile
import unittest

from safety_experiment import load_safety_config,run_safety_experiment
from safety_filter import HOCBFSafetyFilter


class TestSafetyConfiguration(unittest.TestCase):
    def test_classroom_demo_has_actual_braking_and_independent_certificate(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_safety_experiment(folder)
        summary=result['summary']
        self.assertTrue(summary['passed'])
        self.assertTrue(summary['independent_random_validation']['passed'])
        self.assertAlmostEqual(summary['unsafe_collision_time_s'],.1048911492965995,places=11)
        self.assertAlmostEqual(summary['minimum_safe_voltage_v'],-3.0916046704129267,places=10)
        self.assertGreaterEqual(summary['first_intervention_s'],.055)
        self.assertLess(summary['safe_maximum_position_mm'],100.)
        self.assertAlmostEqual(summary['safe_final_position_mm'],98.8,places=9)
        self.assertEqual(summary['alpha_sum_per_s'],98.)
        self.assertEqual(summary['alpha_product_per_s2'],2401.)

    def test_nondefault_config_refreshes_dynamics_guards_and_validation(self):
        config=load_safety_config()
        config.update(x_min=10.,x_max=110.,margin=2.,k_gain=2100.,damp_c=12.,
                      kappa1=31.,kappa2=55.,proposed_action_v=7.)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'alternate.json'
            path.write_text(json.dumps(config),encoding='utf-8')
            result=run_safety_experiment(Path(folder)/'output',config_path=path)
        summary=result['summary']
        self.assertTrue(summary['passed'])
        self.assertEqual(summary['physical_limit_mm'],[10.,110.])
        self.assertEqual(summary['guard_limit_mm'],[12.,108.])
        self.assertEqual(summary['alpha_sum_per_s'],86.)
        self.assertEqual(summary['alpha_product_per_s2'],1705.)
        self.assertEqual(summary['k_gain_mm_s2_per_v'],2100.)
        self.assertEqual(summary['damping_per_s'],12.)
        self.assertAlmostEqual(summary['safe_final_position_mm'],108.,places=7)
        self.assertTrue(summary['independent_random_validation']['passed'])

    def test_legacy_filter_constructor_defaults_are_preserved(self):
        filt=HOCBFSafetyFilter()
        self.assertEqual(filt.k_gain,1850.)
        self.assertEqual(filt.kappa1,24.)
        self.assertEqual(filt.kappa2,24.)


if __name__=='__main__':
    unittest.main()
