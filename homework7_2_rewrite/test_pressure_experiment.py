"""短时域差异须来自闭环压力动态，并保持安全和最终收敛。"""
import unittest
import numpy as np
from work_7_2 import load_config,simulate


class PressureExperimentTests(unittest.TestCase):
    def test_short_horizon_delays_pressure_release_and_still_settles(self):
        # Identical object, weights and constraints; only N changes. The
        # requirement is an observable temporal difference, not a plot offset.
        config=load_config()
        short=simulate(5,config);normal=simulate(20,config)
        p_short=np.interp(.080,short['data'][:,0],short['data'][:,5])
        p_normal=np.interp(.080,normal['data'][:,0],normal['data'][:,5])
        self.assertGreater(p_short-p_normal,8.)
        self.assertGreater(short['metrics']['p2_above_20mpa_ms']-
                           normal['metrics']['p2_above_20mpa_ms'],1.)
        self.assertGreater(short['metrics']['overshoot_pct'],
                           2*normal['metrics']['overshoot_pct'])
        for result in (short,normal):
            self.assertLessEqual(result['metrics']['peak_p2_mpa'],21.)
            self.assertGreaterEqual(result['metrics']['min_pressure_mpa'],0.)
            self.assertLess(abs(result['metrics']['final_error_mm']),.005)
            self.assertLess(abs(result['data'][-1,6]),.01)


if __name__=='__main__':unittest.main()
