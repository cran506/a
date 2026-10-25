"""Physical unit/sign checks and independent integration of the graybox."""
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from model import HydraulicGreyBox
from verify import independent_rhs


class GreyBoxTests(unittest.TestCase):
    def setUp(self):
        self.p = json.loads(Path(__file__).with_name('config.json').read_text())['physical']
        self.model = HydraulicGreyBox(self.p)

    def test_force_balance_and_si_unit_signs(self):
        self.assertAlmostEqual(self.model.preload_n, 4000.)
        np.testing.assert_allclose(self.model.rhs(self.model.equilibrium, 0.), 0., atol=1e-11)
        state = np.array([20., .3, 6., 3.])
        expected = independent_rhs(self.p, state, 2.)
        np.testing.assert_allclose(self.model.rhs(state, 2.), expected, atol=1e-10)
        self.assertAlmostEqual(expected[0], 300.)  # m/s -> mm/s
        a1, a2 = self.p['area1_m2'], self.p['area2_m2']
        force = a1*6e6-a2*3e6-4000.-self.p['viscous_ns_m']*.3
        self.assertAlmostEqual(expected[1], force/self.p['mass_kg'])

    def test_jacobian_and_zero_order_hold_match_independent_ivp(self):
        state = self.model.equilibrium+np.array([12., .12, 1.2, .6])
        u, dt = 1.7, .001
        ac, bc, cc = self.model.linearize(state, u)
        epsilon = 1e-5
        fd = np.column_stack([(independent_rhs(self.p, state+epsilon*np.eye(4)[i], u)-
                               independent_rhs(self.p, state-epsilon*np.eye(4)[i], u))/(2*epsilon)
                              for i in range(4)])
        np.testing.assert_allclose(ac, fd, atol=2e-7, rtol=1e-7)
        du = (independent_rhs(self.p, state, u+epsilon)-independent_rhs(self.p, state, u-epsilon))/(2*epsilon)
        np.testing.assert_allclose(bc, du, atol=2e-7, rtol=1e-7)
        integrated = solve_ivp(lambda t,z: independent_rhs(self.p,z,u), (0.,dt), state,
                               method='DOP853', rtol=1e-12, atol=1e-13).y[:, -1]
        np.testing.assert_allclose(self.model.step(state,u,dt), integrated, atol=1e-10, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
