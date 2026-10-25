"""Physical checks for the optional Chapter 7 classroom hydraulic circuit."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from hydraulic_plant import GreyBoxModel, HydraulicPlant


HERE = Path(__file__).resolve().parent


def teaching_settings():
    return json.loads((HERE/'teaching_plant_config.json').read_text(encoding='utf-8'))


class TeachingCircuitTests(unittest.TestCase):
    def test_extension_return_bypass_reduces_real_cruise_backpressure(self):
        original = HydraulicPlant()
        teaching = HydraulicPlant(config=teaching_settings())
        initial = [0., 0., 1.05, 2., 0.]
        for plant in (original, teaching):
            plant.reset(initial)
            for _ in range(35):
                plant.step(10., .001)
        self.assertGreater(original.state[3], 3.)
        self.assertGreater(teaching.state[3], 0.)
        self.assertLess(teaching.state[3], .5)
        self.assertLess(teaching.state[3], original.state[3]/5.)

    def test_pilot_recovers_pressure_by_real_flow_without_position_reset(self):
        plant = HydraulicPlant(config=teaching_settings())
        plant.reset([50., 0., 7.15, 14.2, 0.])
        for _ in range(20):
            plant.step(0., .001)
        self.assertLess(abs(plant.state[0]-50.), .05)
        self.assertLess(abs(plant.state[1]), .01)
        self.assertAlmostEqual(plant.state[2], 1.05, delta=.12)
        self.assertAlmostEqual(plant.state[3], 2., delta=.12)
        self.assertGreater(float(np.min(plant.pressure_trace[:, 1:])), 0.)

    def test_auxiliary_flow_matches_predictor_and_port_gradient(self):
        settings = teaching_settings()
        true_original, true_class = HydraulicPlant(), HydraulicPlant(config=settings)
        gray_original, gray_class = GreyBoxModel(), GreyBoxModel(config=settings)
        states = ([40., 1.1, .5, .2, 10.], [50., .01, 7., 14., .2])
        for values in states:
            z = np.array(values)
            physical = np.array([.05+z[0]/1000., z[1], z[2]*1e6, z[3]*1e6])
            valve = z[4]/10.
            actual = (true_class.twin._state_derivative(0., physical, valve, False, 1)
                      -true_original.twin._state_derivative(0., physical, valve, False, 1))[2:]
            predicted = (gray_class.rhs(z, z[4])-gray_original.rhs(z, z[4]))[2:4]*1e6
            np.testing.assert_allclose(actual, predicted, rtol=1e-3, atol=2e6)

        # The classroom bypass depends on spool voltage as well as pressure.
        # Verify the exported Kq2 against the true physical RHS derivative.
        z = np.array([40., 1.1, .5, .3, 7.5])
        physical = np.array([.09, 1.1, .5e6, .3e6])
        h = 1e-3
        high = true_class.twin._state_derivative(0., physical, (z[4]+h)/10., False, 1)[3]
        low = true_class.twin._state_derivative(0., physical, (z[4]-h)/10., False, 1)[3]
        volume = (true_class.twin.p['V0_2']
                  +true_class.twin.p['A2']*(true_class.twin.p['L']-physical[0]))
        numerical = (high-low)/(2*h)*volume/true_class.twin._get_beta(physical[3])
        reported = gray_class.get_jacobian_coeffs(z)['Kq2_voltage_m3_s_v']
        self.assertAlmostEqual(reported/numerical, 1., delta=1e-4)


if __name__ == '__main__':
    unittest.main()
