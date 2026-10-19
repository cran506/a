"""作业 6-3 的机理与独立数值校验，不对照图片反推响应曲线。"""
import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from deadzone_model import DeadzoneVelocityPlant, deadzone
from model_assisted_controller import ModelAssistedController, inverse_deadzone
from input_deadzone_model import InputDeadzoneVelocityPlant


class DeadzoneMechanismTests(unittest.TestCase):
    def test_input_deadzone_cascade_matches_independent_ode_and_split_steps(self):
        plant, split = InputDeadzoneVelocityPlant(), InputDeadzoneVelocityPlant()
        initial = np.array([.21,.003])
        plant.reset(*initial); split.reset(*initial)
        for command, duration in ((1.4,.001),(-1.2,.003),(.3,.002)):
            drive = command-.5 if command>.5 else command+.5 if command<-.5 else 0.
            reference = solve_ivp(lambda t,s: [(drive-s[0])/.0008,(.01*s[0]-s[1])/.012],
                (0.,duration), initial, method='DOP853',rtol=1e-12,atol=1e-14)
            plant.step(command,duration)
            split.step(command,duration/2);split.step(command,duration/2)
            actual = [plant.valve_voltage_v,plant.velocity_m_s]
            np.testing.assert_allclose(actual,reference.y[:,-1],atol=2e-12,rtol=1e-10)
            np.testing.assert_allclose(actual,[split.valve_voltage_v,split.velocity_m_s],atol=1e-14,rtol=1e-12)
            initial = reference.y[:,-1]

    def test_dynamic_residual_is_causally_delayed_and_has_zero_dc_gain(self):
        config = dict(gain_m_s_v=.01,tau_velocity_s=.012,washout_gain=17.,washout_tau_s=.002,
                      washout_order=2,feedback_delay_samples=6)
        c = ModelAssistedController(dict(kp=0.,ts=.001),config)
        for k in range(7):
            c.update([0.,.001,0.],[0.,0.])
            self.assertAlmostEqual(c.last['delayed_error_equivalent_v'],0. if k<6 else .1)
            if k<6:self.assertEqual(c.last['washout_feedback_voltage_v'],0.)
        self.assertGreater(abs(c.last['washout_feedback_voltage_v']),0.)
        for _ in range(300):c.update([0.,.001,0.],[0.,0.])
        self.assertLess(abs(c.last['washout_feedback_voltage_v']),1e-12)

    def test_feedback_counterfactual_preserves_memory_and_disables_all_contributions(self):
        config = dict(gain_m_s_v=.01,tau_velocity_s=.012,washout_gain=17.,washout_tau_s=.002,
                      washout_order=2,feedback_delay_samples=6)
        c = ModelAssistedController(dict(kp=1.,ki=2.,ts=.001),config,mode='hard')
        for _ in range(10):c.update([0.,.001,0.],[0.,0.])
        self.assertGreater(c.lowpass1,0.)
        c.feedback_enabled=False
        voltage, ff, residual=c.update([0.,.001,0.],[0.,0.])
        self.assertGreater(c.lowpass1,0.)
        self.assertEqual(len(c.error_history),7)
        self.assertEqual(c.last['feedback_linear_v'],0.)
        self.assertAlmostEqual(voltage,ff)
        self.assertAlmostEqual(residual,0.)

    def test_real_deadzone_handles_scalar_array_and_both_boundaries(self):
        values = np.array([-1., -.5, -.1, 0., .1, .5, 1.])
        expected = np.array([-.5, 0., 0., 0., 0., 0., .5])
        np.testing.assert_array_equal(deadzone(values), expected)
        self.assertIsInstance(deadzone(.3), float)
        self.assertEqual(deadzone(.5), 0.)
        self.assertEqual(deadzone(-.5), 0.)
        np.testing.assert_allclose(deadzone(-values), -deadzone(values), atol=0.)

    def test_hard_undercompensation_leaves_a_real_point_one_volt_gap(self):
        linear = np.array([-.101, -.099, 0., .099, .101])
        compensated = inverse_deadzone(linear, .4, mode='hard')
        np.testing.assert_allclose(deadzone(compensated), [-.001, 0., 0., 0., .001], atol=1e-15)
        self.assertEqual(inverse_deadzone(0., .4, mode='hard'), 0.)
        self.assertEqual(inverse_deadzone(1e-12, .4, mode='hard'), 0.)

    def test_hard_overcompensation_creates_opposite_nonzero_effective_limits(self):
        epsilon = 1e-7  # Above the declared zero-protection epsilon.
        right = deadzone(inverse_deadzone(epsilon, .6, mode='hard'))
        left = deadzone(inverse_deadzone(-epsilon, .6, mode='hard'))
        self.assertAlmostEqual(right, .1+epsilon, delta=1e-15)
        self.assertAlmostEqual(left, -.1-epsilon, delta=1e-15)
        self.assertGreater(right-left, .2)
        self.assertEqual(inverse_deadzone(0., .6, mode='hard'), 0.)
        linear = np.array([-2., -.02, .02, 2.])
        np.testing.assert_allclose(deadzone(inverse_deadzone(linear, .5, mode='hard')),
                                   linear, atol=1e-15)

    def test_finite_tanh_slope_remains_continuous_but_leaves_a_residual_deadzone(self):
        roots = []
        for beta in (6., 18., 60.):
            with self.subTest(beta=beta):
                boundary = brentq(lambda value: value+.6*math.tanh(beta*value)-.5, 0., .5)
                roots.append(boundary)
                self.assertGreater(boundary, 0.)
                self.assertEqual(deadzone(inverse_deadzone(.99*boundary, .6, 'tanh', beta)), 0.)
                self.assertGreater(deadzone(inverse_deadzone(1.01*boundary, .6, 'tanh', beta)), 0.)
                epsilon = 1e-7
                right = inverse_deadzone(epsilon, .6, 'tanh', beta)
                left = inverse_deadzone(-epsilon, .6, 'tanh', beta)
                self.assertAlmostEqual((right-left)/(2*epsilon), 1.+.6*beta, delta=1e-8)
                self.assertLess(abs(right-left), 1e-4)
        self.assertGreater(roots[0], roots[1])
        self.assertGreater(roots[1], roots[2])

    def test_segmented_solution_matches_independent_high_accuracy_ode(self):
        # The first case crosses +0.5 V and -0.5 V in the same ZOH interval.
        # Equal/nearly equal time constants also test the convolution limit.
        cases = ((.0008, .012, .004), (.012, .012, .04),
                 (.012*(1.+1e-10), .012, .04))
        for tau_valve, tau_velocity, duration in cases:
            with self.subTest(tau_valve=tau_valve, tau_velocity=tau_velocity):
                command, valve0, velocity0, gain = -1.4, 1.7, .003, .01
                plant = DeadzoneVelocityPlant(gain_m_s_v=gain,
                    tau_valve_s=tau_valve, tau_velocity_s=tau_velocity)
                plant.reset(valve0, velocity0)

                def rhs(time, state):
                    valve, velocity = state
                    # Independent piecewise physical equation, without using
                    # the production deadzone helper or integration routine.
                    effective = valve-.5 if valve > .5 else valve+.5 if valve < -.5 else 0.
                    return [(command-valve)/tau_valve, (gain*effective-velocity)/tau_velocity]

                numerical = solve_ivp(rhs, (0., duration), [valve0, velocity0], method='DOP853',
                    rtol=1e-11, atol=[1e-12, 1e-14], max_step=duration/400)
                self.assertTrue(numerical.success)
                plant.step(command, duration)
                np.testing.assert_allclose([plant.valve_voltage_v, plant.velocity_m_s],
                    numerical.y[:, -1], rtol=2e-8, atol=2e-11)
                expected_valve = command+(valve0-command)*math.exp(-duration/tau_valve)
                self.assertAlmostEqual(plant.valve_voltage_v, expected_valve, delta=2e-12)

    def test_exact_zoh_has_half_step_equivalence_across_both_deadzone_edges(self):
        whole, halves = DeadzoneVelocityPlant(), DeadzoneVelocityPlant()
        whole.reset(.9, .001)
        halves.reset(.9, .001)
        for command, duration in ((-1.1, .003), (1.4, .004), (.3, .001), (-.5, .001)):
            whole.step(command, duration)
            halves.step(command, duration/2)
            halves.step(command, duration/2)
            np.testing.assert_allclose([whole.valve_voltage_v, whole.velocity_m_s, whole.time_s],
                [halves.valve_voltage_v, halves.velocity_m_s, halves.time_s], atol=2e-14, rtol=1e-12)

    def test_plant_is_left_right_symmetric_without_injected_velocity(self):
        positive, negative = DeadzoneVelocityPlant(), DeadzoneVelocityPlant()
        positive.reset(.8, .002)
        negative.reset(-.8, -.002)
        for command in (1.3, -.9, .1, -1.7, .55):
            positive.step(command, .001)
            negative.step(-command, .001)
            self.assertAlmostEqual(positive.valve_voltage_v, -negative.valve_voltage_v, delta=1e-14)
            self.assertAlmostEqual(positive.velocity_m_s, -negative.velocity_m_s, delta=1e-14)
        stopped = DeadzoneVelocityPlant()
        for command in (.4, -.3, .49):
            self.assertEqual(stopped.step(command, .003), 0.)

    def test_zero_time_constant_valve_and_blocked_valve_have_expected_velocity(self):
        gain, tau, duration = .01, .012, .003
        instantaneous = DeadzoneVelocityPlant(gain_m_s_v=gain, tau_valve_s=0., tau_velocity_s=tau)
        instantaneous.reset(-2., .004)
        actual = instantaneous.step(1.2, duration)
        expected = .004*math.exp(-duration/tau)+gain*.7*(1-math.exp(-duration/tau))
        self.assertAlmostEqual(actual, expected, delta=1e-15)
        self.assertEqual(instantaneous.valve_voltage_v, 1.2)
        blocked = DeadzoneVelocityPlant()
        blocked.reset(.1, .004)
        self.assertAlmostEqual(blocked.step(.25, duration), .004*math.exp(-duration/tau), delta=1e-15)

    def test_feedback_corrected_signal_enters_the_inverse_map_before_sign_choice(self):
        controller = ModelAssistedController(dict(kp=1., ts=.001),
            dict(gain_m_s_v=.01, tau_velocity_s=.012), estimate_v=.6, mode='hard')
        # Positive desired speed gives +0.05 V feedforward.  Faster measured
        # speed contributes -0.15 V feedback, so total linear demand is -0.10 V.
        voltage, feedforward, residual = controller.update([0., .0005, 0.], [0., .002])
        self.assertAlmostEqual(controller.last['linear_voltage_v'], -.1, delta=1e-15)
        self.assertAlmostEqual(voltage, -.7, delta=1e-15)
        self.assertAlmostEqual(feedforward, .65, delta=1e-15)
        self.assertAlmostEqual(feedforward+residual, voltage, delta=1e-15)
        self.assertGreater(feedforward, 0.)
        self.assertLess(voltage, 0.)

    def test_acceleration_feedforward_and_nonlinear_residual_decompose_applied_output(self):
        for mode in ('hard', 'tanh'):
            with self.subTest(mode=mode):
                controller = ModelAssistedController(dict(kp=2., ts=.001),
                    dict(gain_m_s_v=.01, tau_velocity_s=.012), estimate_v=.6, mode=mode, beta=18.)
                voltage, ff, residual = controller.update([0., .001, .002], [0., .0003])
                self.assertAlmostEqual(controller.last['feedforward_linear_v'], .1024, delta=1e-15)
                self.assertAlmostEqual(controller.last['feedback_linear_v'], .14, delta=1e-15)
                self.assertAlmostEqual(controller.last['linear_voltage_v'], .2424, delta=1e-15)
                self.assertAlmostEqual(voltage, ff+residual, delta=1e-15)
                if mode == 'hard':
                    self.assertAlmostEqual(voltage, .8424, delta=1e-15)
                else:
                    self.assertGreater(abs(residual-.14), 1e-3)
                # Saturation belongs to the controller, not the plant or plot.
                voltage, ff, residual = controller.update([0., 1., 0.], [0., 0.])
                self.assertEqual(voltage, 10.)
                self.assertAlmostEqual(voltage, ff+residual, delta=1e-12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
