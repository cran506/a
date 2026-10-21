"""七项必要性质检查：python -m unittest -v test_models。"""
import math
import unittest

import numpy as np

from friction_model import sliding_friction
from model_assisted_controller import ModelAssistedController
from twin_adapter import ClosedLoopTwin, classroom_config


class PhysicalModelTests(unittest.TestCase):
    def setUp(self):
        self.p = classroom_config()
        self.p.update(ambient_pressure=1e5, makeup_area=.001)
        self.pid = dict(kp=4., ki=450., kd=.08, ts=.001, alpha=.85, integral_separation=False)

    def controller(self, **parameters):
        return ModelAssistedController(self.pid, self.p, pressure_alpha=0.,
                                       valve_tau=0., gain_prediction_s=0., **parameters)

    def test_deadzone_inverse_restores_signed_linear_signal(self):
        controller = self.controller(zero_flow_band_v=0.)
        for linear in (-1.3, -.02, 0., .02, 1.3):
            command = controller._deadzone_inverse(linear, moving=linear != 0.)
            effective = math.copysign(max(abs(command) - controller.deadzone_v, 0.), command)
            self.assertAlmostEqual(effective, linear, places=12)

    def test_stribeck_is_odd_and_has_correct_static_limits(self):
        for velocity in (.001, .03, .2):
            self.assertAlmostEqual(sliding_friction(velocity, self.p),
                                   -sliding_friction(-velocity, self.p), places=12)
        self.assertEqual(sliding_friction(0., self.p, direction=1), self.p["Fs"])
        self.assertEqual(sliding_friction(0., self.p, direction=-1), -self.p["Fs"])

    def test_subthreshold_force_has_true_static_equilibrium(self):
        twin = ClosedLoopTwin(self.p, load_force=0., valve_tau=0.)
        state = np.array([.05, 0., self.p["Pr"] + self.p["Fs"] / (2 * self.p["A1"]), self.p["Pr"]])
        twin.reset(state)
        derivative = twin._dynamics(0., state, 0.)
        self.assertEqual(derivative[0], 0.)
        self.assertEqual(derivative[1], 0.)
        self.assertLess(abs(twin.net_drive_force()), self.p["Fs"])
        self.assertAlmostEqual(twin.friction_force, twin.net_drive_force())

    def assert_force_rate_inverse_matches_physical_continuity(self, requested_rate):
        """用独立 Twin RHS 验证逆模型的 N/s 与电压换算，而非回代自身表达式。"""
        controller = self.controller()
        twin = ClosedLoopTwin(self.p, load_force=0., valve_tau=0.)
        state = np.array([.08, .1, 8e6, 3e6])
        controller.pressure_filtered = state[2:].copy()
        linear = controller._inverse_force_rate(state[0], state[1], requested_rate)
        voltage = linear + math.copysign(controller.deadzone_v, linear)
        # Twin 接收归一化阀输入；公开控制器接收 V，10 V 对应归一化1。
        derivative = twin._state_derivative(0., state, voltage / 10., False, 1)
        actual_rate = self.p["A1"] * derivative[2] - self.p["A2"] * derivative[3]
        self.assertAlmostEqual(actual_rate / requested_rate, 1., places=11)
        self.assertGreater(controller.last["inverse_gain_n_s_v"], 0.)

    def test_positive_force_rate_inverse_has_correct_voltage_units(self):
        self.assert_force_rate_inverse_matches_physical_continuity(8e6)

    def test_negative_force_rate_inverse_has_correct_valve_topology(self):
        self.assert_force_rate_inverse_matches_physical_continuity(-8e6)

    def test_inverse_dynamics_counts_viscous_friction_once(self):
        controller = self.controller()
        velocity, acceleration, external = .2, .3, 1000.
        controller.update((0., velocity, acceleration), (0., velocity),
                          controller.nominal_pressures, jerk_d=0., known_force_n=external,
                          origin_m=.01)
        expected = self.p["mass"] * acceleration + sliding_friction(velocity, self.p) + external
        self.assertAlmostEqual(controller.last["planned_inverse_force_n"], expected)
        no_viscosity = dict(self.p, sigma=0.)
        self.assertAlmostEqual(sliding_friction(velocity, self.p) - sliding_friction(velocity, no_viscosity),
                               self.p["sigma"] * velocity)

    def test_startup_gain_uses_current_velocity_and_latches_until_reset(self):
        controller = self.controller(startup_force_gain_s=385., startup_velocity_threshold=1e-4)
        pressure = (self.p["Pr"], self.p["Pr"])
        controller.update((0., 0., 0.), (0., 0.), pressure, jerk_d=10., origin_m=.01)
        self.assertTrue(controller.last["startup_mode"])
        self.assertEqual(controller.last["force_error_gain_s"], 385.)
        controller.update((0., .001, 0.), (0., .001), pressure, jerk_d=0., origin_m=.01)
        self.assertFalse(controller.last["startup_mode"])
        self.assertEqual(controller.last["force_error_gain_s"], controller.force_error_gain_s)
        controller.update((0., 0., 0.), (0., 0.), pressure, jerk_d=0., origin_m=.01)
        self.assertFalse(controller.last["startup_mode"])
        controller.reset()
        self.assertFalse(controller.motion_launched)


if __name__ == "__main__":
    unittest.main()
