"""第5章数值与接口验证；运行：python -m unittest discover -s chapter5 -v。"""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from controllers import DigitalPID, PositionalPID
from trajectory import generate_scurve, time_grid
from twin_adapter import ClosedLoopTwin, QuantizedSensor


class TrajectoryTests(unittest.TestCase):
    def test_exact_sampling_and_motion_endpoint(self):
        t, position, velocity, acceleration = generate_scurve()
        self.assertEqual(len(t), 1401)
        np.testing.assert_array_equal(t, np.arange(1401) * .001)
        self.assertEqual(t[1000], 1.)
        self.assertAlmostEqual(t[-1], 1.4, places=14)
        np.testing.assert_array_equal([position[0], velocity[0], acceleration[0]], [0., 0., 0.])
        np.testing.assert_array_equal(position[1000:], np.full(401, .1))
        np.testing.assert_array_equal(velocity[1000:], np.zeros(401))
        np.testing.assert_array_equal(acceleration[1000:], np.zeros(401))
        self.assertTrue(np.all(np.diff(position) >= -1e-14))

    def test_analytic_peaks_and_cruise_boundaries(self):
        _, position, velocity, acceleration = generate_scurve()
        # 对称七段规划：vmax = D / (T - Ta)，amax = vmax / (Tj + Tca)。
        maximum_velocity = .1 / .65
        maximum_acceleration = maximum_velocity / .25
        self.assertAlmostEqual(float(np.max(velocity)), maximum_velocity, places=12)
        self.assertAlmostEqual(float(np.max(acceleration)), maximum_acceleration, places=12)
        self.assertAlmostEqual(float(np.min(acceleration)), -maximum_acceleration, places=12)
        self.assertAlmostEqual(position[350], .35 * maximum_velocity / 2, places=12)
        self.assertAlmostEqual(position[650], .1 - position[350], places=12)
        np.testing.assert_allclose(velocity[350:651], maximum_velocity, rtol=0, atol=1e-12)
        np.testing.assert_allclose(acceleration[350:651], 0., rtol=0, atol=1e-12)

    def test_derivative_consistency_and_continuity_at_all_seven_boundaries(self):
        dt = .0001
        _, position, velocity, acceleration = generate_scurve(T_total=1.1, dt=dt)
        maximum_velocity = .1 / .65
        maximum_acceleration = maximum_velocity / .25
        maximum_jerk = maximum_acceleration / .1
        # 独立有限差分检查各数组确为同一轨迹的连续导数。
        np.testing.assert_allclose(np.gradient(position, dt)[1:-1], velocity[1:-1], rtol=0, atol=2e-7)
        np.testing.assert_allclose(np.gradient(velocity, dt)[1:-1], acceleration[1:-1], rtol=0, atol=2e-4)
        self.assertLessEqual(float(np.max(np.abs(np.diff(acceleration)))) / dt, maximum_jerk + 1e-8)
        for boundary in (.1, .25, .35, .65, .75, .9, 1.):
            index = round(boundary / dt)
            with self.subTest(boundary=boundary):
                # 允许跃度跳变；位置、速度、加速度跨边界变化仍受各自导数约束。
                self.assertLessEqual(abs(position[index + 1] - position[index - 1]), 2 * dt * maximum_velocity + 1e-12)
                self.assertLessEqual(abs(velocity[index + 1] - velocity[index - 1]), 2 * dt * maximum_acceleration + 1e-12)
                self.assertLessEqual(abs(acceleration[index + 1] - acceleration[index - 1]), 2 * dt * maximum_jerk + 1e-12)

    def test_distance_and_duration_scaling_and_zero_distance(self):
        t, position, velocity, acceleration = generate_scurve()
        t2, position2, velocity2, acceleration2 = generate_scurve(dist=-.2, T_move=2., T_total=2.8, dt=.002)
        np.testing.assert_allclose(t2, 2 * t, rtol=0, atol=1e-14)
        np.testing.assert_allclose(position2, -2 * position, rtol=0, atol=1e-14)
        np.testing.assert_allclose(velocity2, -velocity, rtol=0, atol=1e-14)
        np.testing.assert_allclose(acceleration2, -.5 * acceleration, rtol=0, atol=1e-14)
        for values in generate_scurve(dist=0.)[1:]:
            np.testing.assert_array_equal(values, np.zeros_like(values))

    def test_invalid_sampling_and_durations_are_rejected(self):
        for duration, dt in ((1., 0.), (0., .001), (1., .003), (math.inf, .001)):
            with self.subTest(duration=duration, dt=dt), self.assertRaises(ValueError):
                time_grid(duration, dt)
        with self.assertRaises(ValueError):
            generate_scurve(T_move=2., T_total=1.)


class PIDTests(unittest.TestCase):
    def test_position_and_incremental_forms_are_equal_without_saturation(self):
        for alpha in (0., .85):
            for separation in (False, True):
                with self.subTest(alpha=alpha, integral_separation=separation):
                    incremental = DigitalPID(4.2, 22., .06, .001, alpha=alpha, integral_separation=separation)
                    positional = PositionalPID(4.2, 22., .06, .001, alpha=alpha, integral_separation=separation)
                    for sample in range(1200):
                        t = sample * .001
                        target = .0003 * math.sin(9 * t)
                        feedback = .0001 * math.cos(13 * t)
                        ui, up = incremental.update(target, feedback), positional.update(target, feedback)
                        self.assertLess(abs(incremental.last_raw), 10.)
                        self.assertLess(abs(positional.last_raw), 10.)
                        self.assertAlmostEqual(ui, up, places=11)

    def test_homework_first_step_has_known_derivative_kick(self):
        for controller in (DigitalPID, PositionalPID):
            for alpha, expected_raw, expected_d in ((0., 642.22, 600.), (.85, 132.22, 90.)):
                with self.subTest(controller=controller.__name__, alpha=alpha):
                    pid = controller(4.2, 22., .06, .001, alpha=alpha, integral_separation=False)
                    self.assertEqual(pid.update(.1, 0.), 10.)
                    self.assertAlmostEqual(pid.last_raw, expected_raw, places=10)
                    self.assertAlmostEqual(pid.last_d, expected_d, places=10)
                    self.assertAlmostEqual(pid.last_i, .22, places=12)
                    self.assertEqual(pid.u_prev, 10.)

    def test_filtered_derivative_impulse_is_fifteen_percent_and_decays(self):
        filtered = DigitalPID(0., 0., .06, .001, out_min=-1000., out_max=1000., alpha=.85)
        unfiltered = DigitalPID(0., 0., .06, .001, out_min=-1000., out_max=1000., alpha=0.)
        filtered.update(.01, 0.)
        unfiltered.update(.01, 0.)
        self.assertAlmostEqual(unfiltered.last_d, 60., places=12)
        self.assertAlmostEqual(filtered.last_d / unfiltered.last_d, .15, places=12)
        for index in range(1, 15):
            filtered.update(.01, 0.)
            self.assertAlmostEqual(filtered.last_d, 9. * .85**index, places=11)

    def test_clipped_memory_prevents_unbounded_integral_state(self):
        incremental = DigitalPID(0., 1000., 0., .001, integral_separation=False)
        positional = PositionalPID(0., 1000., 0., .001, integral_separation=False)
        for _ in range(12):
            self.assertEqual(incremental.update(.1, 0.), 10.)
            self.assertEqual(positional.update(.1, 0.), 10.)
            self.assertEqual(incremental.u_prev, 10.)
        self.assertEqual(incremental.last_raw, 20.)
        self.assertEqual(positional.i_total, 120.)
        self.assertEqual(positional.last_raw, 120.)
        # 使用纯积分器隔离饱和记忆，不让微分冲击掩盖 windup。
        self.assertEqual(incremental.update(-.1, 0.), 0.)
        self.assertEqual(incremental.u_prev, 0.)
        for _ in range(11):
            self.assertEqual(positional.update(-.1, 0.), 10.)
        self.assertEqual(positional.update(-.1, 0.), 0.)
        for _ in range(12):
            incremental.update(-.1, 0.)
        self.assertEqual(incremental.u_prev, -10.)

    def test_integral_separation_is_explicit_and_disabled_for_homework(self):
        for controller in (DigitalPID, PositionalPID):
            separated = controller(0., 22., 0., .001, integral_separation=True)
            full = controller(0., 22., 0., .001, integral_separation=False)
            separated.update(.1, 0.)
            full.update(.1, 0.)
            self.assertAlmostEqual(separated.last_i, .011, places=12)
            self.assertAlmostEqual(full.last_i, .22, places=12)
            separated.reset()
            separated.update(.00149, 0.)
            self.assertAlmostEqual(separated.last_i, 22. * .149 * .001, places=12)
            separated.reset()
            separated.update(.0015, 0.)
            self.assertAlmostEqual(separated.last_i, 22. * .15 * .001 * .05, places=12)

    def test_zero_gain_and_reset(self):
        for controller in (DigitalPID, PositionalPID):
            pid = controller(0., 0., 0., .001, integral_separation=False)
            for target, feedback in ((.1, 0.), (-.1, 0.), (.2, .3), (0., 0.)):
                self.assertEqual(pid.update(target, feedback), 0.)
                self.assertEqual(pid.last_raw, 0.)
            pid.reset()
            self.assertEqual((pid.e_prev, pid.d_filt_prev, pid.u_prev, pid.last_raw, pid.last_d, pid.last_i), (0.,) * 6)
        positional = PositionalPID(0., 10., 0., .001, integral_separation=False)
        positional.update(.1, 0.)
        self.assertGreater(positional.i_total, 0.)
        positional.reset()
        self.assertEqual(positional.i_total, 0.)

    def test_invalid_parameters_and_nonfinite_samples_are_rejected(self):
        for kwargs in ({"ts": 0.}, {"alpha": 1.}, {"alpha": -.1}, {"out_min": 10.}, {"kd": math.nan}):
            parameters = dict(kp=4.2, ki=22., kd=.06, ts=.001)
            parameters.update(kwargs)
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                DigitalPID(**parameters)
        for controller in (DigitalPID, PositionalPID):
            pid = controller(4.2, 22., .06, .001)
            for target, feedback in ((math.nan, 0.), (0., math.inf)):
                with self.assertRaises(ValueError):
                    pid.update(target, feedback)


class SensorTests(unittest.TestCase):
    def test_adc_resolution_rounding_and_clipping(self):
        sensor = QuantizedSensor()
        self.assertEqual(sensor.levels, 4096)
        self.assertAlmostEqual(sensor.lsb * 1000, .0732421875, places=12)
        self.assertEqual(sensor.sample(-1.), 0.)
        self.assertEqual(sensor.sample(.3), 4095 * sensor.lsb)
        self.assertEqual(sensor.sample(1.), 4095 * sensor.lsb)
        for value, code in ((.24, 0), (.76, 1), (2.49, 2), (2.51, 3)):
            self.assertEqual(sensor.sample(value * sensor.lsb), code * sensor.lsb)
        for position in np.linspace(0., 4095 * sensor.lsb, 1001):
            self.assertLessEqual(abs(sensor.sample(position) - position), sensor.lsb / 2 + 1e-15)

    def test_sensor_configuration_validation(self):
        for kwargs in ({"bits": 0}, {"bits": 25}, {"span": 0.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QuantizedSensor(**kwargs)


class TwinAdapterTests(unittest.TestCase):
    @staticmethod
    def frozen_solver(fun, t_span, y0, **_kwargs):
        # 调用区间两端的微分函数，但不移动状态；隔离接口/切换逻辑与刚性求解误差。
        y0 = np.asarray(y0, dtype=float)
        fun(t_span[0], y0)
        fun(t_span[1], y0)
        return SimpleNamespace(success=True, y=y0[:, None], nfev=2, message="isolated solver")

    def test_voltage_is_clipped_and_normalized_before_plant(self):
        for voltage, normalized in ((5., .5), (-5., -.5), (10., 1.), (-10., -1.), (30., 1.), (-30., -1.)):
            with self.subTest(voltage=voltage):
                twin = ClosedLoopTwin(load_force=0., valve_tau=0.)
                observed = []
                twin._dynamics = lambda t, y, u: observed.append(u) or np.zeros(4)
                with patch("twin_adapter.solve_ivp", side_effect=self.frozen_solver):
                    state = twin.step_voltage(voltage, .001)
                self.assertEqual(observed, [normalized, normalized])
                self.assertEqual(twin.valve_state, normalized)
                self.assertEqual(state.shape, (4,))
                self.assertAlmostEqual(twin.t, .001)

    def test_actual_radau_step_matches_first_order_valve_solution(self):
        twin = ClosedLoopTwin(load_force=0., valve_tau=.015)
        state = twin.step_voltage(5., .001)
        expected_valve = .5 * (1 - math.exp(-.001 / .015))
        self.assertAlmostEqual(twin.valve_state, expected_valve, delta=2e-9)
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertTrue(np.all(state[2:] > 0.))
        self.assertGreater(twin.nfev, 0)

    def test_load_window_is_half_open_and_latched_for_whole_sample(self):
        twin = ClosedLoopTwin(load_force=180000., valve_tau=0.)
        observed = []
        twin._dynamics = lambda t, y, u: observed.append(twin.active_force) or np.zeros(4)
        with patch("twin_adapter.solve_ivp", side_effect=self.frozen_solver):
            for start, expected in ((.349, 0.), (.350, 180000.), (.699, 180000.), (.700, 0.)):
                with self.subTest(start=start):
                    twin.t = start
                    observed.clear()
                    twin.step_voltage(0., .001)
                    self.assertEqual(observed, [expected, expected])
                    self.assertEqual(twin.active_force, expected)

    def test_load_enters_newton_equation_with_correct_units_and_sign(self):
        twin = ClosedLoopTwin(load_force=180000., valve_tau=0.)
        state = np.array([.05, 0., twin.p['Pr'], twin.p['Pr']])
        twin.active_force = 0.
        unloaded = twin._dynamics(0., state, 0.)
        twin.active_force = 180000.
        loaded = twin._dynamics(0., state, 0.)
        self.assertAlmostEqual(loaded[1] - unloaded[1], -180000. / twin.p['mass'], places=12)
        np.testing.assert_array_equal(loaded[[0, 2, 3]], unloaded[[0, 2, 3]])

    def test_reversed_pressure_difference_drives_return_flow(self):
        twin = ClosedLoopTwin(load_force=0., valve_tau=0.)
        twin.p['C_ip'] = twin.p['C_ep'] = 0.
        # 压力超过供油源时应回流，不能让 abs(压差) 强迫油继续流入高压腔。
        extension = twin._dynamics(0., np.array([.05, 0., twin.p['Ps'] + 1e6, twin.p['Pr']]), .5)
        retraction = twin._dynamics(0., np.array([.05, 0., twin.p['Pr'], twin.p['Ps'] + 1e6]), -.5)
        self.assertLess(extension[2], 0.)
        self.assertLess(retraction[3], 0.)

    def test_solver_failure_and_invalid_pressure_are_not_silently_logged(self):
        twin = ClosedLoopTwin(load_force=0., valve_tau=0.)
        failed = SimpleNamespace(success=False, y=twin.state[:, None], nfev=1, message="test failure")
        with patch("twin_adapter.solve_ivp", return_value=failed), self.assertRaisesRegex(RuntimeError, "test failure"):
            twin.step_voltage(0., .001)
        bad_state = twin.state.copy()
        bad_state[2] = -1.
        invalid = SimpleNamespace(success=True, y=bad_state[:, None], nfev=1, message="invalid pressure")
        with patch("twin_adapter.solve_ivp", return_value=invalid), self.assertRaisesRegex(RuntimeError, "绝对压力非正"):
            twin.step_voltage(0., .001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
