"""Mechanism checks for chapter 7, without running a full positioning episode.

Run: python -m unittest -v test_chapter7.py
The small linear fixture has a known absolute-pressure response.  It tests
constraint mapping independently of the hydraulic model and its Jacobian.
"""
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'chapter7-mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(Path(tempfile.gettempdir())/'chapter7-cache'))

import numpy as np

from hydraulic_plant import GreyBoxModel, HydraulicPlant, STATE_UNITS
from mpc_core import HydraulicMPC, MPCSolveError, augment_incremental_model, condense_affine


class KnownLinearModel:
    """Five-state model with analytically known 0.02 MPa/V pressure change."""
    parameters = {'deadzone': .05}

    def linearize_discrete(self, state, command, dt):
        a = np.eye(5)
        a[4, 4] = 0.
        return a, np.array([.1, 0., 0., .02, 1.]), np.zeros(5)

    def rollout(self, state, commands, dt):
        state = np.asarray(state).copy()
        result = []
        for command in commands:
            state = np.array([state[0]+.1*command, state[1], state[2],
                              state[3]+.02*command, command])
            result.append(state.copy())
        return np.asarray(result)


class ChapterSevenTests(unittest.TestCase):
    def test_education_units_and_exact_valve_zoh_without_false_motion(self):
        plant = HydraulicPlant()
        initial = np.array([12., 0., .1, .1, 0.])
        np.testing.assert_allclose(plant.reset(initial), initial, atol=1e-12)
        self.assertEqual(STATE_UNITS, ('mm', 'm/s', 'MPa', 'MPa', 'V'))
        self.assertAlmostEqual(plant.twin.state[0], plant.origin_m+.012)
        self.assertAlmostEqual(plant.twin.state[2], initial[2]*1e6)
        command, dt = .25, .001  # Entire interval stays inside the real deadzone.
        actual = plant.step(command, dt)
        expected = command*(1-np.exp(-dt/plant.parameters['valve_tau']))
        self.assertAlmostEqual(actual[4], expected, delta=2e-7)
        self.assertAlmostEqual(plant.twin.valve_state, actual[4]/10., delta=1e-12)
        np.testing.assert_allclose(actual[:2], initial[:2], atol=1e-12, rtol=0.)
        self.assertEqual(actual[1], 0.)

    def test_precharged_pressure_can_change_while_static_friction_holds(self):
        plant = HydraulicPlant()
        p = plant.parameters
        p2 = 2e6
        force = .5*p['Fs']
        p1 = (p2*p['A2']+p['ambient_pressure']*(p['A1']-p['A2'])+force)/p['A1']
        initial = plant.reset([0., 0., p1/1e6, p2/1e6, 0.])
        self.assertAlmostEqual(plant.twin.net_drive_force(), force, delta=1e-7)
        actual = plant.step(0., .003)
        np.testing.assert_array_equal(actual[:2], initial[:2])
        self.assertTrue(plant.twin.sticking)
        self.assertEqual(plant.friction_events, [])
        self.assertGreater(np.max(abs(actual[2:4]-initial[2:4])), 1e-5)
        self.assertTrue(np.all(plant.internal_state_trace[:, 2] == 0.))
        self.assertLess(abs(plant.twin.net_drive_force()), p['Fs'])

    def test_jacobian_is_the_actual_rhs_and_contains_physical_flow_gain(self):
        model = GreyBoxModel()
        state, command = np.array([20., .15, 7., 1.8, 4.]), 2.
        a, b, c = model.continuous_linearization(state, command)
        p = model.parameters
        self.assertAlmostEqual(a[0, 1], 1000., delta=1e-6)
        self.assertAlmostEqual(a[1, 2], 1e6*p['A1']/p['mass'], delta=1e-7)
        self.assertAlmostEqual(a[1, 3], -1e6*p['A2']/p['mass'], delta=1e-7)
        self.assertAlmostEqual(b[4, 0], 1/p['valve_tau'])
        np.testing.assert_allclose(a@state+b[:, 0]*command+c,
                                   model.rhs(state, command), atol=1e-8)
        # Independent hydraulic derivation on this positive, open valve branch.
        pressure_pa = state[2]*1e6
        kq_voltage = p['Cd']*p['w']*p['xv_max']/10*np.sqrt(2*(p['Ps']-pressure_pa)/p['rho'])
        volume = p['V0_1']+p['A1']*(model.origin_m+state[0]/1000)
        expected_pressure_gain = model.model._get_beta(pressure_pa)/volume/1e6*kq_voltage
        self.assertAlmostEqual(a[2, 4], expected_pressure_gain,
                               delta=abs(expected_pressure_gain)*1e-5)
        self.assertAlmostEqual(model.get_jacobian_coeffs(state)['Kq1_voltage_m3_s_v'],
                               kq_voltage, delta=kq_voltage*1e-12)
        # A second, independently sized directional perturbation checks all columns.
        perturbation = np.array([.002, 2e-6, 3e-5, -2e-5, 4e-5])
        measured_change = (model.rhs(state+perturbation, command+3e-5)
                           -model.rhs(state-perturbation, command-3e-5))/2
        np.testing.assert_allclose(measured_change, a@perturbation+b[:, 0]*3e-5,
                                   rtol=2e-5, atol=2e-7)

    def test_affine_zoh_preserves_independent_exact_valve_transition(self):
        model = GreyBoxModel()
        state, command, dt = np.array([20., .15, 7., 1.8, 4.]), 2., .001
        ad, bd, cd = model.linearize_discrete(state, command, dt)
        predicted = ad@state+bd[:, 0]*command+cd
        expected_valve = command+(state[4]-command)*np.exp(-dt/model.valve_tau)
        self.assertAlmostEqual(predicted[4], expected_valve, delta=1e-10)
        self.assertEqual(ad.shape, (5, 5))
        self.assertEqual(bd.shape, (5, 1))
        self.assertTrue(np.all(np.isfinite(predicted)))

    def test_time_varying_affine_condensation_matches_physical_recursion(self):
        rng = np.random.default_rng(7)
        physical_models, augmented_models = [], []
        for _ in range(4):
            a = np.eye(5)+rng.normal(0., .02, (5, 5))
            b, c = rng.normal(0., .03, 5), rng.normal(0., .1, 5)
            physical_models.append((a, b, c))
            augmented_models.append(augment_incremental_model(a, b, c))
        state, previous = rng.normal(size=5), 1.4
        initial = np.r_[state, previous]
        increments = rng.normal(size=4)
        independently_computed = []
        for (a, b, c), increment in zip(physical_models, increments):
            previous += increment
            state = a@state+b*previous+c
            independently_computed.extend(np.r_[state, previous])
        f, phi, offset = condense_affine(augmented_models)
        np.testing.assert_allclose(f@initial+phi@increments+offset,
                                   independently_computed, atol=1e-12)
        self.assertEqual(f.shape, (24, 6))
        self.assertEqual(phi.shape, (24, 4))

    def test_absolute_pressure_constraint_and_fresh_feedback_with_warm_start(self):
        for full in (False, True):
            with self.subTest(full=full):
                model = KnownLinearModel()
                controller = HydraulicMPC(model=model, enable_pressure_constraint=full)
                state, previous = np.array([0., 0., .1, 20.4, 0.]), 0.
                for _ in range(2):
                    command = controller.solve(state, previous, 50.)
                    self.assertLessEqual(abs(command), controller.u_max)
                    self.assertLessEqual(abs(command-previous), controller.delta_u_max+1e-12)
                    prediction = np.asarray(controller.last['prediction'])
                    np.testing.assert_allclose(prediction,
                        model.rollout(state, controller.last['command_sequence_v'], controller.dt), atol=1e-12)
                    if full:
                        self.assertLessEqual(np.max(prediction[:, 3]),
                                             controller.p_safe-controller.pressure_margin+controller.nonlinear_tolerance)
                    state = model.rollout(state, [command], controller.dt)[0]
                    previous = command
                self.assertEqual(controller.last['status'], 'solved')
                self.assertLessEqual(controller.last['linear_constraint_violation'], controller.primal_tolerance)

    def test_infeasible_pressure_problem_reports_failure_without_held_command(self):
        controller = HydraulicMPC(model=KnownLinearModel())
        with self.assertRaises(MPCSolveError):
            controller.solve(np.array([0., 0., .1, 22., 0.]), 0., 50.)
        self.assertEqual(controller.last['status'], 'failed')
        self.assertIn('infeasible', controller.last['osqp_status'].lower())
        self.assertIsNone(controller.nominal_sequence)

    def test_wrong_pressure_jacobian_is_rejected_by_nonlinear_validation(self):
        class WrongPressureDerivative(KnownLinearModel):
            def linearize_discrete(self, state, command, dt):
                a, b, c = super().linearize_discrete(state, command, dt)
                b[3] = 0.
                return a, b, c

            def rollout(self, state, commands, dt):
                state, output = state.copy(), []
                for command in commands:
                    state = np.array([state[0]+.1*command, state[1], state[2],
                                      state[3]+.02*abs(command), command])
                    output.append(state.copy())
                return np.asarray(output)

        controller = HydraulicMPC(model=WrongPressureDerivative())
        with self.assertRaises(MPCSolveError):
            controller.solve(np.array([0., 0., .1, 20.4, 0.]), 0., 50.)
        self.assertEqual(controller.last['status'], 'failed')
        self.assertGreater(controller.last['prediction_violation'], 0.)
        self.assertIsNone(controller.nominal_sequence)

    def test_short_real_startup_has_bounded_incremental_commands(self):
        for full in (False, True):
            with self.subTest(full=full):
                plant, model = HydraulicPlant(), GreyBoxModel()
                controller = HydraulicMPC(model=model, enable_pressure_constraint=full)
                previous = 0.
                for _ in range(3):
                    command = controller.solve(plant.state, previous, 50.)
                    self.assertLessEqual(abs(command), controller.u_max)
                    self.assertLessEqual(abs(command-previous), controller.delta_u_max+1e-12)
                    actual = plant.step(command, controller.dt)
                    self.assertTrue(np.all(np.isfinite(actual)))
                    self.assertGreater(min(actual[2:4]), 0.)
                    self.assertIn(controller.last['status'], ('solved', 'validated_shifted_plan'))
                    self.assertEqual(len(controller.last['prediction']), controller.horizon)
                    previous = command

    def test_intersample_water_hammer_extremum_is_recorded_without_pressure_clipping(self):
        plant = HydraulicPlant()
        # A physically moving cylinder is braked over one deliberately long
        # observation interval. Its endpoints alone miss the pressure danger.
        initial = plant.reset([0., 1., 1.05, 2., 10.])
        dt = .008
        actual = plant.step(-10., dt)
        trace = plant.last_pressure_trace
        peak_index = int(np.argmax(trace[:, 2]))
        peak = trace[peak_index]
        self.assertGreater(peak[0], 0.)
        self.assertLess(peak[0], dt)
        self.assertGreater(peak[2]/1e6, 21.)
        self.assertGreater(peak[2]/1e6, max(initial[3], actual[3])+.1)
        self.assertAlmostEqual(plant.last_interval_peak['p2_pa'], peak[2], delta=1e-7)
        self.assertAlmostEqual(plant.last_interval_peak['p2_time_s'], peak[0], delta=1e-12)
        self.assertGreater(len(trace), 32)
        self.assertGreater(np.min(trace[:, 1:]), 0.)
        # Check the extremum using the actual physical continuity equation,
        # including the integrated valve state, independently of the peak log.
        state_trace = plant.internal_state_trace
        row = state_trace[np.argmax(state_trace[:, 4])]
        pressure_rate = plant.twin._dynamics(row[0], row[1:5], row[5])[3]
        self.assertAlmostEqual(pressure_rate, 0., delta=100.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
