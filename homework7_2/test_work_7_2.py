"""Behavioral checks for the Chapter 7-2 horizon experiment.

Run with ``python -m unittest -v test_work_7_2`` from this folder.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ch7_test_mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "ch7_test_cache"))

import numpy as np

from cvxpy_mpc import CVXPYHydraulicMPC
from hydraulic_plant import GreyBoxModel, HydraulicPlant
from mpc_core import HydraulicMPC, MPCSolveError
from safety_restoration import restore_pressure_feasible
from simulation import load_config, simulate


class HorizonExperimentTests(unittest.TestCase):
    def test_declared_auxiliary_circuit_unloads_both_chambers_without_pressure_reset(self):
        # At the 50 mm stop, P1/P2 are force-balanced but a closed-centre valve
        # traps both pressures.  The optional teaching pilot must remove fluid
        # through its explicit modelled paths, leaving position stationary.
        here = Path(__file__).resolve().parent
        auxiliary = json.loads((here / "teaching_plant_config.json").read_text())
        initial = [50.0, 0.0, 7.15, 14.2, 0.0]
        teaching = HydraulicPlant(config=auxiliary)
        legacy = HydraulicPlant()
        teaching.reset(initial)
        legacy.reset(initial)
        for _ in range(20):
            teaching.step(0.0, 0.001)
            legacy.step(0.0, 0.001)
        self.assertLess(abs(teaching.state[0]-initial[0]), 0.05)
        self.assertAlmostEqual(teaching.state[2], 1.05, delta=0.1)
        self.assertAlmostEqual(teaching.state[3], 2.0, delta=0.1)
        self.assertGreater(legacy.state[3], 10.0)
        self.assertGreater(np.min(teaching.pressure_trace[:, 1:]), 0.0)

    def test_terminal_velocity_cost_matches_qp_and_nonlinear_plan_score(self):
        state = np.array([0., 0., 1., 2., 0.])
        reference = np.zeros((2, 2))
        predicted = np.array([[1., .2, 1., 2., 0.],
                              [2., -.4, 1., 2., 0.]])
        commands = np.array([1., -1.])
        common = dict(horizon=2, model=object(), q_position=100.,
                      q_velocity=20., move_weight=.5)
        ordinary = HydraulicMPC(terminal_velocity_factor=1., **common)
        terminal = HydraulicMPC(terminal_velocity_factor=10., **common)
        self.assertAlmostEqual(ordinary._trajectory_cost(
            predicted, reference, commands, 0.), 506.5)
        self.assertAlmostEqual(terminal._trajectory_cost(
            predicted, reference, commands, 0.), 535.3)
        np.testing.assert_allclose(terminal._tracking_weights(),
                                   [[100., 20.], [100., 200.]])

        # A diagonal prediction response makes the terminal QP curvature
        # analytic: v1=delta_u1 and v2=2*delta_u2.
        f = np.zeros((12, 6))
        phi = np.zeros((12, 2))
        phi[1, 0], phi[7, 1] = 1., 2.
        offset = np.zeros(12)
        for controller, final_velocity_weight in ((ordinary, 20.),
                                                  (terminal, 200.)):
            controller._prediction = lambda z, nominal: (f, phi, offset)
            h, gradient, *_ = controller._qp(
                state, 0., reference, np.zeros(2))
            self.assertAlmostEqual(h[0, 0], 2.*(20.+.5))
            self.assertAlmostEqual(h[1, 1], 2.*(4.*final_velocity_weight+.5))
            np.testing.assert_allclose(gradient, 0.)

    def test_terminal_stopping_tail_matches_cost_and_local_qp(self):
        # Use a declared affine prediction so the exact trajectory cost and
        # the local Gauss-Newton QP can be checked independently of the plant.
        common = dict(horizon=3, dt=.001, model=object(), q_position=100.,
                      q_velocity=20., move_weight=.01,
                      terminal_stop_time_s=.008,
                      terminal_stop_smoothing_mm=.5)
        base = HydraulicMPC(**common)
        tail = HydraulicMPC(terminal_tail_weight0=35.,
                            terminal_tail_decay_s=.004, **common)
        longer = HydraulicMPC(horizon=20, dt=.001, model=object(),
                              terminal_tail_weight0=35.,
                              terminal_tail_decay_s=.004)
        expected_weight = 35.*np.exp(-3*.001/.004)
        self.assertAlmostEqual(tail.terminal_tail_weight, expected_weight)
        self.assertLess(longer.terminal_tail_weight, tail.terminal_tail_weight)

        f = np.zeros((18, 6))
        phi = np.zeros((18, 3))
        free = np.zeros(18)
        for stage in range(3):
            free[6*stage] = .4*stage
            free[6*stage+1] = .02
            phi[6*stage, :stage+1] = .11
            phi[6*stage+1, :stage+1] = .015
        for controller in (base, tail):
            controller._prediction = lambda z, nominal: (f, phi, free)

        previous = 0.
        nominal = np.array([.2, .4, .6])
        delta = np.diff(np.r_[previous, nominal])
        reference = np.zeros((3, 2))
        predicted = (free+phi@delta).reshape(3, 6)[:, :5]
        overrun = predicted[-1, 0]+1000.*.008*predicted[-1, 1]
        softplus = .5*np.logaddexp(0., overrun/.5)
        expected_tail_cost = expected_weight*softplus**2
        self.assertAlmostEqual(
            tail._trajectory_cost(predicted, reference, nominal, previous)
            -base._trajectory_cost(predicted, reference, nominal, previous),
            expected_tail_cost)

        state = np.zeros(5)
        base_h, base_g, *_ = base._qp(state, previous, reference, nominal)
        tail_h, tail_g, *_ = tail._qp(state, previous, reference, nominal)
        response = phi[12]+1000.*.008*phi[13]
        slope = 1./(1.+np.exp(-overrun/.5))
        scaled_weight = expected_weight/tail.objective_scale
        expected_hessian = 2.*scaled_weight*slope**2*np.outer(response, response)
        np.testing.assert_allclose(tail_h-base_h, expected_hessian, atol=1e-10)
        expected_gradient = 2.*scaled_weight*softplus*slope*response
        np.testing.assert_allclose(
            (tail_h@delta+tail_g)-(base_h@delta+base_g),
            expected_gradient, atol=1e-10)

        # The QP uses a local squared-residual approximation: its gradient
        # must equal the derivative of the exact nonlinear cost at nominal.
        def exact_cost(increments):
            states = (free+phi@increments).reshape(3, 6)[:, :5]
            return tail._trajectory_cost(
                states, reference, np.cumsum(increments), previous)

        step = 1e-6
        numerical = np.array([
            (exact_cost(delta+step*np.eye(3)[j])
             -exact_cost(delta-step*np.eye(3)[j]))/(2*step)
            for j in range(3)])
        np.testing.assert_allclose(
            tail.objective_scale*(tail_h@delta+tail_g), numerical,
            rtol=1e-8, atol=1e-6)

    def test_cvxpy_reuses_dpp_problem_and_refreshes_optimization_data(self):
        # Both optima are available analytically, so a stale CVXPY parameter
        # or an incorrectly mapped lower/upper inequality changes the answer.
        controller = CVXPYHydraulicMPC(horizon=2, model=object())
        hessian = np.diag([2.0, 4.0])
        rows = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        lower = np.array([-np.inf, -1.0, -1.0])

        first, first_info = controller._solve_qp(
            hessian, np.array([-8.0, -4.0]), rows, lower,
            np.array([3.0, 2.0, 2.0]), np.zeros(2))
        self.assertTrue(first_info["cvxpy_dpp"])
        np.testing.assert_allclose(first, [2.0, 1.0], atol=1e-4)
        self.assertEqual(len(controller._cvx_cache), 1)
        cached_problem = next(iter(controller._cvx_cache.values()))[0]
        self.assertTrue(cached_problem.is_dpp())

        second_upper = np.array([2.0, 2.0, 2.0])
        second, second_info = controller._solve_qp(
            hessian, np.array([-2.0, -12.0]), rows, lower,
            second_upper, first)
        np.testing.assert_allclose(second, [0.0, 2.0], atol=1e-4)
        self.assertIs(next(iter(controller._cvx_cache.values()))[0], cached_problem)
        self.assertLessEqual(float(np.max(rows @ second-second_upper)), 1e-5)
        self.assertLessEqual(second_info["linear_constraint_violation"], 1e-5)

    def test_restoration_validates_real_model_and_rejects_unrecoverable_pressure(self):
        configuration = load_config()
        arguments = dict(origin_m=configuration["origin_m"],
                         valve_tau=configuration["valve_tau"],
                         makeup_area=configuration["makeup_area"],
                         rtol=configuration["rtol"])
        plant = HydraulicPlant(**arguments)
        model = GreyBoxModel(plant.parameters, **arguments)
        controller = CVXPYHydraulicMPC(
            horizon=5, dt=configuration["ts"], model=model,
            q_position=configuration["q_position"],
            q_velocity=configuration["q_velocity"],
            move_weight=configuration["move_weight"],
            delta_u_max=configuration["max_du_step"],
            p_safe=configuration["p_safe"],
            pressure_margin=configuration["pressure_margin"])
        # State sampled immediately before the short-horizon failure in a
        # genuine 50 mm positioning run. The fallback must find a safe plan.
        state = np.array([47.94304007713968, 1.200157261063793,
                          1.1813207599206568, 5.292159010072882,
                          8.5164399137673])
        previous = 5.500000039943047
        controller.nominal_sequence = np.full(5, 10.0)
        voltage = restore_pressure_feasible(
            controller, state, previous, configuration["target_mm"],
            failure=MPCSolveError("local MPC plan failed nonlinear validation"))
        planned = controller.nominal_sequence
        predicted = model.rollout(state, planned, configuration["ts"])
        self.assertAlmostEqual(voltage, planned[0])
        self.assertTrue(controller.last["restoration_used"])
        self.assertGreater(controller.last["restoration_feasible_candidates"], 0)
        self.assertLessEqual(float(np.max(np.abs(planned))), 10.0+1e-8)
        self.assertLessEqual(float(np.max(np.abs(np.diff(np.r_[previous, planned])))),
                             configuration["max_du_step"]+1e-8)
        self.assertLessEqual(float(np.max(predicted[:, 3])),
                             configuration["p_safe"]-configuration["pressure_margin"]+1e-6)

        previous_plan = planned.copy()
        with self.assertRaises(MPCSolveError):
            restore_pressure_feasible(
                controller, np.array([47.94, 1.2, 1.18, 40.0, 8.5]),
                5.5, configuration["target_mm"],
                failure=MPCSolveError("pressure already far above the limit"))
        self.assertEqual(controller.last["status"], "restoration_failed")
        np.testing.assert_allclose(controller.nominal_sequence, previous_plan)

    def test_short_horizon_closed_loop_uses_dense_physical_pressure_trace(self):
        configuration = load_config()
        configuration["duration"] = 0.085  # Includes breakaway, braking, and restoration.
        result = simulate(5, configuration)
        metrics = result["metrics"]
        self.assertEqual(metrics["status"], "completed", metrics["failure"])
        self.assertGreaterEqual(metrics["restoration_count"], 1)
        self.assertGreater(metrics["overshoot_pct"], 0.5)
        self.assertLessEqual(metrics["peak_pressure_mpa"], configuration["p_safe"]+1e-6)
        self.assertAlmostEqual(metrics["peak_pressure_mpa"],
                               float(np.max(result["pressure_trace"]["p2_mpa"])))
        self.assertGreater(len(result["pressure_trace"]["t_s"]),
                           len(result["data"]["t_s"]))
        self.assertLessEqual(metrics["max_abs_voltage_v"], 10.0+1e-8)
        self.assertLessEqual(metrics["max_abs_increment_v"],
                             configuration["max_du_step"]+1e-8)

    def test_n10_refinement_is_applied_and_remains_pressure_safe(self):
        # N=10 previously kept a poor shifted plan for most samples, allowing
        # very large position error. This checks the complete closed-loop
        # consequence of choosing a lower-cost nonlinear-feasible candidate.
        configuration = load_config()
        result = simulate(10, configuration)
        metrics = result["metrics"]
        self.assertEqual(metrics["status"], "completed", metrics["failure"])
        self.assertGreaterEqual(metrics["refinement_count"], 1)
        self.assertLess(metrics["overshoot_pct"], 2.0)
        self.assertLessEqual(float(np.max(result["pressure_trace"]["p2_mpa"])),
                             configuration["p_safe"]+1e-6)

        data = result["data"]
        selected_rows = np.flatnonzero(data["status_code"] == 4)
        self.assertEqual(len(selected_rows), metrics["refinement_count"])
        refined_logs = [entry for entry in result["solver_logs"]
                        if entry["status_code"] == 4]
        self.assertEqual(len(refined_logs), len(selected_rows))
        for row, log in zip(selected_rows, refined_logs):
            self.assertAlmostEqual(float(data["t_s"][row]), log["time_s"])
            self.assertEqual(log["status"], "refined_pressure_feasible")
            self.assertTrue(log["plan_refinement_used"])
            self.assertLess(log["plan_refinement_selected_cost"],
                            log["plan_refinement_original_cost"])
            self.assertAlmostEqual(float(data["u_v"][row]), log["voltage_v"], places=8)
            self.assertAlmostEqual(float(data["u_v"][row]),
                                   log["command_sequence_v"][0], places=8)
            self.assertLessEqual(float(np.max(np.asarray(log["prediction"])[:, 3])),
                                 configuration["p_safe"]-configuration["pressure_margin"]+1e-5)

    def test_invalid_horizon_scan_is_rejected_before_running_plant(self):
        configuration = load_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            for malformed in ([5, 5, 20], [0, 20], [20, 30]):
                candidate = dict(configuration, horizons=malformed)
                path.write_text(json.dumps(candidate), encoding="utf-8")
                with self.subTest(horizons=malformed), self.assertRaises(ValueError):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
