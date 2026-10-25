"""独立验证五维增广预测与稀疏 QP 的教材等价形式。"""
import unittest

import numpy as np

from mpc import (CondensedHydraulicMPC, HydraulicMPC, augment_model,
                 condensed_objective, prediction_matrices)


class MPCTests(unittest.TestCase):
    def test_affine_augmented_prediction_matches_direct_state_recursion(self):
        a = np.array([[1., 2., 0., 0.], [0., .8, .1, -.2],
                      [0., 0., .7, 0.], [0., 0., 0., .6]])
        b, c = np.array([.2, .4, .1, -.1]), np.array([0., .1, .3, .2])
        state, previous = np.array([1., .2, 3., 4.]), .6
        increments = np.array([.1, -.3, .5])
        aa, ba, ca = augment_model(a, b, c)
        f, phi, bias = prediction_matrices(aa, ba, ca, 3)
        direct, voltage = [], previous
        for increment in increments:
            voltage += increment
            state = a@state+b*voltage+c
            direct.extend([*state, voltage])
        predicted = f@np.array([1., .2, 3., 4., previous])+phi@increments+bias
        np.testing.assert_allclose(predicted, direct, atol=1e-12)
        self.assertEqual(f.shape, (15, 5))
        self.assertEqual(phi.shape, (15, 3))

    def test_sparse_qp_matches_condensed_optimum_and_reuses_dpp_problem(self):
        # x_next=x+u; pressure states remain safely constant. This produces
        # an analytically checkable constrained QP without a hydraulic model.
        a, b, c = np.eye(4), np.array([1., 0., 0., 0.]), np.zeros(4)
        state = np.array([0., 0., 1., 2.])
        controller = HydraulicMPC(2, q_position=1., q_velocity=0.,
                                 move_weight=1., du_max=1.)
        problem_identity = id(controller.problem)
        first, info = controller.solve(state, 0., 10., a, b, c)
        np.testing.assert_allclose(info['increment_sequence_v'], [1., 1.], atol=2e-5)
        self.assertAlmostEqual(first, 1., places=4)
        h, g, *_ = condensed_objective(a, b, c, 2, state, 0., 10., 1., 0., 1.)
        du = np.array(info['increment_sequence_v'])
        sparse_cost = info['objective']
        constant = 2*10.**2
        self.assertAlmostEqual(.5*du@h@du+g@du+constant, sparse_cost, places=4)
        self.assertTrue(info['dpp'])
        self.assertEqual(info['qp_calls'], 1)
        _, second = controller.solve(state, 0., .1, a, b, c)
        self.assertEqual(id(controller.problem), problem_identity)
        self.assertLess(second['max_constraint_violation'], 1e-4)
        # The unconstrained condensed optimum is interior for target .1.
        h2, g2, *_ = condensed_objective(a, b, c, 2, state, 0., .1, 1., 0., 1.)
        np.testing.assert_allclose(second['increment_sequence_v'],
                                   -np.linalg.solve(h2, g2), atol=2e-5)

    def test_pressure_hard_bound_changes_control_decision(self):
        a, b = np.eye(4), np.array([1., 0., 0., 1.])
        state = np.array([0., 0., 1., 20.5])
        controller = HydraulicMPC(1, q_position=100., q_velocity=0., move_weight=.1)
        voltage, info = controller.solve(state, 0., 10., a, b)
        self.assertAlmostEqual(voltage, .5, places=4)
        self.assertLessEqual(info['prediction'][0][3], 21.+2e-5)

    def test_held_input_substep_constraint_prevents_hidden_pressure_peak(self):
        # A sampled pressure response can return to its initial value while
        # having a nonzero held-input response midway through the interval.
        # Endpoint-only constraints cannot see this peak.
        ad, bd = np.eye(4), np.array([1., 0., 0., 0.])
        ah, bh, ch = np.eye(4), np.array([.5, 0., 0., 1.]), np.zeros(4)
        state = np.array([0., 0., 1., 20.5])
        common = dict(horizon=1, q_position=100., q_velocity=0., move_weight=.1)
        endpoint_only = HydraulicMPC(**common)
        voltage_unsafe, _ = endpoint_only.solve(state, 0., 10., ad, bd)
        self.assertGreater((ah@state+bh*voltage_unsafe+ch)[3], 21.)
        safeguarded = HydraulicMPC(pressure_substeps=((ah, bh, ch),), **common)
        voltage, info = safeguarded.solve(state, 0., 10., ad, bd)
        self.assertAlmostEqual(voltage, .5, places=4)
        self.assertAlmostEqual(info['prediction'][0][3], 20.5, places=4)
        self.assertLessEqual(info['max_substep_pressure_mpa'], 21.+2e-5)
        self.assertGreaterEqual(info['min_substep_pressure_mpa'], 0.)
        self.assertTrue(info['dpp'])
        self.assertEqual(info['qp_calls'], 1)

    def test_constant_h_condensed_and_sparse_qps_have_same_solution(self):
        ad = np.array([[1., .4, 0., 0.], [0., .7, .1, -.1],
                       [0., 0., .8, 0.], [0., 0., 0., .9]])
        bd, cd = np.array([.1, .2, .1, -.1]), np.array([0., 0., .2, .2])
        state = np.array([0., 0., 1., 2.])
        common = dict(horizon=4, q_position=2., q_velocity=.5,
                      move_weight=.3, du_max=.7,
                      pressure_substeps=((np.eye(4), .5*bd, .5*cd),))
        sparse, condensed = HydraulicMPC(**common), CondensedHydraulicMPC(**common)
        for reference in (3., -.5):
            u_sparse, info_sparse = sparse.solve(state, .2, reference, ad, bd, cd)
            u_condensed, info_condensed = condensed.solve(state, .2, reference, ad, bd, cd)
            self.assertAlmostEqual(u_sparse, u_condensed, delta=3e-5)
            self.assertAlmostEqual(info_sparse['objective'], info_condensed['objective'], delta=1e-4)
            np.testing.assert_allclose(info_sparse['prediction'], info_condensed['prediction'], atol=3e-5)
            self.assertEqual(info_condensed['decision_variables'], 4)
            self.assertTrue(info_condensed['dpp'])
            self.assertEqual(info_condensed['qp_calls'], 1)
        with self.assertRaises(ValueError):
            condensed.solve(state, .2, 3., ad*1.001, bd, cd)


if __name__ == '__main__':
    unittest.main()
