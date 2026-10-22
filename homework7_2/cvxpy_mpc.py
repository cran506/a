"""CVXPY solution of the same affine, constrained QP used by HydraulicMPC.

The physical model, LTV condensation, sequential linearization, and nonlinear
pressure check live in :mod:`mpc_core`.  Only the QP backend differs here.
Each controller instance caches a DPP-compliant CVXPY problem so repeated
rolling-horizon solves reuse the canonicalization and OSQP workspace.
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np

from mpc_core import HydraulicMPC, MPCSolveError


class CVXPYHydraulicMPC(HydraulicMPC):
    """HydraulicMPC with a cached CVXPY/OSQP optimizer and checked KKT data.

    The public ``solve(z, u_prev, ref_traj)`` interface and ``last`` diagnostics
    are inherited unchanged.  ``solver_time_ms`` measures the entire CVXPY
    call, including Python overhead; ``backend_solve_time_ms`` records the
    solver's own reported time for comparison.
    """

    def reset(self):
        super().reset()
        self._cvx_cache = {}

    def _problem(self, n, m, finite_upper, finite_lower):
        key = (n, m, tuple(finite_upper), tuple(finite_lower))
        if key in self._cvx_cache:
            return self._cvx_cache[key]

        decision = cp.Variable(n, name="delta_u")
        cholesky = cp.Parameter((n, n), name="hessian_cholesky")
        gradient = cp.Parameter(n, name="gradient")
        matrix = cp.Parameter((m, n), name="constraints")
        objective = cp.Minimize(0.5 * cp.sum_squares(cholesky @ decision)
                                + gradient @ decision)
        constraints = []
        upper_value = lower_value = upper_con = lower_con = None
        if len(finite_upper):
            upper_value = cp.Parameter(len(finite_upper), name="upper_bounds")
            upper_con = matrix[finite_upper, :] @ decision <= upper_value
            constraints.append(upper_con)
        if len(finite_lower):
            lower_value = cp.Parameter(len(finite_lower), name="lower_bounds")
            lower_con = matrix[finite_lower, :] @ decision >= lower_value
            constraints.append(lower_con)
        problem = cp.Problem(objective, constraints)
        if not problem.is_dpp():
            raise RuntimeError("CVXPY QP formulation lost DPP compliance")
        cache = (problem, decision, cholesky, gradient, matrix,
                 upper_value, lower_value, upper_con, lower_con)
        self._cvx_cache[key] = cache
        return cache

    def _solve_qp(self, h, g, a, lower, upper, warm):
        h = np.asarray(h, dtype=float)
        g = np.asarray(g, dtype=float).reshape(-1)
        a = np.asarray(a, dtype=float)
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        warm = np.asarray(warm, dtype=float).reshape(-1)
        n = len(g)
        m = len(lower)
        if (h.shape != (n, n) or a.shape != (m, n) or upper.shape != (m,)
                or warm.shape != (n,) or not np.all(np.isfinite(h))
                or not np.all(np.isfinite(g)) or not np.all(np.isfinite(a))
                or not np.all(np.isfinite(warm)) or np.any(np.isnan(lower))
                or np.any(np.isnan(upper)) or np.any(np.isposinf(lower))
                or np.any(np.isneginf(upper)) or np.any(lower > upper)):
            raise MPCSolveError("Invalid QP dimensions, coefficients, or bounds")
        finite_upper = np.flatnonzero(np.isfinite(upper))
        finite_lower = np.flatnonzero(np.isfinite(lower))
        try:
            factor = np.linalg.cholesky((h + h.T) * 0.5).T
        except np.linalg.LinAlgError as exc:
            raise MPCSolveError("QP Hessian is not positive definite") from exc
        (problem, decision, cholesky, gradient, matrix,
         upper_value, lower_value, upper_con, lower_con) = self._problem(
             n, m, finite_upper, finite_lower)
        cholesky.value = factor
        gradient.value = g
        matrix.value = a
        if upper_value is not None:
            upper_value.value = upper[finite_upper]
        if lower_value is not None:
            lower_value.value = lower[finite_lower]
        decision.value = warm

        start = time.perf_counter()
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                          eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                          max_iter=self.max_iter, polishing=self.polishing)
        except cp.error.SolverError as exc:
            elapsed_ms = 1000 * (time.perf_counter() - start)
            info = dict(osqp_status="solver_error", osqp_iterations=0,
                        solver_time_ms=elapsed_ms, qp_solver="cvxpy-osqp",
                        fallback_solver_used=False, solver_failure=str(exc))
            raise MPCSolveError(f"CVXPY/OSQP failed: {exc}", info) from exc
        elapsed_ms = 1000 * (time.perf_counter() - start)
        stats = problem.solver_stats
        backend_time = getattr(stats, "solve_time", None)
        info = dict(osqp_status=str(problem.status),
                    osqp_iterations=int(stats.num_iters or 0),
                    solver_time_ms=elapsed_ms,
                    backend_solve_time_ms=(1000 * float(backend_time)
                                           if backend_time is not None else None),
                    qp_solver="cvxpy-osqp", fallback_solver_used=False,
                    cvxpy_dpp=True)
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or decision.value is None:
            raise MPCSolveError(f"CVXPY/OSQP rejected: {problem.status}", info)
        vector = np.asarray(decision.value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(vector)):
            raise MPCSolveError("CVXPY/OSQP returned nonfinite controls", info)

        dual = np.zeros(m)
        upper_dual = lower_dual = None
        if upper_con is not None:
            upper_dual = np.asarray(upper_con.dual_value, dtype=float).reshape(-1)
            dual[finite_upper] += upper_dual
        if lower_con is not None:
            lower_dual = np.asarray(lower_con.dual_value, dtype=float).reshape(-1)
            dual[finite_lower] -= lower_dual
        if not np.all(np.isfinite(dual)):
            raise MPCSolveError("CVXPY/OSQP returned nonfinite multipliers", info)
        mapped = a @ vector
        upper_violation = (np.max(mapped[finite_upper] - upper[finite_upper])
                           if len(finite_upper) else 0.0)
        lower_violation = (np.max(lower[finite_lower] - mapped[finite_lower])
                           if len(finite_lower) else 0.0)
        violation = float(max(0.0, upper_violation, lower_violation))
        stationarity = float(np.max(np.abs(h @ vector + g + a.T @ dual)))
        complementarity = float(max(
            np.max(np.abs((upper[finite_upper] - mapped[finite_upper]) * upper_dual))
            if upper_dual is not None else 0.0,
            np.max(np.abs((mapped[finite_lower] - lower[finite_lower]) * lower_dual))
            if lower_dual is not None else 0.0))
        dual_sign_violation = float(max(
            0.0,
            -np.min(upper_dual) if upper_con is not None else 0.0,
            -np.min(lower_dual) if lower_con is not None else 0.0))
        info.update(linear_constraint_violation=violation,
                    kkt_stationarity=stationarity,
                    kkt_complementarity=complementarity,
                    dual_sign_violation=dual_sign_violation)
        # An active-set multiplier has gradient units, while slack has row
        # units.  Normalize their product by the QP coefficient scales.
        finite_limits = np.r_[upper[finite_upper], lower[finite_lower]]
        slack_scale = 1.0 + float(np.max(np.abs(finite_limits))) if len(finite_limits) else 1.0
        complementarity_limit = self.dual_tolerance * (1 + np.max(np.abs(g))) * slack_scale
        if (violation > self.primal_tolerance
                or stationarity > self.dual_tolerance * (1 + np.max(np.abs(g)))
                or complementarity > complementarity_limit
                or dual_sign_violation > self.dual_tolerance):
            raise MPCSolveError("CVXPY/OSQP failed independent KKT checks", info)
        return vector, info


__all__ = ["CVXPYHydraulicMPC"]
