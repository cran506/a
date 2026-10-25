"""课件第七章的五维增广模型与单次稀疏 QP 滚动控制。

物理状态采用 [位置 mm, 速度 m/s, P1 MPa, P2 MPa]。本模块不包含
对象参数、后备控制或按 N 改变的权重；每次 solve 只调用一次 CVXPY。
"""
from __future__ import annotations

import time

import cvxpy as cp
import numpy as np


class MPCSolveError(RuntimeError):
    """当前 QP 未给出满足声明精度的可行控制序列。"""


def augment_model(ad, bd, cd=None):
    """将绝对输入模型增广为以 Δu 为输入的模型（课件式 7.4）。"""
    ad = np.asarray(ad, dtype=float)
    n = ad.shape[0]
    bd = np.asarray(bd, dtype=float).reshape(n)
    cd = np.zeros(n) if cd is None else np.asarray(cd, dtype=float).reshape(n)
    if ad.shape != (n, n) or not all(np.all(np.isfinite(v)) for v in (ad, bd, cd)):
        raise ValueError("Invalid affine state model")
    a_aug = np.zeros((n+1, n+1))
    a_aug[:n, :n], a_aug[:n, n], a_aug[n, n] = ad, bd, 1.
    b_aug = np.r_[bd, 1.]
    c_aug = np.r_[cd, 0.]
    return a_aug, b_aug, c_aug


def prediction_matrices(a_aug, b_aug, c_aug, horizon):
    """返回堆叠预测 Ξ=F ξ0+Φ ΔU+b，预测点为 k+1 到 k+N。"""
    if int(horizon) != horizon or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    a_aug = np.asarray(a_aug, dtype=float)
    n = a_aug.shape[0]
    b_aug = np.asarray(b_aug, dtype=float).reshape(n)
    c_aug = np.asarray(c_aug, dtype=float).reshape(n)
    if a_aug.shape != (n, n):
        raise ValueError("A_aug must be square")
    horizon = int(horizon)
    f, phi, bias = [], [], []
    power, response, offset = np.eye(n), np.zeros((n, horizon)), np.zeros(n)
    for index in range(horizon):
        power = a_aug @ power
        response = a_aug @ response
        response[:, index] += b_aug
        offset = a_aug @ offset+c_aug
        f.append(power.copy())
        phi.append(response.copy())
        bias.append(offset.copy())
    return np.vstack(f), np.vstack(phi), np.concatenate(bias)


def condensed_objective(ad, bd, cd, horizon, state, previous_voltage, reference,
                        q_position=100., q_velocity=20., move_weight=.1):
    """教学用凝聚 H,g，与下面稀疏 QP 的目标函数严格等价（差一常数）。"""
    aa, ba, ca = augment_model(ad, bd, cd)
    f, phi, bias = prediction_matrices(aa, ba, ca, horizon)
    initial = np.r_[np.asarray(state, dtype=float).reshape(4), previous_voltage]
    selected = np.array([5*i+j for i in range(horizon) for j in (0, 1)])
    free = (f@initial+bias)[selected]
    response = phi[selected]
    target = np.tile([reference, 0.], horizon)
    weights = np.tile([q_position, q_velocity], horizon)
    hessian = 2*(response.T@(weights[:, None]*response)+move_weight*np.eye(horizon))
    gradient = 2*response.T@(weights*(free-target))
    return hessian, gradient, f, phi, bias


def _stat(info, name):
    value = getattr(info, name, None)
    return None if value is None else float(value)


class HydraulicMPC:
    """四物理状态、五维增广、每采样时刻一次 QP 的液压 MPC。

    预测时域内冻结本周期的离散仿射模型 Ad,Bd,cd。每组实验使用相同
    的阶段权重和物理边界，仅 horizon 改变。电压变化率这里选用课件
    第 10 页的硬上下界形式。硬压力约束不会通过罚函数或削顶放松。
    """
    def __init__(self, horizon=20, dt=.001, *, q_position=100., q_velocity=20.,
                 move_weight=.1, du_max=4.5, pmax=21., pressure_lower=0.,
                 constrain_p1=True, voltage_limit=10., eps_abs=1e-6,
                 eps_rel=1e-6, max_iter=20000, feasibility_tolerance=2e-4,
                 pressure_substeps=()):
        if int(horizon) != horizon or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        scalars = (dt, q_position, q_velocity, move_weight, du_max, pmax,
                   voltage_limit, eps_abs, eps_rel, feasibility_tolerance)
        if (not np.all(np.isfinite(scalars)) or dt <= 0.
                or min(q_position, q_velocity) < 0. or move_weight <= 0.
                or min(du_max, pmax, voltage_limit, eps_abs, eps_rel,
                       feasibility_tolerance) <= 0.
                or voltage_limit > 10. or max_iter < 1
                or (pressure_lower is not None and
                    (not np.isfinite(pressure_lower) or pressure_lower >= pmax))):
            raise ValueError("Invalid MPC weights, bounds or solver settings")
        self.horizon, self.dt = int(horizon), float(dt)
        self.q_position, self.q_velocity = float(q_position), float(q_velocity)
        self.move_weight, self.du_max = float(move_weight), float(du_max)
        self.pmax, self.pressure_lower = float(pmax), pressure_lower
        self.constrain_p1, self.voltage_limit = bool(constrain_p1), float(voltage_limit)
        self.eps_abs, self.eps_rel = float(eps_abs), float(eps_rel)
        self.max_iter, self.feasibility_tolerance = int(max_iter), float(feasibility_tolerance)
        self.pressure_substeps = []
        for model in pressure_substeps:
            if len(model) != 3:
                raise ValueError("Each pressure substep must contain Ah,Bh,ch")
            ah, bh, ch = (np.asarray(value, dtype=float) for value in model)
            if (ah.shape != (4, 4) or bh.size != 4 or ch.size != 4
                    or not all(np.all(np.isfinite(v)) for v in (ah, bh, ch))):
                raise ValueError("Invalid constant pressure substep model")
            self.pressure_substeps.append((ah.copy(), bh.reshape(4).copy(), ch.reshape(4).copy()))

        n = self.horizon
        self.x = cp.Variable((4, n+1), name="physical_state")
        self.u = cp.Variable(n, name="voltage")
        self.du = cp.Variable(n, name="voltage_increment")
        self.initial = cp.Parameter(4, name="measured_state")
        self.previous = cp.Parameter(name="previous_voltage")
        self.reference = cp.Parameter(name="target_mm")
        self.ad = cp.Parameter((4, 4), name="Ad")
        self.bd = cp.Parameter(4, name="Bd")
        self.cd = cp.Parameter(4, name="affine_offset")
        constraints = [self.x[:, 0] == self.initial,
                       self.u[0] == self.previous+self.du[0],
                       cp.abs(self.u) <= self.voltage_limit,
                       cp.abs(self.du) <= self.du_max]
        if n > 1:
            constraints.append(self.u[1:] == self.u[:-1]+self.du[1:])
        for k in range(n):
            constraints.append(self.x[:, k+1] == self.ad@self.x[:, k]+
                               self.bd*self.u[k]+self.cd)
        # Apply bounds to future nodes. Current measured pressures are fixed
        # data; an already unsafe initial state is reported, not made feasible
        # by silently changing the measurement.
        pressure_rows = [2, 3] if self.constrain_p1 else [3]
        self.pressure_rows = pressure_rows
        constraints.append(self.x[pressure_rows, 1:] <= self.pmax)
        if self.pressure_lower is not None:
            constraints.append(self.x[[2, 3], 1:] >= self.pressure_lower)
        # Optional exact held-input maps at fractions of the sample interval.
        # These guard intersample pressure extrema without adding a second QP
        # or changing the sampling rate. Their matrices are fixed constants.
        for ah, bh, ch in self.pressure_substeps:
            middle = ah@self.x[:, :-1]+bh[:, None]@cp.reshape(self.u, (1, n), order='C')+ch[:, None]
            constraints.append(middle[pressure_rows, :] <= self.pmax)
            if self.pressure_lower is not None:
                constraints.append(middle[[2, 3], :] >= self.pressure_lower)
        objective = (self.q_position*cp.sum_squares(self.x[0, 1:]-self.reference)
                     +self.q_velocity*cp.sum_squares(self.x[1, 1:])
                     +self.move_weight*cp.sum_squares(self.du))
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        if not self.problem.is_dpp():
            raise RuntimeError("Sparse MPC parameterization must satisfy DPP")
        self.last = None

    def solve(self, state, u_prev, reference, ad, bd, cd=None):
        """求一次 QP，返回 (首个电压, 诊断字典)，不执行物理状态更新。"""
        state = np.asarray(state, dtype=float).reshape(4)
        ad = np.asarray(ad, dtype=float).reshape(4, 4)
        bd = np.asarray(bd, dtype=float).reshape(4)
        cd = np.zeros(4) if cd is None else np.asarray(cd, dtype=float).reshape(4)
        if not all(np.all(np.isfinite(v)) for v in (state, ad, bd, cd, u_prev, reference)):
            raise ValueError("Nonfinite model or measurement")
        self.initial.value, self.previous.value = state, float(u_prev)
        self.reference.value = float(reference)
        self.ad.value, self.bd.value, self.cd.value = ad, bd, cd
        tic = time.perf_counter()
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                               eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                               max_iter=self.max_iter, polishing=False,
                               adaptive_rho=True)
        except cp.error.SolverError as exc:
            raise MPCSolveError(f"OSQP failed: {exc}") from exc
        wall_ms = (time.perf_counter()-tic)*1000.
        stats = self.problem.solver_stats
        extra = getattr(stats.extra_stats, "info", None)
        info = dict(status=self.problem.status, qp_calls=1,
                    cvxpy_wall_ms=wall_ms,
                    backend_solve_ms=(None if _stat(extra, "solve_time") is None
                                      else 1000.*_stat(extra, "solve_time")),
                    backend_run_ms=(None if _stat(extra, "run_time") is None
                                    else 1000.*_stat(extra, "run_time")),
                    solver_stats_ms=(None if stats.solve_time is None
                                     else float(stats.solve_time)*1000.),
                    iterations=None if stats.num_iters is None else int(stats.num_iters),
                    primal_residual=_stat(extra, "prim_res"),
                    dual_residual=_stat(extra, "dual_res"),
                    dpp=self.problem.is_dpp(),
                    objective=None if self.problem.value is None else float(self.problem.value))
        self.last = info
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise MPCSolveError(f"QP has status {self.problem.status}")
        x, u, du = (np.asarray(v.value, dtype=float) for v in (self.x, self.u, self.du))
        if not all(np.all(np.isfinite(v)) for v in (x, u, du)):
            raise MPCSolveError("QP returned nonfinite trajectory")
        dynamic_error = np.max(np.abs(x[:, 1:]-(ad@x[:, :-1]+bd[:, None]*u+cd[:, None])))
        initial_error = np.max(np.abs(x[:, 0]-state))
        increment_error = np.max(np.abs(np.diff(np.r_[u_prev, u])-du))
        voltage_violation = max(0., float(np.max(np.abs(u)))-self.voltage_limit)
        increment_violation = max(0., float(np.max(np.abs(du)))-self.du_max)
        pressure_violation = max(0., float(np.max(x[self.pressure_rows, 1:]))-self.pmax)
        if self.pressure_lower is not None:
            pressure_violation = max(pressure_violation,
                                     self.pressure_lower-float(np.min(x[[2, 3], 1:])))
        substep_pressure_max = substep_pressure_min = None
        if self.pressure_substeps:
            middle_states = np.hstack([ah@x[:, :-1]+bh[:, None]*u+ch[:, None]
                                       for ah, bh, ch in self.pressure_substeps])
            substep_pressure_max = float(np.max(middle_states[self.pressure_rows]))
            substep_pressure_min = float(np.min(middle_states[[2, 3]]))
            pressure_violation = max(pressure_violation, substep_pressure_max-self.pmax)
            if self.pressure_lower is not None:
                pressure_violation = max(pressure_violation,
                                         self.pressure_lower-substep_pressure_min)
        violation = max(float(dynamic_error), float(initial_error), float(increment_error),
                        voltage_violation, increment_violation, pressure_violation)
        info.update(prediction=x[:, 1:].T.tolist(), command_sequence_v=u.tolist(),
                    increment_sequence_v=du.tolist(), dynamics_residual=float(dynamic_error),
                    voltage_violation_v=voltage_violation,
                    increment_violation_v=increment_violation,
                    pressure_violation_mpa=pressure_violation,
                    max_substep_pressure_mpa=substep_pressure_max,
                    min_substep_pressure_mpa=substep_pressure_min,
                    max_constraint_violation=violation)
        if violation > self.feasibility_tolerance:
            raise MPCSolveError(f"QP constraint residual {violation:.3g} exceeds tolerance")
        return float(u[0]), info


class CondensedHydraulicMPC(HydraulicMPC):
    """固定工作点模型的教材凝聚 QP：仅 N 个 Δu 决策变量。

    固定线性化工作点是 LTV 模型的一个特例。首次 solve 构造常数 H/M；
    后续只刷新 g 和约束右端。若对象矩阵改变则明确报错，避免把旧矩阵
    用于新工作点。与稀疏形式的目标与硬约束相同。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constant_model = None
        self._condensed_problem = None

    def _build_condensed(self, ad, bd, cd):
        n = self.horizon
        self._constant_model = (ad.copy(), bd.copy(), cd.copy())
        hessian, _, self._f, self._phi, self._bias = condensed_objective(
            ad, bd, cd, n, np.zeros(4), 0., 0., self.q_position,
            self.q_velocity, self.move_weight)
        self.hessian = hessian
        self._sum = np.tril(np.ones((n, n)))
        # A common scalar does not change the optimum. Normalizing curvature
        # prevents large position-unit weights from dominating OSQP scaling.
        self.objective_scale = max(1., float(np.max(np.diag(hessian))))

        upper_f, upper_phi, upper_b = [], [], []
        lower_f, lower_phi, lower_b = [], [], []

        def add_pressure_maps(free_map, response, offset, stride):
            rows = np.array([stride*k+j for k in range(n) for j in self.pressure_rows])
            upper_f.append(free_map[rows])
            upper_phi.append(response[rows])
            upper_b.append(offset[rows])
            if self.pressure_lower is not None:
                rows = np.array([stride*k+j for k in range(n) for j in (2, 3)])
                lower_f.append(free_map[rows])
                lower_phi.append(response[rows])
                lower_b.append(offset[rows])

        add_pressure_maps(self._f, self._phi, self._bias, 5)
        # Held-input fractional-step maps use the state at the start of each
        # interval and the same voltage as its endpoint propagation.
        previous_selector = np.array([0., 0., 0., 0., 1.])
        for ah, bh, ch in self.pressure_substeps:
            sf, sp, sb = [], [], []
            for k in range(n):
                if k == 0:
                    start_f = np.c_[np.eye(4), np.zeros(4)]
                    start_phi = np.zeros((4, n))
                    start_b = np.zeros(4)
                else:
                    rows = slice(5*(k-1), 5*(k-1)+4)
                    start_f, start_phi, start_b = self._f[rows], self._phi[rows], self._bias[rows]
                sf.append(ah@start_f+np.outer(bh, previous_selector))
                sp.append(ah@start_phi+np.outer(bh, self._sum[k]))
                sb.append(ah@start_b+ch)
            add_pressure_maps(np.vstack(sf), np.vstack(sp), np.concatenate(sb), 4)
        self._upper_f, self._upper_phi, self._upper_b = (
            np.vstack(upper_f), np.vstack(upper_phi), np.concatenate(upper_b))
        if lower_f:
            self._lower_f, self._lower_phi, self._lower_b = (
                np.vstack(lower_f), np.vstack(lower_phi), np.concatenate(lower_b))
        else:
            self._lower_f = self._lower_phi = self._lower_b = None
        matrices = [np.eye(n), -np.eye(n), self._sum, -self._sum, self._upper_phi]
        if self._lower_phi is not None:
            matrices.append(-self._lower_phi)
        self.constraint_matrix = np.vstack(matrices)
        self.decision = cp.Variable(n, name="delta_u_condensed")
        self.gradient = cp.Parameter(n, name="gradient_condensed")
        self.bound = cp.Parameter(len(self.constraint_matrix), name="bound_condensed")
        objective = (.5*cp.quad_form(self.decision, cp.psd_wrap(
            hessian/self.objective_scale))+self.gradient@self.decision)
        self.problem = cp.Problem(cp.Minimize(objective),
                                  [self.constraint_matrix@self.decision <= self.bound])
        if not self.problem.is_dpp():
            raise RuntimeError("Constant-H condensed problem must satisfy DPP")
        self._condensed_problem = self.problem

    def solve(self, state, u_prev, reference, ad, bd, cd=None):
        state = np.asarray(state, dtype=float).reshape(4)
        ad = np.asarray(ad, dtype=float).reshape(4, 4)
        bd = np.asarray(bd, dtype=float).reshape(4)
        cd = np.zeros(4) if cd is None else np.asarray(cd, dtype=float).reshape(4)
        if not all(np.all(np.isfinite(v)) for v in (state, ad, bd, cd, u_prev, reference)):
            raise ValueError("Nonfinite model or measurement")
        if self._constant_model is None:
            self._build_condensed(ad, bd, cd)
        elif not all(np.array_equal(old, new) for old, new in zip(self._constant_model, (ad, bd, cd))):
            raise ValueError("CondensedHydraulicMPC requires one constant working-point model")
        initial = np.r_[state, u_prev]
        free = self._f@initial+self._bias
        selected = np.array([5*k+j for k in range(self.horizon) for j in (0, 1)])
        error = free[selected]-np.tile([reference, 0.], self.horizon)
        weights = np.tile([self.q_position, self.q_velocity], self.horizon)
        response = self._phi[selected]
        gradient = 2.*response.T@(weights*error)
        bounds = [np.full(self.horizon, self.du_max), np.full(self.horizon, self.du_max),
                  np.full(self.horizon, self.voltage_limit-u_prev),
                  np.full(self.horizon, self.voltage_limit+u_prev),
                  self.pmax-(self._upper_f@initial+self._upper_b)]
        if self._lower_f is not None:
            bounds.append(self._lower_f@initial+self._lower_b-self.pressure_lower)
        self.gradient.value, self.bound.value = gradient/self.objective_scale, np.concatenate(bounds)
        tic = time.perf_counter()
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                               eps_abs=self.eps_abs, eps_rel=self.eps_rel,
                               max_iter=self.max_iter, polishing=False, adaptive_rho=True)
        except cp.error.SolverError as exc:
            raise MPCSolveError(f"OSQP failed: {exc}") from exc
        wall_ms = (time.perf_counter()-tic)*1000.
        stats = self.problem.solver_stats
        extra = getattr(stats.extra_stats, "info", None)
        info = dict(status=self.problem.status, qp_calls=1,
                    formulation="constant-H condensed incremental QP",
                    decision_variables=self.horizon, objective_scale=self.objective_scale,
                    cvxpy_wall_ms=wall_ms,
                    backend_solve_ms=(None if _stat(extra, "solve_time") is None
                                      else 1000.*_stat(extra, "solve_time")),
                    backend_run_ms=(None if _stat(extra, "run_time") is None
                                    else 1000.*_stat(extra, "run_time")),
                    solver_stats_ms=(None if stats.solve_time is None else float(stats.solve_time)*1000.),
                    iterations=None if stats.num_iters is None else int(stats.num_iters),
                    primal_residual=_stat(extra, "prim_res"),
                    dual_residual=_stat(extra, "dual_res"), dpp=self.problem.is_dpp(),
                    qp_reduced_objective=None if self.problem.value is None else float(self.problem.value))
        self.last = info
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise MPCSolveError(f"QP has status {self.problem.status}")
        du = np.asarray(self.decision.value, dtype=float)
        if not np.all(np.isfinite(du)):
            raise MPCSolveError("QP returned nonfinite trajectory")
        u = u_prev+self._sum@du
        predicted_aug = (free+self._phi@du).reshape(self.horizon, 5)
        x = np.column_stack([state, predicted_aug[:, :4].T])
        dynamic_error = float(np.max(np.abs(x[:, 1:]-(ad@x[:, :-1]+bd[:, None]*u+cd[:, None]))))
        voltage_violation = max(0., float(np.max(np.abs(u)))-self.voltage_limit)
        increment_violation = max(0., float(np.max(np.abs(du)))-self.du_max)
        pressure_violation = max(0., float(np.max(x[self.pressure_rows, 1:]))-self.pmax)
        if self.pressure_lower is not None:
            pressure_violation = max(pressure_violation,
                                     self.pressure_lower-float(np.min(x[[2, 3], 1:])))
        substep_pressure_max = substep_pressure_min = None
        if self.pressure_substeps:
            middle = np.hstack([ah@x[:, :-1]+bh[:, None]*u+ch[:, None]
                                for ah, bh, ch in self.pressure_substeps])
            substep_pressure_max = float(np.max(middle[self.pressure_rows]))
            substep_pressure_min = float(np.min(middle[[2, 3]]))
            pressure_violation = max(pressure_violation, substep_pressure_max-self.pmax)
            if self.pressure_lower is not None:
                pressure_violation = max(pressure_violation, self.pressure_lower-substep_pressure_min)
        inequality_violation = max(0., float(np.max(self.constraint_matrix@du-self.bound.value)))
        violation = max(dynamic_error, voltage_violation, increment_violation,
                        pressure_violation, inequality_violation)
        objective = float(np.sum((predicted_aug[:, 0]-reference)**2*self.q_position+
                                 predicted_aug[:, 1]**2*self.q_velocity)+self.move_weight*du@du)
        info.update(prediction=predicted_aug[:, :4].tolist(), command_sequence_v=u.tolist(),
                    increment_sequence_v=du.tolist(), dynamics_residual=dynamic_error,
                    objective=objective, voltage_violation_v=voltage_violation,
                    increment_violation_v=increment_violation,
                    pressure_violation_mpa=pressure_violation,
                    max_substep_pressure_mpa=substep_pressure_max,
                    min_substep_pressure_mpa=substep_pressure_min,
                    max_constraint_violation=violation)
        if violation > self.feasibility_tolerance:
            raise MPCSolveError(f"QP constraint residual {violation:.3g} exceeds tolerance")
        return float(u[0]), info
