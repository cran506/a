"""Declared short-horizon recovery after an MPC candidate fails validation.

This is a finite candidate search, not a claim of recursive feasibility.  It
uses the same grey-box nonlinear rollout and the same input/pressure limits as
``HydraulicMPC``.  The physical plant's dense pressure trace must still be
checked separately after simulation.
"""
from __future__ import annotations

import time

import numpy as np

from mpc_core import MPCSolveError


def restore_pressure_feasible(
    controller,
    state,
    previous_voltage,
    reference,
    *,
    failure: Exception | None = None,
    grid_points: int = 41,
) -> float:
    """Choose the best pressure-feasible candidate after an MPC failure.

    ``state`` is the measured five-state vector in the MPC model's units and
    ``reference`` accepts the same forms as ``HydraulicMPC.solve``.  Every
    tested voltage sequence obeys the controller's absolute and slew bounds.
    Each one is then independently checked by its nonlinear model rollout.
    The selected sequence becomes the controller's warm plan for the next
    sample.  A failed search raises ``MPCSolveError``; it never applies an
    unvalidated hold command.

    The search includes the shifted previous plan, if present, and a uniform
    grid of constant voltage targets over the allowed input interval.  A
    constant target is projected step by step, so the first command and all
    future commands satisfy the same slew constraint as the original QP.
    This fallback should be counted and reported as a separate safety policy,
    rather than reported as a successful QP or a proof of future feasibility.
    """
    if isinstance(grid_points, bool) or int(grid_points) != grid_points or grid_points < 3:
        raise ValueError("grid_points must be an integer of at least three")
    start = time.perf_counter()
    z = np.asarray(state, dtype=float).reshape(-1)
    u_prev = float(previous_voltage)
    if z.shape != (controller.nx_physical,) or not np.all(np.isfinite(z)) or not np.isfinite(u_prev):
        raise ValueError("Expected finite measured physical state and previous voltage")
    if not controller.u_min <= u_prev <= controller.u_max:
        raise ValueError("Previous voltage is outside the controller's input bounds")
    target = controller._reference(reference)

    failed = dict(getattr(controller, "last", {}) or {})
    failed_attempts = failed.get("qp_diagnostics", [])
    failure_text = str(failure) if failure is not None else str(failed.get("error", ""))
    candidates = []
    if controller.nominal_sequence is not None:
        shifted = np.r_[controller.nominal_sequence[1:], controller.nominal_sequence[-1]]
        candidates.append(("shifted_plan", controller._project_inputs(shifted, u_prev)))
    voltage_grid = np.unique(np.r_[
        np.linspace(controller.u_min, controller.u_max, int(grid_points)),
        np.clip(u_prev, controller.u_min, controller.u_max),
        np.clip(0.0, controller.u_min, controller.u_max),
    ])
    for voltage in voltage_grid:
        constant = np.full(controller.horizon, voltage, dtype=float)
        candidates.append(("constant_grid", controller._project_inputs(constant, u_prev)))

    best = None
    seen = set()
    checked = 0
    feasible_count = 0
    minimum_violation = float("inf")
    for source, sequence in candidates:
        key = tuple(np.round(sequence, 12))
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        valid, predicted, pressure_violation, input_violation = controller._nonlinear_validation(
            z, sequence, u_prev
        )
        minimum_violation = min(minimum_violation, pressure_violation)
        if not valid:
            continue
        feasible_count += 1
        cost = controller._trajectory_cost(predicted, target, sequence, u_prev)
        if best is None or cost < best[0]:
            best = (cost, source, sequence, predicted, pressure_violation, input_violation)

    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    common = dict(
        restoration_used=True,
        restoration_failure=failure_text,
        restoration_grid_points=int(grid_points),
        restoration_candidates_checked=checked,
        restoration_feasible_candidates=feasible_count,
        restoration_wall_ms=elapsed_ms,
        restoration_smallest_pressure_violation_mpa=minimum_violation,
        qp_diagnostics=failed_attempts,
        iterations=sum(int(attempt.get("osqp_iterations", 0)) for attempt in failed_attempts),
        solver_time_ms=sum(float(attempt.get("solver_time_ms", 0.0)) for attempt in failed_attempts),
        runtime_ms=float(failed.get("runtime_ms", 0.0)) + elapsed_ms,
    )
    if best is None:
        controller.last = dict(failed, **common, status="restoration_failed")
        raise MPCSolveError(
            f"No nonlinear-pressure-feasible restoration candidate among {checked} tested sequences",
            diagnostics=common,
        ) from failure

    cost, source, sequence, predicted, pressure_violation, input_violation = best
    voltage = float(sequence[0])
    controller.nominal_sequence = sequence.copy()
    controller._previous_applied = voltage
    controller._last_du = np.diff(np.r_[u_prev, sequence])
    controller.last = dict(
        common,
        status="restored_pressure_feasible",
        restoration_source=source,
        feasible_shifted_plan_used=source == "shifted_plan",
        qp_solver="restoration_grid",
        osqp_status=str(failed.get("osqp_status", "failed")),
        candidate_voltage_v=voltage,
        voltage_v=voltage,
        actual_du=float(voltage-u_prev),
        actual_du_v=float(voltage-u_prev),
        input_violation_v=float(input_violation),
        prediction_violation=float(pressure_violation),
        prediction_violation_mpa=float(pressure_violation),
        predicted_peak_p1_mpa=float(np.max(predicted[:, 2])),
        predicted_peak_p2_mpa=float(np.max(predicted[:, 3])),
        command_sequence_v=sequence.tolist(),
        prediction=predicted.tolist(),
        nonlinear_cost=cost,
    )
    return voltage


def refine_shifted_plan(
    controller,
    state,
    previous_voltage,
    reference,
    *,
    grid_points: int = 41,
    min_relative_improvement: float = 0.01,
    min_absolute_improvement: float = 1e-3,
) -> float:
    """Optionally replace a validated shifted plan with a better safe plan.

    Call immediately after ``controller.solve`` only if it reported
    ``feasible_shifted_plan_used``.  The original plan is reevaluated and kept
    as a candidate.  Projected constant-voltage grid candidates use precisely
    the same nonlinear pressure/input validation and stage tracking cost as
    ``HydraulicMPC.solve``.  Replacement requires both absolute and relative
    improvement; otherwise the original command and warm plan are untouched.

    This is a declared candidate-search policy following a failed/local QP,
    not an assertion that the QP solved or that recursive feasibility holds.
    ``controller.last`` retains the original QP diagnostics and separately
    records the selection decision and its additional computation time.
    """
    if isinstance(grid_points, bool) or int(grid_points) != grid_points or grid_points < 3:
        raise ValueError("grid_points must be an integer of at least three")
    if not np.isfinite(min_relative_improvement) or not 0 <= min_relative_improvement < 1:
        raise ValueError("min_relative_improvement must lie in [0, 1)")
    if not np.isfinite(min_absolute_improvement) or min_absolute_improvement < 0:
        raise ValueError("min_absolute_improvement must be nonnegative")
    previous_info = dict(getattr(controller, "last", {}) or {})
    if not previous_info.get("feasible_shifted_plan_used"):
        return float(previous_info.get("voltage_v", previous_voltage))

    start = time.perf_counter()
    z = np.asarray(state, dtype=float).reshape(-1)
    u_prev = float(previous_voltage)
    if (z.shape != (controller.nx_physical,) or not np.all(np.isfinite(z))
            or not np.isfinite(u_prev) or not controller.u_min <= u_prev <= controller.u_max):
        raise ValueError("Expected finite measured state and bounded previous voltage")
    if controller.nominal_sequence is None:
        raise MPCSolveError("A validated shifted plan must have a current command sequence")
    original = np.asarray(controller.nominal_sequence, dtype=float).copy()
    if original.shape != (controller.horizon,) or not np.all(np.isfinite(original)):
        raise MPCSolveError("Invalid shifted command sequence")
    target = controller._reference(reference)

    def candidate_cost(sequence, predicted):
        return controller._trajectory_cost(predicted, target, sequence, u_prev)

    valid, original_prediction, original_violation, original_input_violation = (
        controller._nonlinear_validation(z, original, u_prev))
    if not valid:
        raise MPCSolveError(
            "Previously validated shifted plan is infeasible under current feedback",
            diagnostics=dict(pressure_violation_mpa=original_violation,
                             input_violation_v=original_input_violation))
    original_cost = candidate_cost(original, original_prediction)
    best = (original_cost, "original_shifted_plan", original, original_prediction,
            original_violation, original_input_violation)
    seen = {tuple(np.round(original, 12))}
    checked, feasible_count, rollout_failures = 1, 1, 0
    voltage_grid = np.unique(np.r_[
        np.linspace(controller.u_min, controller.u_max, int(grid_points)),
        np.clip(u_prev, controller.u_min, controller.u_max),
        np.clip(0.0, controller.u_min, controller.u_max),
    ])
    for voltage in voltage_grid:
        sequence = controller._project_inputs(
            np.full(controller.horizon, voltage, dtype=float), u_prev)
        key = tuple(np.round(sequence, 12))
        if key in seen:
            continue
        seen.add(key)
        checked += 1
        try:
            valid, predicted, pressure_violation, input_violation = (
                controller._nonlinear_validation(z, sequence, u_prev))
        except (MPCSolveError, RuntimeError, ValueError):
            # A candidate whose physical rollout is undefined cannot be
            # selected. Preserve the accepted original plan and count it.
            rollout_failures += 1
            continue
        if not valid:
            continue
        feasible_count += 1
        cost = candidate_cost(sequence, predicted)
        if cost < best[0]:
            best = (cost, "constant_grid", sequence, predicted,
                    pressure_violation, input_violation)

    elapsed_ms = 1000.0*(time.perf_counter()-start)
    selected_cost, source, sequence, predicted, violation, input_violation = best
    improvement = original_cost-selected_cost
    threshold = max(float(min_absolute_improvement),
                    float(min_relative_improvement)*max(abs(original_cost), 1.0))
    replace = source != "original_shifted_plan" and improvement > threshold
    diagnostics = dict(
        plan_refinement_checked=True,
        plan_refinement_used=bool(replace),
        plan_refinement_source=source if replace else "original_shifted_plan",
        plan_refinement_grid_points=int(grid_points),
        plan_refinement_candidates_checked=checked,
        plan_refinement_feasible_candidates=feasible_count,
        plan_refinement_rollout_failures=rollout_failures,
        plan_refinement_original_cost=original_cost,
        plan_refinement_best_candidate_cost=selected_cost,
        plan_refinement_selected_cost=selected_cost if replace else original_cost,
        plan_refinement_absolute_improvement=improvement if replace else 0.0,
        plan_refinement_relative_improvement=(improvement/max(abs(original_cost), 1.0)
                                              if replace else 0.0),
        plan_refinement_required_improvement=threshold,
        plan_refinement_original_voltage_v=float(original[0]),
        plan_refinement_wall_ms=elapsed_ms,
        runtime_ms=float(previous_info.get("runtime_ms", 0.0))+elapsed_ms,
    )
    if not replace:
        controller.last = dict(previous_info, **diagnostics)
        return float(original[0])

    applied = float(sequence[0])
    controller.nominal_sequence = sequence.copy()
    controller._previous_applied = applied
    controller._last_du = np.diff(np.r_[u_prev, sequence])
    controller.last = dict(
        previous_info, **diagnostics,
        status="refined_pressure_feasible",
        shifted_plan_replaced=True,
        # The original QP did return a validated shifted plan. Keep that fact
        # for downstream status classification; the actual command is below.
        feasible_shifted_plan_used=True,
        candidate_voltage_v=applied,
        voltage_v=applied,
        actual_du=float(applied-u_prev),
        actual_du_v=float(applied-u_prev),
        input_violation_v=float(input_violation),
        prediction_violation=float(violation),
        prediction_violation_mpa=float(violation),
        predicted_peak_p1_mpa=float(np.max(predicted[:, 2])),
        predicted_peak_p2_mpa=float(np.max(predicted[:, 3])),
        command_sequence_v=sequence.tolist(),
        prediction=predicted.tolist(),
        nonlinear_cost=selected_cost,
    )
    return applied
