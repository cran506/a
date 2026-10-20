"""Independent ODE and barrier verification; no reference-figure curve fitting.

Run ``python test_physics_safety.py`` to save the measured validation summary.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import unittest
import numpy as np
from scipy.integrate import solve_ivp

from hydraulic_twin import HydraulicTwinCore
from safety_experiment import run_safety_experiment
from safety_filter import HOCBFSafetyFilter, SafetyInfeasibleError


def validate_physics():
    twin = HydraulicTwinCore()
    equilibria = []
    for velocity, load in ((.035, 0.), (.05, 0.), (.065, 5000.),
                            (-.05, 0.), (0., 0.), (0., 5000.), (0., -1000.)):
        state, voltage = twin.equilibrium_for_velocity(velocity, load)
        derivative = twin.rhs(state, voltage, load)
        equilibria.append({"velocity_m_s": velocity, "load_n": load,
            "voltage_v": voltage, "state_si": state.tolist(),
            "mechanical_residual_m_s2": float(abs(derivative[1])),
            "pressure_residual_pa_s": float(np.max(abs(derivative[2:])))})

    initial, _ = twin.equilibrium_for_velocity(.05)
    # Every jump is a declared held-input or fault event. Radau and RK4 each
    # stop at those same events; no solver interpolates across a discontinuity.
    commands = ([(3.51, 0., 1.)] * 20 + [(9., 5000., .5)] * 10
                + [(6.68, 5000., .5)] * 70 + [(3.51, 0., 1.)] * 50)
    reference = initial.copy()
    reference_history = []
    for voltage, load, beta in commands:
        solution = solve_ivp(lambda _t, y: twin.rhs(y, voltage, load, beta),
            (0., .001), reference, method="Radau", rtol=2e-10,
            atol=[1e-12, 1e-10, .01, .01])
        if not solution.success:
            raise AssertionError("Independent Radau integration failed")
        reference = solution.y[:, -1]
        reference_history.append(reference.copy())
    reference_history = np.array(reference_history)
    errors = {}
    trajectories = {}
    for count in (8, 16, 32):
        twin.reset(initial)
        history = []
        for voltage, load, beta in commands:
            state, _ = twin.step(voltage, load_n=load,
                                 beta_multiplier=beta, substeps=count)
            history.append(state)
        history = np.array(history)
        trajectories[count] = history
        errors[str(count)] = np.max(abs(history-reference_history), axis=0).tolist()

    twin.reset(initial)
    start = time.perf_counter()
    substeps = []
    for index in range(600):
        state, info = twin.step(3.5098639505149896,
            load_n=5000. if index >= 100 else 0.,
            beta_multiplier=.5 if index >= 100 else 1.)
        substeps.append(info["integration_substeps"])
    elapsed = time.perf_counter()-start
    eight = np.array(errors["8"])
    thirty_two = np.array(errors["32"])
    checks = {
        "all_equilibrium_force_balances": all(x["mechanical_residual_m_s2"] < 1e-9 for x in equilibria),
        "all_equilibrium_flow_balances": all(x["pressure_residual_pa_s"] < .01 for x in equilibria),
        "rk4_radau_position_agreement_better_than_1um": bool(eight[0] < 1e-6),
        "rk4_radau_speed_agreement_better_than_0_01mm_s": bool(eight[1] < 1e-5),
        "rk4_radau_pressure_agreement_better_than_0_001MPa": bool(max(eight[2:]) < 1000),
        "subdivision_improves_all_state_errors": bool(np.all(thirty_two < eight)),
        "all_tested_pressures_positive": bool(np.min(trajectories[8][:, 2:]) > 0),
        "fault_run_remains_finite": bool(np.all(np.isfinite(state))),
        "fault_reduces_speed_at_fixed_voltage": bool(0 < state[1] < .04),
    }
    return {"experiment": "independent_Radau_and_RK4_comparison", "dt_s": .001,
        "fault_load_n": 5000., "fault_beta_multiplier": .5,
        "equilibria": equilibria, "comparison_duration_s": .15,
        "max_absolute_state_error_si_by_substeps": errors,
        "fixed_voltage_fault_final_state_si": state.tolist(),
        "benchmark_600_native_steps_seconds": elapsed,
        "mean_internal_substeps": float(np.mean(substeps)),
        "checks": checks, "passed": bool(all(checks.values()))}


class TestPhysicsAndSafety(unittest.TestCase):
    def test_configuration_and_explicit_override_precedence(self):
        config = dict(m=300., beta_e=1.2e9, F_c=1400.,
                      initial_position_m=.002, initial_velocity_m_s=.04)
        twin = HydraulicTwinCore(config=config)
        self.assertEqual(twin.p["m"], 300.)
        self.assertEqual(twin.p["beta_e"], 1.2e9)
        self.assertEqual(twin.p["F_c"], 1400.)
        self.assertEqual(twin.state[0], .002)
        self.assertEqual(twin.state[1], .04)
        overridden = HydraulicTwinCore(m=275., config=config)
        self.assertEqual(overridden.p["m"], 275.)
        self.assertEqual(overridden.p["F_c"], 1400.)

    def test_radau_equilibrium_and_subdivision(self):
        result = validate_physics()
        self.assertTrue(result["passed"], result)

    def test_invalid_command_does_not_advance_or_modify_state(self):
        twin = HydraulicTwinCore()
        before = twin.state.copy()
        with self.assertRaises(ValueError):
            twin.step(10.1)
        np.testing.assert_array_equal(twin.state, before)
        self.assertEqual(twin.t, 0.)

    def test_nonfinite_action_and_uncertified_state_rejected(self):
        filt = HOCBFSafetyFilter()
        with self.assertRaises(ValueError):
            filt.filter_action(20., 0., float("nan"))
        with self.assertRaises(SafetyInfeasibleError):
            filt.filter_action(98., 100., 8.5)

    def test_safe_action_transparent(self):
        filt = HOCBFSafetyFilter()
        self.assertEqual(filt.filter_action(50., 0., .1), .1)
        self.assertFalse(filt.last_info.intervened)


if __name__ == "__main__":
    import sys
    if "--write-summary" in sys.argv:
        sys.argv.remove("--write-summary")
        out = Path(__file__).resolve().parent/"outputs"
        out.mkdir(exist_ok=True)
        result = validate_physics()
        (out/"physics_validation.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        safety = run_safety_experiment(out)
        if not result["passed"] or not safety["summary"]["passed"]:
            raise SystemExit("Independent verification failed")
    unittest.main()
