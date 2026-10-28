"""Check that trajectory restoration satisfies the original model equations."""

import argparse
import json
from pathlib import Path

import amigo as am
import numpy as np
from scipy.sparse import csr_matrix
from amigo.algorithm.ipm_state import InteriorPointState

from integrated.model.joint import (
    prepare_joint_cases,
    assemble_joint_model,
    build_directory,
)
from integrated.optimization.driver import CadreOptimizer
from integrated.optimization.checkpoint import project_into_bounds
from integrated.optimization.joint import initialize_adjoints
from integrated.solvers.evaluator import ConsistentEvaluator
from integrated.optimization.projection import (
    TrajectoryProjector,
    ProjectedFilterLineSearch,
    BatteryRestorer,
    BatteryRestorationCase,
    ProjectedEvaluator,
    TrajectoryAdjointProjector,
)


def check_price_refit(model, point, projector):
    optimizer = CadreOptimizer(model, model.create_vector())
    probe = model.create_vector()
    probe.get_vector().copy(point)
    variables = optimizer.optimizer.create_opt_vector(probe.get_vector())
    variables.get_zl().zero()
    variables.get_zu().zero()
    adjoints = TrajectoryAdjointProjector(projector.joint, optimizer.optimizer)
    probe.get_vector().get_array()[adjoints.periodicity] = np.linspace(1000, 2000, 6)
    probe.get_vector().copy_host_to_device()
    adjoints.project(variables)
    before = float(np.max(abs(adjoints.residual.get_array()[adjoints.design])))
    probe.get_vector().copy_device_to_host()
    original = probe.get_vector().get_array()[projector.primal].copy()
    improved = TrajectoryAdjointProjector(
        projector.joint, optimizer.optimizer, refit_prices=True
    )
    improved.project(variables)
    after = float(np.max(abs(improved.residual.get_array()[improved.design])))
    state_error = float(np.max(abs(improved.residual.get_array()[improved.states])))
    assert improved.price_refits == 1 and after < before
    assert state_error < 1e-6
    probe.get_vector().copy_device_to_host()
    np.testing.assert_array_equal(
        probe.get_vector().get_array()[projector.primal], original
    )
    for dual in [variables.get_zl(), variables.get_zu()]:
        dual.copy_device_to_host()
        np.testing.assert_array_equal(dual.get_array(), 0)
    return {
        "price_refit_design_residual_before": before,
        "price_refit_design_residual_after": after,
        "price_refit_state_stationarity_error": state_error,
        "price_refit_preserves_primals_and_bound_duals": True,
    }


def check_full_power_correction(
    model, point, projector, *, restore=False, case_position=0
):
    restorer = BatteryRestorer(projector, full_power=True)
    case = projector.joint.cases[case_position]
    oracle = BatteryRestorationCase(
        restorer,
        case,
        restorer.partitions[case_position],
        point,
        adjust_initial_soc=False,
        full_power=True,
        reference=point.get_array().copy(),
    )
    zero = np.zeros(case.m)
    oracle.evaluate(zero)
    direction = np.random.default_rng(718).normal(size=case.m) * 0.01
    expected = np.r_[
        oracle.equality_jacobian @ direction,
        oracle.inequality_jacobian @ direction,
    ]
    probes = []
    step = 1e-5
    for sign in [1, -1]:
        oracle.evaluate(sign * step * direction, derivatives=False)
        probes.append(np.r_[oracle.equality, oracle.inequality])
    finite = (probes[0] - probes[1]) / (2 * step)
    error = float(np.max(abs(finite - expected)) / max(1, np.max(abs(expected))))
    assert error < 2e-6, error
    report = {
        "full_power_correction_case": case.case_index,
        "full_power_correction_columns": case.m,
        "full_power_correction_derivative_error": error,
    }
    if restore:
        probe = model.create_vector()
        probe.get_vector().copy(point)
        controls = model.get_indices(
            f"case{case.case_index}.control_points.CP_P_comm[:]"
        )
        values = probe.get_vector().get_array()
        values[controls] += 0.001 * np.minimum(
            values[controls], projector.upper.get_array()[controls] - values[controls]
        )
        probe.get_vector().copy_host_to_device()
        projector.project(probe.get_vector())
        repair = BatteryRestorationCase(
            restorer,
            case,
            restorer.partitions[case_position],
            probe.get_vector(),
            adjust_initial_soc=False,
            full_power=True,
            reference=point.get_array().copy(),
        )
        initial_soc = float(values[repair.controls[-1]])
        assert repair.restore(probe.get_vector(), max_iterations=12), repair.message
        assert repair.within_reference_gaps(point.get_array())
        assert probe.get_vector().get_array()[repair.controls[-1]] == initial_soc
        unchanged = np.setdiff1d(projector.joint.design_indices(), controls)
        np.testing.assert_array_equal(
            probe.get_vector().get_array()[unchanged], point.get_array()[unchanged]
        )
        gradient = model.problem.create_vector()
        model.problem.update(probe.get_vector())
        model.problem.gradient(0.0, probe.get_vector(), gradient)
        gradient.copy_device_to_host()
        native_error = float(
            np.max(abs(gradient.get_array()[model.problem.get_constraint_indices()]))
        )
        assert native_error < 1e-7, native_error
        report["full_power_correction_periodicity_error"] = float(
            abs(repair.equality[0])
        )
        report["full_power_correction_minimum_margin"] = float(
            np.min(repair.inequality)
        )
        report["full_power_correction_preserves_initial_soc"] = True
        report["full_power_correction_preserves_reference_gap"] = True
        report["full_power_correction_keeps_other_design_controls"] = True
        report["full_power_correction_native_constraint_error"] = native_error
    return report


def check_bound_dual_scaling(model, point, projector):
    optimizer = CadreOptimizer(model, model.create_vector())
    probe = model.create_vector()
    probe.get_vector().copy(point)
    variables = optimizer.optimizer.create_opt_vector(probe.get_vector())
    primal = projector.primal
    probe.get_vector().get_array()[primal] = project_into_bounds(
        probe.get_vector().get_array()[primal],
        projector.lower.get_array()[primal],
        projector.upper.get_array()[primal],
        1e-6,
    )
    probe.get_vector().copy_host_to_device()
    previous = probe.get_vector().get_array()[primal].copy()
    products = []
    for dual, bounds, sign in [
        (variables.get_zl(), projector.lower, 1),
        (variables.get_zu(), projector.upper, -1),
    ]:
        bound = bounds.get_array()[primal]
        finite = np.isfinite(bound)
        dual.get_array()[:] = 0
        dual.get_array()[finite] = 0.5
        dual.copy_host_to_device()
        products.append(sign * (previous[finite] - bound[finite]) * 0.5)
    soc = model.get_indices("case0.battery.SOC[:]")[0]
    probe.get_vector().get_array()[soc] += 1e-4
    probe.get_vector().copy_host_to_device()
    evaluator = ProjectedEvaluator(
        model.problem, optimizer.optimizer, projector, lambda: None
    )
    assert evaluator.rescale_bound_duals(variables, previous)
    assert evaluator.satisfies_gap_fraction(probe.get_vector(), previous)
    preserved = probe.get_vector().get_array()[soc]
    lower_soc = projector.lower.get_array()[soc]
    original_soc = previous[np.flatnonzero(primal == soc)[0]]
    probe.get_vector().get_array()[soc] = lower_soc + 0.001 * (original_soc - lower_soc)
    probe.get_vector().copy_host_to_device()
    assert not evaluator.satisfies_gap_fraction(probe.get_vector(), previous)
    probe.get_vector().get_array()[soc] = preserved
    probe.get_vector().copy_host_to_device()
    errors = []
    current = probe.get_vector().get_array()[primal]
    for expected, (dual, bounds, sign) in zip(
        products,
        [
            (variables.get_zl(), projector.lower, 1),
            (variables.get_zu(), projector.upper, -1),
        ],
    ):
        bound = bounds.get_array()[primal]
        finite = np.isfinite(bound)
        dual.copy_device_to_host()
        actual = sign * (current[finite] - bound[finite]) * dual.get_array()[finite]
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)
        errors.append(float(np.max(abs(actual - expected))))
    adjoints = TrajectoryAdjointProjector(projector.joint, optimizer.optimizer)
    energy_rows = model.get_indices(
        [
            f"case{case.case_index}.battery_periodicity.res[:]"
            for case in projector.joint.cases
        ]
    )
    probe.get_vector().get_array()[energy_rows] = -0.2
    probe.get_vector().copy_host_to_device()
    original = probe.get_vector().get_array().copy()
    adjoints.project(variables)
    probe.get_vector().copy_device_to_host()
    np.testing.assert_array_equal(
        probe.get_vector().get_array()[primal], original[primal]
    )
    np.testing.assert_array_equal(
        probe.get_vector().get_array()[energy_rows], original[energy_rows]
    )
    model.problem.update(probe.get_vector())
    model.problem.gradient(1.0, probe.get_vector(), adjoints.gradient)
    optimizer.optimizer.compute_dual_residual(
        variables, adjoints.gradient, adjoints.residual
    )
    adjoints.residual.copy_device_to_host()
    state_error = float(np.max(abs(adjoints.residual.get_array()[adjoints.states])))
    assert state_error < 1e-6, state_error
    search = ProjectedFilterLineSearch(
        optimizer.get_options({}), model.problem, optimizer.optimizer
    )
    step, trial = (
        optimizer.optimizer.create_opt_vector(),
        optimizer.optimizer.create_opt_vector(),
    )
    step.get_solution().zero()
    step.get_zl().zero()
    step.get_zu().zero()
    constraints = np.asarray(model.problem.get_constraint_indices())
    step.get_solution().get_array()[constraints] = 0.02
    step.get_zl().get_array()[:] = 0.01
    for vector in [step.get_solution(), step.get_zl()]:
        vector.copy_host_to_device()
    search.optimizer.apply_step_update(0.25, 0.75, variables, step, trial)
    trial.get_solution().copy_device_to_host()
    np.testing.assert_allclose(
        trial.get_solution().get_array()[constraints],
        probe.get_vector().get_array()[constraints] + 0.005,
        atol=1e-14,
    )
    finite = np.isfinite(projector.lower.get_array()[primal])
    trial.get_zl().copy_device_to_host()
    np.testing.assert_allclose(
        trial.get_zl().get_array()[finite],
        variables.get_zl().get_array()[finite] + 0.0075,
        atol=1e-14,
    )
    return {
        "trial_complementarity_product_error": max(errors),
        "repaired_trial_state_stationarity_error": state_error,
        "repaired_trial_gap_fraction_checked": True,
        "equality_multiplier_uses_primal_step": True,
    }


def check_centered_bound_duals(model, point, projector):
    optimizer = CadreOptimizer(model, model.create_vector())
    variables = optimizer.optimizer.create_opt_vector()
    probe = variables.get_solution()
    probe.copy(point)
    probe.copy_device_to_host()
    indices = np.asarray(projector.primal)
    lo, hi = projector.lower.get_array(), projector.upper.get_array()
    probe.get_array()[indices] = project_into_bounds(
        probe.get_array()[indices], lo[indices], hi[indices], 1e-6
    )
    probe.copy_host_to_device()
    original = probe.get_array().copy()
    barrier = [1e-5]
    evaluator = ProjectedEvaluator(
        model.problem,
        optimizer.optimizer,
        projector,
        lambda: variables,
        barrier_provider=lambda: barrier[0],
    )
    maximum_error = 0.0
    for value in [1e-5, 1e-7]:
        barrier[0] = value
        assert evaluator.rescale_bound_duals(variables, original[indices])
        for dual, bounds, sign in [
            (variables.get_zl(), lo, 1),
            (variables.get_zu(), hi, -1),
        ]:
            finite = np.isfinite(bounds[indices])
            gap = sign * (probe.get_array()[indices][finite] - bounds[indices][finite])
            dual.copy_device_to_host()
            products = gap * dual.get_array()[finite]
            np.testing.assert_allclose(products, value, rtol=1e-13, atol=1e-20)
            np.testing.assert_array_equal(dual.get_array()[~finite], 0)
            maximum_error = max(maximum_error, float(np.max(abs(products - value))))
        np.testing.assert_array_equal(probe.get_array(), original)
    first = indices[np.flatnonzero(np.isfinite(lo[indices]))[0]]
    probe.get_array()[first] = lo[first]
    probe.copy_host_to_device()
    assert not evaluator.rescale_bound_duals(variables, original[indices])
    return {
        "centered_bound_complementarity_error": maximum_error,
        "barrier_updates_recenter_duals": True,
        "noninterior_centering_rejected": True,
    }


def check_battery_restoration(joint, point, projector, interior_start=False):
    print("Checking battery-restoration sensitivities...", flush=True)
    restorer = BatteryRestorer(projector, interior=interior_start)
    errors = []
    for adjust_soc in [True, False]:
        oracle = BatteryRestorationCase(
            restorer,
            joint.cases[0],
            restorer.partitions[0],
            point,
            adjust_initial_soc=adjust_soc,
        )
        zero = np.zeros(oracle.mapping.shape[1])
        oracle.evaluate(zero)
        direction = np.random.default_rng(291).normal(size=len(zero)) * 0.01
        exact = np.r_[
            oracle.equality_jacobian @ direction, oracle.inequality_jacobian @ direction
        ]
        probes = []
        for sign in [1, -1]:
            oracle.evaluate(sign * 1e-5 * direction)
            probes.append(np.r_[oracle.equality, oracle.inequality])
        finite = (probes[0] - probes[1]) / 2e-5
        error = float(np.max(abs(exact - finite)) / max(1, np.max(abs(finite))))
        assert error < 1e-6, error
        errors.append(error)
    derivative_error = max(errors)
    original = point.get_array().copy()
    print("Restoring SOC periodicity...", flush=True)
    assert restorer.restore(
        point, verbose=True, adjust_initial_soc=interior_start
    ), restorer.last_message
    point.copy_device_to_host()
    changed = np.concatenate(
        [
            joint.model.get_indices(
                [
                    f"case{case.case_index}.control_points.CP_P_comm[:]",
                    f"case{case.case_index}.battery_ic.initial_soc[:]",
                ]
            )
            for case in joint.cases
        ]
    )
    fixed_design = np.setdiff1d(joint.design_indices(), changed)
    np.testing.assert_array_equal(
        point.get_array()[fixed_design], original[fixed_design]
    )
    restored = joint.model.create_vector()
    restored.get_vector().copy(point)
    report = joint.replay(restored)
    periodicity = max(
        abs(case["battery_feasibility"]["periodicity_error"])
        for case in report["cases"]
    )
    assert periodicity < 1e-9, periodicity
    if interior_start:
        for case in report["cases"]:
            battery = case["battery_feasibility"]
            assert battery["SOC_min"] >= 0.215 - 1e-9
            assert battery["SOC_max"] <= 0.985 + 1e-9
            assert battery["current_min_A"] >= -9.85 - 1e-9
            assert battery["current_max_A"] <= 4.85 + 1e-9
    initial_soc_ids = joint.model.get_indices(
        [f"case{case.case_index}.battery_ic.initial_soc[:]" for case in joint.cases]
    )
    initial_socs = point.get_array()[initial_soc_ids].copy()
    power = joint.model.get_indices(
        f"case{joint.cases[0].case_index}.control_points.CP_P_comm[:]"
    )
    point.get_array()[power] *= 1.0001
    point.copy_host_to_device()
    projector.project(point)
    assert restorer.restore(
        point,
        max_iterations=3,
        adjust_initial_soc=False,
    ), restorer.last_message
    point.copy_device_to_host()
    np.testing.assert_array_equal(point.get_array()[initial_soc_ids], initial_socs)
    optimizer = CadreOptimizer(joint.model, joint.model.create_vector())
    options = optimizer.get_options({"initial_barrier_param": 1e-5})
    ipm = InteriorPointState(point, options, joint.model.problem, optimizer.optimizer)
    ipm.current.get_zl().zero()
    ipm.current.get_zu().zero()
    optimizer.evaluator = ConsistentEvaluator(joint.model.problem, optimizer.optimizer)
    adjoints = initialize_adjoints(joint, optimizer, ipm)
    dual = joint.model.problem.create_vector()
    optimizer.optimizer.compute_dual_residual(ipm.current, ipm.gradient, dual)
    dual.copy_device_to_host()
    states = np.setdiff1d(
        joint.model.problem.get_primal_indices(), joint.design_indices()
    )
    state_error = float(np.max(abs(dual.get_array()[states])))
    assert state_error < 1e-6, state_error
    design = joint.design_indices()
    lower = joint.model.get_lower().get_vector().get_array()[design]
    upper = joint.model.get_upper().get_vector().get_array()[design]
    design_error = float(np.linalg.norm(dual.get_array()[design] * (upper - lower)))
    assert abs(design_error - adjoints["design_residual_after"]) < 1e-7 * max(
        1, design_error
    )
    return {
        "restoration_parameters_per_case": 2 if interior_start else 1,
        "initial_battery_interior_margins": interior_start,
        "battery_restoration_derivative_error": derivative_error,
        "restored_periodicity_error": periodicity,
        "restoration_keeps_other_design_controls": True,
        "trial_restoration_keeps_initial_soc": True,
        **adjoints,
        "native_state_stationarity_error": state_error,
        "native_design_residual": design_error,
    }


def check_filter_fallback(model, point):
    """Exercise AMIGO's minimum-step fallback with finite and invalid probes."""
    optimizer = CadreOptimizer(model, model.create_vector())
    options = optimizer.get_options({})
    state = InteriorPointState(point, options, model.problem, optimizer.optimizer)
    state.max_alpha_primal = state.max_alpha_dual = 1e-12
    for vector in [state.current, state.step]:
        vector.get_zl().zero()
        vector.get_zu().zero()
    state.step.get_solution().zero()
    control = model.get_indices("case0.control_points.CP_P_comm[:]")[0]
    state.step.get_solution().get_array()[control] = 1e6
    state.step.get_solution().copy_host_to_device()
    original = state.current.get_solution().get_array().copy()

    class TrialEvaluator:
        invalid_trial = False
        acceptable = False

        def evaluate_objective_and_infeasibility(self, state):
            state.objective_value = state.log_barrier_value = 0
            state.con_infeasibility = 1

        def evaluate_directional_derivative(self, state):
            return 0

        def evaluate_gradient_from_point(self, scale, point, gradient):
            gradient.zero()

        def evaluate_objective_and_infeasibility_from_point(self, *args):
            if self.invalid_trial:
                return np.inf, np.inf, np.inf
            return (-1, 0, 0) if self.acceptable else (1, 0, 2)

    evaluator = TrialEvaluator()
    for invalid in [False, True]:
        evaluator.invalid_trial = invalid
        search = ProjectedFilterLineSearch(options, model.problem, optimizer.optimizer)
        try:
            search.line_search(None, evaluator, state)
        except RuntimeError as error:
            assert "passed the filter" in str(error)
        else:
            raise AssertionError("A rejected minimum step was accepted")
        state.current.get_solution().copy_device_to_host()
        np.testing.assert_array_equal(
            state.current.get_solution().get_array(), original
        )
    evaluator.invalid_trial = False
    state.max_alpha_primal = state.max_alpha_dual = 1.0
    exhausted = ProjectedFilterLineSearch(
        {**options, "max_line_search_iterations": 2}, model.problem, optimizer.optimizer
    )
    try:
        exhausted.line_search(None, evaluator, state)
    except RuntimeError as error:
        assert "passed the filter" in str(error)
    else:
        raise AssertionError("An exhausted line search was allowed to continue")
    state.current.get_solution().copy_device_to_host()
    np.testing.assert_array_equal(state.current.get_solution().get_array(), original)
    state.max_alpha_primal = state.max_alpha_dual = 1e-12
    evaluator.invalid_trial, evaluator.acceptable = False, True
    search = ProjectedFilterLineSearch(options, model.problem, optimizer.optimizer)
    assert search.line_search(None, evaluator, state).success
    state.max_alpha_primal, state.max_alpha_dual = 1e-12, 1e-30
    centered = ProjectedFilterLineSearch(
        options, model.problem, optimizer.optimizer, centered_duals=True
    )
    info = centered.line_search(None, evaluator, state)
    assert info.success and info.alpha_primal >= 0.9e-12
    state.current.get_solution().get_array()[:] = original
    state.current.get_solution().copy_host_to_device()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=Path("/tmp/cadre-joint-final-check/build")
    )
    parser.add_argument("--reuse-build", action="store_true")
    parser.add_argument("--nodes", type=int, default=8)
    parser.add_argument("--control-points", type=int, default=4)
    parser.add_argument("--full-window", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--restore-periodicity", action="store_true")
    parser.add_argument("--interior-start", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.restore_periodicity and not args.full_window:
        parser.error("Periodicity restoration tests require --full-window")
    if args.interior_start and not args.restore_periodicity:
        parser.error("--interior-start requires --restore-periodicity")
    cases, options = prepare_joint_cases(
        list(range(6)),
        nodes=args.nodes,
        controls=args.control_points,
        full_duration=args.full_window,
    )
    joint = assemble_joint_model(cases, options, objective_scale=1e-3)
    if args.reuse_build:
        with build_directory(args.build_dir):
            joint.model._built = True
            joint.model.initialize(order_type=am.OrderingType.NATURAL)
    else:
        joint.initialize(args.build_dir)
    model = joint.model
    state = model.get_initial_point()
    point = state.get_vector()
    if args.checkpoint:
        with np.load(args.checkpoint, allow_pickle=False) as saved:
            point.get_array()[:] = saved["solution"]
        point.copy_host_to_device()
    projector = TrajectoryProjector(joint)
    projector.project(point)
    constraints = model.problem.get_constraint_indices()
    periodicity = model.get_indices(
        [f"case{c.case_index}.battery_periodicity.res[:]" for c in cases]
    )
    rows = np.setdiff1d(constraints, periodicity)
    point.fill_at(constraints, 0.0)
    gradient = model.problem.create_vector()
    model.problem.update(point)
    model.problem.gradient(1.0, point, gradient)
    gradient.copy_device_to_host()
    residual = float(np.max(abs(gradient.get_array()[rows])))
    assert residual < 1e-7, residual
    native = model.create_matrix()
    model.eval_hessian(state, native)
    native.copy_data_device_to_host()
    n, m, _, rowp, cols = native.get_nonzero_structure()
    matrix = csr_matrix((native.get_data().copy(), cols, rowp), shape=(n, m))
    original = point.get_array().copy()
    direction = np.zeros(n)
    design = joint.design_indices()
    direction[design] = np.random.default_rng(514).normal(size=len(design)) * 0.01
    probes = []
    step = 1e-5
    for sign in [1, -1]:
        point.get_array()[:] = original + sign * step * direction
        point.copy_host_to_device()
        projector.project(point)
        probes.append(point.get_array().copy())
    tangent = (probes[0] - probes[1]) / (2 * step)
    scale = max(1.0, np.max(abs(matrix[rows] @ direction)))
    tangent_error = float(np.max(abs(matrix[rows] @ tangent)) / scale)
    assert tangent_error < 1e-6, tangent_error
    point.get_array()[:] = original
    point.copy_host_to_device()
    restoration = {}
    if args.interior_start:
        restoration = check_battery_restoration(
            joint, point, projector, interior_start=True
        )
        original = point.get_array().copy()
    dual_checks = check_bound_dual_scaling(model, point, projector)
    centered_checks = check_centered_bound_duals(model, point, projector)
    price_checks = check_price_refit(model, point, projector)
    check_filter_fallback(model, point)
    if args.interior_start:
        # The accepted synthetic filter probe deliberately changes its point.
        point.get_array()[:] = original
        point.copy_host_to_device()
    if args.restore_periodicity and not args.interior_start:
        point.get_array()[:] = original
        point.copy_host_to_device()
        restoration = check_battery_restoration(joint, point, projector)
    if args.restore_periodicity:
        point.fill_at(constraints, 0.0)
        model.problem.update(point)
        model.problem.gradient(1.0, point, gradient)
        gradient.copy_device_to_host()
        restored_residual = float(np.max(abs(gradient.get_array()[constraints])))
        assert restored_residual < 1e-7, restored_residual
        restoration["restored_native_constraint_error"] = restored_residual
    report = {
        **check_full_power_correction(
            model,
            point,
            projector,
            restore=args.restore_periodicity,
            case_position=3 if args.restore_periodicity else 0,
        ),
        "nodes": args.nodes,
        "control_points": args.control_points,
        "full_window": args.full_window,
        "nonperiodic_constraint_max_error": residual,
        "trajectory_tangent_relative_error": tangent_error,
        "six_periodicity_constraints_retained": len(periodicity) == 6,
        "rejected_minimum_steps_rolled_back": True,
        "accepted_filter_step_retained": True,
        "exhausted_line_search_rejected": True,
        **dual_checks,
        **centered_checks,
        **price_checks,
        **restoration,
    }
    filename = (
        "projection_full_window_validation.json"
        if args.full_window
        else "projection_validation.json"
    )
    path = args.output or Path("integrated/results/validation") / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
