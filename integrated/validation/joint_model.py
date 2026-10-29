"""Check joint assembly, derivatives, and a consistent KKT solve without optimizing."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import amigo as am
import numpy as np
from scipy.sparse import csr_matrix

from cadre_paths import source_provenance
from integrated.model.joint import (
    assemble_joint_model,
    build_directory,
    prepare_joint_cases,
)
from integrated.solvers.native_ldl import temporal_kkt_ordering
from integrated.optimization.driver import CadreOptimizer
from integrated.solvers.joint import (
    BoxStepBounds,
    JointKKTSolver,
    StepBoundIterationLimit,
    joint_partitions,
)
from integrated.validation.helpers import Matrix, Vector


def check_joint(joint):
    model = joint.model
    design = joint.design_indices()
    assert len(design) == len(np.unique(design)) == joint.design_variable_count
    shared = model.get_indices("shared_surface.cellInstd[0, :]")
    fin = int(model.get_indices("shared_angles.finAngle[:]")[0])
    antenna = int(model.get_indices("shared_angles.antAngle[:]")[0])
    lower = model.get_lower().get_vector().get_array()
    upper = model.get_upper().get_vector().get_array()

    for name, lo, hi in [
        ("shared_surface.cellInstd[0, :]", 0, 1),
        ("shared_angles.finAngle[:]", 0, np.pi / 2),
        ("shared_angles.antAngle[:]", -np.pi / 4, np.pi / 4),
    ]:
        indices = model.get_indices(name)
        np.testing.assert_array_equal(lower[indices], lo)
        np.testing.assert_array_equal(upper[indices], hi)

    for case in joint.cases:
        prefix = f"case{case.case_index}."
        np.testing.assert_array_equal(
            shared, model.get_indices(prefix + "surface.cellInstd[0, :]")
        )
        assert fin == model.get_indices(prefix + "angles.finAngle[:]")[0]
        assert antenna == model.get_indices(prefix + "angles.antAngle[:]")[0]

        for name, lo, hi in [
            ("control_points.CP_P_comm[:]", 0, 25),
            ("control_points.CP_Isetpt[:, :]", 0, 0.4),
            ("control_points.CP_gamma[:]", 0, np.pi / 2),
            ("battery_ic.initial_soc[:]", 0.2, 1),
            ("battery.SOC[:]", 0.2, 1),
            ("battery.I_bat[:]", -10, 5),
        ]:
            indices = model.get_indices(prefix + name)
            np.testing.assert_array_equal(lower[indices], lo)
            np.testing.assert_array_equal(upper[indices], hi)

    order = temporal_kkt_ordering(model, joint.cases[0].n)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(lower)))

    state = model.get_initial_point()
    vector = state.get_vector()
    values = vector.get_array()
    original = values.copy()
    replay = joint.replay(state)
    replay_error = max(
        error
        for case in replay["cases"]
        for error in case["reference_max_errors"].values()
    )
    assert replay_error < 1e-7, replay_error
    gradient = model.create_vector()
    duals = model.problem.get_constraint_indices()
    final_data, periodicity = [], []

    for index, case in enumerate(joint.cases):
        prefix = f"case{case.case_index}."
        ids = np.atleast_1d(model.get_indices(prefix + f"comm_data.Data[{case.n-1}]"))
        final_data.extend(ids)
        values[ids] = index + 1
        np.testing.assert_array_equal(
            ids, np.atleast_1d(model.get_indices(f"objective.Data[0, {index}]"))
        )
        values[model.get_indices(prefix + "battery.SOC[0]")] = 0.4
        values[model.get_indices(prefix + f"battery.SOC[{case.n-1}]")] = 0.6
        periodicity.extend(model.get_indices(prefix + "battery_periodicity.res[:]"))

    values[duals] = 0
    model.problem.update(vector)
    objective = model.problem.lagrangian(1.0, vector)
    expected = -len(joint.cases) * (len(joint.cases) + 1) / 2 * joint.objective_scale
    assert abs(objective - expected) < 1e-12, (objective, expected)
    model.eval_gradient(state, gradient)
    np.testing.assert_allclose(
        gradient.get_vector().get_array()[final_data],
        -joint.objective_scale,
        atol=1e-12,
    )
    assert len(set(periodicity)) == len(joint.cases)
    np.testing.assert_allclose(
        gradient.get_vector().get_array()[periodicity], 0.2, atol=1e-12
    )

    values[:] = original
    values[shared] = 0.5
    values[fin], values[antenna] = 1.02, 0.17

    for case in joint.cases:
        prefix = f"case{case.case_index}."
        values[model.get_indices(prefix + "values.Gamma[:]")] = np.linspace(
            0.4, 1.0, case.n
        )

        for k in range(5):
            values[model.get_indices(prefix + f"thermal.temperature[:, {k}]")] = 280 + k

        for panel in range(12):
            values[model.get_indices(prefix + f"values.Isetpt[:, {panel}]")] = 0.12

    rng = np.random.default_rng(974)
    values[duals] = rng.normal(size=len(duals)) * 0.1
    point = values.copy()
    joint.set_point_provider(lambda: vector.get_array())
    model.problem.update(vector)
    matrix = model.create_matrix()
    model.eval_hessian(state, matrix)
    matrix.copy_data_device_to_host()
    n, m, _, rowp, cols = matrix.get_nonzero_structure()
    native = csr_matrix(
        (np.array(matrix.get_data(), copy=True), cols, rowp), shape=(n, m)
    )
    hessian = native + joint.hessian_correction((n, m))
    direction = rng.normal(size=len(values))

    for expression, indices in model._staged_fixed_vars:
        fixed = model.get_indices(expression)
        direction[fixed if indices is None else fixed[indices]] = 0

    step = 1e-5
    probes = []

    for sign in [1, -1]:
        values[:] = point + sign * step * direction
        model.problem.update(vector)
        model.eval_gradient(state, gradient)
        probes.append(gradient.get_vector().get_array().copy())

    values[:] = point
    model.problem.update(vector)
    finite = (probes[0] - probes[1]) / (2 * step)
    error = np.max(np.abs(hessian @ direction - finite)) / max(
        1, np.max(np.abs(finite))
    )
    asymmetry = np.max(np.abs((hessian - hessian.T).data), initial=0)
    assert np.all(np.isfinite(hessian.data))
    assert error < 2e-6, error
    assert asymmetry < 1e-8, asymmetry
    partitions, shared_design = joint_partitions(joint)
    solver = JointKKTSolver(
        partitions,
        shared_design,
        model.problem.get_primal_indices(),
        duals,
        joint.hessian_correction,
    )
    diagonal = np.zeros(n)
    diagonal[model.problem.get_primal_indices()] = 0.7
    solver.factor(Matrix(native), Vector(diagonal))
    # Use a consistent perturbation of this physical KKT system. Arbitrary
    # equality rows at a short contact-free window can require 1e13-sized
    # multipliers even with a direct pivoted solve.
    known_step = rng.normal(size=n)
    rhs = solver.target @ known_step
    solution = Vector(np.zeros(n))
    solver.solve(Vector(rhs), solution)
    selective = JointKKTSolver(
        partitions,
        shared_design,
        model.problem.get_primal_indices(),
        duals,
        joint.hessian_correction,
        selective_curvature=True,
        case_workers=2,
    )
    selective.factor(Matrix(native), Vector(diagonal))
    selective_rhs = selective.target @ known_step
    selective.solve(Vector(selective_rhs), solution)
    state_penalty = np.zeros(n)
    state_penalty[shared_design] = 0.2
    for case in joint.cases:
        for name, weight in [("battery.SOC[:]", 100), ("battery.I_bat[:]", 0.5)]:
            ids = model.get_indices(f"case{case.case_index}.{name}")
            state_penalty[ids[len(ids) // 2]] = weight
    selective.refactor_limited_blocks(state_penalty)
    selective.record_step_shift(1.0, state_penalty)
    selective_rhs = selective.target @ known_step
    selective.solve(Vector(selective_rhs), solution)
    state_penalty_residual = float(
        np.linalg.norm(selective.target @ solution.get_array() - selective_rhs, np.inf)
        / max(1, np.linalg.norm(selective_rhs, np.inf))
    )
    assert state_penalty_residual < 1e-7
    spectral = JointKKTSolver(
        partitions,
        shared_design,
        model.problem.get_primal_indices(),
        duals,
        joint.hessian_correction,
        spectral_curvature=True,
    )
    spectral.factor(Matrix(native), Vector(diagonal))
    spectral_rhs = spectral.target @ known_step
    spectral.solve(Vector(spectral_rhs), solution)
    primal = np.asarray(model.problem.get_primal_indices())
    bounded = primal[np.isfinite(lower[primal]) | np.isfinite(upper[primal])]
    independent = np.concatenate(
        [part.controls for part in partitions] + [shared_design]
    )
    box = JointKKTSolver(
        partitions,
        shared_design,
        primal,
        duals,
        joint.hessian_correction,
        spectral_curvature=True,
        box_qp=True,
        limited_indices=independent,
        step_limits=0.1 * (upper[independent] - lower[independent]),
        step_bounds=BoxStepBounds(
            bounded,
            lower[bounded],
            upper[bounded],
            0.1 * (upper[bounded] - lower[bounded]),
            lambda: values,
        ),
    )
    box.factor(Matrix(native), Vector(diagonal))
    response_error = 0.0
    for position in np.linspace(0, len(bounded) - 1, 12, dtype=int):
        probe = np.zeros(n)
        probe[bounded[position]] = 1
        original_response = box._solve(probe)[bounded]
        cached_response = box.bound_inverse_response(position)
        difference = np.max(abs(original_response - cached_response)) / max(
            1, np.max(abs(original_response))
        )
        response_error = max(response_error, float(difference))
    assert response_error < 1e-7
    box_rhs = rng.normal(size=n)
    box_rhs[duals] = 0
    box.solve(Vector(box_rhs), solution)
    box_residual = box.max_linear_residual
    box.factor(Matrix(native), Vector(diagonal))
    with patch(
        "integrated.solvers.joint.active_box_constraints",
        side_effect=StepBoundIterationLimit(1),
    ):
        box.solve(Vector(box_rhs), solution)
    assert box.box_qp_fallbacks == 1
    assert box.design_step_ratio(solution.get_array()) <= 1 + 1e-6
    fallback_residual = float(
        np.linalg.norm(box.target @ solution.get_array() - box_rhs, np.inf)
        / max(1, np.linalg.norm(box_rhs, np.inf))
    )
    assert fallback_residual < 1e-7
    controlled = JointKKTSolver(
        partitions,
        shared_design,
        primal,
        duals,
        joint.hessian_correction,
        limited_indices=independent,
        step_limits=0.1 * (upper[independent] - lower[independent]),
        step_bounds=BoxStepBounds(
            bounded,
            lower[bounded],
            upper[bounded],
            0.1 * (upper[bounded] - lower[bounded]),
            lambda: values,
            symmetric=True,
        ),
    )
    controlled.factor(Matrix(native), Vector(diagonal))
    feasible_direction = known_step.copy()
    feasible_direction[primal] *= 1e-4
    negative, positive = controlled.step_bounds.limits()
    feasible_direction[bounded] = (
        0.01 * np.minimum(negative, positive) * np.tanh(known_step[bounded])
    )
    feasible_direction[design] = 0
    control_rhs = controlled.target @ feasible_direction
    control_rhs[independent] += 100 * rng.normal(size=len(independent))
    assert np.max(abs(control_rhs[duals])) > 0
    controlled.solve(Vector(control_rhs), solution)
    control_residual = float(
        np.linalg.norm(controlled.target @ solution.get_array() - control_rhs, np.inf)
        / max(1, np.linalg.norm(control_rhs, np.inf))
    )
    assert control_residual < 1e-7
    assert controlled.design_step_ratio(solution.get_array()) <= 1 + 1e-6
    backend = CadreOptimizer(model, model.create_vector()).optimizer
    current = backend.create_opt_vector(vector)
    for dual, bounds, sign in [
        (current.get_zl(), lower, 1),
        (current.get_zu(), upper, -1),
    ]:
        dual.zero()
        finite = np.isfinite(bounds[primal])
        gaps = sign * (values[primal[finite]] - bounds[primal[finite]])
        dual.get_array()[finite] = 1 / gaps
        dual.copy_host_to_device()

    def check_positive_dual_step(direction):
        native_direction = model.problem.create_vector()
        native_direction.get_array()[:] = direction
        native_direction.copy_host_to_device()
        full_step = backend.create_opt_vector()
        for barrier in [1.0, 0.1, 1e-6]:
            backend.compute_update(barrier, current, native_direction, full_step)
            alpha_primal, _, alpha_dual, _ = backend.compute_max_step(
                0.995, current, full_step
            )
            assert alpha_primal >= 0.5 and alpha_dual >= 0.5

    check_positive_dual_step(solution.get_array())
    controlled.box_qp = True
    controlled.factor(Matrix(native), Vector(diagonal))
    controlled.solve(Vector(control_rhs), solution)
    full_space_box_residual = float(
        np.linalg.norm(controlled.target @ solution.get_array() - control_rhs, np.inf)
        / max(1, np.linalg.norm(control_rhs, np.inf))
    )
    assert full_space_box_residual < 1e-7
    assert controlled.design_step_ratio(solution.get_array()) <= 1 + 1e-6
    check_positive_dual_step(solution.get_array())
    controlled.box_qp = False
    controlled.spectral_curvature = controlled.selective_curvature = True
    controlled.factor(Matrix(native), Vector(diagonal))
    controlled.solve(Vector(control_rhs), solution)
    spectral_control_residual = float(
        np.linalg.norm(controlled.target @ solution.get_array() - control_rhs, np.inf)
        / max(1, np.linalg.norm(control_rhs, np.inf))
    )
    assert spectral_control_residual < 1e-7
    assert controlled.design_step_ratio(solution.get_array()) <= 1 + 1e-6
    check_positive_dual_step(solution.get_array())
    return {
        "cadre_source": source_provenance(),
        "cases": [case.case_index for case in joint.cases],
        "design_variables": joint.design_variable_count,
        "objective_scale": joint.objective_scale,
        "shared_design_variables": 86,
        "shared_aliases_verified": True,
        "design_and_battery_bounds_verified": True,
        "scoped_temporal_permutation_verified": True,
        "summed_objective_error": float(abs(objective - expected)),
        "separate_soc_periodicity_verified": True,
        "independent_initial_replay_max_error": float(replay_error),
        "assembled_hessian_direction_relative_error": float(error),
        "assembled_hessian_asymmetry": float(asymmetry),
        "joint_kkt_relative_residual": solver.max_linear_residual,
        "selective_joint_kkt_relative_residual": selective.max_linear_residual,
        "state_penalty_kkt_relative_residual": state_penalty_residual,
        "spectral_kkt_relative_residual": spectral.max_linear_residual,
        "active_box_kkt_relative_residual": box_residual,
        "active_box_fallback_kkt_relative_residual": fallback_residual,
        "active_box_fallback_step_bounds_passed": True,
        "full_space_control_bound_kkt_residual": control_residual,
        "full_space_control_bounds_with_equality_defects": True,
        "half_dual_step_positive_with_symmetric_limits": True,
        "full_space_active_box_kkt_residual": full_space_box_residual,
        "full_space_active_box_half_dual_step_positive": True,
        "full_space_spectral_control_bound_kkt_residual": spectral_control_residual,
        "full_space_spectral_half_dual_step_positive": True,
        "active_box_largest_set": box.largest_active_set,
        "cached_bound_response_relative_error": response_error,
        "parallel_case_workers_checked": selective.case_workers,
        "optimization_run": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=Path("/tmp/cadre-joint-validation/build")
    )
    parser.add_argument(
        "--reuse-build",
        action="store_true",
        help="Reuse a build from the identical source revision and six-case configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("integrated/results/validation/joint_model_validation.json"),
    )
    args = parser.parse_args(argv)
    cases, options = prepare_joint_cases(list(range(6)), nodes=8, controls=4)
    joint = assemble_joint_model(cases, options, objective_scale=1e-3)

    if args.reuse_build:
        with build_directory(args.build_dir):
            joint.model._built = True
            joint.model.initialize(order_type=am.OrderingType.NATURAL)
    else:
        joint.initialize(args.build_dir)

    report = check_joint(joint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
