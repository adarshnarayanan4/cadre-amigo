"""Check joint KKT solves, curvature correction, and case partition failures."""

import argparse
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from integrated.solvers.joint import (
    BoxStepBounds,
    CasePartition,
    JointKKTSolver,
    positive_spectral_factor,
    active_box_constraints,
    StepBoundIterationLimit,
)
from integrated.validation.helpers import Matrix, Vector, saved_bound_response


def check_system(
    indefinite=False,
    normal=False,
    limit_design=False,
    selective=False,
    workers=1,
    bound_aware=False,
    spectral=False,
    box_qp=False,
    force_box_fallback=False,
    control_bounds_only=False,
    nonzero_constraint_rhs=False,
    symmetric_bounds=False,
):
    rng = np.random.default_rng(216)
    size, shared = 35, np.array([33, 34])
    matrix = np.zeros((size, size))
    partitions, primal, constraints = [], list(shared), []
    for case in range(3):
        indices = np.arange(case * 11, (case + 1) * 11)
        states, duals, controls = indices[:4], indices[4:8], indices[8:]
        partitions.append(CasePartition(indices, controls))
        primal.extend(np.r_[states, controls])
        constraints.extend(duals)
        jacobian = np.eye(4) + rng.normal(size=(4, 4)) * 0.1
        matrix[np.ix_(duals, states)] = jacobian
        matrix[np.ix_(duals, controls)] = rng.normal(size=(4, 3))
        matrix[np.ix_(duals, shared)] = rng.normal(size=(4, 2))
        if not normal:
            matrix[np.ix_(states, states)] = np.eye(4) * 0.2
            matrix[np.ix_(controls, controls)] = np.eye(3) * (-10 if indefinite else 5)
    matrix += matrix.T - np.diag(matrix.diagonal())
    if not normal:
        matrix[np.ix_(shared, shared)] = np.eye(2) * (-5 if indefinite else 10)
    diagonal = np.zeros(size)
    if normal:
        diagonal[primal] = 1
    limited = np.array([8, 19, 30, 33]) if limit_design else None
    limits = np.array([0.2, 0.15, 0.3, 0.1]) if limit_design else None
    bounds = None
    if bound_aware:
        limited = np.concatenate([part.controls for part in partitions] + [shared])
        limits = np.full(len(limited), 0.1)
        point = np.zeros(size)
        point[primal] = np.resize([0.01, 0.99, 0.5], len(primal))
        bounded = limited if control_bounds_only else np.asarray(primal)
        bounds = BoxStepBounds(
            bounded,
            np.zeros(len(bounded)),
            np.ones(len(bounded)),
            np.full(len(bounded), 0.1),
            lambda: point,
            symmetric=control_bounds_only or symmetric_bounds,
        )
    solver = JointKKTSolver(
        partitions,
        shared,
        primal,
        constraints,
        lambda shape: csr_matrix(shape),
        chunk=2,
        limited_indices=limited,
        step_limits=limits,
        selective_curvature=selective,
        spectral_curvature=spectral,
        box_qp=box_qp,
        case_workers=workers,
        step_bounds=bounds,
    )
    solver.factor(Matrix(matrix), Vector(diagonal))
    if box_qp:
        for position, index in enumerate(bounds.indices):
            probe = np.zeros(size)
            probe[index] = 1
            np.testing.assert_allclose(
                solver.bound_inverse_response(position),
                solver._solve(probe)[bounds.indices],
                rtol=1e-8,
                atol=1e-9,
            )
    original_target = solver.target.toarray()
    rhs = rng.normal(size=size) * (100 if limit_design or bound_aware else 1)
    if bound_aware and not nonzero_constraint_rhs:
        # A replayed trajectory has homogeneous linearized state equations.
        rhs[constraints] = 0
    elif bound_aware and symmetric_bounds:
        # Keep a nonzero equality defect small enough for bounded state steps.
        rhs[constraints] *= 1e-5
    solution = Vector(np.zeros(size))
    if force_box_fallback:
        with patch(
            "integrated.solvers.joint.active_box_constraints",
            side_effect=StepBoundIterationLimit(1),
        ):
            solver.solve(Vector(rhs), solution)
        assert solver.box_qp_fallbacks == 1
    else:
        solver.solve(Vector(rhs), solution)
    expected = np.linalg.solve(solver.target.toarray(), rhs)
    np.testing.assert_allclose(solution.get_array(), expected, rtol=1e-8, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(solver.target.toarray())
    assert np.count_nonzero(eigenvalues < 0) == len(constraints)
    assert (solver.regularization_max > 0) == (
        indefinite or limit_design or bound_aware
    )
    if limit_design or bound_aware:
        limited = solver.limited_indices
        assert (
            np.max(abs(solution.get_array()[limited]) / solver.step_limits) <= 1 + 1e-8
        )
        correction = solver.target.toarray() - original_target
        unmodified = (
            constraints if bound_aware else np.setdiff1d(np.arange(size), limited)
        )
        np.testing.assert_allclose(correction[unmodified], 0, atol=1e-12)
        np.testing.assert_allclose(
            correction - np.diag(np.diag(correction)), 0, atol=1e-12
        )
        if bound_aware:
            assert solver.design_step_ratio(solution.get_array()) <= 1 + 1e-8
            states = np.setdiff1d(primal, limited)
            if control_bounds_only:
                np.testing.assert_allclose(np.diag(correction)[states], 0, atol=1e-12)
            else:
                assert np.max(np.diag(correction)[states]) > 0
        # A second RHS must use the same regularized matrix and state equations.
        rhs = rng.normal(size=size)
        solver.solve(Vector(rhs), solution)
        np.testing.assert_allclose(
            solution.get_array(),
            np.linalg.solve(solver.target.toarray(), rhs),
            atol=1e-8,
            rtol=1e-8,
        )
        if bound_aware and not indefinite:
            # Retrying curvature must remove the previous state penalties too.
            solver.strengthen_curvature_safeguard()
            rhs[constraints] = 0
            solver.solve(Vector(rhs), solution)
            np.testing.assert_allclose(
                solution.get_array(),
                np.linalg.solve(solver.target.toarray(), rhs),
                atol=1e-8,
                rtol=1e-8,
            )
    # An invalid partition must fail before a numerical result is accepted.
    solver.partitions = partitions[:-1]
    try:
        solver.factor(Matrix(matrix), Vector(diagonal))
    except ValueError:
        pass
    else:
        raise AssertionError("Uncovered case rows were accepted")
    return {
        "relative_linear_residual": solver.max_linear_residual,
        "box_qp_fallbacks": solver.box_qp_fallbacks,
        "negative_eigenvalues": int(np.count_nonzero(eigenvalues < 0)),
        "expected_negative_eigenvalues": len(constraints),
    }


def check_coupled_curvature(selective=False, spectral=False):
    # Positive private blocks can still have negative curvature through sharing.
    matrix = np.diag([0.2, 0.0, 1.0, 1e-4, 1.0])
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[2, 4] = matrix[4, 2] = 2
    solver = JointKKTSolver(
        [CasePartition(np.arange(4), np.array([2, 3]))],
        [4],
        [0, 2, 3, 4],
        [1],
        lambda shape: csr_matrix(shape),
        selective_curvature=selective,
        spectral_curvature=spectral,
    )
    solver.factor(Matrix(matrix), Vector(np.zeros(5)))
    correction = (solver.target.toarray() - matrix).diagonal()[[2, 3, 4]]
    if selective or spectral:
        np.testing.assert_allclose(correction[:2], 0, atol=1e-12)
        assert correction[2] > 0
    else:
        np.testing.assert_allclose(correction / [1, 1e-4, 1], solver.curvature_shift)
    assert solver.curvature_shift > 0
    assert np.count_nonzero(np.linalg.eigvalsh(solver.target.toarray()) < 0) == 1
    # Rounding-level negative curvature must not force the larger design shift.
    matrix[2, 4] = matrix[4, 2] = 1 + 1e-13
    solver.factor(Matrix(matrix), Vector(np.zeros(5)))
    assert 0 < solver.curvature_shift < 1e-8
    assert np.count_nonzero(np.linalg.eigvalsh(solver.target.toarray()) < 0) == 1
    previous_shift = solver.curvature_shift
    solver.strengthen_curvature_safeguard()
    assert solver.curvature_shift > previous_shift
    np.testing.assert_allclose(
        solver.target.diagonal() - matrix.diagonal(),
        solver.regularization_diagonal,
        atol=1e-15,
    )
    rhs = solver.target @ np.ones(5)
    result = Vector(np.zeros(5))
    solver.solve(Vector(rhs), result)
    np.testing.assert_allclose(solver.target @ result.get_array(), rhs, atol=1e-7)
    return {
        "shared_coupling_curvature_checked": True,
        "consistent_design_shift": True,
        "rounding_curvature_uses_small_shift": True,
        "stronger_safeguard_refactors_same_kkt_system": True,
        "positive_private_curvature_preserved": selective,
    }


def check_spectral_modes():
    rng = np.random.default_rng(618)
    modes, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    matrix = modes @ np.diag([-3, -0.1, 0.5, 2, 5]) @ modes.T
    _, scale, correction = positive_spectral_factor(matrix)
    scaled = scale[:, None] * matrix * scale[None, :]
    eigenvalues, modes = np.linalg.eigh(scaled)
    update = scale[:, None] * correction * scale[None, :]
    positive = modes[:, eigenvalues > 1e-6]
    np.testing.assert_allclose(update @ positive, 0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix + correction).min() > 0
    assert np.max(abs(correction - np.diag(correction.diagonal()))) > 1e-3
    return {"positive_modes_preserved": True, "negative_modes_corrected": True}


def check_box_subproblems():
    """Compare active-bound steps with exhaustive solutions of small convex QPs."""
    rng = np.random.default_rng(980)
    largest_iterations = 0
    for _ in range(20):
        modes, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        hessian = modes @ np.diag([0.1, 0.5, 2, 10]) @ modes.T
        rhs = rng.normal(size=4) * 3
        lower, upper = -rng.uniform(0.01, 0.5, 4), rng.uniform(0.01, 0.5, 4)
        inverse = np.linalg.inv(hessian)
        free = inverse @ rhs
        active, targets, multipliers, iterations = active_box_constraints(
            free, lower, upper, lambda index: inverse[:, index]
        )
        step = free - inverse[:, active] @ multipliers
        assert np.all(step >= lower - 1e-9) and np.all(step <= upper + 1e-9)
        assert np.min(multipliers * targets, initial=0) >= -1e-9
        best, best_value = None, np.inf
        for status in product([-1, 0, 1], repeat=4):
            status = np.asarray(status)
            fixed, remaining = np.flatnonzero(status), np.flatnonzero(status == 0)
            candidate = np.where(status < 0, lower, np.where(status > 0, upper, 0))
            candidate[remaining] = np.linalg.solve(
                hessian[np.ix_(remaining, remaining)],
                rhs[remaining] - hessian[np.ix_(remaining, fixed)] @ candidate[fixed],
            )
            if np.any(candidate < lower - 1e-9) or np.any(candidate > upper + 1e-9):
                continue
            value = 0.5 * candidate @ hessian @ candidate - rhs @ candidate
            if value < best_value:
                best, best_value = candidate, value
        np.testing.assert_allclose(step, best, rtol=1e-8, atol=1e-8)
        correction = np.zeros(4)
        correction[active] = multipliers / targets
        np.testing.assert_allclose(
            (hessian + np.diag(correction)) @ step, rhs, atol=1e-8
        )
        largest_iterations = max(largest_iterations, iterations)
    assert largest_iterations > 5  # At least one case releases an active bound.
    try:
        active_box_constraints(free, lower, upper, lambda index: inverse[:, index], 0)
    except StepBoundIterationLimit as error:
        assert error.iterations == 0
    else:
        raise AssertionError("An unfinished QP was accepted")
    hessian = np.array([[1e-6, 1e-4], [1e-4, 1.0]])
    rhs = np.array([1e4, 1.0])
    inverse = np.linalg.inv(hessian)
    active, targets, multipliers, _ = active_box_constraints(
        inverse @ rhs,
        np.array([-1.0, -0.5]),
        np.array([1e-9, 0.5]),
        lambda index: inverse[:, index],
    )
    correction = np.zeros(2)
    correction[active] = np.maximum(multipliers / targets, 0)
    recovered = np.linalg.solve(hessian + np.diag(correction), rhs)
    np.testing.assert_allclose(recovered, [1e-9, 0.5], rtol=1e-6, atol=1e-12)
    # Coupled states can have dependent inverse responses and a nonzero affine
    # offset. Starting at zero would select incompatible active bounds here.
    inverse = np.ones((2, 2))
    free = np.array([2.0, 2.2])
    active, _, multipliers, _ = active_box_constraints(
        free,
        np.full(2, -0.5),
        np.array([0.3, 0.4]),
        lambda index: inverse[:, index],
        feasible_start=np.array([-0.1, 0.1]),
    )
    np.testing.assert_allclose(free - inverse[:, active] @ multipliers, [0.2, 0.4])

    size = 251
    identity = np.eye(size)
    reports = []
    active, targets, multipliers, _ = active_box_constraints(
        np.full(size, 2.0),
        np.full(size, -1.0),
        np.full(size, 1.0),
        lambda index: identity[:, index],
        progress=lambda iteration, count: reports.append((iteration, count)),
    )
    assert reports == [(250, 250)]
    np.testing.assert_allclose(2.0 - identity[:, active] @ multipliers, 1.0)
    return {
        "active_set_progress_reports": True,
        "exhaustive_qp_comparisons": 20,
        "largest_iterations": largest_iterations,
        "large_cancellation_recovered_by_refactorization": True,
        "dependent_bounds_with_affine_offset": True,
    }


def check_affine_start_fallback():
    # x - u = 0.02 has a bounded feasible step, but setting u = 0 puts x
    # outside its step bounds. The active set must defer to diagonal damping.
    matrix = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    bounded = np.array([0, 2, 3])
    point = np.array([0.01, 0.0, 0.5, 0.5])
    solver = JointKKTSolver(
        [CasePartition(np.array([0, 1, 2]), np.array([2]))],
        np.array([3]),
        bounded,
        np.array([1]),
        lambda shape: csr_matrix(shape),
        limited_indices=np.array([2, 3]),
        step_limits=0.1,
        box_qp=True,
        step_bounds=BoxStepBounds(
            bounded,
            np.zeros(3),
            np.ones(3),
            np.full(3, 0.1),
            lambda: point,
            symmetric=True,
        ),
    )
    solver.factor(Matrix(matrix), Vector(np.zeros(4)))
    rhs = np.array([0.0, 0.02, 0.0, 0.0])
    solution = Vector(np.zeros(4))
    solver.solve(Vector(rhs), solution)
    assert solver.box_qp_fallbacks == 1 and solver.box_qp_iterations == 0
    assert solver.design_step_ratio(solution.get_array()) <= 1 + 1e-6
    np.testing.assert_allclose(solver.target @ solution.get_array(), rhs, atol=1e-12)
    return {
        "fallbacks": solver.box_qp_fallbacks,
        "equations_and_step_bounds_passed": True,
    }


def check_original_recovery(selective=False):
    matrix = np.diag([0.2, 0.0, 1.0, 1e-4, 1.0])
    matrix[0, 1] = matrix[1, 0] = 1
    matrix[2, 4] = matrix[4, 2] = 1 + 1e-10
    for descent in [True, False]:
        solver = JointKKTSolver(
            [CasePartition(np.arange(4), np.array([2, 3]))],
            [4],
            [0, 2, 3, 4],
            [1],
            lambda shape: csr_matrix(shape),
            selective_curvature=selective,
        )
        solver.factor(Matrix(matrix), Vector(np.zeros(5)))
        rhs = np.array([0, 0, 1e-3, 0, 1e-3 if descent else -1e-3])
        result = Vector(np.zeros(5))
        solver.solve(Vector(rhs), result)
        assert (solver.recovered_newton_steps == 1) == descent
        np.testing.assert_allclose(solver.target @ result.get_array(), rhs, atol=1e-9)
        if descent:
            np.testing.assert_allclose(solver.target.toarray(), matrix)
    return {
        "accurate_descent_step_recovered": True,
        "negative_curvature_step_rejected": True,
    }


def check_step_capture():
    """Round-trip the inverse used by a truncated QP and finish its replay."""
    original = active_box_constraints

    def limited(*args, **kwargs):
        kwargs["max_iterations"] = 1
        return original(*args, **kwargs)

    with TemporaryDirectory(prefix="cadre-step-bounds-") as directory:
        folder = Path(directory)
        with (
            patch.object(JointKKTSolver, "diagnostic_directory", folder, create=True),
            patch(
                "integrated.solvers.joint.active_box_constraints", side_effect=limited
            ),
        ):
            check_system(bound_aware=True, box_qp=True)
        with np.load(
            folder / "step_bounds_diagnostic.npz", allow_pickle=False
        ) as saved:
            response = saved_bound_response(saved)
            maximum_error = 0.0
            for column, index in enumerate(saved["response_indices"]):
                expected = saved["response_values"][:, column]
                error = np.max(abs(response(int(index)) - expected))
                maximum_error = max(maximum_error, float(error))
            free, lower, upper, initial = [
                saved[name]
                for name in ("unconstrained", "lower", "upper", "feasible_start")
            ]
        assert maximum_error < 1e-12, maximum_error
        active, _, multipliers, iterations = original(
            free, lower, upper, response, feasible_start=initial
        )
        step = free - np.column_stack([response(int(i)) for i in active]) @ multipliers
        assert np.all(step >= lower - 1e-8) and np.all(step <= upper + 1e-8)
        assert iterations > 1 and not list(folder.glob("*.tmp.npz"))
        # Diagnostic storage must not prevent the checked fallback from running.
        (folder / "step_bounds_diagnostic.npz").unlink()
        with (
            patch.object(JointKKTSolver, "diagnostic_directory", folder, create=True),
            patch(
                "integrated.solvers.joint.active_box_constraints", side_effect=limited
            ),
            patch(
                "integrated.solvers.joint.np.savez_compressed",
                side_effect=OSError("disk full"),
            ),
            patch("integrated.solvers.joint.print_status"),
        ):
            fallback = check_system(bound_aware=True, box_qp=True)
        assert fallback["box_qp_fallbacks"] >= 1
    return {
        "inverse_roundtrip_error": maximum_error,
        "replayed_qp_iterations": iterations,
        "fallback_survives_diagnostic_write_failure": True,
    }


def replay_step_bounds(path, max_iterations):
    """Replay a captured QP only; this does not run or validate an optimization."""
    with np.load(path, allow_pickle=False) as saved:
        response = saved_bound_response(saved)
        response_error = 0.0
        for column, index in enumerate(saved["response_indices"]):
            expected = saved["response_values"][:, column]
            error = np.max(abs(response(int(index)) - expected)) / max(
                1, np.max(abs(expected))
            )
            response_error = max(response_error, float(error))
        assert response_error < 1e-12, response_error
        free, lower, upper, initial = [
            saved[name]
            for name in ("unconstrained", "lower", "upper", "feasible_start")
        ]
    try:
        active, targets, multipliers, iterations = active_box_constraints(
            free,
            lower,
            upper,
            response,
            max_iterations=max_iterations,
            feasible_start=initial,
        )
        work = multipliers * targets
        assert np.min(work, initial=0) >= -1e-8 * max(1, np.max(abs(work), initial=0))
        result = {
            "converged_subproblem": True,
            "active_bounds": len(active),
            "iterations": iterations,
        }
    except StepBoundIterationLimit as error:
        result = {"converged_subproblem": False, "iterations": error.iterations}
    print(
        json.dumps(
            {
                "optimization_run": False,
                "cached_response_error": response_error,
                **result,
            },
            indent=2,
        )
    )
    return 0 if result["converged_subproblem"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        help="Replay a captured step-bound subproblem without building CADRE",
    )
    parser.add_argument("--max-qp-iterations", type=int, default=2000)
    args = parser.parse_args()
    if args.max_qp_iterations < 1:
        parser.error("--max-qp-iterations must be positive")
    if args.replay:
        return replay_step_bounds(args.replay, args.max_qp_iterations)
    report = {
        "step_capture_and_replay": check_step_capture(),
        "affine_start_outside_box_fallback": check_affine_start_fallback(),
        "active_box_subproblems": check_box_subproblems(),
        "full_space_control_bounds": check_system(
            bound_aware=True, control_bounds_only=True, nonzero_constraint_rhs=True
        ),
        "spectral_full_space_control_bounds": check_system(
            indefinite=True,
            bound_aware=True,
            control_bounds_only=True,
            nonzero_constraint_rhs=True,
            spectral=True,
        ),
        "active_box_kkt_solve": check_system(bound_aware=True, box_qp=True),
        "full_space_active_box_kkt_solve": check_system(
            bound_aware=True,
            box_qp=True,
            symmetric_bounds=True,
            nonzero_constraint_rhs=True,
        ),
        "full_space_active_box_fallback": check_system(
            bound_aware=True,
            box_qp=True,
            symmetric_bounds=True,
            nonzero_constraint_rhs=True,
            force_box_fallback=True,
        ),
        "spectral_active_box_kkt_solve": check_system(
            indefinite=True, bound_aware=True, spectral=True, box_qp=True
        ),
        "active_box_fallback": check_system(
            bound_aware=True, box_qp=True, force_box_fallback=True
        ),
        "spectral_active_box_fallback": check_system(
            indefinite=True,
            bound_aware=True,
            spectral=True,
            box_qp=True,
            force_box_fallback=True,
        ),
        "spectral_modes": check_spectral_modes(),
        "spectral_indefinite_solve": check_system(indefinite=True, spectral=True),
        "spectral_bound_aware_solve": check_system(
            indefinite=True, bound_aware=True, spectral=True
        ),
        "spectral_symmetric_bounds_with_equality_defects": check_system(
            indefinite=True,
            bound_aware=True,
            spectral=True,
            symmetric_bounds=True,
            nonzero_constraint_rhs=True,
        ),
        "spectral_coupled_curvature": check_coupled_curvature(spectral=True),
        "positive_curvature": check_system(),
        "indefinite_curvature": check_system(indefinite=True),
        "multiplier_initialization": check_system(normal=True),
        "design_step_limit": check_system(limit_design=True),
        "design_step_limit_indefinite": check_system(
            indefinite=True, limit_design=True
        ),
        "coupled_curvature": check_coupled_curvature(),
        "original_recovery": check_original_recovery(),
        "selective_coupled_curvature": check_coupled_curvature(selective=True),
        "selective_original_recovery": check_original_recovery(selective=True),
        "selective_parallel_solve": check_system(
            indefinite=True, selective=True, workers=2
        ),
        "bound_aware_solve": check_system(bound_aware=True, selective=True, workers=2),
        "bound_aware_indefinite_solve": check_system(
            bound_aware=True, indefinite=True, selective=True, workers=2
        ),
    }
    path = Path("integrated/results/validation/joint_solver_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
