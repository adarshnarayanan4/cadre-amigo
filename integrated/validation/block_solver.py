"""Verify shared-block KKT solves against a dense reference for each solver path."""

from dataclasses import dataclass
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.linalg import cho_factor as real_cholesky
from scipy.sparse import csr_matrix

from integrated.reporting.console import print_validation
from integrated.solvers.block_solver import SparseLUSolver, TrajectoryKKTSolver
from integrated.validation.helpers import Matrix, Vector


@dataclass
class DenseKKTCase:
    """A reproducible KKT system and its dense-reference comparison helpers."""

    matrix: np.ndarray
    shared_indices: np.ndarray
    primal_count: int
    rhs: Vector

    def factor(self, solver, matrix=None, diagonal=None):
        if matrix is None:
            matrix = self.matrix

        if diagonal is None:
            diagonal = np.zeros(len(matrix))

        solver.factor(Matrix(matrix), Vector(diagonal))

    def solve_error(self, solver, expected_matrix=None):
        if expected_matrix is None:
            expected_matrix = self.matrix

        answer = Vector(np.zeros(len(self.matrix)))
        solver.solve(self.rhs, answer)

        expected = np.linalg.solve(expected_matrix, self.rhs.array)

        return float(np.max(np.abs(answer.array - expected)))

    def initialization_diagonal(self):
        diagonal = np.zeros(len(self.matrix))
        diagonal[: self.primal_count] = 1.0
        diagonal[self.shared_indices] = 1.0

        return diagonal


def make_shared_block_case(rng):
    """Use 129 shared variables to exercise the iterative Schur threshold."""
    primal_count = 24
    constraint_count = 8
    shared_count = 129

    hessian = rng.normal(size=(primal_count, primal_count))
    hessian = hessian @ hessian.T + np.eye(primal_count)

    jacobian = rng.normal(size=(constraint_count, primal_count))

    core = np.block(
        [
            [hessian, jacobian.T],
            [jacobian, np.zeros((constraint_count, constraint_count))],
        ]
    )

    coupling = 0.01 * rng.normal(size=(len(core), shared_count))

    shared_block = 3 * np.eye(shared_count)

    matrix = np.block([[core, coupling], [coupling.T, shared_block]])
    shared_indices = np.arange(len(core), len(matrix))

    rhs = Vector(rng.normal(size=len(matrix)))

    return DenseKKTCase(matrix, shared_indices, primal_count, rhs)


def validate_schur_cache(case):
    """Check direct solves, cached iteration, fallback, and cache separation."""
    solver = SparseLUSolver(shared_indices=case.shared_indices)
    case.factor(solver)

    errors = {"direct_vs_dense_max_error": case.solve_error(solver)}

    changed = case.matrix.copy()
    primal = slice(0, case.primal_count)
    changed[primal, primal] += 0.002 * np.eye(case.primal_count)
    case.factor(solver, changed)

    assert solver.iterative_schur

    errors["iterative_vs_dense_max_error"] = case.solve_error(solver, changed)

    # Accept an iteration-limit flag when the complete KKT residual is accurate.
    exact = np.linalg.solve(changed, case.rhs.array)

    with patch(
        "integrated.solvers.block_solver.gmres",
        return_value=(exact[case.shared_indices], 3),
    ):
        errors["iteration_limit_with_accurate_solution_max_error"] = case.solve_error(
            solver, changed
        )

    assert solver.iterative_schur

    with patch(
        "integrated.solvers.block_solver.gmres",
        return_value=(np.zeros(len(case.shared_indices)), 1),
    ):
        errors["fallback_vs_dense_max_error"] = case.solve_error(solver, changed)

    assert not solver.iterative_schur

    diagonal = case.initialization_diagonal()
    case.factor(solver, diagonal=diagonal)

    errors["normal_equations_vs_dense_max_error"] = case.solve_error(
        solver, case.matrix + np.diag(diagonal)
    )
    case.factor(solver, changed)

    assert solver.iterative_schur

    errors["cached_newton_after_normal_equations_max_error"] = case.solve_error(
        solver, changed
    )

    return errors


def validate_curvature_safeguards(case):
    """Check regularization, chunked elimination, and positive curvature."""
    indices = case.shared_indices
    indefinite = case.matrix.copy()
    indefinite[indices[0], indices[0]] = -0.2
    safeguarded = SparseLUSolver(
        shared_indices=indices, safeguard_shared=True, coupling_chunk_size=256
    )
    case.factor(safeguarded, indefinite)

    assert safeguarded.shared_regularization is not None

    corrected = indefinite.copy()
    corrected[np.ix_(indices, indices)] += safeguarded.shared_regularization
    errors = {
        "safeguarded_vs_regularized_dense_max_error": case.solve_error(
            safeguarded, corrected
        )
    }

    schur = (
        safeguarded.shared_block - safeguarded.shared_local @ safeguarded.core_coupling
    )

    assert np.min(np.linalg.eigvalsh(schur)) > 0

    chunked = SparseLUSolver(
        shared_indices=indices, safeguard_shared=True, coupling_chunk_size=17
    )
    case.factor(chunked, indefinite)

    assert chunked.core_coupling is None

    errors["chunked_safeguarded_vs_dense_max_error"] = case.solve_error(
        chunked, corrected
    )

    positive = SparseLUSolver(shared_indices=indices, safeguard_shared=True)
    case.factor(positive)

    assert positive.shared_regularization is None

    errors["positive_curvature_unchanged_max_error"] = case.solve_error(positive)

    return errors


def validate_roundoff_recovery(case):
    """Recover the original Newton step after a rounding-level Cholesky failure."""
    attempts = 0

    def fail_first_cholesky(matrix, **kwargs):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise np.linalg.LinAlgError("Simulated roundoff-level Cholesky failure")

        return real_cholesky(matrix, **kwargs)

    solver = SparseLUSolver(shared_indices=case.shared_indices, safeguard_shared=True)

    with patch(
        "integrated.solvers.block_solver.cho_factor", side_effect=fail_first_cholesky
    ):
        case.factor(solver)

    assert solver.curvature_shift == 1e-8

    error = case.solve_error(solver)

    assert solver.solve_unregularized

    return {"original_newton_recovery_max_error": error}


def validate_partition_switching(case):
    """Switch from the smaller initialization partition to all shared controls."""
    solver = SparseLUSolver(
        shared_indices=case.shared_indices,
        normal_shared_indices=case.shared_indices[-7:],
    )

    diagonal = case.initialization_diagonal()
    case.factor(solver, diagonal=diagonal)

    assert len(solver.shared_indices) == 7

    errors = {
        "smaller_initialization_partition_max_error": case.solve_error(
            solver, case.matrix + np.diag(diagonal)
        )
    }
    case.factor(solver)

    assert len(solver.shared_indices) == len(case.shared_indices)

    errors["larger_newton_partition_max_error"] = case.solve_error(solver)

    return errors


def validate_trajectory_inverse(rng):
    """Check permuted trajectory elimination with vector and matrix right sides."""
    state_count = 11

    jacobian = rng.normal(size=(state_count, state_count)) + 8 * np.eye(state_count)

    hessian = rng.normal(size=(state_count, state_count))
    hessian = hessian + hessian.T

    trajectory = np.block(
        [
            [hessian, jacobian.T],
            [jacobian, np.zeros((state_count, state_count))],
        ]
    )
    # Include an isolated prescribed variable after the states and constraints.
    trajectory = np.pad(trajectory, ((0, 1), (0, 1)))
    trajectory[-1, -1] = 2.0

    fixed_index = 2 * state_count
    permutation = rng.permutation(len(trajectory))
    reordered = trajectory[np.ix_(permutation, permutation)]

    state_indices = np.flatnonzero(
        (permutation < state_count) | (permutation == fixed_index)
    )
    constraint_indices = np.flatnonzero(
        (permutation >= state_count) & (permutation < fixed_index)
    )

    inverse = TrajectoryKKTSolver(
        csr_matrix(reordered), state_indices, constraint_indices
    )

    errors = {}

    for columns in [None, 7]:
        shape = len(trajectory) if columns is None else (len(trajectory), columns)
        rhs = rng.normal(size=shape)
        errors[f"trajectory_inverse_{columns}_rhs_max_error"] = float(
            np.max(np.abs(inverse.solve(rhs) - np.linalg.solve(reordered, rhs)))
        )

    return errors


def main():
    """Run every solver path and save the existing numerical error report."""
    rng = np.random.default_rng(31)
    case = make_shared_block_case(rng)

    errors = {}

    errors.update(validate_schur_cache(case))

    errors.update(validate_curvature_safeguards(case))

    errors.update(validate_roundoff_recovery(case))

    errors.update(validate_partition_switching(case))

    errors.update(validate_trajectory_inverse(rng))

    print_validation("BLOCK SOLVER VALIDATION", errors)

    assert max(errors.values()) < 1e-9

    Path("integrated/results/validation").mkdir(parents=True, exist_ok=True)

    Path("integrated/results/validation/block_solver_validation.json").write_text(
        json.dumps(errors, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
