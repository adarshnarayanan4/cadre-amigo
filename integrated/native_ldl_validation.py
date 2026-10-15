"""Verify external-Hessian expansion, permutation, and native LDL inertia."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from integrated.solvers.native_ldl import NativeLDLSolver
from integrated.validation_helpers import Matrix, Vector


def main():
    rng = np.random.default_rng(601)
    primal, constraints = 7, 4

    j = rng.normal(size=(constraints, primal))

    base = np.block(
        [[2 * np.eye(primal), j.T], [j, np.zeros((constraints, constraints))]]
    )
    n = len(base)

    amplitude = [0.0]

    def external(shape):
        return csr_matrix(([amplitude[0], amplitude[0]], ([0, 1], [1, 0])), shape=shape)

    solver = NativeLDLSolver(external, permutation=rng.permutation(n))

    diagonal = Vector(np.r_[np.full(primal, 0.05), np.zeros(constraints)])

    rhs = Vector(rng.normal(size=n))
    answer = Vector(np.zeros(n))

    errors = {}

    for value in [0.0, 0.4]:
        amplitude[0] = value

        assert solver.factor(Matrix(base), diagonal) == 0

        expected = base + external((n, n)).toarray() + np.diag(diagonal.array)

        eigenvalues = np.linalg.eigvalsh(expected)

        assert solver.get_inertia() == (
            int(np.sum(eigenvalues > 0)),
            int(np.sum(eigenvalues < 0)),
        )

        solver.solve(rhs, answer)

        errors[f"expanded_hessian_{value}_dense_max_error"] = float(
            np.max(abs(answer.array - np.linalg.solve(expected, rhs.array)))
        )

    baseline = Vector(np.r_[np.full(primal, 0.01), np.zeros(constraints)])
    perturbed = Vector(np.r_[np.full(primal, 0.05), np.full(constraints, -0.02)])

    metric = np.r_[np.linspace(0.25, 2.0, primal), np.ones(constraints)]

    scaled = NativeLDLSolver(
        external,
        permutation=rng.permutation(n),
        regularization_metric=metric,
        diagonal_provider=lambda: baseline,
    )

    assert scaled.factor(Matrix(base), perturbed) == 0

    expected = (
        base
        + external((n, n)).toarray()
        + np.diag(baseline.array + (perturbed.array - baseline.array) * metric)
    )

    scaled.solve(rhs, answer)

    errors["metric_regularization_preserves_barrier_max_error"] = float(
        np.max(abs(answer.array - np.linalg.solve(expected, rhs.array)))
    )

    assert max(errors.values()) < 1e-9

    print(errors)

    Path("integrated/results/native_ldl_validation.json").write_text(
        json.dumps(errors, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
