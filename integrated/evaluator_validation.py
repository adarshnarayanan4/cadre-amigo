"""Regress objective-only derivatives and current-point external Hessians."""

import json
from pathlib import Path
from types import SimpleNamespace

from amigo.algorithm.evaluator import Evaluator
import numpy as np

from integrated.cadre_reporting import print_validation
from integrated.solvers.evaluator import ConsistentEvaluator
from integrated.validation_helpers import Vector


class WorkVector(Vector):
    def copy(self, other):
        self.array[:] = other.array

    def fill_at(self, indices, value):
        self.array[indices] = value


class Problem:
    def create_vector(self):
        return WorkVector(np.zeros(2))

    def create_constraint_vector(self):
        return WorkVector(np.zeros(1))

    def get_constraint_indices(self):
        return np.array([1])

    def update(self, point):
        self.cached = point.array.copy()

    def gradient(self, alpha, point, gradient):
        x, multiplier = point.array
        gradient.array[:] = [-alpha + 2 * multiplier * x, x * x - 1]

    def hessian(self, alpha, point, matrix):
        matrix[:] = [[2 * point.array[1], 2 * self.cached[0]], [2 * self.cached[0], 0]]

    def dot(self, a, b):
        return float(a.array @ b.array)


def main():
    problem = Problem()

    backend = SimpleNamespace(
        get_kkt_element_counts=lambda: (1, 1, 0),
        compute_log_barrier_derivative=lambda *args: 0,
    )

    point = WorkVector([2.0, 11.0])
    direction = WorkVector([0.3, 7.0])

    state = SimpleNamespace(
        current=SimpleNamespace(get_solution=lambda: point),
        step=SimpleNamespace(get_solution=lambda: direction),
        residual_current=True,
        step_current=True,
        obj_scale=1.0,
        mu=0,
        hessian_current=False,
        hessian=np.zeros((2, 2)),
    )

    problem.update(point)

    legacy = Evaluator(problem, backend).evaluate_directional_derivative(state)

    evaluator = ConsistentEvaluator(problem, backend)
    corrected = evaluator.evaluate_directional_derivative(state)

    h = 1e-5
    finite = (
        -(point.array[0] + h * direction.array[0])
        + (point.array[0] - h * direction.array[0])
    ) / (2 * h)

    assert abs(legacy - finite) > 20
    assert abs(corrected - finite) < 1e-9
    assert np.array_equal(problem.cached, point.array)

    state.obj_scale = 2.0

    assert abs(evaluator.evaluate_directional_derivative(state) - 2 * finite) < 1e-9

    problem.cached[:] = [9, 0]

    evaluator.evaluate_hessian(state)

    expected = np.array([[22, 4], [4, 0]])
    error = float(np.max(abs(state.hessian - expected)))

    assert error == 0

    result = {
        "legacy_directional_error": abs(legacy - finite),
        "corrected_directional_error": abs(corrected - finite),
        "external_hessian_cache_error": error,
    }

    print_validation("EVALUATOR VALIDATION", result)

    Path("integrated/results/evaluator_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
