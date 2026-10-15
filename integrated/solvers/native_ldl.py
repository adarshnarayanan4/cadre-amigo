"""AMIGO LDL with analytic external Hessians and explicit temporal ordering."""

import amigo as am
from amigo.algorithm.solvers.linear_solver import LinearSolver
import numpy as np
from scipy.sparse import csr_matrix


def temporal_kkt_ordering(model, nodes):
    """Place each variable after its last local use; retain global designs last."""
    size = model.create_vector().get_vector().get_size()

    stamps = np.full(size, -1, dtype=int)

    for name, group in model.comp.items():
        # Spline-kernel aliases supply the actual last use of each control.
        if name == "control_points":
            continue

        count = int(group.size)
        times = (
            np.arange(count) + nodes - count
            if count in (nodes, nodes - 1, nodes - 2)
            else np.full(count, nodes - 1)
        )

        for variable in group.vars:
            indices = np.asarray(model.get_indices(f"{name}.{variable}"))

            if indices.size:
                np.maximum.at(
                    stamps, indices.ravel(), np.repeat(times, indices.size // count)
                )

    stamps[stamps < 0] = nodes - 1

    return np.argsort(stamps, kind="stable")


def primal_regularization_metric(model, point):
    """Nominal variable units for a positive inertia-correction metric."""
    scales = np.ones(len(point))
    primal = np.zeros(len(point), dtype=bool)
    primal[model.problem.get_primal_indices()] = True

    for name, group in model.comp.items():
        for variable in group.get_input_names():
            indices = np.asarray(model.get_indices(f"{name}.{variable}")).ravel()
            indices = indices[primal[indices]]

            if len(indices):
                nominal = max(1.0, float(np.max(np.abs(point[indices]))))
                scales[indices] = np.maximum(scales[indices], nominal)

    lower, upper = model.problem.get_lower(), model.problem.get_upper()
    lower.copy_device_to_host()

    upper.copy_device_to_host()

    width = upper.get_array() - lower.get_array()
    bounded = primal & np.isfinite(width) & (width > 0)
    scales[bounded] = np.maximum(width[bounded], np.abs(point[bounded]))

    return 1 / scales**2


class NativeLDLSolver(LinearSolver):
    """Expand external-Hessian sparsity for native LDL and full inertia checks."""

    def __init__(
        self,
        hessian_correction=None,
        permutation=None,
        regularization_metric=None,
        diagonal_provider=None,
    ):
        self.hessian_correction = hessian_correction
        self.permutation = permutation

        self.ldl = None
        self.last_flag = None
        self.source_hessian = None
        self.mapping_cache = [None, None]

        self.regularization_metric = regularization_metric
        self.diagonal_provider = diagonal_provider

    def factor(self, hessian, diagonal):
        hessian.copy_data_device_to_host()

        diagonal.copy_device_to_host()

        if hessian is not self.source_hessian:
            self.source_hessian = hessian
            self.source_structure = hessian.get_nonzero_structure()

        n, m, _, rowp, cols = self.source_structure
        original = csr_matrix(
            (np.array(hessian.get_data(), copy=True), cols, rowp), shape=(n, m)
        )

        correction = (
            self.hessian_correction((n, m)).tocsr()
            if self.hessian_correction
            else csr_matrix((n, m))
        )

        original.eliminate_zeros()

        correction.eliminate_zeros()

        if self.ldl is None:
            self._initialize_pattern(original, correction)

        positions_by_slot = []

        for slot, matrix in enumerate((original, correction)):
            positions = self._map_entries(matrix, slot)

            if positions is None:
                # A previously zero nonlinear entry became nonzero. Expand
                # symbolic storage rather than assuming zeros remain zero.
                self._initialize_pattern(original, correction)

                positions_by_slot = [
                    self._map_entries(a, i)
                    for i, a in enumerate((original, correction))
                ]
                break

            positions_by_slot.append(positions)

        values = self.native_matrix.get_data()
        values[:] = 0

        for matrix, positions in zip((original, correction), positions_by_slot):
            values[positions] += matrix.data

        effective_diagonal = diagonal.get_array().copy()

        if self.regularization_metric is not None:
            base = self.diagonal_provider()
            base.copy_device_to_host()

            effective_diagonal = (
                base.get_array()
                + (effective_diagonal - base.get_array()) * self.regularization_metric
            )

        self.native_diagonal.get_array()[:] = effective_diagonal[self.permutation]
        self.native_diagonal.copy_host_to_device()

        self.last_matrix = original + correction
        self.last_matrix.setdiag(self.last_matrix.diagonal() + effective_diagonal)

        self.last_flag = self.ldl.factor(self.native_diagonal)

        if self.last_flag:
            print("Native LDL factor flag:", self.last_flag, flush=True)

        return self.last_flag

    def _initialize_pattern(self, original, correction):
        n, m = original.shape

        if self.permutation is None:
            self.permutation = np.arange(n)

        self.inverse_permutation = np.argsort(self.permutation)
        pattern = original.copy()
        pattern.data[:] = 1

        extra = correction.copy()
        extra.data[:] = 1

        pattern = pattern + extra

        if hasattr(self, "original_pattern"):
            pattern = pattern + self.original_pattern
            pattern.data[:] = 1

        self.original_pattern = pattern.copy()
        self.mapping_cache = [None, None]

        pattern.setdiag(1)

        pattern = pattern[self.permutation][:, self.permutation].tocsr()
        pattern.sort_indices()

        self.pattern_keys = (
            np.repeat(np.arange(n, dtype=np.int64), np.diff(pattern.indptr)) * n
            + pattern.indices
        )

        self.native_matrix = am.CSRMat(
            n,
            m,
            pattern.indptr.astype(np.int32),
            pattern.indices.astype(np.int32),
            np.zeros(pattern.nnz),
        )

        self.native_diagonal = am.Vector(n)
        # METIS ordering crashes for this environment's matrix/library.
        # The supplied permutation avoids that dependency entirely.
        self.ldl = am.SparseLDL(
            self.native_matrix,
            am.SolverType.LDL,
            ustab=0.01,
            pivot_tol=1e-14,
            order=am.OrderingType.NATURAL,
        )

        print(
            "Native LDL symbolic initialized:",
            n,
            "rows,",
            pattern.nnz,
            "entries",
            flush=True,
        )

    def _map_entries(self, matrix, slot):
        cached = self.mapping_cache[slot]

        if (
            cached is not None
            and np.array_equal(matrix.indptr, cached[0])
            and np.array_equal(matrix.indices, cached[1])
        ):
            return cached[2]

        n = matrix.shape[0]
        rows = np.repeat(np.arange(n), np.diff(matrix.indptr))
        keys = (
            self.inverse_permutation[rows].astype(np.int64) * n
            + self.inverse_permutation[matrix.indices]
        )

        positions = np.searchsorted(self.pattern_keys, keys)

        if np.any(positions >= len(self.pattern_keys)) or not np.array_equal(
            self.pattern_keys[positions], keys
        ):
            return None

        self.mapping_cache[slot] = (
            matrix.indptr.copy(),
            matrix.indices.copy(),
            positions,
        )

        return positions

    def solve(self, rhs, solution):
        if self.last_flag:
            raise RuntimeError(
                f"Cannot solve incomplete native LDL factorization: {self.last_flag}"
            )

        rhs.copy_device_to_host()

        values = rhs.get_array()

        def solve_array(b):
            ordered = np.array(b[self.permutation], copy=True)
            self.ldl.solve(ordered)

            result = np.empty_like(ordered)
            result[self.permutation] = ordered

            return result

        result = solve_array(values)
        scale = max(1.0, np.linalg.norm(values, np.inf))
        error = np.linalg.norm(values - self.last_matrix @ result, np.inf) / scale

        for _ in range(4):
            if error < 1e-10:
                break

            candidate = result + solve_array(values - self.last_matrix @ result)
            next_error = (
                np.linalg.norm(values - self.last_matrix @ candidate, np.inf) / scale
            )

            if not np.isfinite(next_error) or next_error >= error:
                break

            result, error = candidate, next_error

        if not np.isfinite(error) or error > 1e-7:
            raise FloatingPointError(f"Inaccurate native LDL solve: {error}")

        solution.get_array()[:] = result
        solution.copy_host_to_device()

    def inertia_enabled(self):
        return True

    def get_inertia(self):
        return (0, 0) if self.last_flag else self.ldl.get_inertia()

    def set_pivot_tolerance(self, tolerance):
        pass
