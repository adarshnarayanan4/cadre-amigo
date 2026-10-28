"""Case-wise KKT elimination for trajectories with shared CADRE designs."""

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable

from amigo.algorithm.solvers.linear_solver import LinearSolver
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, solve_triangular
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu

from integrated.reporting.console import print_kkt_matrix, print_status
from integrated.solvers.block_solver import TrajectoryKKTSolver

SMALL_CURVATURE_SHIFTS = (0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8)


def scaled_cholesky(matrix):
    """Factor positive design curvature after diagonal scaling."""
    symmetric = (matrix + matrix.T) / 2
    scale = 1 / np.sqrt(np.maximum(abs(np.diag(symmetric)), 1e-12))
    factor = cho_factor(scale[:, None] * symmetric * scale[None, :], lower=True)
    return factor, scale


def scaled_solve(factor, scale, rhs):
    weights = scale if rhs.ndim == 1 else scale[:, None]
    return weights * cho_solve(factor, weights * rhs)


def positive_diagonal_factor(matrix):
    """Correct negative modes while preserving a sparse diagonal KKT update."""
    symmetric = (matrix + matrix.T) / 2
    metric = np.maximum(abs(np.diag(symmetric)), 1e-12)
    scale = 1 / np.sqrt(metric)
    scaled = scale[:, None] * symmetric * scale[None, :]
    correction = np.zeros(len(matrix))
    try:
        factor = cho_factor(scaled, lower=True)
    except np.linalg.LinAlgError:
        eigenvalues, modes = eigh(scaled)
        floor = 1e-8 * max(1.0, np.max(abs(eigenvalues)))
        increments = np.maximum(abs(eigenvalues), floor) - eigenvalues
        selected = increments > 0
        modes = modes[:, selected] * np.sqrt(increments[selected])
        # This diagonal majorizes the mode correction: diag(row sums) - C
        # is positive semidefinite for the symmetric correction C.
        correction = np.sum(abs(modes @ modes.T), axis=1)
        scaled[np.diag_indices_from(scaled)] += correction
        factor = cho_factor(scaled, lower=True)
    return factor, scale, correction * metric


def positive_spectral_factor(matrix):
    """Correct nonpositive modes without damping the positive eigenspace."""
    symmetric = (matrix + matrix.T) / 2
    scale = 1 / np.sqrt(np.maximum(abs(np.diag(symmetric)), 1e-12))
    scaled = scale[:, None] * symmetric * scale[None, :]
    correction = np.zeros_like(scaled)
    try:
        factor = cho_factor(scaled, lower=True)
    except np.linalg.LinAlgError:
        eigenvalues, modes = eigh(scaled)
        floor = 1e-8 * max(1.0, np.max(abs(eigenvalues)))
        increments = np.maximum(abs(eigenvalues), floor) - eigenvalues
        selected = increments > 0
        weighted = modes[:, selected] * np.sqrt(increments[selected])
        correction = weighted @ weighted.T
        factor = cho_factor(scaled + correction, lower=True)
    return factor, scale, correction / scale[:, None] / scale[None, :]


@dataclass
class CasePartition:
    indices: np.ndarray
    controls: np.ndarray


@dataclass
class BoxStepBounds:
    """Maximum trial displacements, including bounds on eliminated states."""

    indices: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    max_step: np.ndarray
    point_provider: Callable[[], np.ndarray]
    fraction: float = 0.8

    def limits(self):
        values = np.asarray(self.point_provider())[self.indices]
        if not np.all(np.isfinite(values)):
            raise FloatingPointError("Step bounds require finite variables")
        negative = self.fraction * (values - self.lower)
        positive = self.fraction * (self.upper - values)
        if np.any(negative <= 0) or np.any(positive <= 0):
            raise FloatingPointError("Step bounds require a strictly interior point")
        return np.minimum(self.max_step, negative), np.minimum(self.max_step, positive)


def append_active_factor(factor, cross):
    """Extend a normalized Cholesky factor by one active bound."""
    if not len(cross):
        return np.ones((1, 1))
    if factor is None:
        return None
    column = solve_triangular(factor, cross, lower=True)
    pivot = 1 - column @ column
    if pivot <= 1e-12:
        return None
    extended = np.zeros((len(cross) + 1, len(cross) + 1), order="F")
    extended[:-1, :-1] = factor
    extended[-1, :-1] = column
    extended[-1, -1] = np.sqrt(pivot)
    return extended


def remove_active_factor(factor, index):
    """Remove an active bound using rotations instead of a full refactorization."""
    if factor is None:
        return None
    upper = np.delete(factor.T, index, axis=1)
    for row in range(index, len(upper) - 1):
        first, second = upper[row, row], upper[row + 1, row]
        radius = np.hypot(first, second)
        if radius == 0:
            return None
        cosine, sine = first / radius, second / radius
        previous = upper[row, row:].copy()
        following = upper[row + 1, row:].copy()
        upper[row, row:] = cosine * previous + sine * following
        upper[row + 1, row:] = -sine * previous + cosine * following
    return upper[:-1].T.copy(order="F")


def active_box_constraints(unconstrained, lower, upper, response, max_iterations=2000):
    """Solve the convex step subproblem through a small active-bound system.

    ``response(i)`` returns column i of the inverse quadratic form, restricted
    to bounded variables. No inverse containing all design variables is formed.
    """
    current = np.zeros_like(unconstrained)
    active, targets, columns = [], [], {}
    storage = np.empty((len(current), 64), order="F")
    factor = None
    for iteration in range(max_iterations):
        if active:
            inverse = storage[:, : len(active)]
            scale = 1 / np.sqrt([columns[index][index] for index in active])
            rhs = scale * (unconstrained[active] - targets)
            if factor is None:
                gram = inverse[active]
                scaled = scale[:, None] * ((gram + gram.T) / 2) * scale[None, :]
                try:
                    factor = np.tril(cho_factor(scaled, lower=True)[0])
                except np.linalg.LinAlgError:
                    multipliers = scale * np.linalg.lstsq(scaled, rhs, rcond=1e-12)[0]
            if factor is not None:
                multipliers = scale * cho_solve((factor, True), rhs)
            candidate = unconstrained - inverse @ multipliers
            error = np.max(abs(candidate[active] - targets)) / max(
                1.0, np.max(abs(unconstrained[active]))
            )
            if not np.isfinite(error) or error > 1e-6:
                raise FloatingPointError("Dependent or inaccurate active step bounds")
            # The Schur subtraction can lose tiny bound distances. Keep the
            # active equations exact in this QP bookkeeping; the returned
            # Newton direction is recomputed from the full regularized KKT.
            candidate[active] = targets
        else:
            candidate, multipliers = unconstrained, np.empty(0)

        direction = candidate - current
        fractions = np.full(len(current), np.inf)
        positive, negative = direction > 0, direction < 0
        fractions[positive] = (upper[positive] - current[positive]) / direction[
            positive
        ]
        fractions[negative] = (lower[negative] - current[negative]) / direction[
            negative
        ]
        fractions[active] = np.inf
        blocking = int(np.argmin(fractions))
        fraction = max(0.0, min(1.0, fractions[blocking]))
        if fraction < 1 - 1e-12:
            current += fraction * direction
            target = upper[blocking] if direction[blocking] > 0 else lower[blocking]
            current[blocking] = target
            if blocking not in columns:
                column = np.asarray(response(blocking))
                if not np.all(np.isfinite(column)) or column[blocking] <= 0:
                    raise FloatingPointError("Nonpositive active-bound response")
                columns[blocking] = column
            column = columns[blocking]
            if len(active) == storage.shape[1]:
                larger = np.empty((len(current), 2 * storage.shape[1]), order="F")
                larger[:, : len(active)] = storage[:, : len(active)]
                storage = larger
            storage[:, len(active)] = column
            cross = np.array(
                [
                    0.5
                    * (column[index] + columns[index][blocking])
                    / np.sqrt(columns[index][index] * column[blocking])
                    for index in active
                ]
            )
            factor = append_active_factor(factor, cross)
            active.append(blocking)
            targets.append(target)
            continue

        current = candidate.copy()
        work = multipliers * targets
        tolerance = 1e-10 * max(1.0, np.max(abs(work), initial=0))
        if len(active) and np.min(work) < -tolerance:
            release = int(np.argmin(work))
            factor = remove_active_factor(factor, release)
            storage[:, release : len(active) - 1] = storage[
                :, release + 1 : len(active)
            ]
            del active[release], targets[release]
            continue
        return (
            np.asarray(active, dtype=int),
            np.asarray(targets),
            multipliers,
            iteration + 1,
        )
    raise FloatingPointError("Active step-bound solve exceeded its iteration limit")


class CaseFactor:
    def __init__(
        self,
        matrix,
        partition,
        shared,
        primal,
        constraints,
        normal,
        chunk,
        bounded_indices=None,
    ):
        self.controls = partition.controls
        self.core = np.setdiff1d(partition.indices, self.controls)
        self.design = np.r_[self.controls, shared]
        core_matrix = matrix[self.core][:, self.core].tocsc()

        if normal:
            self.factor = splu(core_matrix)
        else:
            self.factor = TrajectoryKKTSolver(
                core_matrix,
                np.flatnonzero(np.isin(self.core, primal)),
                np.flatnonzero(np.isin(self.core, constraints)),
            )

        self.coupling = matrix[self.core][:, self.design].tocsr()
        self.transpose_coupling = matrix[self.design][:, self.core].tocsr()
        if bounded_indices is not None and self.coupling[self.factor.fixed].nnz:
            raise ValueError("Cached bound responses require isolated prescribed rows")
        reduced = matrix[self.design][:, self.design].toarray()
        private_count = len(self.controls)
        if bounded_indices is not None:
            local = np.flatnonzero(np.isin(self.core, bounded_indices))
            self.bound_core = self.core[local]
            self.bound_core_lookup = {
                int(index): row for row, index in enumerate(self.bound_core)
            }
            self.control_lookup = {
                int(index): row for row, index in enumerate(self.controls)
            }
            self.fixed_lookup = {
                int(self.core[index]): row
                for row, index in enumerate(self.factor.fixed)
            }
            self.bound_response = np.empty((len(local), len(self.design)))
        # The shared diagonal belongs to the complete model, only once.
        reduced[private_count:, private_count:] = 0

        for start in range(0, len(self.design), chunk):
            stop = min(start + chunk, len(self.design))
            response = self.factor.solve(
                self.coupling[:, start:stop].toarray(order="F")
            )
            reduced[:, start:stop] -= self.transpose_coupling @ response
            if bounded_indices is not None:
                self.bound_response[:, start:stop] = -response[local]

        private = reduced[:private_count, :private_count]
        self.private_hessian = (private + private.T) / 2
        self.shared_hessian = reduced[private_count:, private_count:].copy()
        self.cross = reduced[:private_count, private_count:]
        self.private_count = private_count
        self.base_cross = self.cross.copy()
        self.state_responses = {}
        if not normal:
            self.base_state_hessian = self.factor.hessian.copy()
            self.base_fixed_diagonal = self.factor.fixed_diagonal.copy()

    def regularized_blocks(self, diagonal, scaling):
        """Condense state penalties through the unchanged trajectory Jacobian."""
        private, cross = self.private_hessian.copy(), self.base_cross.copy()
        shared = np.zeros_like(self.shared_hessian)
        core_diagonal = diagonal[self.core] * scaling[self.core] ** 2
        state_diagonal = core_diagonal[self.factor.primal]
        selected = np.flatnonzero(state_diagonal)
        missing = [index for index in selected if index not in self.state_responses]
        if hasattr(self, "bound_response"):
            uncached = []
            for index in missing:
                row = self.bound_core_lookup.get(
                    int(self.core[self.factor.primal[index]])
                )
                if row is None:
                    uncached.append(index)
                else:
                    self.state_responses[index] = self.bound_response[row]
            missing = uncached
        if missing:
            probes = np.zeros((len(self.core), len(missing)))
            probes[self.factor.primal[missing], np.arange(len(missing))] = 1
            responses = -(self.coupling.T @ self.factor.solve(probes)).T
            self.state_responses.update(zip(missing, responses))
        if len(selected):
            response = np.array([self.state_responses[index] for index in selected])
            weighted = response * np.sqrt(state_diagonal[selected, None])
            update = weighted.T @ weighted
            private += update[: self.private_count, : self.private_count]
            cross += update[: self.private_count, self.private_count :]
            shared += update[self.private_count :, self.private_count :]

        self.factor.hessian = self.base_state_hessian.copy()
        self.factor.hessian.setdiag(self.base_state_hessian.diagonal() + state_diagonal)
        self.factor.fixed_diagonal = (
            self.base_fixed_diagonal + core_diagonal[self.factor.fixed]
        )
        return private, cross, shared

    def reset_state_penalties(self):
        self.cross = self.base_cross.copy()
        self.factor.hessian = self.base_state_hessian.copy()
        self.factor.fixed_diagonal = self.base_fixed_diagonal.copy()

    def solve_private(self, rhs):
        return scaled_solve(self.private_factor, self.scale, rhs)

    def reduce_rhs(self, rhs):
        response = self.factor.solve(rhs[self.core])
        reduced = -self.transpose_coupling @ response
        private_rhs = rhs[self.controls] + reduced[: self.private_count]
        shared_rhs = reduced[self.private_count :] - self.cross.T @ self.solve_private(
            private_rhs
        )
        return private_rhs, shared_rhs

    def recover(self, rhs, private_rhs, shared_step, result):
        private_step = self.solve_private(private_rhs - self.cross @ shared_step)
        result[self.controls] = private_step
        result[self.core] = self.factor.solve(
            rhs[self.core] - self.coupling @ np.r_[private_step, shared_step]
        )


class JointKKTSolver(LinearSolver):
    """Eliminate each case's states and controls before the 86 shared variables.

    Each case keeps one power control with the trajectory to enforce its SOC
    periodicity. Positive reduced design curvature supplies the Hessian
    correction; every solve is checked against that corrected full KKT matrix.
    No dense matrix containing all cases' controls is formed.
    """

    def __init__(
        self,
        partitions,
        shared,
        primal,
        constraints,
        hessian_correction,
        chunk=64,
        limited_indices=None,
        step_limits=None,
        case_workers=1,
        selective_curvature=False,
        spectral_curvature=False,
        box_qp=False,
        step_bounds=None,
    ):
        self.partitions = partitions
        self.shared = np.asarray(shared)
        self.primal = np.asarray(primal)
        self.constraints = np.asarray(constraints)
        self.hessian_correction = hessian_correction
        self.chunk = chunk
        if case_workers < 1:
            raise ValueError("Require at least one case worker")
        self.case_workers = max(1, min(int(case_workers), len(partitions)))
        self.selective_curvature = selective_curvature or spectral_curvature
        self.spectral_curvature = spectral_curvature
        self.step_bounds = step_bounds
        self.box_qp = box_qp
        if box_qp and step_bounds is None:
            raise ValueError("Box step subproblems require explicit step bounds")
        self.box_qp_iterations = 0
        self.largest_active_set = 0
        self.factor_count = 0
        self.max_linear_residual = 0.0
        self.previous_design_shift = 0.01
        self.limited_indices = np.asarray(
            [] if limited_indices is None else limited_indices, dtype=int
        )
        if len(np.unique(self.limited_indices)) != len(self.limited_indices):
            raise ValueError("Limited design indices must be distinct")
        self.step_limits = np.broadcast_to(
            0.2 if step_limits is None else step_limits, self.limited_indices.shape
        ).copy()
        if not np.all(np.isfinite(self.step_limits)) or np.any(self.step_limits <= 0):
            raise ValueError("Design step limits must be positive and finite")
        self.max_step_limits = self.step_limits.copy()
        self.previous_step_shift = 0.0
        if step_bounds is not None:
            independent = np.concatenate(
                [shared] + [part.controls for part in partitions]
            )
            if not np.array_equal(np.sort(independent), np.sort(self.limited_indices)):
                raise ValueError(
                    "Bound-aware steps require all independent design variables"
                )
            positions = {int(index): k for k, index in enumerate(step_bounds.indices)}
            try:
                self.bound_positions = np.array(
                    [positions[int(index)] for index in self.limited_indices]
                )
            except KeyError as error:
                raise ValueError(
                    "Missing bounds for an independent design variable"
                ) from error
        self.limited_steps = 0
        self.state_limited_steps = 0
        self.largest_design_step_ratio = 0.0
        self.recovered_newton_steps = 0

    def map_cases(self, function, cases):
        """Run independent case work, retaining deterministic summation order."""
        if self.case_workers == 1:
            return list(map(function, cases))
        with ThreadPoolExecutor(max_workers=self.case_workers) as workers:
            return list(workers.map(function, cases))

    def factor(self, hessian, diagonal):
        self.step_shift = 0.0
        self.limit_checked = False
        hessian.copy_data_device_to_host()
        diagonal.copy_device_to_host()
        n, m, _, rowp, cols = hessian.get_nonzero_structure()
        matrix = csr_matrix((hessian.get_data().copy(), cols, rowp), shape=(n, m))
        matrix += self.hessian_correction(matrix.shape)
        diagonal_values = diagonal.get_array()
        matrix.setdiag(matrix.diagonal() + diagonal_values)
        matrix.eliminate_zeros()
        if not np.all(np.isfinite(matrix.data)):
            raise FloatingPointError("Nonfinite joint KKT entries")
        original = matrix.copy()
        if self.spectral_curvature:
            self.unregularized_target = original.copy()
        self.unregularized_diagonal = original.diagonal().copy()
        self.scaling = np.ones(n)
        for _ in range(8):
            row_max = abs(matrix).max(axis=1).toarray().ravel()
            step = 1 / np.sqrt(np.maximum(row_max, 1e-30))
            self.scaling *= step
            matrix = matrix.multiply(step[:, None]).multiply(step[None, :]).tocsr()

        self.factor_count += 1
        self.normal = bool(
            np.any(diagonal_values == 1)
            and np.all((diagonal_values == 0) | (diagonal_values == 1))
        )
        if self.factor_count <= 2:
            print_kkt_matrix(
                original.shape,
                original.nnz,
                np.max(np.abs(original.data)),
                self.factor_count,
                "normal_equations" if self.normal else "newton",
            )

        owner = np.full(n, -1, dtype=int)
        owner[self.shared] = len(self.partitions)
        for index, partition in enumerate(self.partitions):
            if np.any(owner[partition.indices] != -1):
                raise ValueError("Joint KKT partitions overlap")
            owner[partition.indices] = index
        self.owner = owner
        if np.any(owner < 0):
            raise ValueError("Joint KKT partitions do not cover the model")
        entries = matrix.tocoo()
        cross_case = (
            (owner[entries.row] != owner[entries.col])
            & (owner[entries.row] != len(self.partitions))
            & (owner[entries.col] != len(self.partitions))
        )
        if np.any(cross_case):
            raise ValueError("Unexpected direct coupling between case trajectories")

        if self.normal:
            # Initial multiplier estimates can lose precision when an energy
            # balance is used as the trajectory pivot. Pivot the complete
            # initialization system instead; Newton steps use case elimination.
            self.normal_factor = splu(matrix.tocsc())
            self.target = original
            self.regularization_max = 0.0
            return 0

        # Release previous factors before forming the next Newton system.
        self.cases = []
        self.shared_hessian = matrix[self.shared][:, self.shared].toarray()

        def make_case(partition):
            return CaseFactor(
                matrix,
                partition,
                self.shared,
                self.primal,
                self.constraints,
                self.normal,
                self.chunk,
                bounded_indices=self.step_bounds.indices if self.box_qp else None,
            )

        self.cases = self.map_cases(make_case, self.partitions)
        if self.box_qp:
            lookup = {
                int(index): row for row, index in enumerate(self.step_bounds.indices)
            }
            self.shared_lookup = {
                int(index): row for row, index in enumerate(self.shared)
            }
            self.shared_bound_positions = np.array(
                [lookup[int(i)] for i in self.shared]
            )
            for case in self.cases:
                case.control_bound_positions = np.array(
                    [lookup[int(i)] for i in case.controls]
                )
                case.core_bound_positions = np.array(
                    [lookup[int(i)] for i in case.bound_core]
                )
        for case in self.cases:
            self.shared_hessian += case.shared_hessian
        self.shared_hessian = (self.shared_hessian + self.shared_hessian.T) / 2
        shift = self.factor_design_blocks()
        self.regularization_diagonal = shift / self.scaling**2
        if self.spectral_curvature:
            original += self.spectral_update()
        else:
            original.setdiag(original.diagonal() + self.regularization_diagonal)
        original.eliminate_zeros()
        self.target = original
        self.regularization_max = float(np.max(self.regularization_diagonal))
        return 0

    def factor_design_blocks(self, minimum_shift=0.0):
        """Apply one curvature safeguard to the complete coupled design system."""
        if self.selective_curvature:
            return self.factor_selective_design_blocks(minimum_shift)
        for case in self.cases:
            case.metric = np.maximum(abs(np.diag(case.private_hessian)), 1e-12)
            case.scale = 1 / np.sqrt(case.metric)
        shared_metric = np.maximum(abs(np.diag(self.shared_hessian)), 1e-12)
        self.shared_metric = shared_metric
        self.shared_scale = 1 / np.sqrt(shared_metric)

        def factor_with_shift(shift):
            schur = self.shared_hessian.copy()
            schur[np.diag_indices_from(schur)] += shift * shared_metric
            for case in self.cases:
                private = (
                    case.scale[:, None] * case.private_hessian * case.scale[None, :]
                )
                private[np.diag_indices_from(private)] += shift
                case.private_factor = cho_factor(private, lower=True)
                schur -= case.cross.T @ case.solve_private(case.cross)
            self.shared_factor = cho_factor(
                self.shared_scale[:, None] * schur * self.shared_scale[None, :],
                lower=True,
            )

        for shift in SMALL_CURVATURE_SHIFTS:
            if shift < minimum_shift:
                continue
            try:
                factor_with_shift(shift)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            shift = max(1e-6, 0.25 * self.previous_design_shift)
            for _ in range(50):
                try:
                    factor_with_shift(shift)
                    # Keep a curvature margin instead of stopping at a nearly
                    # singular positive matrix.
                    shift *= 2
                    factor_with_shift(shift)
                    break
                except np.linalg.LinAlgError:
                    shift *= 2
            else:
                raise FloatingPointError("Could not correct joint design curvature")
        self.previous_design_shift = shift

        diagonal = np.zeros(len(self.scaling))
        for case in self.cases:
            correction = shift * case.metric
            case.curvature_diagonal = correction
            diagonal[case.controls] = correction
            case.private_hessian[
                np.diag_indices_from(case.private_hessian)
            ] += correction
        correction = shift * shared_metric
        self.shared_curvature_diagonal = correction
        diagonal[self.shared] = correction
        self.shared_hessian[np.diag_indices_from(self.shared_hessian)] += correction
        self.curvature_shift = float(shift)
        return diagonal

    def factor_selective_design_blocks(self, minimum_shift):
        """Correct local curvature, then the coupled shared Schur complement."""
        diagonal = np.zeros(len(self.scaling))
        factor_positive = (
            positive_spectral_factor
            if self.spectral_curvature
            else positive_diagonal_factor
        )

        def factor_private(case):
            case.metric = np.maximum(abs(np.diag(case.private_hessian)), 1e-12)
            floor = minimum_shift * case.metric
            private = case.private_hessian.copy()
            private[np.diag_indices_from(private)] += floor
            case.private_factor, case.scale, extra = factor_positive(private)
            if self.spectral_curvature:
                case.curvature_matrix = extra + np.diag(floor)
                case.curvature_diagonal = case.curvature_matrix.diagonal().copy()
                case.private_hessian += case.curvature_matrix
            else:
                case.curvature_diagonal = floor + extra
                case.private_hessian[
                    np.diag_indices_from(private)
                ] += case.curvature_diagonal
            return case.curvature_diagonal

        corrections = self.map_cases(factor_private, self.cases)
        normalized = []
        for case, correction in zip(self.cases, corrections):
            diagonal[case.controls] = correction
            normalized.append(correction / case.metric)
        self.shared_metric = np.maximum(abs(np.diag(self.shared_hessian)), 1e-12)
        floor = minimum_shift * self.shared_metric
        schur = self.shared_hessian.copy()
        schur[np.diag_indices_from(schur)] += floor
        for case in self.cases:
            schur -= case.cross.T @ case.solve_private(case.cross)
        self.shared_factor, self.shared_scale, extra = factor_positive(schur)
        if self.spectral_curvature:
            self.shared_curvature_matrix = extra + np.diag(floor)
            self.shared_curvature_diagonal = (
                self.shared_curvature_matrix.diagonal().copy()
            )
            self.shared_hessian += self.shared_curvature_matrix
        else:
            self.shared_curvature_diagonal = floor + extra
            self.shared_hessian[
                np.diag_indices_from(schur)
            ] += self.shared_curvature_diagonal
        diagonal[self.shared] = self.shared_curvature_diagonal
        normalized.append(self.shared_curvature_diagonal / self.shared_metric)
        self.curvature_shift = float(np.max(np.concatenate(normalized), initial=0))
        return diagonal

    def spectral_update(self):
        """Assemble case-local design corrections in the original KKT coordinates."""
        blocks = [(case.controls, case.curvature_matrix) for case in self.cases]
        blocks.append((self.shared, self.shared_curvature_matrix))
        rows, columns, values = [], [], []
        for indices, block in blocks:
            local_rows, local_columns = np.nonzero(block)
            row, column = indices[local_rows], indices[local_columns]
            rows.append(row)
            columns.append(column)
            values.append(
                block[local_rows, local_columns]
                / self.scaling[row]
                / self.scaling[column]
            )
        return coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(len(self.scaling), len(self.scaling)),
        ).tocsr()

    def _solve_scaled(self, rhs):
        if self.normal:
            return self.normal_factor.solve(rhs)
        result = np.empty_like(rhs)
        shared_rhs = rhs[self.shared].copy()
        private_rhs = []
        for case in self.cases:
            private, contribution = case.reduce_rhs(rhs)
            private_rhs.append(private)
            shared_rhs += contribution
        shared_step = scaled_solve(self.shared_factor, self.shared_scale, shared_rhs)
        result[self.shared] = shared_step
        for case, private in zip(self.cases, private_rhs):
            case.recover(rhs, private, shared_step, result)
        return result

    def _solve(self, rhs):
        return self.scaling * self._solve_scaled(self.scaling * rhs)

    def _limited_inverse(self):
        """Extract selected design responses without full trajectory responses."""
        count = len(self.limited_indices)
        inverse = np.zeros((count, count))
        shared_response = np.zeros((count, len(self.shared)))
        weights = self.scaling[self.limited_indices] / self.step_limits
        covered = np.zeros(count, dtype=bool)

        for case in self.cases:
            positions = np.flatnonzero(np.isin(self.limited_indices, case.controls))
            if len(positions) == 0:
                continue
            lookup = {int(index): offset for offset, index in enumerate(case.controls)}
            local = np.array([lookup[int(self.limited_indices[i])] for i in positions])
            columns = np.zeros((len(case.controls), len(local)))
            columns[local, np.arange(len(local))] = 1
            inverse[np.ix_(positions, positions)] = case.solve_private(columns)[local]
            shared_response[positions] = case.solve_private(case.cross)[local]
            covered[positions] = True

        positions = np.flatnonzero(np.isin(self.limited_indices, self.shared))
        lookup = {int(index): offset for offset, index in enumerate(self.shared)}
        for position in positions:
            shared_response[position, lookup[int(self.limited_indices[position])]] = -1
        covered[positions] = True
        if not np.all(covered):
            raise ValueError(
                "Limited variables must belong to independent design blocks"
            )
        inverse += shared_response @ scaled_solve(
            self.shared_factor, self.shared_scale, shared_response.T
        )
        inverse *= weights[:, None] * weights[None, :]
        return (inverse + inverse.T) / 2

    def _limit_design_step(self, step, values):
        """Add enough design curvature to respect the requested step limits."""
        self.limit_checked = True
        if self.box_qp:
            self.limit_box_quadratic_step(step)
            return
        if self.step_bounds is not None:
            self.limit_bound_aware_step(step, values)
            return
        normalized = step[self.limited_indices] / self.step_limits
        if np.max(abs(normalized), initial=0) <= 1:
            return

        eigenvalues, modes = eigh(self._limited_inverse())
        if eigenvalues[0] < -1e-8 * max(1, eigenvalues[-1]):
            raise FloatingPointError("Design inverse has negative curvature")
        eigenvalues = np.maximum(eigenvalues, 0)
        projected = modes.T @ normalized

        def maximum_step(shift):
            response = modes @ (projected / (1 + shift * eigenvalues))
            return np.max(abs(response))

        lower, upper = 0.0, 1 / max(1, eigenvalues[-1])
        for _ in range(100):
            if maximum_step(upper) <= 0.99:
                break
            lower, upper = upper, 2 * upper
        else:
            raise FloatingPointError("Could not bound the design Newton direction")
        for _ in range(12):
            middle = (lower + upper) / 2
            if maximum_step(middle) <= 0.99:
                upper = middle
            else:
                lower = middle

        correction = np.zeros(len(self.scaling))
        correction[self.limited_indices] = upper / self.step_limits**2
        self.refactor_limited_blocks(correction)
        self.record_step_shift(upper, correction)

    def record_step_shift(self, shift, correction):
        self.step_shift = shift
        self.target.setdiag(self.target.diagonal() + correction)
        self.target.eliminate_zeros()
        self.regularization_diagonal += correction
        self.regularization_max = float(np.max(self.regularization_diagonal))
        self.limited_steps += 1

    def limit_box_quadratic_step(self, step):
        """Limit only active directions, retaining feasible tangential movement."""
        negative, positive = self.step_bounds.limits()
        self.negative_step_bounds, self.positive_step_bounds = negative, positive
        if self.design_step_ratio(step) <= 1:
            return
        bounded = self.step_bounds.indices

        active, targets, multipliers, iterations = active_box_constraints(
            step[bounded],
            -0.98 * negative,
            0.98 * positive,
            self.bound_inverse_response,
        )
        self.box_qp_iterations += iterations
        self.largest_active_set = max(self.largest_active_set, len(active))
        if self.factor_count <= 2:
            print_status(
                "BOUND-CONSTRAINED NEWTON STEP",
                [("Active bounds", len(active)), ("QP iterations", iterations)],
            )
        correction = np.zeros(len(step))
        correction[bounded[active]] = np.maximum(multipliers / targets, 0)
        self.refactor_limited_blocks(correction)
        self.record_step_shift(1.0, correction)
        self.state_limited_steps += bool(
            np.any(~np.isin(bounded[active], self.limited_indices))
        )

    def bound_inverse_response(self, position):
        """Apply the reduced inverse and cached state sensitivities to one bound."""
        index = int(self.step_bounds.indices[position])
        owner = self.owner[index]
        private_rhs = [np.zeros(len(case.controls)) for case in self.cases]
        shared_rhs = np.zeros(len(self.shared))
        if owner == len(self.cases):
            shared_rhs[self.shared_lookup[index]] = self.scaling[index]
        else:
            case = self.cases[owner]
            if index in case.fixed_lookup:
                result = np.zeros(len(self.step_bounds.indices))
                result[position] = (
                    self.scaling[index] ** 2
                    / case.factor.fixed_diagonal[case.fixed_lookup[index]]
                )
                return result
            if index in case.control_lookup:
                private_rhs[owner][case.control_lookup[index]] = self.scaling[index]
            else:
                row = (
                    case.bound_response[case.bound_core_lookup[index]]
                    * self.scaling[index]
                )
                private_rhs[owner] = row[: case.private_count].copy()
                shared_rhs[:] = row[case.private_count :]
            shared_rhs -= case.cross.T @ case.solve_private(private_rhs[owner])

        shared_step = scaled_solve(self.shared_factor, self.shared_scale, shared_rhs)
        result = np.empty(len(self.step_bounds.indices))
        result[self.shared_bound_positions] = shared_step * self.scaling[self.shared]
        for case, rhs in zip(self.cases, private_rhs):
            private_step = case.solve_private(rhs - case.cross @ shared_step)
            result[case.control_bound_positions] = (
                private_step * self.scaling[case.controls]
            )
            result[case.core_bound_positions] = (
                case.bound_response @ np.r_[private_step, shared_step]
            ) * self.scaling[case.bound_core]
        return result

    def design_step_ratio(self, step):
        if self.step_bounds is None:
            return float(
                np.max(abs(step[self.limited_indices]) / self.step_limits, initial=0)
            )
        values = step[self.step_bounds.indices]
        limits = np.where(
            values >= 0, self.positive_step_bounds, self.negative_step_bounds
        )
        return float(np.max(abs(values) / limits, initial=0))

    def limit_bound_aware_step(self, step, values):
        """Limit states through their design response, without a dense global inverse."""
        negative, positive = self.step_bounds.limits()
        self.negative_step_bounds, self.positive_step_bounds = negative, positive
        negative = np.minimum(self.max_step_limits, negative[self.bound_positions])
        positive = np.minimum(self.max_step_limits, positive[self.bound_positions])
        self.step_limits = np.where(step[self.limited_indices] >= 0, positive, negative)
        if self.design_step_ratio(step) <= 1:
            return

        # Penalize each case's most restrictive eliminated state directly.
        # Otherwise a tight SOC bound can damp every independent control just
        # to prevent that one state from moving outward.
        state_indices, state_negative, state_positive = [], [], []
        bounded = self.step_bounds.indices
        full_negative, full_positive = (
            self.negative_step_bounds,
            self.positive_step_bounds,
        )
        ratios = abs(step[bounded]) / np.where(
            step[bounded] >= 0, full_positive, full_negative
        )
        for case in self.cases:
            positions = np.flatnonzero(np.isin(bounded, case.core) & (ratios > 1))
            if len(positions):
                position = positions[np.argmax(ratios[positions])]
                state_indices.append(bounded[position])
                state_negative.append(full_negative[position])
                state_positive.append(full_positive[position])
        penalized = np.r_[self.limited_indices, state_indices].astype(int)
        negative = np.r_[negative, state_negative]
        positive = np.r_[positive, state_positive]
        limits = np.where(step[penalized] >= 0, positive, negative)

        # A regularized direction can turn inward. Release the old outward
        # penalty before choosing the final diagonal correction.
        for _ in range(4):
            weights = limits.copy()

            def trial(shift):
                correction = np.zeros(len(step))
                correction[penalized] = shift / weights**2
                try:
                    self.refactor_limited_blocks(correction)
                    candidate = self._solve(values)
                except FloatingPointError:
                    return np.inf, correction, None, None
                desired = np.where(candidate[penalized] >= 0, positive, negative)
                return self.design_step_ratio(candidate), correction, candidate, desired

            lower, upper = 0.0, max(1e-6, 0.25 * self.previous_step_shift)
            for _ in range(40):
                ratio, _, _, _ = trial(upper)
                if ratio <= 0.99:
                    break
                lower, upper = upper, 10 * upper
            else:
                raise FloatingPointError(
                    "Could not bound the design and trajectory step"
                )
            if lower == 0:
                for _ in range(30):
                    probe = upper / 10
                    ratio, _, _, _ = trial(probe)
                    if ratio > 0.99:
                        lower = probe
                        break
                    upper = probe
            for _ in range(8):
                middle = (lower + upper) / 2
                ratio, _, _, _ = trial(middle)
                if ratio <= 0.99:
                    upper = middle
                else:
                    lower = middle
            ratio, correction, candidate, desired = trial(upper)
            changed = (desired != weights) & (
                abs(candidate[penalized]) > 1e-8 * np.minimum(positive, negative)
            )
            self.previous_step_shift = upper
            if not np.any(changed):
                break
            limits = np.where(changed, desired, weights)
        self.step_limits = desired[: len(self.limited_indices)]
        self.record_step_shift(upper, correction)
        self.state_limited_steps += bool(state_indices)

    def refactor_limited_blocks(self, diagonal):
        """Refactor the bounded system instead of subtracting large inverse steps."""
        shared = self.shared_hessian.copy()
        shared[np.diag_indices_from(shared)] += (
            diagonal[self.shared] * self.scaling[self.shared] ** 2
        )

        def factor_private(case):
            private, cross, state_shared = case.regularized_blocks(
                diagonal, self.scaling
            )
            case.cross = cross
            private[np.diag_indices_from(private)] += (
                diagonal[case.controls] * self.scaling[case.controls] ** 2
            )
            try:
                case.private_factor, case.scale = scaled_cholesky(private)
            except np.linalg.LinAlgError:
                raise FloatingPointError(
                    "Step regularization lost positive design curvature"
                )
            return case.cross.T @ case.solve_private(case.cross) - state_shared

        for contribution in self.map_cases(factor_private, self.cases):
            shared -= contribution
        try:
            self.shared_factor, self.shared_scale = scaled_cholesky(shared)
        except np.linalg.LinAlgError:
            raise FloatingPointError(
                "Step regularization lost positive shared curvature"
            )

    def solve(self, rhs, solution):
        rhs.copy_device_to_host()
        values = rhs.get_array()
        result = self._solve(values)
        new_direction = not self.normal and not self.limit_checked
        if new_direction:
            self._limit_design_step(result, values)
            if self.step_shift > 0:
                result = self._solve(values)
        if new_direction:
            result = self.recover_original_step(values, result)
        scale = max(1.0, np.linalg.norm(values, np.inf))
        error = np.linalg.norm(values - self.target @ result, np.inf) / scale
        best_result, best_error = result.copy(), error
        for _ in range(8):
            if error < 1e-10:
                break
            result += self._solve(values - self.target @ result)
            error = np.linalg.norm(values - self.target @ result, np.inf) / scale
            if np.isfinite(error) and error < best_error:
                best_result, best_error = result.copy(), error
        result, error = best_result, best_error
        tolerance = 1e-5 if self.normal else 1e-7
        if (
            new_direction
            and (not np.isfinite(error) or error > tolerance)
            and self.curvature_shift < 1e-8
        ):
            self.strengthen_curvature_safeguard()
            return self.solve(rhs, solution)
        if not np.isfinite(error) or error > tolerance:
            raise FloatingPointError(f"Inaccurate joint KKT solve: {error:.3e}")
        self.max_linear_residual = max(self.max_linear_residual, float(error))
        if new_direction and len(self.limited_indices):
            step_ratio = self.design_step_ratio(result)
            if step_ratio > 1 + 1e-6:
                raise FloatingPointError(
                    f"Design direction exceeded its limit: {step_ratio:.3e}"
                )
            self.largest_design_step_ratio = max(
                self.largest_design_step_ratio, step_ratio
            )
        solution.get_array()[:] = result
        solution.copy_host_to_device()

    def strengthen_curvature_safeguard(self):
        """Retry an inaccurate small-shift solve with stronger design curvature."""
        previous = self.curvature_shift
        for case in self.cases:
            case.reset_state_penalties()
            if self.spectral_curvature:
                case.private_hessian -= case.curvature_matrix
            else:
                case.private_hessian[
                    np.diag_indices_from(case.private_hessian)
                ] -= case.curvature_diagonal
        if self.spectral_curvature:
            self.shared_hessian -= self.shared_curvature_matrix
        else:
            self.shared_hessian[
                np.diag_indices_from(self.shared_hessian)
            ] -= self.shared_curvature_diagonal
        minimum = next(shift for shift in SMALL_CURVATURE_SHIFTS if shift > previous)
        diagonal = self.factor_design_blocks(minimum_shift=minimum)
        self.regularization_diagonal = diagonal / self.scaling**2
        self.regularization_max = float(np.max(self.regularization_diagonal))
        if self.spectral_curvature:
            self.target = self.unregularized_target + self.spectral_update()
        else:
            self.target.setdiag(
                self.unregularized_diagonal + self.regularization_diagonal
            )
        self.step_shift = 0.0
        self.limit_checked = False

    def recover_original_step(self, values, result):
        """Remove rounding-level damping when a checked Newton solve permits it."""
        if (
            self.spectral_curvature
            or self.step_shift
            or not 0 < self.curvature_shift <= 1e-8
        ):
            return result
        size = np.linalg.norm(values, np.inf)
        # Eliminating bound duals can enlarge the RHS even near stationarity.
        # Gate on equality defects and verify the original system after solving.
        if np.max(abs(values[self.constraints]), initial=0) > 1e-7:
            return result
        tolerance = min(1e-7, max(1e-12, 0.01 * size))
        saved_diagonal = self.target.diagonal().copy()
        self.target.setdiag(self.unregularized_diagonal)
        accepted = False
        try:
            operator = LinearOperator(
                self.target.shape,
                matvec=lambda point: self.scaling
                * (self.target @ (self.scaling * point)),
                dtype=float,
            )
            preconditioner = LinearOperator(
                self.target.shape, matvec=self._solve_scaled, dtype=float
            )

            class ResidualSatisfied(Exception):
                pass

            def check_original_residual(scaled):
                candidate = self.scaling * scaled
                if np.max(abs(values - self.target @ candidate)) <= tolerance:
                    raise ResidualSatisfied(scaled.copy())

            try:
                scaled, _ = gmres(
                    operator,
                    self.scaling * values,
                    x0=result / self.scaling,
                    M=preconditioner,
                    rtol=0,
                    atol=0.5 * tolerance * np.min(self.scaling),
                    restart=100,
                    maxiter=10,
                    callback=check_original_residual,
                    callback_type="x",
                )
            except ResidualSatisfied as converged:
                scaled = converged.args[0]
            candidate = self.scaling * scaled
            error = np.max(abs(values - self.target @ candidate))
            curvature = (
                candidate[self.primal] @ values[self.primal]
                - candidate[self.constraints] @ values[self.constraints]
            )
            bounded = self.design_step_ratio(candidate) <= 1
            if np.isfinite(error) and error <= tolerance and curvature >= 0 and bounded:
                accepted = True
                self.regularization_diagonal.fill(0)
                self.regularization_max = 0.0
                self.recovered_newton_steps += 1
                print_status(
                    "RECOVERED ORIGINAL NEWTON SYSTEM",
                    [("Linear residual", f"{error:.3e}")],
                )
                return candidate
        finally:
            if not accepted:
                self.target.setdiag(saved_diagonal)
            self.target.eliminate_zeros()
        return result


def joint_partitions(joint):
    model = joint.model
    shared = model.get_indices(
        [
            "shared_surface.cellInstd[0, :]",
            "shared_angles.finAngle[:]",
            "shared_angles.antAngle[:]",
        ]
    )
    partitions = []
    for case in joint.cases:
        prefix = f"case{case.case_index}."
        names = [
            f"{name}.{variable}"
            for name, group in model.comp.items()
            if name.startswith(prefix)
            for variable in group.vars
        ]
        indices = np.setdiff1d(np.unique(model.get_indices(names)), shared)
        controls = model.get_indices(
            [
                prefix + name
                for name in [
                    "control_points.CP_P_comm[:]",
                    "control_points.CP_gamma[:]",
                    "control_points.CP_Isetpt[:, :]",
                    "battery_ic.initial_soc[:]",
                ]
            ]
        )
        pivot = int(np.argmax(case.B.T @ case.CommLOS))
        energy_control = model.get_indices(prefix + "control_points.CP_P_comm[:]")[
            pivot
        ]
        partitions.append(CasePartition(indices, controls[controls != energy_control]))
    return partitions, shared
