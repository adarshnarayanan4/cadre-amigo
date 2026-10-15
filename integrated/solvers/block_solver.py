"""Sparse LU and trajectory elimination for CADRE KKT systems."""

from amigo.algorithm.solvers.linear_solver import LinearSolver
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, lu_factor, lu_solve
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, gmres, splu


class TrajectoryKKTSolver:
    """Invert [H, J.T; J, 0] through a square trajectory Jacobian.

    With independent design controls removed, J determines the trajectory.
    Its factorization serves both state and adjoint solves, regardless of H's
    curvature. Prescribed variables appear as isolated diagonal rows.
    """

    def __init__(self, matrix, primal, constraints):
        matrix = matrix.tocsr()
        primal = np.asarray(primal)
        self.constraints = np.asarray(constraints)
        jacobian = matrix[self.constraints][:, primal].tocsr()
        active = np.asarray(jacobian.getnnz(axis=0)).ravel() != 0
        self.primal = primal[active]
        self.fixed = primal[~active]

        if len(self.primal) != len(self.constraints):
            raise ValueError(
                "Trajectory Jacobian is not square after control elimination"
            )

        if np.any(np.diff(matrix.indptr)[self.fixed] != 1):
            raise ValueError(
                "Unconstrained local variables are not isolated prescribed rows"
            )

        self.fixed_diagonal = matrix.diagonal()[self.fixed]

        self.hessian = matrix[self.primal][:, self.primal].tocsr()

        self.jacobian_lu = splu(jacobian[:, active].tocsc())
        self.L, self.U = self.jacobian_lu.L, self.jacobian_lu.U

    def solve(self, rhs):
        result = np.empty_like(rhs)

        state_step = self.jacobian_lu.solve(rhs[self.constraints])
        result[self.primal] = state_step

        result[self.constraints] = self.jacobian_lu.solve(
            rhs[self.primal] - self.hessian @ state_step, trans="T"
        )

        denominator = (
            self.fixed_diagonal if rhs.ndim == 1 else self.fixed_diagonal[:, None]
        )
        result[self.fixed] = rhs[self.fixed] / denominator

        return result


class SparseLUSolver(LinearSolver):
    """Pivoted sparse LU for AMIGO's complete primal-dual KKT matrix.

    SuperLU supplies sparse ordering and numerical pivoting. AMIGO retains
    its barrier updates, line search, and KKT convergence checks; this
    backend does not provide inertia-based Hessian correction.
    """

    def __init__(
        self,
        hessian_correction=None,
        shared_indices=None,
        safeguard_shared=False,
        coupling_chunk_size=128,
        normal_shared_indices=None,
        primal_indices=None,
        constraint_indices=None,
    ):
        self.hessian_correction = hessian_correction
        self.shared_indices = shared_indices
        self.full_shared_indices = shared_indices
        self.normal_shared_indices = normal_shared_indices

        self.structure = None

        self.lu = None
        self.factor_count = 0

        self.schur_cache = {}

        self.schur_scale_cache = {}

        self.schur_cholesky_cache = {}

        self.safeguard_shared = safeguard_shared
        self.shared_regularization = None
        self.coupling_chunk_size = coupling_chunk_size

        self.primal_indices = primal_indices
        self.constraint_indices = constraint_indices

    def factor(self, hessian, diagonal):
        self.shared_regularization = None
        self.curvature_shift = 0.0

        if self.structure is None:
            self.structure = hessian.get_nonzero_structure()

        nrows, ncols, _, rowp, cols = self.structure
        diagonal.copy_device_to_host()

        diagonal_values = diagonal.get_array()

        self.schur_key = (
            "normal_equations"
            if np.any(diagonal_values == 1.0)
            and np.all((diagonal_values == 0.0) | (diagonal_values == 1.0))
            else "newton"
        )

        self.shared_indices = (
            self.normal_shared_indices
            if self.schur_key == "normal_equations"
            and self.normal_shared_indices is not None
            else self.full_shared_indices
        )

        hessian.copy_data_device_to_host()

        matrix = csr_matrix(
            (np.array(hessian.get_data(), copy=True), cols, rowp),
            shape=(nrows, ncols),
            copy=True,
        )

        if self.hessian_correction is not None:
            matrix = matrix + self.hessian_correction(matrix.shape)

        matrix.setdiag(matrix.diagonal() + diagonal.get_array())

        self.factor_count += 1

        if self.factor_count <= 2:
            print(
                "KKT matrix:",
                matrix.shape,
                "nnz:",
                matrix.nnz,
                "max magnitude:",
                np.max(np.abs(matrix.data)),
                flush=True,
            )

        if not np.all(np.isfinite(matrix.data)):
            positions = np.flatnonzero(~np.isfinite(matrix.data))[:10]
            bad_rows = np.searchsorted(matrix.indptr, positions, side="right") - 1
            raise FloatingPointError(
                f"Nonfinite KKT entries at {list(zip(bad_rows, matrix.indices[positions]))}"
            )

        matrix.eliminate_zeros()

        self.unregularized_kkt = matrix

        self.lu = None  # Release the previous factors before allocating new ones.

        if self.shared_indices is None:
            self.lu = splu(matrix.tocsc(), permc_spec="COLAMD", diag_pivot_thresh=1.0)
        else:
            self._factor_shared_blocks(matrix)

        return 0

    def _factor_shared_blocks(self, matrix):
        """Eliminate local states and reuse a shared-block preconditioner if available."""
        nrows = matrix.shape[0]
        # Eliminate local trajectory states first, then solve the small
        # dense Schur complement of globally shared surface allocations.
        # This is an exact block factorization of the same KKT matrix.
        shared = np.asarray(self.shared_indices)
        local = np.setdiff1d(np.arange(nrows), shared)
        core = matrix[local][:, local].tocsc()
        self.core_coupling = None

        if self.primal_indices is not None and self.schur_key == "newton":
            position = np.full(nrows, -1, dtype=int)
            position[local] = np.arange(len(local))
            primal = position[np.asarray(self.primal_indices)]

            primal = primal[primal >= 0]

            constraints = position[np.asarray(self.constraint_indices)]

            if np.any(constraints < 0):
                raise ValueError(
                    "Trajectory elimination requires all equality rows to remain local"
                )

            self.lu = TrajectoryKKTSolver(core, primal, constraints)
        else:
            self.lu = splu(core, permc_spec="COLAMD", diag_pivot_thresh=1.0)

        self.local_shared = matrix[local][:, shared].tocsr()
        self.shared_local = matrix[shared][:, local].tocsr()
        self.shared_block = matrix[shared][:, shared].toarray()
        self.local_indices = local
        self.kkt_matrix = matrix

        # A previous dense shared-block factor is a useful preconditioner
        # when successive IPM iterates are close. Rebuild it if the tight
        # linear residual test fails; the Newton system stays unchanged.
        self.schur_lu = self.schur_cache.get(self.schur_key)
        self.schur_scaling = self.schur_scale_cache.get(self.schur_key)
        self.schur_cholesky = self.schur_cholesky_cache.get(self.schur_key, False)

        self.iterative_schur = (
            len(shared) > 128
            and self.schur_lu is not None
            and not self.safeguard_shared
        )

        if not self.iterative_schur:
            self._refresh_schur()

    def _form_schur(self):
        """Form the reduced matrix, limiting memory for large control sets."""
        if self.local_shared.nnz == 0:
            self.core_coupling = None
            schur = self.shared_block
        elif self.local_shared.shape[1] <= self.coupling_chunk_size:
            self.core_coupling = self.lu.solve(self.local_shared.toarray(order="F"))
            schur = self.shared_block - self.shared_local @ self.core_coupling
        else:
            # Form the reduced matrix without retaining a trajectory-by-control
            # dense inverse product (8.7 GB for the full 4,284-control case).
            self.core_coupling = None
            schur = self.shared_block.copy()

            for start in range(0, self.local_shared.shape[1], self.coupling_chunk_size):
                stop = start + self.coupling_chunk_size
                coupling = self.lu.solve(
                    self.local_shared[:, start:stop].toarray(order="F")
                )
                schur[:, start:stop] -= self.shared_local @ coupling

        return schur

    def _refresh_schur(self):
        """Scale, factor, and cache the current shared-variable Schur complement."""
        schur = self._form_schur()
        self.schur_scaling = 1 / np.sqrt(np.maximum(np.abs(np.diag(schur)), 1e-12))
        scaled_schur = self.schur_scaling[:, None] * schur * self.schur_scaling[None, :]
        self.schur_cholesky = False

        if self.safeguard_shared and self.schur_key == "newton":
            self._factor_safeguarded_schur(scaled_schur)
        else:
            self.schur_lu = lu_factor(scaled_schur)

        self.schur_cache[self.schur_key] = self.schur_lu
        self.schur_scale_cache[self.schur_key] = self.schur_scaling
        self.schur_cholesky_cache[self.schur_key] = self.schur_cholesky
        self.iterative_schur = False

    def _factor_safeguarded_schur(self, scaled_schur):
        """Shift nonpositive shared curvature and apply the shift to the full KKT."""
        symmetric = (scaled_schur + scaled_schur.T) / 2
        shift = 0.0

        try:
            self.schur_lu = cho_factor(symmetric, lower=True)
        except np.linalg.LinAlgError:
            shift = 1e-8
            shifted = symmetric + shift * np.eye(len(symmetric))

            try:
                self.schur_lu = cho_factor(shifted, lower=True)
            except np.linalg.LinAlgError:
                minimum = float(
                    eigh(symmetric, eigvals_only=True, subset_by_index=[0, 0])[0]
                )
                shift = max(1e-8, -2 * minimum)
                self.schur_lu = cho_factor(
                    symmetric + shift * np.eye(len(symmetric)), lower=True
                )

        self.schur_cholesky = True

        if shift:
            self.curvature_shift = shift
            diagonal_correction = shift / self.schur_scaling**2
            self.shared_regularization = np.diag(diagonal_correction)
            self.shared_block[
                np.diag_indices_from(self.shared_block)
            ] += diagonal_correction
            self.kkt_matrix = self.kkt_matrix + csr_matrix(
                (diagonal_correction, (self.shared_indices, self.shared_indices)),
                shape=self.kkt_matrix.shape,
            )

            print(
                "Shared curvature safeguard; scaled diagonal shift:",
                shift,
                flush=True,
            )

    def _schur_product(self, values):
        """Apply the Schur complement without forming a dense matrix."""
        local_response = self.lu.solve(self.local_shared @ values)

        return self.shared_block @ values - self.shared_local @ local_response

    def _apply_schur_preconditioner(self, values):
        return self.schur_scaling * lu_solve(self.schur_lu, self.schur_scaling * values)

    def _solve_block(self, values):
        local_solution = self.lu.solve(values[self.local_indices])
        shared_rhs = values[self.shared_indices] - self.shared_local @ local_solution

        if self.iterative_schur:
            operator = LinearOperator(
                self.shared_block.shape, matvec=self._schur_product
            )
            preconditioner = LinearOperator(
                self.shared_block.shape, matvec=self._apply_schur_preconditioner
            )
            shared_solution, _ = gmres(
                operator,
                shared_rhs,
                M=preconditioner,
                rtol=1e-11,
                atol=1e-13,
                restart=50,
                maxiter=3,
            )
            result = np.empty_like(values)
            result[self.shared_indices] = shared_solution
            result[self.local_indices] = self.lu.solve(
                values[self.local_indices] - self.local_shared @ shared_solution
            )
            error = np.linalg.norm(self.kkt_matrix @ result - values, np.inf) / max(
                1.0, np.linalg.norm(values, np.inf)
            )

            # GMRES tests its Schur residual, which can be poorly scaled.
            # Accept using the independently checked complete KKT residual.
            if np.isfinite(error) and error < 1e-10:
                return result

            print(
                "Refreshing shared-block factorization; linear residual:",
                error,
                flush=True,
            )

            self._refresh_schur()

        dense_solve = cho_solve if self.schur_cholesky else lu_solve
        shared_solution = self.schur_scaling * dense_solve(
            self.schur_lu, self.schur_scaling * shared_rhs
        )

        result = np.empty_like(values)

        if self.core_coupling is not None:
            local_solution = local_solution - self.core_coupling @ shared_solution
        elif self.local_shared.nnz:
            local_solution = self.lu.solve(
                values[self.local_indices] - self.local_shared @ shared_solution
            )

        result[self.local_indices] = local_solution
        result[self.shared_indices] = shared_solution

        return result

    def _recover_newton_step(self, values, result):
        """Try the unshifted system when regularization only corrects roundoff."""
        target = self.kkt_matrix
        self.solve_unregularized = False

        # A rounding-level shift is a preconditioner, not a reason to
        # damp feasible control directions indefinitely. Recover the
        # original Newton step when a checked Krylov solve permits it.
        if self.schur_key != "newton" or not 0 < self.curvature_shift <= 1e-8:
            return result, target

        operator = LinearOperator(
            self.kkt_matrix.shape,
            matvec=lambda v: self.unregularized_kkt @ v,
            dtype=float,
        )
        preconditioner = LinearOperator(
            self.kkt_matrix.shape, matvec=self._solve_block, dtype=float
        )
        candidate, _ = gmres(
            operator,
            values,
            x0=result,
            M=preconditioner,
            rtol=1e-11,
            atol=1e-13,
            restart=30,
            maxiter=3,
        )
        original_error = np.linalg.norm(
            values - self.unregularized_kkt @ candidate, np.inf
        ) / max(1.0, np.linalg.norm(values, np.inf))

        shared_step = candidate[self.shared_indices]
        reduced_product = (
            self.shared_block @ shared_step
            - np.diag(self.shared_regularization) * shared_step
            - self.shared_local @ self.lu.solve(self.local_shared @ shared_step)
        )
        curvature = float(shared_step @ reduced_product)

        if np.isfinite(original_error) and original_error < 1e-9 and curvature >= 0:
            result, target = candidate, self.unregularized_kkt
            self.solve_unregularized = True

            print(
                "Recovered original Newton system; linear residual:",
                original_error,
                flush=True,
            )

        return result, target

    def _refine_solution(self, values, result, target):
        """Improve the full-system residual and enforce its acceptance tolerance."""
        scale = max(1.0, np.linalg.norm(values, np.inf))
        error = np.linalg.norm(values - target @ result, np.inf) / scale

        for _ in range(3):
            if error < 1e-10:
                break

            candidate = result + self._solve_block(values - target @ result)
            next_error = np.linalg.norm(values - target @ candidate, np.inf) / scale

            if not np.isfinite(next_error) or next_error >= error:
                break

            result, error = candidate, next_error

        # Least-squares multipliers are only an initial estimate; their
        # normal equations can lose more precision than the Newton system.
        tolerance = 1e-5 if self.schur_key == "normal_equations" else 1e-7

        if self.schur_key == "normal_equations" and error > 1e-7:
            print(
                "Approximate multiplier initialization; linear residual:",
                error,
                flush=True,
            )

        if not np.isfinite(error) or error > tolerance:
            raise FloatingPointError(f"Inaccurate block KKT solve: {error}")

        return result

    def solve(self, rhs, solution):
        """Solve the factored system and copy the answer back to AMIGO."""
        rhs.copy_device_to_host()

        values = rhs.get_array()

        if self.shared_indices is None:
            result = self.lu.solve(values)
        else:
            result = self._solve_block(values)
            result, target = self._recover_newton_step(values, result)
            result = self._refine_solution(values, result, target)

        solution.get_array()[:] = result
        solution.copy_host_to_device()
