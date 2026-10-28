"""Keep trial trajectories consistent with the independent design controls."""

import numpy as np
from amigo.algorithm.line_search.filter_line_search import FilterLineSearch
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from integrated.reference.joint_replay import replay_joint
from integrated.solvers.evaluator import ConsistentEvaluator
from integrated.solvers.joint import joint_partitions
from integrated.reporting.console import print_status

MINIMUM_GAP_FRACTION = 0.005
POWER_REPAIR_GAP_FRACTION = 0.2


class TrajectoryProjector:
    def __init__(self, joint):
        self.joint = joint
        self.buffer = joint.model.create_vector()
        self.lower = joint.model.get_lower().get_vector()
        self.upper = joint.model.get_upper().get_vector()
        self.lower.copy_device_to_host()
        self.upper.copy_device_to_host()
        self.primal = joint.model.problem.get_primal_indices()
        self.indices = {}
        self.evaluations = 0

    def project(self, point, cases=None):
        self.buffer.get_vector().copy(point)
        trajectories = replay_joint(self.joint, self.buffer, collect=True, cases=cases)
        point.copy_device_to_host()
        values = point.get_array()
        for name, array in trajectories.items():
            if name not in self.indices:
                self.indices[name] = self.joint.model.get_indices(name)
            indices = self.indices[name]
            values[indices] = np.asarray(array).reshape(indices.shape)
        self.evaluations += 1
        primal = values[self.primal]
        lower, upper = (
            self.lower.get_array()[self.primal],
            self.upper.get_array()[self.primal],
        )
        finite = np.all(np.isfinite(values))
        interior = np.all(primal > lower) and np.all(primal < upper)
        point.copy_host_to_device()
        return bool(finite and interior)


class BatteryRestorer:
    """Restore SOC periodicity with bounded power and initial-SOC corrections.

    Each case scales its power profile, with optional initial-SOC adjustment.
    Sensitivities come from the native Jacobian, excluding the periodicity row.
    The original AMIGO objective and KKT equations are unchanged.
    """

    def __init__(
        self, projector, interior=False, periodicity_tolerance=1e-9, full_power=False
    ):
        self.projector = projector
        self.interior = interior
        if not np.isfinite(periodicity_tolerance) or periodicity_tolerance <= 0:
            raise ValueError("Periodicity tolerance must be positive and finite")
        self.periodicity_tolerance = periodicity_tolerance
        self.full_power = full_power
        self.full_power_solves = 0
        self.reported_full_power = False
        self.joint = projector.joint
        self.model = self.joint.model
        self.constraints = self.model.problem.get_constraint_indices()
        self.design = self.joint.design_indices()
        self.partitions, _ = joint_partitions(self.joint)
        self.matrix = self.model.create_matrix()
        self.calls = 0
        self.case_solves = 0
        self.last_message = ""

    def restore(
        self,
        point,
        max_iterations=20,
        verbose=False,
        adjust_initial_soc=True,
        reference=None,
    ):
        self.calls += 1
        self.last_message = ""
        point.copy_device_to_host()
        original = point.get_array().copy()
        try:
            for case, partition in zip(self.joint.cases, self.partitions):
                oracle = BatteryRestorationCase(
                    self, case, partition, point, adjust_initial_soc=adjust_initial_soc
                )
                success = oracle.restore(point, max_iterations=max_iterations)
                if (
                    success
                    and self.full_power
                    and not adjust_initial_soc
                    and reference is not None
                ):
                    success = oracle.within_reference_gaps(reference)
                if not success and self.full_power and not adjust_initial_soc:
                    self.full_power_solves += 1
                    oracle = BatteryRestorationCase(
                        self,
                        case,
                        partition,
                        point,
                        adjust_initial_soc=False,
                        full_power=True,
                        reference=reference,
                    )
                    success = oracle.restore(
                        point, max_iterations=max(8, max_iterations)
                    )
                    if success and reference is not None:
                        success = oracle.within_reference_gaps(reference)
                    if success and not self.reported_full_power:
                        self.reported_full_power = True
                        print_status(
                            "BATTERY PROFILE REPAIR",
                            [
                                ("Case", case.case_index),
                                ("Power controls", case.m),
                                ("SOC periodicity", f"{oracle.equality[0]:.3e}"),
                            ],
                        )
                if verbose:
                    print(
                        f"  Case {case.case_index}: SOC periodicity {oracle.equality[0]:.3e}",
                        flush=True,
                    )
                if not success:
                    self.last_message = f"Case {case.case_index}: {oracle.message}"
                    break
            else:
                if self.projector.project(point):
                    return True
                self.last_message = "Restored trajectory is outside strict bounds"
        except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
            self.last_message = str(error)
        point.get_array()[:] = original
        point.copy_host_to_device()
        return False


class _BatteryFeasible(Exception):
    """Stop SQP once the correction meets the physical acceptance checks."""


class BatteryRestorationCase:
    def __init__(
        self,
        restorer,
        case,
        partition,
        point,
        adjust_initial_soc=True,
        full_power=False,
        reference=None,
    ):
        self.restorer, self.case = restorer, case
        self.reference = reference
        self.reference_fraction = (
            1.01 * POWER_REPAIR_GAP_FRACTION if full_power else MINIMUM_GAP_FRACTION
        )
        self.model = restorer.model
        self.state = self.model.create_vector()
        self.vector = self.state.get_vector()
        self.vector.copy(point)
        self.vector.fill_at(restorer.constraints, 0.0)
        prefix = f"case{case.case_index}."
        self.controls = self.model.get_indices(
            [
                prefix + "control_points.CP_P_comm[:]",
                prefix + "battery_ic.initial_soc[:]",
            ]
        )
        self.periodicity = self.model.get_indices(prefix + "battery_periodicity.res[:]")
        self.rows = np.setdiff1d(
            np.intersect1d(partition.indices, restorer.constraints), self.periodicity
        )
        self.states = np.setdiff1d(
            np.intersect1d(partition.indices, restorer.projector.primal),
            restorer.design,
        )
        self.required_margins = np.zeros(len(self.states))
        if restorer.interior:
            for name, margin in [("SOC", 0.015), ("I_bat", 0.15)]:
                indices = self.model.get_indices(prefix + f"battery.{name}[:]")
                self.required_margins[np.isin(self.states, indices)] = margin
        self.lower = restorer.projector.lower.get_array()
        self.upper = restorer.projector.upper.get_array()
        self.start = point.get_array()[self.controls].copy()
        self.width = self.upper[self.controls] - self.lower[self.controls]
        power_scale = np.linalg.norm(self.start[:-1] / self.width[:-1])
        if power_scale <= 0 or np.any(self.start[:-1] <= 0):
            raise ValueError(
                "Battery restoration requires positive interior power controls"
            )
        lower_gap = self.start - self.lower[self.controls]
        upper_gap = self.upper[self.controls] - self.start
        lower_margin = np.minimum(1e-10 * self.width, 0.1 * lower_gap)
        upper_margin = np.minimum(1e-10 * self.width, 0.1 * upper_gap)
        if reference is not None:
            lower_margin = np.maximum(
                lower_margin,
                self.reference_fraction
                * (reference[self.controls] - self.lower[self.controls]),
            )
            upper_margin = np.maximum(
                upper_margin,
                self.reference_fraction
                * (self.upper[self.controls] - reference[self.controls]),
            )
        lo = -lower_gap + lower_margin
        hi = upper_gap - upper_margin
        if full_power:
            columns = len(self.controls) - 1 + int(adjust_initial_soc)
            self.mapping = np.zeros((len(self.controls), columns))
            self.mapping[:-1, : len(self.controls) - 1] = np.diag(self.width[:-1])
            self.step_lower = lo[:-1] / self.width[:-1]
            self.step_upper = hi[:-1] / self.width[:-1]
        else:
            self.mapping = np.zeros(
                (len(self.controls), 2 if adjust_initial_soc else 1)
            )
            self.mapping[:-1, 0] = self.start[:-1] / power_scale
            self.step_lower = np.array([np.max(lo[:-1] / self.mapping[:-1, 0])])
            self.step_upper = np.array([np.min(hi[:-1] / self.mapping[:-1, 0])])
        if adjust_initial_soc:
            self.mapping[-1, -1] = self.width[-1]
            self.step_lower = np.r_[self.step_lower, lo[-1] / self.width[-1]]
            self.step_upper = np.r_[self.step_upper, hi[-1] / self.width[-1]]
        self.last = None
        self.message = ""
        self.margins = None

    def evaluate(self, normalized, derivatives=True):
        if self.last is None or not np.array_equal(normalized, self.last):
            self.vector.copy_device_to_host()
            self.vector.get_array()[self.controls] = (
                self.start + self.mapping @ normalized
            )
            self.vector.copy_host_to_device()
            self.restorer.projector.project(self.vector, cases=[self.case])
            self.vector.copy_device_to_host()
            values = self.vector.get_array()
            final_soc = self.model.get_indices(
                f"case{self.case.case_index}.battery.SOC[:]"
            )[-1]
            self.equality = np.array([values[final_soc] - values[self.controls[-1]]])
            states = self.states
            self.finite_lower = np.isfinite(self.lower[states])
            self.finite_upper = np.isfinite(self.upper[states])
            self.scale = np.ones(len(states))
            both = self.finite_lower & self.finite_upper
            self.scale[both] = np.maximum(
                1, self.upper[states[both]] - self.lower[states[both]]
            )
            lo, hi = self.finite_lower, self.finite_upper
            lower_gaps = (values[states[lo]] - self.lower[states[lo]]) / self.scale[lo]
            upper_gaps = (self.upper[states[hi]] - values[states[hi]]) / self.scale[hi]
            gaps = np.r_[lower_gaps, upper_gaps]
            if self.margins is None:
                self.margins = np.where(gaps > 0, np.minimum(1e-10, 0.1 * gaps), 1e-10)
                required = self.required_margins / self.scale
                self.margins = np.maximum(
                    self.margins, np.r_[required[lo], required[hi]]
                )
                if self.reference is not None:
                    old_lower = (
                        self.reference[states[lo]] - self.lower[states[lo]]
                    ) / self.scale[lo]
                    old_upper = (
                        self.upper[states[hi]] - self.reference[states[hi]]
                    ) / self.scale[hi]
                    old_gaps = np.r_[old_lower, old_upper]
                    self.margins = np.maximum(
                        np.r_[required[lo], required[hi]],
                        np.maximum(
                            np.minimum(1e-10, 0.1 * old_gaps),
                            self.reference_fraction * old_gaps,
                        ),
                    )
            self.inequality = gaps - self.margins
            if not np.all(np.isfinite(self.inequality)) or not np.all(
                np.isfinite(self.equality)
            ):
                raise FloatingPointError("Nonfinite battery restoration trajectory")
            self.last = normalized.copy()
            self.derivatives_current = False
        if derivatives and not self.derivatives_current:
            self.differentiate()
            self.derivatives_current = True

    def differentiate(self):
        self.model.problem.update(self.vector)
        native = self.restorer.matrix
        self.model.problem.hessian(1.0, self.vector, native)
        native.copy_data_device_to_host()
        n, m, _, rowp, cols = native.get_nonzero_structure()
        matrix = csr_matrix((native.get_data().copy(), cols, rowp), shape=(n, m))
        jacobian = matrix[self.rows][:, self.states].tocsr()
        jacobian.eliminate_zeros()
        active = np.asarray(jacobian.getnnz(axis=0)).ravel() != 0
        jacobian = jacobian[:, active].tocsc()
        if jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError(
                "Battery restoration requires a square trajectory Jacobian"
            )
        rhs = -matrix[self.rows][:, self.controls] @ self.mapping
        response = splu(jacobian).solve(rhs)
        error = np.max(abs(jacobian @ response - rhs)) / max(1.0, np.max(abs(rhs)))
        if not np.isfinite(error) or error > 1e-7:
            raise FloatingPointError(f"Inaccurate restoration sensitivity: {error:.3e}")
        sensitivity = np.zeros((len(self.states), self.mapping.shape[1]))
        sensitivity[active] = response
        self.equality_jacobian = (
            matrix[self.periodicity][:, self.states] @ sensitivity
            + matrix[self.periodicity][:, self.controls] @ self.mapping
        )
        lo, hi = self.finite_lower, self.finite_upper
        self.inequality_jacobian = np.vstack(
            [
                sensitivity[lo] / self.scale[lo, None],
                -sensitivity[hi] / self.scale[hi, None],
            ]
        )

    def quantity(self, normalized, name):
        self.evaluate(normalized, derivatives=name.endswith("_jacobian"))
        return getattr(self, name)

    def feasible(self):
        return (
            np.max(abs(self.equality)) < self.restorer.periodicity_tolerance
            and np.min(self.inequality) >= -1e-12
            and np.all(self.inequality + self.margins > 0)
        )

    def within_reference_gaps(self, reference, fraction=POWER_REPAIR_GAP_FRACTION):
        """Keep nonlinear repair from undoing the linear step's interior margin."""
        self.vector.copy_device_to_host()
        indices = np.r_[self.controls, self.states]
        values = self.vector.get_array()[indices]
        for bounds, sign in [(self.lower, 1), (self.upper, -1)]:
            bound = bounds[indices]
            finite = np.isfinite(bound)
            old = sign * (reference[indices[finite]] - bound[finite])
            new = sign * (values[finite] - bound[finite])
            if np.any(new + 1e-14 * np.maximum(1, abs(bound[finite])) < fraction * old):
                return False
        return True

    def stop_when_feasible(self, normalized):
        self.evaluate(normalized, derivatives=False)
        if self.feasible():
            raise _BatteryFeasible

    def restore(self, point, max_iterations=20):
        zero = np.zeros(self.mapping.shape[1])
        self.evaluate(zero, derivatives=False)
        if self.feasible():
            return True
        self.restorer.case_solves += 1
        try:
            result = minimize(
                lambda z: 0.5 * z @ z,
                np.clip(zero, self.step_lower, self.step_upper),
                jac=lambda z: z,
                bounds=list(zip(self.step_lower, self.step_upper)),
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda z: self.quantity(z, "equality"),
                        "jac": lambda z: self.quantity(z, "equality_jacobian"),
                    },
                    {
                        "type": "ineq",
                        "fun": lambda z: self.quantity(z, "inequality"),
                        "jac": lambda z: self.quantity(z, "inequality_jacobian"),
                    },
                ],
                method="SLSQP",
                options={"maxiter": max_iterations, "ftol": 1e-12},
                callback=self.stop_when_feasible,
            )
        except _BatteryFeasible:
            self.message = "Feasible battery trajectory"
        else:
            self.evaluate(result.x, derivatives=False)
            self.message = str(result.message)
        if not self.feasible():
            self.message += (
                f"; periodicity {self.equality[0]:.3e}"
                f"; minimum bound margin {np.min(self.inequality):.3e}"
            )
            return False
        self.vector.get_array()[self.restorer.constraints] = point.get_array()[
            self.restorer.constraints
        ]
        self.vector.copy_host_to_device()
        point.copy(self.vector)
        return True


class TrajectoryAdjointProjector:
    """Recover state adjoints and optionally improve periodicity multipliers."""

    def __init__(self, joint, optimizer, *, refit_prices=False):
        self.model, self.optimizer = joint.model, optimizer
        self.refit_prices = refit_prices
        self.price_refits = 0
        constraints = self.model.problem.get_constraint_indices()
        self.periodicity = np.asarray(
            self.model.get_indices(
                [
                    f"case{case.case_index}.battery_periodicity.res[:]"
                    for case in joint.cases
                ]
            )
        )
        self.design = np.asarray(joint.design_indices())
        self.rows = np.setdiff1d(constraints, self.periodicity)
        self.states = np.setdiff1d(
            self.model.problem.get_primal_indices(), joint.design_indices()
        )
        self.gradient = self.model.problem.create_vector()
        self.residual = self.model.problem.create_vector()
        self.matrix = self.model.create_matrix()
        lower, upper = (
            self.model.get_lower().get_vector(),
            self.model.get_upper().get_vector(),
        )
        lower.copy_device_to_host()
        upper.copy_device_to_host()
        self.design_widths = (
            upper.get_array()[self.design] - lower.get_array()[self.design]
        )

    def refresh_residual(self, variables, obj_scale):
        point = variables.get_solution()
        self.model.problem.update(point)
        self.model.problem.gradient(obj_scale, point, self.gradient)
        self.optimizer.compute_dual_residual(variables, self.gradient, self.residual)
        self.residual.copy_device_to_host()
        return self.residual.get_array()

    def project(self, variables, obj_scale=1.0):
        point = variables.get_solution()
        point.copy_device_to_host()
        point.get_array()[self.rows] = 0.0
        point.copy_host_to_device()
        self.refresh_residual(variables, obj_scale)
        self.model.problem.hessian(obj_scale, point, self.matrix)
        self.matrix.copy_data_device_to_host()
        n, m, _, rowp, cols = self.matrix.get_nonzero_structure()
        matrix = csr_matrix((self.matrix.get_data().copy(), cols, rowp), shape=(n, m))
        jacobian = matrix[self.rows][:, self.states].tocsr()
        jacobian.eliminate_zeros()
        active = np.asarray(jacobian.getnnz(axis=0)).ravel() != 0
        jacobian = jacobian[:, active].tocsc()
        if jacobian.shape[0] != jacobian.shape[1]:
            raise ValueError("Trial adjoints require a square trajectory Jacobian")
        self.residual.copy_device_to_host()
        rhs = -self.residual.get_array()[self.states[active]]
        factor = splu(jacobian)
        multipliers = factor.solve(rhs, trans="T")
        for _ in range(2):
            remainder = rhs - jacobian.T @ multipliers
            if np.max(abs(remainder)) < 1e-10:
                break
            multipliers += factor.solve(remainder, trans="T")
        error = np.max(abs(jacobian.T @ multipliers - rhs)) / max(1, np.max(abs(rhs)))
        if not np.isfinite(error) or error > 1e-7:
            raise FloatingPointError(f"Inaccurate trial adjoints: {error:.3e}")
        point.copy_device_to_host()
        point.get_array()[self.rows] = multipliers
        point.copy_host_to_device()
        residual = self.refresh_residual(variables, obj_scale)
        actual = np.max(abs(residual[self.states]))
        if not np.isfinite(actual) or actual > 1e-6:
            raise FloatingPointError(
                f"Trial state stationarity after adjoint solve: {actual:.3e}"
            )
        if self.refit_prices:
            self.improve_energy_prices(variables, obj_scale, factor, matrix, active)

    def improve_energy_prices(self, variables, obj_scale, factor, matrix, active):
        """Remove trust-step bias only when the original dual residual improves."""
        reduced = self.residual.get_array()[self.design].copy()
        before = np.max(abs(reduced))
        if before < 1e-4:
            return
        response = factor.solve(
            -matrix[self.periodicity][:, self.states[active]].T.toarray(), trans="T"
        )
        energy_gradient = (
            matrix[self.periodicity][:, self.design].toarray()
            + response.T @ matrix[self.rows][:, self.design]
        )
        change = np.linalg.lstsq(
            (energy_gradient * self.design_widths).T,
            -reduced * self.design_widths,
            rcond=1e-10,
        )[0]
        predicted = reduced + change @ energy_gradient
        if (
            not np.all(np.isfinite(predicted))
            or np.max(abs(predicted)) >= 0.999 * before
        ):
            return
        point = variables.get_solution()
        point.copy_device_to_host()
        previous_rows = point.get_array()[self.rows].copy()
        previous_prices = point.get_array()[self.periodicity].copy()
        point.get_array()[self.rows] += response @ change
        point.get_array()[self.periodicity] += change
        point.copy_host_to_device()
        residual = self.refresh_residual(variables, obj_scale)
        if (
            np.all(np.isfinite(residual))
            and np.max(abs(residual[self.states])) <= 1e-6
            and np.max(abs(residual[self.design])) < before
        ):
            self.price_refits += 1
            return
        point.get_array()[self.rows] = previous_rows
        point.get_array()[self.periodicity] = previous_prices
        point.copy_host_to_device()
        self.refresh_residual(variables, obj_scale)


class ProjectedEvaluator(ConsistentEvaluator):
    def __init__(
        self,
        problem,
        optimizer,
        projector,
        current_provider,
        restorer=None,
        adjoints=None,
        barrier_provider=None,
    ):
        super().__init__(problem, optimizer)
        self.projector = projector
        self.current_provider = current_provider
        self.invalid_trial = False
        self.restorer = restorer
        self.adjoints = adjoints
        self.barrier_provider = barrier_provider
        self.last_trial_error = ""
        self.trial_rejections = {}

    def note_rejection(self, kind, detail):
        self.last_trial_error = detail
        count = self.trial_rejections.get(kind, 0) + 1
        self.trial_rejections[kind] = count
        if self.adjoints is not None and self.adjoints.refit_prices and count == 1:
            print_status("TRIAL BACKTRACK", [("Reason", kind), ("Detail", detail)])

    def evaluate_gradient_from_point(self, obj_scale, variables, gradient):
        self.invalid_trial = False
        self.last_trial_error = ""
        if variables is not self.current_provider():
            point = variables.get_solution()
            point.copy_device_to_host()
            previous = point.get_array()[self.projector.primal].copy()
            base = self.current_provider().get_solution()
            base.copy_device_to_host()
            reference = base.get_array()[self.projector.primal].copy()
            try:
                self.invalid_trial = not self.projector.project(
                    variables.get_solution()
                )
                if self.restorer is not None:
                    self.invalid_trial = not self.restorer.restore(
                        variables.get_solution(),
                        max_iterations=3,
                        adjust_initial_soc=False,
                        reference=base.get_array().copy(),
                    )
                    if self.invalid_trial:
                        self.note_rejection(
                            "Battery restoration", self.restorer.last_message
                        )
                elif self.invalid_trial:
                    self.note_rejection(
                        "Trajectory bounds", "Replayed state is outside strict bounds"
                    )
                if not self.invalid_trial:
                    self.invalid_trial = not self.satisfies_gap_fraction(
                        point, reference
                    )
                    if self.invalid_trial:
                        self.note_rejection(
                            "Bound distance", "Replayed state is too close to a bound"
                        )
                if not self.invalid_trial:
                    self.invalid_trial = not self.rescale_bound_duals(
                        variables, previous
                    )
                    if self.invalid_trial:
                        self.note_rejection(
                            "Bound multipliers",
                            "Cannot recover strictly positive bound products",
                        )
                if not self.invalid_trial and self.adjoints is not None:
                    self.adjoints.project(variables, obj_scale)
            except (
                FloatingPointError,
                ValueError,
                RuntimeError,
                OverflowError,
            ) as error:
                self.note_rejection("Numerical evaluation", str(error))
                self.invalid_trial = True
            if self.invalid_trial:
                gradient.zero()
                return
        super().evaluate_gradient_from_point(obj_scale, variables, gradient)

    def satisfies_gap_fraction(self, point, reference):
        point.copy_device_to_host()
        current = point.get_array()[self.projector.primal]
        for bounds, sign in [(self.projector.lower, 1), (self.projector.upper, -1)]:
            bound = bounds.get_array()[self.projector.primal]
            finite = np.isfinite(bound)
            old_gap = sign * (reference[finite] - bound[finite])
            new_gap = sign * (current[finite] - bound[finite])
            tolerance = 1e-14 * np.maximum(1, abs(bound[finite]))
            if np.any(new_gap <= 0) or np.any(
                new_gap + tolerance < MINIMUM_GAP_FRACTION * old_gap
            ):
                return False
        return True

    def rescale_bound_duals(self, variables, previous):
        if self.barrier_provider is not None:
            return self.center_bound_duals(variables, self.barrier_provider())
        """Preserve each trial gap-times-multiplier product after replay."""
        point = variables.get_solution()
        point.copy_device_to_host()
        current = point.get_array()[self.projector.primal]
        for dual, bounds, sign in [
            (variables.get_zl(), self.projector.lower, 1),
            (variables.get_zu(), self.projector.upper, -1),
        ]:
            bound = bounds.get_array()[self.projector.primal]
            finite = np.isfinite(bound)
            old_gap = sign * (previous[finite] - bound[finite])
            new_gap = sign * (current[finite] - bound[finite])
            if np.any(old_gap <= 0) or np.any(new_gap <= 0):
                return False
            dual.copy_device_to_host()
            values = dual.get_array()
            values[finite] *= old_gap / new_gap
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                return False
            dual.copy_host_to_device()
        return True

    def center_bound_duals(self, variables, barrier):
        """Represent each finite bound dual by its log-barrier value mu / gap."""
        point = variables.get_solution()
        point.copy_device_to_host()
        indices = self.projector.primal
        current = point.get_array()[indices]
        for dual, bounds, sign in [
            (variables.get_zl(), self.projector.lower, 1),
            (variables.get_zu(), self.projector.upper, -1),
        ]:
            bound = bounds.get_array()[indices]
            finite = np.isfinite(bound)
            gaps = sign * (current[finite] - bound[finite])
            if np.any(gaps <= 0):
                return False
            dual.copy_device_to_host()
            dual.get_array()[:] = 0
            dual.get_array()[finite] = barrier / gaps
            if not np.all(np.isfinite(dual.get_array())):
                return False
            dual.copy_host_to_device()
        return True

    def evaluate_objective_and_infeasibility_from_point(
        self, mu, obj_scale, variables, gradient
    ):
        if variables is not self.current_provider() and self.invalid_trial:
            return np.inf, np.inf, np.inf
        return super().evaluate_objective_and_infeasibility_from_point(
            mu, obj_scale, variables, gradient
        )


class _PrimalMultiplierStep:
    """Keep equality multipliers on the repaired primal step's line parameter."""

    def __init__(self, backend, problem):
        self.backend = backend
        self.constraints = np.asarray(problem.get_constraint_indices())

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def apply_step_update(self, alpha_primal, alpha_dual, current, step, trial):
        self.backend.apply_step_update(alpha_primal, alpha_dual, current, step, trial)
        point, base, direction = (
            trial.get_solution(),
            current.get_solution(),
            step.get_solution(),
        )
        for vector in [point, base, direction]:
            vector.copy_device_to_host()
        point.get_array()[self.constraints] = (
            base.get_array()[self.constraints]
            + alpha_primal * direction.get_array()[self.constraints]
        )
        point.copy_host_to_device()


class ProjectedFilterLineSearch(FilterLineSearch):
    """Use AMIGO's filter while rejecting every invalid projected trial."""

    def __init__(self, options, problem, optimizer, centered_duals=False):
        super().__init__(options, problem, optimizer)
        self.centered_duals = centered_duals
        self.saved = optimizer.create_opt_vector()
        self.optimizer = _PrimalMultiplierStep(optimizer, problem)
        self.accepted_trial = False

    def check_acceptance(self, base, trial_barr, trial_theta, alpha_test):
        self.accepted_trial = super().check_acceptance(
            base, trial_barr, trial_theta, alpha_test
        )
        return self.accepted_trial

    def compute_alpha_min(self, base):
        return min(super().compute_alpha_min(base), 1e-12)

    def line_search(self, solver, evaluator, state):
        self.saved.copy(state.current)
        self.accepted_trial = False
        if self.centered_duals:
            # Trial bound duals are recomputed from mu / gap; their unused
            # Newton extrapolation must not shorten the primal step.
            state.max_alpha_dual = state.max_alpha_primal
        fraction = min(1.0, (1 - MINIMUM_GAP_FRACTION) / state.tau)
        state.max_alpha_primal *= fraction
        state.max_alpha_dual *= fraction
        info = super().line_search(solver, evaluator, state)
        if not info.success or evaluator.invalid_trial or not self.accepted_trial:
            # AMIGO permits its minimum step even after filter rejection.
            # Preserve the last accepted point when that fallback is reached.
            state.current.copy(self.saved)
            state.invalidate()
            self.problem.update(state.current.get_solution())
            reason = getattr(evaluator, "last_trial_error", "")
            raise RuntimeError(
                f"No bounded trajectory trial passed the filter. {reason}"
            )
        return info
