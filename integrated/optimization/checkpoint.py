"""Checkpoint loading and continuation callbacks for CADRE optimization."""

import json
import shutil
import time

from amigo.algorithm.multiplier_initialization import MultiplierInitializer
import numpy as np

from integrated.reporting.console import print_heading, print_status
from integrated.solvers.evaluator import ConsistentEvaluator


def project_into_bounds(values, lower, upper, push):
    """Project finite bounds using the same relative margin on cold and warm starts."""
    width = upper - lower
    lower_target = np.full(values.shape, -np.inf)
    upper_target = np.full(values.shape, np.inf)
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    lower_target[finite_lower] = lower[finite_lower] + push * np.minimum(
        np.maximum(1, np.abs(lower[finite_lower])), width[finite_lower]
    )
    upper_target[finite_upper] = upper[finite_upper] - push * np.minimum(
        np.maximum(1, np.abs(upper[finite_upper])), width[finite_upper]
    )
    return np.clip(values, lower_target, upper_target)


def load_checkpoint(optimizer, args, checkpoint_signature):
    """Restore a compatible point and retain saved barrier/bound multipliers."""
    warm_bound_duals = None
    warm_same_problem = False
    checkpoint_signature = {"case_index": 0, **checkpoint_signature}

    if args.warm_start:
        with np.load(args.warm_start, allow_pickle=False) as saved:
            # Older checkpoints were produced only for case 0.
            saved_signature = {"case_index": 0, **json.loads(str(saved["signature"]))}
            warm_same_problem = saved_signature == checkpoint_signature
            differences = {
                key
                for key in checkpoint_signature
                if saved_signature.get(key) != checkpoint_signature[key]
            }

            releasing = (
                args.release_design_controls
                and checkpoint_signature["design_physics"]
                and saved_signature.get("design_physics")
                and differences <= {"optimize_roll", "optimize_allocation"}
                and all(
                    checkpoint_signature[key] and not saved_signature[key]
                    for key in differences
                )
            )

            if saved_signature != checkpoint_signature and not releasing:
                raise ValueError("Warm-start model options do not match this run")

            initial = optimizer.x_init

            if saved["solution"].shape != initial.get_array().shape:
                raise ValueError("Warm-start vector shape does not match this model")
            if not np.all(np.isfinite(saved["solution"])):
                raise ValueError("Warm-start solution contains nonfinite values")

            initial.get_array()[:] = saved["solution"]
            initial.copy_host_to_device()

            if warm_same_problem and "zl" in saved and "zu" in saved:
                warm_bound_duals = (saved["zl"].copy(), saved["zu"].copy())
                if any(
                    not np.all(np.isfinite(values)) or np.any(values < 0)
                    for values in warm_bound_duals
                ):
                    raise ValueError(
                        "Warm-start bound multipliers must be finite and nonnegative"
                    )

            if args.initial_barrier is None and "barrier_param" in saved:
                args.initial_barrier = float(saved["barrier_param"])
                if not np.isfinite(args.initial_barrier) or args.initial_barrier <= 0:
                    raise ValueError(
                        "Warm-start barrier parameter must be positive and finite"
                    )

        print_heading("WARM-START CHECKPOINT")
        print(f"  {args.warm_start}\n", flush=True)

    return warm_same_problem, warm_bound_duals


class CheckpointManager:
    """Restore the first iterate and atomically save latest/best checkpoints."""

    def __init__(
        self,
        model,
        optimizer,
        args,
        signature,
        same_problem,
        bound_duals,
        design_physics,
    ):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.signature = signature
        self.warm_same_problem = same_problem
        self.warm_bound_duals = bound_duals
        self.design_physics = design_physics

        self.start_time = time.perf_counter()
        self.best_error = np.inf
        self.initialization_adjusted = False

        args.output.mkdir(parents=True, exist_ok=True)

    def __call__(self, state):
        """Run after initialization and each continuing/final solver iteration."""
        model, optimizer, args = self.model, self.optimizer, self.args
        checkpoint_signature = self.signature
        warm_same_problem = self.warm_same_problem
        warm_bound_duals = self.warm_bound_duals
        design_physics = self.design_physics

        if not isinstance(optimizer.evaluator, ConsistentEvaluator):
            optimizer.evaluator = ConsistentEvaluator(
                model.problem, optimizer.optimizer
            )

        if (
            (args.center_bound_multipliers or args.warm_start)
            and state.iter == 0
            and not self.initialization_adjusted
        ):
            self.initialization_adjusted = True
            indices = model.problem.get_primal_indices()
            point = state.current.get_solution()

            if args.warm_start:
                optimizer.x_init.copy_device_to_host()

                point.copy_device_to_host()

                point.get_array()[:] = optimizer.x_init.get_array()

                optimizer.lower.copy_device_to_host()

                optimizer.upper.copy_device_to_host()

                lower = optimizer.lower.get_array()[indices]

                upper = optimizer.upper.get_array()[indices]

                point.get_array()[indices] = project_into_bounds(
                    point.get_array()[indices], lower, upper, args.warm_start_push
                )
                point.copy_host_to_device()

            point.copy_device_to_host()

            values = point.get_array()[indices]

            preserve_multipliers = bool(
                args.warm_start and not args.center_bound_multipliers
            )
            recover_shadows = preserve_multipliers and not warm_same_problem

            if recover_shadows:
                # The checkpoint contains constraint multipliers. Their
                # Lagrangian gradient recovers positive active-bound shadows
                # for the old controls and estimates them for released controls.
                state.invalidate()

                optimizer.evaluator.evaluate_gradient(state)

                state.gradient.copy_device_to_host()

                lagrangian_gradient = state.gradient.get_array()[indices].copy()

            for side, (dual, bounds, sign) in enumerate(
                (
                    (state.current.get_zl(), optimizer.lower, 1),
                    (state.current.get_zu(), optimizer.upper, -1),
                )
            ):
                bounds.copy_device_to_host()

                bound_values = bounds.get_array()[indices]
                finite = np.isfinite(bound_values)
                gaps = sign * (values - bound_values)

                if np.any(gaps[finite] <= 0):
                    raise ValueError(
                        "Cannot center bound multipliers outside strict bounds"
                    )

                dual.copy_device_to_host()

                dual.get_array()[:] = 0
                dual.get_array()[finite] = state.mu / gaps[finite]

                if preserve_multipliers and warm_bound_duals is not None:
                    if warm_bound_duals[side].shape != dual.get_array().shape:
                        raise ValueError(
                            "Warm-start bound-multiplier shape does not match"
                        )

                    dual.get_array()[:] = warm_bound_duals[side]
                elif recover_shadows:
                    width = (
                        optimizer.upper.get_array()[indices]
                        - optimizer.lower.get_array()[indices]
                    )
                    near = finite & (
                        gaps
                        <= 0.02
                        * np.where(
                            np.isfinite(width),
                            width,
                            np.maximum(1, np.abs(bound_values)),
                        )
                    )
                    dual.get_array()[near] = np.maximum(
                        dual.get_array()[near], sign * lagrangian_gradient[near]
                    )

                dual.copy_host_to_device()

            state.invalidate()

            if not preserve_multipliers:
                MultiplierInitializer(
                    optimizer.run_options, model, model.problem, optimizer.optimizer
                ).initialize_multipliers(optimizer.evaluator, optimizer.solver, state)

            optimizer.evaluator.evaluate_objective_and_infeasibility(state)
            optimizer.evaluator.evaluate_residual(state)

            print_status(
                (
                    "WARM-START MULTIPLIERS"
                    if preserve_multipliers
                    else "CENTERED BOUND MULTIPLIERS"
                ),
                [
                    ("Complementarity", f"{state.complementarity:.3e}"),
                    ("Primal residual", f"{state.primal_infeas:.3e}"),
                    ("Dual residual", f"{state.dual_infeas:.3e}"),
                ],
            )

        # Repeated least-squares projection is useful early on. Near a
        # stationary feasible point let Newton refine the multipliers.

        if (
            design_physics
            and optimizer.run_options["recompute_multipliers"]
            and state.primal_infeas < 1e-6
            and state.dual_infeas < 1e-4
        ):
            optimizer.run_options["recompute_multipliers"] = False

            print_status(
                "NEAR FEASIBILITY",
                [("Multiplier updates", "Retaining Newton updates")],
            )

        solution = state.current.get_solution()
        solution.copy_device_to_host()

        temporary = args.output / "checkpoint.tmp.npz"

        lower_duals, upper_duals = state.current.get_zl(), state.current.get_zu()
        lower_duals.copy_device_to_host()

        upper_duals.copy_device_to_host()

        solver = optimizer.solver
        diagnostics = {
            name: float(getattr(solver, name))
            for name in (
                "curvature_shift",
                "step_shift",
                "regularization_max",
                "max_linear_residual",
                "largest_design_step_ratio",
                "box_qp_iterations",
                "box_qp_iteration_budget",
                "box_qp_fallbacks",
            )
            if hasattr(solver, name)
        }

        np.savez_compressed(
            temporary,
            solution=solution.get_array(),
            zl=lower_duals.get_array(),
            zu=upper_duals.get_array(),
            signature=json.dumps(checkpoint_signature),
            iteration=state.iter,
            barrier_param=state.mu,
            elapsed_s=time.perf_counter() - self.start_time,
            primal_infeasibility=state.primal_infeas,
            dual_infeasibility=state.dual_infeas,
            nlp_error=state.kkt_error,
            complementarity=state.complementarity,
            linear_solver_diagnostics=json.dumps(diagnostics),
        )

        temporary.replace(args.output / "checkpoint.npz")

        if state.kkt_error < self.best_error:
            self.best_error = state.kkt_error
            temporary_best = args.output / "best_checkpoint.tmp.npz"
            shutil.copyfile(args.output / "checkpoint.npz", temporary_best)

            temporary_best.replace(args.output / "best_checkpoint.npz")
