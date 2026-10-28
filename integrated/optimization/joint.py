"""Run and resume a joint CADRE optimization with shared design variables."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from cadre_paths import source_provenance
from integrated.model.joint import build_directory
from integrated.optimization.checkpoint import (
    CheckpointManager,
    load_checkpoint,
    project_into_bounds,
)
from integrated.optimization.driver import CadreOptimizer
from integrated.solvers.block_solver import SparseLUSolver
from integrated.reporting.console import print_heading, print_convergence, print_values
from integrated.solvers.joint import BoxStepBounds, JointKKTSolver, joint_partitions
from integrated.optimization.projection import (
    TrajectoryProjector,
    ProjectedEvaluator,
    ProjectedFilterLineSearch,
    BatteryRestorer,
    TrajectoryAdjointProjector,
)
from integrated.solvers.native_ldl import (
    NativeLDLSolver,
    primal_regularization_metric,
    temporal_kkt_ordering,
)


def model_signature(joint, time_grid):
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    # Include callback and initialization code as well as compiled components.
    files = [root / "cadre_paths.py", root / "cadre_source.json"]
    for folder in [
        "integrated/model",
        "integrated/reference",
        "attitude",
        "battery",
        "communication",
        "parameters",
        "power",
        "reaction_wheel",
        "thermal",
    ]:
        files.extend(sorted((root / folder).glob("*.py")))
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "formulation": "joint-cadre-v1",
        "cases": [case.case_index for case in joint.cases],
        "nodes": joint.cases[0].n,
        "controls": joint.cases[0].m,
        "objective_scale": joint.objective_scale,
        "time_grid": time_grid,
        "model_source_sha256": digest.hexdigest(),
        "amigo_version": version("amigo"),
        "cadre_source": source_provenance(),
    }


def initialize_joint(joint, args):
    signature = model_signature(joint, args.time_grid)
    build = args.output / "build"
    marker = build / "model_signature.json"
    if args.reuse_build:
        if not marker.is_file() or json.loads(marker.read_text()) != signature:
            raise ValueError(
                "Build signature does not match this model; omit --reuse-build"
            )
        with build_directory(build):
            import amigo as am

            joint.model._built = True
            joint.model.initialize(order_type=am.OrderingType.NATURAL)
    else:
        joint.initialize(build)
        marker.write_text(json.dumps(signature, indent=2) + "\n")
    return signature


def make_joint_solver(joint, optimizer, args, point_provider):
    """Configure factorization and step limits for the selected optimization path."""
    model = joint.model
    primal = np.asarray(model.problem.get_primal_indices())
    constraints = model.problem.get_constraint_indices()
    if args.linear_solver == "native":
        return NativeLDLSolver(
            hessian_correction=joint.hessian_correction,
            permutation=temporal_kkt_ordering(model, joint.cases[0].n),
            regularization_metric=primal_regularization_metric(
                model, optimizer.x_init.get_array()
            ),
            diagonal_provider=lambda: optimizer.state.diagonal,
        )
    partitions, shared = joint_partitions(joint)
    if args.linear_solver == "sparse":
        return SparseLUSolver(
            hessian_correction=joint.hessian_correction,
            shared_indices=np.r_[partitions[0].controls, shared],
            safeguard_shared=True,
            primal_indices=primal,
            constraint_indices=constraints,
        )
    roll = model.get_indices(
        [f"case{case.case_index}.control_points.CP_gamma[:]" for case in joint.cases]
    )
    initial_soc = model.get_indices(
        [f"case{case.case_index}.battery_ic.initial_soc[:]" for case in joint.cases]
    )
    limited = (
        np.concatenate([part.controls for part in partitions] + [shared])
        if args.primal_barrier
        else np.r_[roll, initial_soc, shared]
    )
    lower, upper = model.get_lower().get_vector(), model.get_upper().get_vector()
    lower.copy_device_to_host()
    upper.copy_device_to_host()
    lo, hi = lower.get_array(), upper.get_array()
    limits = 0.1 * (hi[limited] - lo[limited])
    bounds = None
    if args.primal_barrier:
        limits[np.isin(limited, roll)] = np.minimum(
            limits[np.isin(limited, roll)], 0.05
        )
        bounded = primal[np.isfinite(lo[primal]) | np.isfinite(hi[primal])]
        max_step = 0.1 * (hi[bounded] - lo[bounded])
        max_step[np.isin(bounded, roll)] = np.minimum(
            max_step[np.isin(bounded, roll)], 0.05
        )
        bounds = BoxStepBounds(
            bounded, lo[bounded], hi[bounded], max_step, point_provider
        )
    solver = JointKKTSolver(
        partitions,
        shared,
        primal,
        constraints,
        joint.hessian_correction,
        limited_indices=limited,
        step_limits=limits,
        case_workers=args.case_workers,
        selective_curvature=args.primal_barrier,
        spectral_curvature=args.spectral_curvature,
        box_qp=args.box_qp,
        step_bounds=bounds,
    )
    solver.design_step_fraction = 0.1
    return solver


def optimize_joint(joint, args, signature):
    point = joint.model.create_vector()
    optimizer = CadreOptimizer(joint.model, point)
    same_problem, bound_duals = load_checkpoint(optimizer, args, signature)
    if args.initial_design:
        load_initial_design(
            joint, optimizer.x_init, args.initial_design, refit=args.refit_controls
        )
        initial_replay = TrajectoryProjector(joint)
        initial_replay.project(optimizer.x_init)
        optimizer.x_init.copy_device_to_host()
        if not np.all(np.isfinite(optimizer.x_init.get_array())):
            raise ValueError("Initial design produced a nonfinite physical trajectory")
    start_projector = TrajectoryProjector(joint) if args.repair_start else None
    start_restorer = (
        BatteryRestorer(
            start_projector,
            interior=True,
            periodicity_tolerance=1e-12 if args.primal_barrier else 1e-9,
        )
        if args.repair_start
        else None
    )

    def current_point():
        vector = optimizer.state.current.get_solution()
        vector.copy_device_to_host()
        return vector.get_array()

    joint.set_point_provider(current_point)
    solver = make_joint_solver(joint, optimizer, args, current_point)

    checkpoint = CheckpointManager(
        joint.model, optimizer, args, signature, same_problem, bound_duals, True
    )
    initialized = False
    projector = None if args.full_space_trials else TrajectoryProjector(joint)
    restorer = (
        BatteryRestorer(
            projector,
            periodicity_tolerance=1e-12 if args.primal_barrier else 1e-9,
            full_power=args.box_qp,
        )
        if args.restore_periodicity
        else None
    )
    trial_adjoints = (
        TrajectoryAdjointProjector(joint, optimizer.optimizer, refit_prices=args.box_qp)
        if projector is not None
        else None
    )
    if args.warm_start and restorer is not None:
        optimizer.x_init.copy_device_to_host()
        original = optimizer.x_init.get_array().copy()
        projector.project(optimizer.x_init)
        if not restorer.restore(optimizer.x_init, adjust_initial_soc=False):
            raise ValueError(
                f"SOC periodicity restoration failed: {restorer.last_message}"
            )
        optimizer.x_init.copy_device_to_host()
        if not args.primal_barrier and not np.allclose(
            original, optimizer.x_init.get_array(), atol=1e-12, rtol=0
        ):
            args.center_bound_multipliers = args.reset_multipliers = True

    def continuation(state):
        nonlocal initialized
        if projector is not None and not isinstance(
            optimizer.evaluator, ProjectedEvaluator
        ):
            optimizer.evaluator = ProjectedEvaluator(
                joint.model.problem,
                optimizer.optimizer,
                projector,
                lambda: optimizer.state.current,
                restorer=restorer,
                adjoints=trial_adjoints,
                barrier_provider=(
                    (lambda: optimizer.state.mu) if args.primal_barrier else None
                ),
            )
        if state.iter == 0 and not initialized and not args.warm_start:
            point_vector = state.current.get_solution()
            optimizer.x_init.copy_device_to_host()
            point_vector.copy_device_to_host()
            point_vector.get_array()[:] = optimizer.x_init.get_array()
            optimizer.lower.copy_device_to_host()
            optimizer.upper.copy_device_to_host()
            primal = joint.model.problem.get_primal_indices()
            point_vector.get_array()[primal] = project_into_bounds(
                point_vector.get_array()[primal],
                optimizer.lower.get_array()[primal],
                optimizer.upper.get_array()[primal],
                args.initial_bound_push,
            )
            point_vector.copy_host_to_device()
            if projector is None and args.initial_design:
                # Named controls do not come with compatible trajectory states.
                # Recompute those states after the control-bound push. Full-space
                # optimization permits equality defects where a bound needs clipping.
                initial_replay.project(point_vector)
                point_vector.copy_device_to_host()
                if not np.all(np.isfinite(point_vector.get_array())):
                    raise FloatingPointError(
                        "Initial design replay produced nonfinite states"
                    )
                point_vector.get_array()[primal] = project_into_bounds(
                    point_vector.get_array()[primal],
                    optimizer.lower.get_array()[primal],
                    optimizer.upper.get_array()[primal],
                    args.initial_bound_push,
                )
                point_vector.copy_host_to_device()
            if start_restorer is not None:
                start_projector.project(point_vector)
                print_heading("INITIAL BATTERY REPAIR")
                if not start_restorer.restore(
                    point_vector,
                    max_iterations=80,
                    verbose=True,
                    adjust_initial_soc=True,
                ):
                    raise ValueError(
                        f"Initial battery repair failed: {start_restorer.last_message}"
                    )
            if projector is not None:
                bounded = projector.project(point_vector)
                if restorer is not None:
                    if not restorer.restore(point_vector, adjust_initial_soc=False):
                        raise ValueError(
                            f"SOC periodicity restoration failed: {restorer.last_message}"
                        )
                elif not bounded:
                    raise ValueError(
                        "Starting trajectory violates model bounds after replay"
                    )
            state.invalidate()
        checkpoint(state)
        if state.iter == 0 and not initialized:
            if not args.warm_start or args.reset_multipliers:
                initialize_adjoints(joint, optimizer, state)
            elif trial_adjoints is not None:
                trial_adjoints.project(state.current, state.obj_scale)
                state.invalidate()
                optimizer.evaluator.evaluate_objective_and_infeasibility(state)
                optimizer.evaluator.evaluate_residual(state)
                print_heading("RESTORED TRAJECTORY MULTIPLIERS")
                print_values(
                    [
                        ("Primal residual", f"{state.primal_infeas:.3e}"),
                        ("Dual residual", f"{state.dual_infeas:.3e}"),
                    ]
                )
            initialized = True
            checkpoint.best_error = np.inf
            checkpoint(state)

    barrier = (
        args.initial_barrier
        if args.initial_barrier is not None
        else 0.01 * joint.objective_scale
    )
    options = {
        "solver": solver,
        "max_iterations": args.max_iterations,
        "convergence_tolerance": 1e-8,
        "initial_barrier_param": barrier,
        "barrier_strategy": "monotone",
        "max_line_search_iterations": 40,
        "recompute_multipliers": False,
        "init_least_squares_multipliers": False,
        "continuation_control": continuation,
    }
    if projector is not None:
        normalized = optimizer.get_options(options)
        options["line_search"] = ProjectedFilterLineSearch(
            normalized,
            joint.model.problem,
            optimizer.optimizer,
            centered_duals=args.primal_barrier,
        )
    print_heading("RUNNING JOINT CADRE OPTIMIZATION")
    print_values(
        [
            ("Cases", len(joint.cases)),
            ("Design variables", joint.design_variable_count),
            ("Objective scale", joint.objective_scale),
            ("Iteration objective", "Includes bound-barrier terms"),
        ]
    )
    history = optimizer.optimize(options)
    solver.initial_battery_restorations = (
        start_restorer.case_solves if start_restorer is not None else 0
    )
    solver.trajectory_projections = (
        projector.evaluations if projector is not None else 0
    )
    solver.battery_restorations = restorer.calls if restorer is not None else 0
    solver.full_power_restorations = (
        restorer.full_power_solves if restorer is not None else 0
    )
    solver.price_refits = (
        trial_adjoints.price_refits if trial_adjoints is not None else 0
    )
    solver.trial_rejections = getattr(optimizer.evaluator, "trial_rejections", {})
    solver.restoration_parameters_per_case = 1 if restorer is not None else 0
    checkpoint(optimizer.state)
    print_convergence(history)
    return point, history, solver


def refit_control_profile(controls, count, upper, *, solar=False):
    """Fit bounded spline controls over the same physical time window."""
    from parameters.parameters_reference import build_bspline_matrix

    controls = np.asarray(controls)
    if (
        controls.ndim not in (1, 2)
        or controls.shape[0] < 4
        or (controls.ndim == 2 and controls.shape[1] != 12)
        or not np.all(np.isfinite(controls))
        or np.any(controls < 0)
        or np.any(controls > upper)
    ):
        raise ValueError("Cannot refit invalid or out-of-bounds source controls")
    if count == len(controls):
        return controls.copy()
    nodes = max(600, 3 * max(count, len(controls)))
    source = build_bspline_matrix(nodes, len(controls))
    target = build_bspline_matrix(nodes, count)
    profiles = (source @ controls).reshape(nodes, -1)
    fitted = []
    for profile in profiles.T:
        # Current leakage during eclipse can consume substantial battery power.
        weights = np.where(profile < 1e-5, 30.0, 1.0) if solar else np.ones(nodes)
        result = lsq_linear(
            target * weights[:, None], profile * weights, bounds=(0, upper)
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(f"Spline control fit failed: {result.message}")
        fitted.append(result.x)
    return np.column_stack(fitted).reshape((count,) + controls.shape[1:])


def load_initial_design(joint, point, directory, *, refit=False):
    """Import controls only; states and all multipliers are initialized afresh."""
    summary = json.loads((directory / "summary.json").read_text())
    source, expected = summary["cadre_source"], source_provenance()
    if any(
        source[key] != expected[key]
        for key in ["repository", "reference_revision", "benchmark_sha256"]
    ):
        raise ValueError("Initial design uses a different CADRE reference")
    if refit and (
        not np.isfinite(summary.get("duration_hours", np.nan))
        or not np.isclose(
            summary["duration_hours"] * 3600,
            joint.cases[0].window_duration,
            rtol=0,
            atol=1e-6,
        )
    ):
        raise ValueError("Control refitting requires matching physical time windows")
    point.copy_device_to_host()
    values = point.get_array()
    arrays = directory / "trajectories.npz"
    if not arrays.is_file():
        arrays = directory / "controls.npz"
    with np.load(arrays, allow_pickle=False) as saved:
        assignments = {
            "shared_surface.cellInstd[0, :]": saved["surface_allocation"].ravel(),
            "shared_angles.finAngle[:]": [summary["fin_angle_rad"]],
            "shared_angles.antAngle[:]": [summary["antenna_angle_rad"]],
        }
        for case in joint.cases:
            for name, suffix, upper in [
                ("CP_P_comm", "[:]", 25),
                ("CP_gamma", "[:]", np.pi / 2),
                ("CP_Isetpt", "[:, :]", 0.4),
            ]:
                expression = f"case{case.case_index}.control_points.{name}{suffix}"
                controls = saved[f"case{case.case_index}_{name}"]
                assignments[expression] = (
                    refit_control_profile(
                        controls, case.m, upper, solar=name == "CP_Isetpt"
                    )
                    if refit
                    else controls
                )
            assignments[f"case{case.case_index}.battery_ic.initial_soc[:]"] = [
                saved[f"case{case.case_index}_SOC"][0]
            ]
        for name, array in assignments.items():
            indices = joint.model.get_indices(name)
            array = np.asarray(array)
            if array.shape != indices.shape or not np.all(np.isfinite(array)):
                raise ValueError(f"Initial design has invalid controls: {name}")
            values[indices] = array.reshape(indices.shape)
    lower, upper = (
        joint.model.get_lower().get_vector(),
        joint.model.get_upper().get_vector(),
    )
    lower.copy_device_to_host()
    upper.copy_device_to_host()
    design = joint.design_indices()
    if np.any(values[design] < lower.get_array()[design]) or np.any(
        values[design] > upper.get_array()[design]
    ):
        raise ValueError("Initial design contains controls outside model bounds")
    point.copy_host_to_device()


def initialize_adjoints(joint, optimizer, state):
    """Seed equality multipliers through the causal trajectory Jacobian.

    A square trajectory solve determines the state adjoints. A six-column
    least-squares fit then seeds the periodicity multipliers to reduce the
    design gradient without disturbing state stationarity.
    """
    model = joint.model
    constraints = model.problem.get_constraint_indices()
    periodicity = model.get_indices(
        [f"case{case.case_index}.battery_periodicity.res[:]" for case in joint.cases]
    )
    rows = np.setdiff1d(constraints, periodicity)
    primal = np.setdiff1d(model.problem.get_primal_indices(), joint.design_indices())
    point = state.current.get_solution()
    point.fill_at(constraints, 0.0)
    state.invalidate()
    evaluator = optimizer.evaluator
    evaluator.evaluate_gradient(state)
    evaluator.evaluate_hessian(state)
    state.hessian.copy_data_device_to_host()
    n, m, _, rowp, cols = state.hessian.get_nonzero_structure()
    matrix = csr_matrix((state.hessian.get_data().copy(), cols, rowp), shape=(n, m))
    jacobian = matrix[rows][:, primal].tocsr()
    jacobian.eliminate_zeros()
    active = np.asarray(jacobian.getnnz(axis=0)).ravel() != 0
    jacobian = jacobian[:, active].tocsc()
    if jacobian.shape[0] != jacobian.shape[1]:
        raise ValueError(
            f"Adjoint initialization requires a square trajectory Jacobian; got {jacobian.shape}"
        )
    optimizer.optimizer.compute_dual_residual(
        state.current, state.gradient, state.residual
    )
    state.residual.copy_device_to_host()
    rhs = -state.residual.get_array()[primal[active]]
    factor = splu(jacobian)
    multipliers = factor.solve(rhs, trans="T")
    error = np.max(np.abs(jacobian.T @ multipliers - rhs)) / max(1, np.max(np.abs(rhs)))
    if not np.isfinite(error) or error > 1e-6:
        raise FloatingPointError(f"Inaccurate trajectory adjoint seed: {error:.3e}")
    design = joint.design_indices()
    state_indices = primal[active]
    response = factor.solve(
        -matrix[periodicity][:, state_indices].T.toarray(), trans="T"
    )
    energy_gradient = (
        matrix[periodicity][:, design].toarray() + response.T @ matrix[rows][:, design]
    )
    reduced = state.residual.get_array()[design] + matrix[design][:, rows] @ multipliers
    optimizer.lower.copy_device_to_host()
    optimizer.upper.copy_device_to_host()
    widths = optimizer.upper.get_array()[design] - optimizer.lower.get_array()[design]
    weighted = energy_gradient * widths
    prices = np.linalg.lstsq(weighted.T, -reduced * widths, rcond=1e-10)[0]
    before = float(np.linalg.norm(reduced * widths))
    after = float(np.linalg.norm((reduced + prices @ energy_gradient) * widths))
    if not np.isfinite(after) or after > before * (1 + 1e-10) + 1e-10:
        raise FloatingPointError(
            "Periodicity adjoint seed increased the design residual"
        )
    multipliers += response @ prices
    equation_error = (
        jacobian.T @ multipliers
        + matrix[periodicity][:, state_indices].T @ prices
        - rhs
    )
    error = float(np.max(abs(equation_error)) / max(1, np.max(abs(rhs))))
    if not np.isfinite(error) or error > 1e-6:
        raise FloatingPointError(f"Inaccurate periodicity adjoint seed: {error:.3e}")
    point.copy_device_to_host()
    point.get_array()[rows] = multipliers
    point.get_array()[periodicity] = prices
    point.copy_host_to_device()
    state.invalidate()
    evaluator.evaluate_objective_and_infeasibility(state)
    evaluator.evaluate_residual(state)
    print_heading("TRAJECTORY ADJOINT INITIALIZATION")
    print_values(
        [
            ("Linear residual", f"{error:.3e}"),
            ("Primal infeasibility", f"{state.primal_infeas:.3e}"),
            ("Dual infeasibility", f"{state.dual_infeas:.3e}"),
            ("Scaled design residual before", f"{before:.3e}"),
            ("Scaled design residual after", f"{after:.3e}"),
        ]
    )
    return {
        "adjoint_equation_error": error,
        "design_residual_before": before,
        "design_residual_after": after,
    }
