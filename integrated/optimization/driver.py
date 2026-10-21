"""Solver selection, checkpoint continuation, and optimization execution."""

from dataclasses import dataclass

import amigo as am
import numpy as np

from integrated.optimization.checkpoint import CheckpointManager, load_checkpoint
from integrated.reporting.console import print_convergence, print_heading, print_status
from integrated.solvers.block_solver import SparseLUSolver
from integrated.solvers.native_ldl import (
    NativeLDLSolver,
    primal_regularization_metric,
    temporal_kkt_ordering,
)


class CadreOptimizer(am.Optimizer):
    """Retain normalized options for the design-stage continuation callback."""

    def get_options(self, options=None):
        self.run_options = super().get_options(options or {})

        return self.run_options


@dataclass
class CadreOptimizationResult:
    """Optimizer output and the settings needed to report its provenance."""

    point: object
    history: dict
    initial_barrier_param: float


def optimize_case(model, case, args):
    """Optimize an initialized model, preserving checkpoint restart behavior."""
    point = model.create_vector()
    optimizer = CadreOptimizer(model, point)

    checkpoint_signature = _checkpoint_signature(case, args)

    warm_same_problem, warm_bound_duals = load_checkpoint(
        optimizer, args, checkpoint_signature
    )

    shared_indices, normal_shared_indices = _shared_control_indices(
        model, optimizer, case, args
    )

    solver = _create_linear_solver(
        model, optimizer, case, args, shared_indices, normal_shared_indices
    )

    initial_barrier = args.initial_barrier

    if initial_barrier is None:
        initial_barrier = 0.01 if case.coupled_voltage else 0.1

    options = {
        "solver": solver,
        "max_iterations": args.max_iterations,
        "convergence_tolerance": 1e-8,
        "initial_barrier_param": initial_barrier,
        "barrier_strategy": "monotone",
        "max_line_search_iterations": 40,
        "recompute_multipliers": not args.optimize_roll,
        "init_least_squares_multipliers": not (
            args.warm_start and not args.center_bound_multipliers
        ),
    }

    checkpoint = CheckpointManager(
        model,
        optimizer,
        args,
        checkpoint_signature,
        warm_same_problem,
        warm_bound_duals,
        case.design_physics,
    )

    options["continuation_control"] = checkpoint

    print_heading("RUNNING P_COMM OPTIMIZATION")

    opt_data = optimizer.optimize(options)

    checkpoint(optimizer.state)

    print_convergence(opt_data)

    return CadreOptimizationResult(point, opt_data, initial_barrier)


def _checkpoint_signature(case, args):
    """Record the model settings that must match a warm-start checkpoint."""
    return {
        "case_index": case.case_index,
        "nodes": case.n,
        "control_points": case.m,
        "fixed_initial_soc": args.fixed_initial_soc,
        "fixed_solar_currents": args.fixed_solar_currents,
        "legacy_fixed_voltage": args.legacy_fixed_voltage,
        "design_physics": case.design_physics,
        "optimize_roll": args.optimize_roll,
        "optimize_allocation": args.optimize_allocation,
    }


def _shared_control_indices(model, optimizer, case, args):
    """Choose the shared variables for Newton and multiplier-initialization solves."""

    if not case.design_physics:
        return None, None

    case.design_external.point_provider = (
        lambda: optimizer.state.current.get_solution().get_array()
    )

    shared_indices = model.get_indices("surface.cellInstd[0, :]")

    if not args.optimize_allocation:
        case.design_external.fixed_global_indices = shared_indices.copy()

    if args.optimize_roll:
        shared_indices = np.concatenate(
            (shared_indices, model.get_indices("control_points.CP_gamma[:]"))
        )

    normal_shared_indices = shared_indices.copy()

    if args.reduce_all_controls:
        shared_indices = _reduced_control_indices(model, case, args)

    return shared_indices, normal_shared_indices


def _reduced_control_indices(model, case, args):
    """Keep one energy-balance control local when eliminating design controls."""
    expressions = ["control_points.CP_P_comm[:]"]

    if case.free_solar:
        expressions.append("control_points.CP_Isetpt[:, :]")

    if args.optimize_roll:
        expressions.append("control_points.CP_gamma[:]")

    if args.optimize_allocation:
        expressions.append("surface.cellInstd[0, :]")

    if not args.fixed_initial_soc:
        expressions.append("battery_ic.initial_soc[0]")

    shared_indices = model.get_indices(expressions)

    # A nearly energy-neutral cycle is almost independent of initial SOC.
    # Use a communication-power control for periodicity instead of inverting
    # that weak SOC sensitivity.
    pivot_control = int(np.argmax(case.B.T @ case.CommLOS))
    energy_index = model.get_indices("control_points.CP_P_comm[:]")[pivot_control]

    shared_indices = shared_indices[shared_indices != energy_index]

    print_status(
        "CONTROL REDUCTION",
        [
            ("Local energy-balance control", f"{pivot_control:,}"),
            ("Reduced control system size", f"{len(shared_indices):,}"),
        ],
    )

    return shared_indices


def _create_linear_solver(
    model, optimizer, case, args, shared_indices, normal_shared_indices
):
    """Configure the selected backend with external Hessians and index partitions."""
    hessian_correction = (
        case.design_external.hessian_correction if case.design_physics else None
    )

    if args.linear_solver == "native":
        return NativeLDLSolver(
            hessian_correction=hessian_correction,
            permutation=temporal_kkt_ordering(model, case.n),
            regularization_metric=primal_regularization_metric(
                model, optimizer.x_init.get_array()
            ),
            diagonal_provider=lambda: optimizer.state.diagonal,
        )

    primal_indices = None
    constraint_indices = None

    if args.reduce_all_controls:
        primal_indices = model.problem.get_primal_indices()
        constraint_indices = model.problem.get_constraint_indices()
    else:
        normal_shared_indices = None

    return SparseLUSolver(
        hessian_correction=hessian_correction,
        shared_indices=shared_indices,
        safeguard_shared=args.optimize_roll,
        normal_shared_indices=normal_shared_indices,
        primal_indices=primal_indices,
        constraint_indices=constraint_indices,
    )
