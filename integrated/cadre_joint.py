"""Build, optimize, or resume CADRE cases with one shared spacecraft design."""

import argparse
import json
import math
from pathlib import Path

from cadre_paths import source_provenance
from integrated.model.joint import prepare_joint_cases, assemble_joint_model
from integrated.optimization.joint import initialize_joint, optimize_joint
from integrated.reporting.joint import save_joint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=int, nargs="+", choices=range(6), default=list(range(6))
    )
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--control-points", type=int, default=300)
    parser.add_argument(
        "--time-grid",
        choices=["full", "prefix"],
        help="Full 12-hour grid, or a prefix at the native time step",
    )
    parser.add_argument(
        "--build", action="store_true", help="Compile and initialize only"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Compile, optimize, and independently validate",
    )
    parser.add_argument(
        "--reuse-build",
        action="store_true",
        help="Reuse a verified build in this output directory",
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument(
        "--objective-scale",
        type=float,
        default=1e-3,
        help="Positive multiplier of the summed data objective",
    )
    parser.add_argument(
        "--linear-solver", choices=["block", "native", "sparse"], default="block"
    )
    parser.add_argument(
        "--case-workers",
        type=int,
        default=1,
        help="Maximum threads for independent case factorizations (default: 1)",
    )
    parser.add_argument(
        "--primal-barrier",
        action="store_true",
        help="Use centered bound duals and bound-aware reduced-space trials",
    )
    parser.add_argument(
        "--spectral-curvature",
        action="store_true",
        help="Correct negative design modes with case-local matrix updates",
    )
    parser.add_argument(
        "--box-qp",
        action="store_true",
        help="Solve an active-bound quadratic step instead of uniform step damping",
    )
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument(
        "--initial-design",
        type=Path,
        help="Use named controls from a saved joint result as a new starting guess",
    )
    parser.add_argument(
        "--refit-controls",
        action="store_true",
        help="Fit imported controls to a new spline grid and repair the starting battery",
    )
    parser.add_argument("--initial-barrier", type=float)
    parser.add_argument(
        "--initial-bound-push",
        type=float,
        default=1e-6,
        help="Interior margin for the replayed cold starting point",
    )
    parser.add_argument("--center-bound-multipliers", action="store_true")
    parser.add_argument(
        "--reset-multipliers",
        action="store_true",
        help="Recenter bound multipliers and recompute trajectory adjoints on restart",
    )
    parser.add_argument("--warm-start-push", type=float, default=0.0)
    parser.add_argument(
        "--repair-start",
        action="store_true",
        help="Repair initial battery limits and periodicity before optimization",
    )
    parser.add_argument(
        "--restore-periodicity",
        action="store_true",
        help="Restore battery periodicity with bounded power-profile corrections",
    )
    parser.add_argument(
        "--full-space-trials",
        action="store_true",
        help="Disable trajectory restoration during line search",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.objective_scale) or args.objective_scale <= 0:
        parser.error("Objective scale must be positive and finite")
    if len(set(args.cases)) != len(args.cases):
        parser.error("Choose distinct benchmark cases")
    if args.linear_solver == "sparse" and len(args.cases) != 1:
        parser.error("The sparse reference solver requires exactly one case")
    if args.case_workers < 1:
        parser.error("Require at least one case worker")
    if args.box_qp and not args.primal_barrier:
        parser.error("--box-qp requires --primal-barrier")
    if args.spectral_curvature and (not args.optimize or args.linear_solver != "block"):
        parser.error("--spectral-curvature requires optimization with the block solver")
    if args.primal_barrier:
        if not args.optimize or args.full_space_trials or args.linear_solver != "block":
            parser.error(
                "--primal-barrier requires projected optimization with the block solver"
            )
        args.restore_periodicity = True
        if not args.warm_start:
            args.repair_start = True
    if not 4 <= args.control_points <= args.nodes <= 1500:
        parser.error("Require 4 <= control-points <= nodes <= 1500")
    if args.max_iterations < 1 or not 0 <= args.warm_start_push < 0.5:
        parser.error("Require positive iterations and 0 <= warm-start-push < 0.5")
    if not 0 < args.initial_bound_push < 0.5:
        parser.error("Require 0 < initial-bound-push < 0.5")
    if args.initial_barrier is not None and (
        not math.isfinite(args.initial_barrier) or args.initial_barrier <= 0
    ):
        parser.error("Initial barrier must be positive and finite")
    if args.warm_start and not args.optimize:
        parser.error("--warm-start requires --optimize")
    if args.repair_start and (not args.optimize or args.warm_start):
        parser.error("--repair-start requires a cold optimization run")
    if args.initial_design and (args.warm_start or not args.optimize):
        parser.error(
            "--initial-design requires --optimize and cannot be combined with --warm-start"
        )
    if args.refit_controls:
        if not args.initial_design:
            parser.error("--refit-controls requires --initial-design")
        args.repair_start = True
    if args.restore_periodicity and args.full_space_trials:
        parser.error("--restore-periodicity requires trajectory restoration")
    if args.reuse_build and not (args.build or args.optimize):
        parser.error("--reuse-build requires --build or --optimize")
    args.time_grid = args.time_grid or ("full" if args.optimize else "prefix")
    args.output = (
        args.output
        or Path("integrated/results")
        / ("joint_full" if args.optimize else "joint_preparation")
    ).resolve()
    args.release_design_controls = False
    if not args.warm_start:
        args.center_bound_multipliers = True
    if args.reset_multipliers or args.primal_barrier:
        args.center_bound_multipliers = True
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        cases, options = prepare_joint_cases(
            args.cases,
            args.nodes,
            args.control_points,
            full_duration=args.time_grid == "full",
            repair=args.optimize and not (args.warm_start or args.initial_design),
        )
        joint = assemble_joint_model(cases, options, args.objective_scale)
        signature = (
            initialize_joint(joint, args) if args.build or args.optimize else None
        )
        report = {
            "cadre_source": source_provenance(),
            "cases": args.cases,
            "nodes_per_case": args.nodes,
            "control_points_per_profile": args.control_points,
            "duration_hours": cases[0].window_duration / 3600,
            "time_grid": args.time_grid,
            "design_variables": joint.design_variable_count,
            "shared_design_variables": 86,
            "objective_scale": args.objective_scale,
            "objective": "Negative sum of final downloaded data",
            "periodicity": "Separate final SOC = initial SOC in every case",
            "compiled_and_initialized": bool(args.build or args.optimize),
            "optimization_requested": args.optimize,
            "model_signature": signature,
        }
        (args.output / "model.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.optimize:
            point, history, solver = optimize_joint(joint, args, signature)
            save_joint(joint, args, signature, point, history, solver)
            (args.output / "failure.json").unlink(missing_ok=True)
        else:
            print(
                f"Joint model: {len(cases)} cases, {joint.design_variable_count:,} design variables. No optimization run."
            )
        return joint
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(
                {"error_type": type(error).__name__, "message": str(error)}, indent=2
            )
            + "\n"
        )
        raise


if __name__ == "__main__":
    main()
