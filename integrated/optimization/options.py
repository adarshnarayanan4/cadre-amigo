"""Command-line configuration for the CADRE optimization."""

import argparse
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    """Parse and validate optimization settings."""
    parser = argparse.ArgumentParser(
        description="Optimize CADRE communication power with coupled thermal and battery dynamics."
    )

    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--control-points", type=int, default=300)
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=200,
        help="Maximum optimizer iterations for this run or warm-start segment (default: 200)",
    )

    parser.add_argument(
        "--fixed-initial-soc",
        action="store_true",
        help="Retain benchmark initial SOC for a comparison run",
    )
    parser.add_argument(
        "--fixed-solar-currents",
        action="store_true",
        help="Update voltage but keep the benchmark solar currents fixed",
    )
    parser.add_argument(
        "--legacy-fixed-voltage",
        action="store_true",
        help="Reproduce the earlier fixed-current, fixed-voltage model",
    )

    parser.add_argument(
        "--optimize-roll",
        action="store_true",
        help="Optimize roll with exposure, gain, and reaction-wheel dependencies restored",
    )
    parser.add_argument(
        "--optimize-allocation",
        action="store_true",
        help="Optimize the 84 shared cell/radiator thermal allocations",
    )
    parser.add_argument(
        "--design-physics",
        action="store_true",
        help="Restore design physics while retaining fixed roll and allocation for comparison",
    )

    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--linear-solver",
        choices=("superlu", "native"),
        default="superlu",
        help="Native LDL enables full-matrix inertia correction and explicit temporal ordering",
    )

    parser.add_argument(
        "--warm-start",
        type=Path,
        help="Warm start from saved primal/dual variables and barrier parameter",
    )
    parser.add_argument(
        "--release-design-controls",
        action="store_true",
        help="Allow a design-physics warm start to release previously fixed roll/allocation controls",
    )

    parser.add_argument(
        "--initial-barrier",
        type=float,
        help="Positive initial IPM barrier parameter; defaults to 0.01 for coupled voltage",
    )
    parser.add_argument(
        "--center-bound-multipliers",
        action="store_true",
        help="Initialize bound multipliers so each finite-bound complementarity product equals the initial barrier",
    )
    parser.add_argument(
        "--warm-start-push",
        type=float,
        default=1e-5,
        help="Strict-bound projection fraction for checkpoint warm starts (default: 1e-5)",
    )

    parser.add_argument(
        "--reduce-all-controls",
        action="store_true",
        help="Reduce free design controls, retaining one communication-power control to enforce energy periodicity",
    )

    args = parser.parse_args(argv)

    if args.max_iterations < 1:
        parser.error("--max-iterations must be positive")

    if args.initial_barrier is not None and (
        not np.isfinite(args.initial_barrier) or args.initial_barrier <= 0
    ):
        parser.error("--initial-barrier must be finite and positive")

    if not np.isfinite(args.warm_start_push) or not 0 < args.warm_start_push < 0.5:
        parser.error("--warm-start-push must lie strictly between 0 and 0.5")

    if args.release_design_controls and not args.warm_start:
        parser.error("--release-design-controls requires --warm-start")

    if args.reduce_all_controls and not (
        args.design_physics or args.optimize_roll or args.optimize_allocation
    ):
        parser.error("--reduce-all-controls requires design physics")

    if args.linear_solver == "native" and args.reduce_all_controls:
        parser.error("Native LDL uses the full KKT system; omit --reduce-all-controls")

    design_physics = (
        args.design_physics or args.optimize_roll or args.optimize_allocation
    )

    if design_physics and args.legacy_fixed_voltage:
        parser.error("Design optimization requires coupled solar voltage")

    coupled_voltage = not args.legacy_fixed_voltage
    free_solar = coupled_voltage and not args.fixed_solar_currents

    if args.output is None:
        name = (
            ("pcomm_solar" if free_solar else "pcomm_coupled_voltage")
            if coupled_voltage
            else (
                "pcomm_constrained" if args.fixed_initial_soc else "pcomm_initial_soc"
            )
        )

        if design_physics:
            name = (
                "pcomm_roll_allocation"
                if args.optimize_roll and args.optimize_allocation
                else (
                    "pcomm_roll"
                    if args.optimize_roll
                    else (
                        "pcomm_allocation"
                        if args.optimize_allocation
                        else "pcomm_design_reference"
                    )
                )
            )

        args.output = Path("integrated/results") / name

    return args
