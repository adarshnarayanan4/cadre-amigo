"""Prepare or compile the joint CADRE model without running an optimization."""

import argparse
import json
from pathlib import Path

from integrated.model.joint import prepare_joint_cases, assemble_joint_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=int, nargs="+", choices=range(6), default=list(range(6))
    )
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--control-points", type=int, default=300)
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also compile and initialize the model; never optimize",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("integrated/results/joint_preparation")
    )
    args = parser.parse_args(argv)
    output = args.output.resolve()
    cases, options = prepare_joint_cases(args.cases, args.nodes, args.control_points)
    joint = assemble_joint_model(cases, options)

    if args.build:
        joint.initialize(output / "build")

    report = {
        "cases": args.cases,
        "nodes_per_case": args.nodes,
        "control_points_per_profile": args.control_points,
        "design_variables": joint.design_variable_count,
        "shared_design_variables": 86,
        "objective": "Negative sum of final downloaded data",
        "periodicity": "Separate final SOC = initial SOC equality in every case",
        "battery_inequalities": "Current and SOC limits at every node",
        "compiled_and_initialized": args.build,
        "optimized": False,
        "validated_optimization": False,
        "initialization": "Reference replay without baseline repair; not guaranteed feasible",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "model.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Joint model: {len(cases)} cases, {joint.design_variable_count:,} design variables. No optimization run."
    )
    return joint


if __name__ == "__main__":
    main()
