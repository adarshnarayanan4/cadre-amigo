"""Replay a saved joint design on a new grid, optionally repairing its battery."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from integrated.model.joint import prepare_joint_cases, assemble_joint_model
from integrated.optimization.checkpoint import project_into_bounds, load_checkpoint
from integrated.optimization.driver import CadreOptimizer
from integrated.optimization.joint import initialize_joint, load_initial_design
from integrated.optimization.projection import (
    BatteryRestorer,
    BatteryRestorationCase,
    TrajectoryProjector,
)
from integrated.reporting.console import print_heading, print_values


def prepare_mesh(args):
    """Keep the physical time window fixed while changing nodes and controls."""
    source = json.loads((args.results / "summary.json").read_text())
    if not np.isclose(source["duration_hours"], 12, rtol=0, atol=1e-9):
        raise ValueError("Mesh replay requires a full 12-hour source result")
    controls = args.control_points or source["controls_per_profile"]
    if not 4 <= controls <= args.nodes <= 1500:
        raise ValueError("Require 4 <= control points <= nodes <= 1500")
    if args.output.resolve() == args.results.resolve():
        raise ValueError(
            "Use a separate output directory to preserve the source result"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    cases, options = prepare_joint_cases(
        [case["case_index"] for case in source["cases"]],
        nodes=args.nodes,
        controls=controls,
        full_duration=True,
    )
    joint = assemble_joint_model(cases, options, source["objective_scale"])
    signature = initialize_joint(
        joint,
        SimpleNamespace(
            output=args.output, time_grid="full", reuse_build=args.reuse_build
        ),
    )
    model = joint.model
    state = model.get_initial_point()
    point = state.get_vector()
    load_initial_design(
        joint, point, args.results, refit=controls != source["controls_per_profile"]
    )
    lower, upper = model.get_lower().get_vector(), model.get_upper().get_vector()
    lower.copy_device_to_host()
    upper.copy_device_to_host()
    if args.repair:
        design = joint.design_indices()
        point.get_array()[design] = project_into_bounds(
            point.get_array()[design],
            lower.get_array()[design],
            upper.get_array()[design],
            1e-6,
        )
        point.copy_host_to_device()
    projector = TrajectoryProjector(joint)
    projector.project(point)
    if args.repair:
        restorer = BatteryRestorer(projector, periodicity_tolerance=1e-10)
        for case, partition in zip(cases, restorer.partitions):
            repair = BatteryRestorationCase(
                restorer,
                case,
                partition,
                point,
                full_power=True,
                adjust_initial_soc=True,
            )
            if not repair.restore(point, max_iterations=80):
                raise RuntimeError(
                    f"Case {case.case_index} mesh repair failed: {repair.message}"
                )
        projector.project(point)
    point.fill_at(model.problem.get_constraint_indices(), 0)
    model.problem.update(point)
    gradient = model.problem.create_vector()
    model.problem.gradient(0.0, point, gradient)
    gradient.copy_device_to_host()
    native_error = float(
        np.max(abs(gradient.get_array()[model.problem.get_constraint_indices()]))
    )
    point.copy_device_to_host()
    values = point.get_array()
    primal = np.asarray(model.problem.get_primal_indices())
    bound_error = float(
        max(
            0.0,
            np.max(lower.get_array()[primal] - values[primal]),
            np.max(values[primal] - upper.get_array()[primal]),
        )
    )
    replay = joint.replay(state)
    checks = {
        "Finite solution": bool(np.all(np.isfinite(values))),
        "Native equalities": native_error < 1e-6,
        "Variable bounds": bound_error < 1e-7,
        "Independent replay": all(
            error < 1e-5
            for case in replay["cases"]
            for error in case["reference_max_errors"].values()
        ),
        "Battery limits and periodicity": all(
            case["battery_feasibility"]["inequality_violation"] < 1e-7
            and abs(case["battery_feasibility"]["periodicity_error"]) < 1e-7
            for case in replay["cases"]
        ),
    }
    passed = all(checks.values())
    summary = {
        "validated_initialization": passed,
        "feasibility_validated": passed,
        "validated": False,
        "optimization_run": False,
        "optimization_converged": False,
        "validation_checks": checks,
        "source_result": str(args.results),
        "cadre_source": signature["cadre_source"],
        "model_signature": signature,
        "nodes_per_case": args.nodes,
        "controls_per_profile": controls,
        "duration_hours": 12.0,
        "objective_scale": joint.objective_scale,
        "design_variables": joint.design_variable_count,
        "controls_held_fixed": not args.repair
        and controls == source["controls_per_profile"],
        "power_and_initial_soc_repaired": args.repair,
        "native_equality_max_error": native_error,
        "bound_violation": bound_error,
        **replay,
    }
    arrays = {
        "surface_allocation": values[
            model.get_indices("shared_surface.cellInstd[0, :]")
        ].reshape(7, 12)
    }
    for case in cases:
        prefix = f"case{case.case_index}_"
        arrays[prefix + "time_s"] = np.arange(case.n) * case.dt
        for key, name in {
            "CP_P_comm": "control_points.CP_P_comm[:]",
            "CP_gamma": "control_points.CP_gamma[:]",
            "CP_Isetpt": "control_points.CP_Isetpt[:, :]",
            "P_comm": "values.P_comm[:]",
            "SOC": "battery.SOC[:]",
            "I_bat": "battery.I_bat[:]",
            "Data": "comm_data.Data[:]",
            "roll": "values.Gamma[:]",
        }.items():
            arrays[prefix + key] = values[
                model.get_indices(f"case{case.case_index}.{name}")
            ].copy()
    np.savez_compressed(args.output / "controls.npz", **arrays)
    if passed and args.repair:
        bound_duals = []
        for bounds, sign in [(lower, 1), (upper, -1)]:
            gaps = sign * (values[primal] - bounds.get_array()[primal])
            finite = np.isfinite(bounds.get_array()[primal])
            if np.any(gaps[finite] <= 0):
                raise ValueError("Cannot checkpoint a mesh start outside strict bounds")
            dual = np.zeros(len(primal))
            dual[finite] = 1e-3 / gaps[finite]
            bound_duals.append(dual)
        temporary = args.output / "checkpoint.tmp.npz"
        np.savez_compressed(
            temporary,
            solution=values,
            zl=bound_duals[0],
            zu=bound_duals[1],
            signature=json.dumps(signature),
            barrier_param=1e-3,
            iteration=0,
        )
        temporary.replace(args.output / "checkpoint.npz")
        optimizer = CadreOptimizer(model, model.create_vector())
        same, saved_duals = load_checkpoint(
            optimizer,
            SimpleNamespace(
                warm_start=args.output / "checkpoint.npz",
                release_design_controls=False,
                initial_barrier=None,
            ),
            signature,
        )
        if not same or saved_duals is None:
            raise ValueError(
                "Prepared checkpoint did not round-trip through the optimizer loader"
            )
        np.testing.assert_array_equal(optimizer.x_init.get_array(), values)
        variables = optimizer.optimizer.create_opt_vector()
        for dual, saved in zip([variables.get_zl(), variables.get_zu()], saved_duals):
            if dual.get_array().shape != saved.shape:
                raise ValueError(
                    "Prepared checkpoint has incompatible bound-multiplier shapes"
                )
        summary["checkpoint_round_trip_checked"] = True
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.repair and passed:
        from integrated.reporting.joint_plots import plot_mesh_comparison

        plot_mesh_comparison(args.results, args.output)
    print_heading("TIME-GRID REPLAY")
    print_values(
        [
            ("Nodes per case", args.nodes),
            ("Control points", controls),
            ("Feasibility", "PASSED" if passed else "FAILED"),
            ("Optimization performed", "No"),
            ("Downloaded data", f"{replay['total_downloaded_data']:,.6f}"),
            ("Native equality error", f"{native_error:.3e}"),
        ]
    )
    if args.repair and not passed:
        raise RuntimeError(
            "Repaired mesh failed its physical validation; see summary.json"
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--control-points", type=int)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--reuse-build", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    prepare_mesh(parser.parse_args())


if __name__ == "__main__":
    main()
