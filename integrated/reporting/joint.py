"""Validate and save a joint result, including every case's trajectories."""

import json

import numpy as np

from integrated.reporting.console import print_heading, print_values


def save_joint(joint, args, signature, point, history, solver):
    replay = joint.replay(point)
    vector = point.get_vector()
    vector.copy_device_to_host()
    values = vector.get_array()
    model = joint.model
    lower, upper = model.get_lower().get_vector(), model.get_upper().get_vector()
    lower.copy_device_to_host()
    upper.copy_device_to_host()
    primal = model.problem.get_primal_indices()
    finite_solution = bool(np.all(np.isfinite(values)))
    bound_error = float(
        max(
            0,
            np.max(lower.get_array()[primal] - values[primal]),
            np.max(values[primal] - upper.get_array()[primal]),
        )
    )
    last = history["iterations"][-1]
    replay_passed = all(
        np.isfinite(error) and error < 1e-5
        for case in replay["cases"]
        for error in case["reference_max_errors"].values()
    )
    feasible = all(
        case["battery_feasibility"]["inequality_violation"] < 1e-7
        and abs(case["battery_feasibility"]["periodicity_error"]) < 1e-7
        for case in replay["cases"]
    )
    baseline = float(sum(case.baseline["Data"][-1] for case in joint.cases))
    total = replay["total_downloaded_data"]
    baseline_feasible = all(
        case.baseline_metrics["inequality_violation"] < 1e-7
        and abs(case.baseline_metrics["periodicity_error"]) < 1e-7
        for case in joint.cases
    )
    improved = bool(total >= baseline - 1e-6)
    checks = {
        "Solver convergence": bool(history.get("converged", False)),
        "Native equalities": bool(last["inf_pr"] < 1e-6),
        "Variable bounds": bool(bound_error < 1e-7),
        "Independent replay": bool(replay_passed),
        "Battery limits and periodicity": bool(feasible),
        "Finite solution": finite_solution,
        "Feasible baseline comparison": bool(improved or not baseline_feasible),
    }
    validated = all(checks.values())
    feasibility_validated = all(
        checks[name]
        for name in [
            "Native equalities",
            "Variable bounds",
            "Independent replay",
            "Battery limits and periodicity",
            "Finite solution",
        ]
    )
    allocation = values[model.get_indices("shared_surface.cellInstd[0, :]")].reshape(
        7, 12
    )
    summary = {
        "validated": validated,
        "feasibility_validated": feasibility_validated,
        "validation_checks": checks,
        "converged": bool(history.get("converged", False)),
        "acceptable": bool(history.get("acceptable", False)),
        "cadre_source": signature["cadre_source"],
        "model_signature": signature,
        "nodes_per_case": joint.cases[0].n,
        "controls_per_profile": joint.cases[0].m,
        "duration_hours": joint.cases[0].window_duration / 3600,
        "time_grid": args.time_grid,
        "design_variables": joint.design_variable_count,
        "objective_scale": joint.objective_scale,
        "solver_barrier_objective": float(last["objective"]),
        "summed_data_objective": -joint.objective_scale * total,
        "initial_barrier_param": float(history["iterations"][0]["barrier_param"]),
        "reset_multipliers": args.reset_multipliers,
        "shared_design_variables": 86,
        "iterations": (
            int(last["iteration"])
            if "iteration" in last
            else int(last.get("iter", len(history["iterations"]) - 1))
        ),
        "nlp_error": float(last["nlp_error"]),
        "primal_infeasibility": float(last["inf_pr"]),
        "dual_infeasibility": float(last["inf_du"]),
        "complementarity": float(last["compl"]),
        "bound_violation": bound_error,
        "finite_solution": finite_solution,
        "baseline_feasible": baseline_feasible,
        "baseline_data": baseline,
        "optimized_data": total,
        "improved_over_baseline": improved,
        "improvement_percent": (
            100 * (total / baseline - 1) if baseline > 0 and baseline_feasible else None
        ),
        "solver_time_s": float(last["time"]),
        "solver_time_scope": "solver phase for this segment",
        "linear_solver": args.linear_solver,
        "primal_barrier": args.primal_barrier,
        "case_workers": getattr(solver, "case_workers", 1),
        "selective_curvature": getattr(solver, "selective_curvature", False),
        "spectral_curvature": getattr(solver, "spectral_curvature", False),
        "box_qp": getattr(solver, "box_qp", False),
        "box_qp_iterations": getattr(solver, "box_qp_iterations", 0),
        "largest_active_step_set": getattr(solver, "largest_active_set", 0),
        "periodicity_price_refits": getattr(solver, "price_refits", 0),
        "projected_trial_rejections": getattr(solver, "trial_rejections", {}),
        "trajectory_projection": not args.full_space_trials,
        "periodicity_restoration": args.restore_periodicity,
        "initial_battery_repair": args.repair_start,
        "battery_restorations": getattr(solver, "battery_restorations", 0),
        "full_power_restorations": getattr(solver, "full_power_restorations", 0),
        "initial_battery_restorations": getattr(
            solver, "initial_battery_restorations", 0
        ),
        "restoration_parameters_per_case": getattr(
            solver, "restoration_parameters_per_case", 0
        ),
        "trajectory_projections": getattr(solver, "trajectory_projections", 0),
        "max_linear_residual": getattr(solver, "max_linear_residual", None),
        "final_hessian_regularization_max": getattr(solver, "regularization_max", None),
        "design_step_fraction": getattr(solver, "design_step_fraction", None),
        "largest_normalized_design_direction": getattr(
            solver, "largest_design_step_ratio", None
        ),
        "design_limited_steps": getattr(solver, "limited_steps", 0),
        "state_limited_steps": getattr(solver, "state_limited_steps", 0),
        "recovered_newton_steps": getattr(solver, "recovered_newton_steps", 0),
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "initial_design": str(args.initial_design) if args.initial_design else None,
        "refit_controls": args.refit_controls,
        "surface_allocation": allocation.tolist(),
        **replay,
    }
    arrays = {"surface_allocation": allocation}
    for case in joint.cases:
        prefix = f"case{case.case_index}."
        arrays[f"case{case.case_index}_time_s"] = np.arange(case.n) * case.dt
        for label, name in {
            "P_comm": "values.P_comm[:]",
            "roll": "values.Gamma[:]",
            "solar_current": "values.Isetpt[:, :]",
            "SOC": "battery.SOC[:]",
            "I_bat": "battery.I_bat[:]",
            "Data": "comm_data.Data[:]",
            "temperature": "thermal.temperature[:, :]",
            "voltage": "power_solar.V_sol[:, :]",
            "CP_P_comm": "control_points.CP_P_comm[:]",
            "CP_gamma": "control_points.CP_gamma[:]",
            "CP_Isetpt": "control_points.CP_Isetpt[:, :]",
        }.items():
            arrays[f"case{case.case_index}_{label}"] = values[
                model.get_indices(prefix + name)
            ].copy()
        arrays[f"case{case.case_index}_baseline_Data"] = case.baseline["Data"]
        for name in ["P_comm", "SOC", "I_bat"]:
            arrays[f"case{case.case_index}_baseline_{name}"] = case.baseline[name]
    np.savez_compressed(args.output / "trajectories.npz", **arrays)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "solver_history.json").write_text(
        json.dumps(history["iterations"], indent=2) + "\n"
    )
    print_heading("JOINT CADRE RESULTS")
    print_values(
        [
            ("Complete validation", "PASSED" if validated else "FAILED"),
            (
                "Trajectory feasibility",
                "PASSED" if feasibility_validated else "FAILED",
            ),
            (
                "Solver convergence",
                "CONVERGED" if summary["converged"] else "NOT CONVERGED",
            ),
            ("NLP error", f"{summary['nlp_error']:.3e}"),
            ("Primal infeasibility", f"{summary['primal_infeasibility']:.3e}"),
            ("Dual infeasibility", f"{summary['dual_infeasibility']:.3e}"),
            ("Complementarity", f"{summary['complementarity']:.3e}"),
            ("Total downloaded data", f"{total:,.6f} CADRE units"),
            (
                (
                    "Feasible baseline"
                    if baseline_feasible
                    else "Reference (not feasible)"
                ),
                f"{baseline:,.6f} CADRE units",
            ),
            ("Solver time", f"{summary['solver_time_s']:.2f} s"),
            ("Results directory", str(args.output)),
        ]
    )
    if not validated:
        print_values(
            [
                (name, "PASSED" if passed else "FAILED")
                for name, passed in checks.items()
            ]
        )
    from integrated.reporting.joint_plots import plot_saved_joint

    plot_saved_joint(args.output)
    # The final summary retains the configuration from the setup report.
    (args.output / "model.json").unlink(missing_ok=True)
    if not validated:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(
            f"Joint optimization failed: {failed}. Diagnostics saved to {args.output}"
        )
    return summary
