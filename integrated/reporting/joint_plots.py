"""Plot joint trajectories, shared designs, and recorded solver residuals."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualizations.style import (
    BLUE,
    GRAY,
    GOLD,
    RED,
    COLORS,
    configure_style,
    save_figure,
    integer_ticks,
)


def plot_saved_joint(results):
    results = Path(results)
    summary = json.loads((results / "summary.json").read_text())
    history = json.loads((results / "solver_history.json").read_text())
    configure_style()
    history = [row for row in history if row["iteration"] > 0]
    status = (
        "Validated optimization"
        if summary["validated"]
        else (
            "Feasible, not converged"
            if summary.get("feasibility_validated", False)
            else "Unvalidated trial"
        )
    )
    with np.load(results / "trajectories.npz", allow_pickle=False) as data:
        for case in summary["cases"]:
            index = case["case_index"]
            prefix = f"case{index}_"
            hours = data[prefix + "time_s"] / 3600
            fig, axes = plt.subplots(
                4, 1, figsize=(13, 13), constrained_layout=True, sharex=True
            )
            for axis, key, title, units, bounds in zip(
                axes,
                ["P_comm", "SOC", "I_bat", "Data"],
                [
                    "Communication power",
                    "Battery state of charge",
                    "Battery current",
                    "Downloaded data",
                ],
                ["Power [W]", "SOC", "Current [A]", "CADRE data units"],
                [(0, 25), (0.2, 1), (-10, 5), None],
            ):
                axis.plot(hours, data[prefix + key], color=BLUE, label=status)
                axis.plot(
                    hours,
                    data[prefix + "baseline_" + key],
                    color=GRAY,
                    label=(
                        "Feasible baseline"
                        if summary["baseline_feasible"]
                        else "Reference (not feasible)"
                    ),
                )
                if bounds:
                    for position, bound in enumerate(bounds):
                        axis.axhline(
                            bound,
                            color=RED,
                            linestyle=":",
                            label="Bounds" if position == 0 else None,
                        )
                axis.set(title=title, ylabel=units, xlim=(hours[0], hours[-1]))
            axes[-1].set_xlabel("Time [hours]")
            metrics = case["battery_feasibility"]
            save_figure(
                fig,
                results,
                f"case_{index}",
                f"CADRE case {index}",
                note=f"{status}. SOC periodicity error: {metrics['periodicity_error']:.2e}; "
                f"battery inequality violation: {metrics['inequality_violation']:.2e}.",
            )

    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    image = axis.imshow(summary["surface_allocation"], vmin=0, vmax=1, aspect="auto")
    axis.set(xlabel="Panel", ylabel="Cell", xticks=np.arange(12), yticks=np.arange(7))
    fig.colorbar(image, ax=axis, label="Cell allocation (0 radiator, 1 solar cell)")
    save_figure(
        fig,
        results,
        "shared_design",
        "Shared spacecraft design",
        note=f"{status}. Fin angle: {np.degrees(summary['fin_angle_rad']):.3f} deg; "
        f"antenna angle: {np.degrees(summary['antenna_angle_rad']):.3f} deg.",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, key, title in zip(
        axes.ravel(),
        ["nlp_error", "inf_pr", "inf_du", "compl"],
        [
            "KKT optimality error",
            "Primal infeasibility",
            "Dual stationarity",
            "Complementarity",
        ],
    ):
        axis.semilogy(
            [row["iteration"] for row in history],
            [max(row[key], np.finfo(float).tiny) for row in history],
            color=BLUE,
        )
        axis.set(title=title, xlabel="Optimizer iteration", ylabel="Residual")
        integer_ticks(axis)
    save_figure(
        fig,
        results,
        "convergence",
        "Joint optimizer convergence",
        note=status
        + "; iteration 0 precedes starting-point restoration and is omitted.",
    )
    for report in sorted(results.glob("mesh_replay_*.json")):
        plot_fixed_control_replay(results, report)


def plot_fixed_control_replay(results, report_path):
    """Show data and battery errors when only the time grid changes."""
    results, report_path = Path(results), Path(report_path)
    summary = json.loads((results / "summary.json").read_text())
    report = json.loads(report_path.read_text())
    configure_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    cases = [case["case_index"] for case in report["cases"]]

    source_arrays = results / "trajectories.npz"
    if not source_arrays.exists():
        source_arrays = results / "controls.npz"
    with np.load(source_arrays, allow_pickle=False) as coarse, np.load(
        report_path.with_suffix(".npz"), allow_pickle=False
    ) as fine:
        prefix = f"case{cases[0]}_"
        source_label = "candidate" if not summary.get("converged") else "solution"
        for saved, color, nodes, label in [
            (coarse, BLUE, summary["nodes_per_case"], source_label),
            (fine, GOLD, report["replay_nodes_per_case"], "replay"),
        ]:
            total = sum(saved[f"case{index}_Data"] for index in cases)
            axes[0].plot(
                saved[prefix + "time_s"] / 3600,
                total,
                color=color,
                label=f"{nodes:,}-node {label}",
            )
    axes[0].set(
        title="Total downloaded data",
        xlabel="Time [hours]",
        ylabel="CADRE data units",
        xlim=(0, 12),
    )

    metrics = [case["battery_feasibility"] for case in report["cases"]]
    violations = [
        max(0, 0.2 - item["SOC_min"], item["SOC_max"] - 1) for item in metrics
    ]
    periodicity = [abs(item["periodicity_error"]) for item in metrics]
    positions = np.arange(len(cases))
    axes[1].bar(
        positions - 0.2, violations, width=0.4, color=BLUE, label="SOC bound violation"
    )
    axes[1].bar(
        positions + 0.2,
        periodicity,
        width=0.4,
        color=GOLD,
        label="SOC periodicity error",
    )
    axes[1].axhline(1e-7, color=RED, linestyle=":", label="Validation tolerance")
    axes[1].set(
        title=f"Battery SOC checks at {report['replay_nodes_per_case']:,} nodes",
        xlabel="Case",
        ylabel="SOC error",
        xticks=positions,
        xticklabels=cases,
    )
    if max(violations + periodicity, default=0) > 1e-5:
        axes[1].set_yscale("log")
        axes[1].set_ylim(bottom=1e-8)
    else:
        axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    current_violations = [
        max(0, -10 - item["current_min_A"], item["current_max_A"] - 5)
        for item in metrics
    ]
    axes[2].bar(positions, current_violations, color=BLUE, label="Current violation")
    axes[2].axhline(1e-7, color=RED, linestyle=":", label="Validation tolerance")
    axes[2].set(
        title=f"Battery current checks at {report['replay_nodes_per_case']:,} nodes",
        xlabel="Case",
        ylabel="Current violation [A]",
        xticks=positions,
        xticklabels=cases,
    )
    if max(current_violations, default=0) > 1e-5:
        axes[2].set_yscale("log")
        axes[2].set_ylim(bottom=1e-8)
    else:
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    note = (
        f"Fixed {report['controls_per_profile']} controls per profile; reference replay, no fine-grid optimization. "
        f"Downloaded data changes {report['downloaded_data_difference_percent']:+.2f}%."
    )
    if summary.get("feasibility_validated") and not summary.get("converged"):
        note += " The source checkpoint has not converged."
    save_figure(
        fig,
        results,
        report_path.stem,
        f"{len(cases)}-case grid check",
        note=note,
    )


def plot_mesh_comparison(source, results):
    """Compare a coarse design, its uncorrected replay, and the repaired start."""
    source, results = Path(source), Path(results)
    original = json.loads((source / "summary.json").read_text())
    summary = json.loads((results / "summary.json").read_text())
    configure_style()
    source_arrays = source / "trajectories.npz"
    if not source_arrays.is_file():
        source_arrays = source / "controls.npz"
    replay_file = source / "mesh_replay.npz"
    replay = None
    if replay_file.is_file() and len(summary["cases"]) == 1:
        with np.load(replay_file, allow_pickle=False) as saved:
            replay = {key: saved[key].copy() for key in saved.files}
    with np.load(source_arrays, allow_pickle=False) as coarse, np.load(
        results / "controls.npz", allow_pickle=False
    ) as fine:
        for case in summary["cases"]:
            prefix = f"case{case['case_index']}_"
            fig, axes = plt.subplots(
                4, 1, figsize=(13, 13), sharex=True, constrained_layout=True
            )
            for axis, key, title, units, bounds in zip(
                axes,
                ["P_comm", "SOC", "I_bat", "Data"],
                [
                    "Communication power",
                    "Battery state of charge",
                    "Battery current",
                    "Downloaded data",
                ],
                ["Power [W]", "SOC", "Current [A]", "CADRE data units"],
                [(0, 25), (0.2, 1), (-10, 5), None],
            ):
                axis.plot(
                    coarse[prefix + "time_s"] / 3600,
                    coarse[prefix + key],
                    color=GRAY,
                    linestyle="--",
                    label=f"{original['nodes_per_case']:,}-node solution",
                )
                if replay is not None:
                    axis.plot(
                        replay["time_s"] / 3600,
                        replay[key],
                        color=GOLD,
                        label=f"{summary['nodes_per_case']:,}-node replay",
                    )
                axis.plot(
                    fine[prefix + "time_s"] / 3600,
                    fine[prefix + key],
                    color=BLUE,
                    label=f"{summary['nodes_per_case']:,}-node feasible start",
                )
                if bounds:
                    for position, bound in enumerate(bounds):
                        axis.axhline(
                            bound,
                            color=RED,
                            linestyle=":",
                            label="Bounds" if position == 0 else None,
                        )
                axis.set(title=title, ylabel=units, xlim=(0, 12))
            if replay is not None:
                inset = axes[2].inset_axes([0.64, 0.18, 0.32, 0.37])
                center = replay["time_s"][np.argmin(replay["I_bat"])] / 3600
                inset.plot(replay["time_s"] / 3600, replay["I_bat"], color=GOLD)
                inset.plot(
                    fine[prefix + "time_s"] / 3600, fine[prefix + "I_bat"], color=BLUE
                )
                inset.axhline(-10, color=RED, linestyle=":")
                inset.set(
                    xlim=(center - 0.2, center + 0.2),
                    ylim=(min(-10.12, replay["I_bat"].min() - 0.02), -9.8),
                    title="Charging-current limit",
                )
                inset.tick_params(labelsize=8)
                inset.set_title("Charging current limit", fontsize=9, pad=4)
            axes[-1].set_xlabel("Time [hours]")
            note = (
                f"The feasible start uses {summary['controls_per_profile']} refitted controls and a battery repair; "
                "no fine-grid optimization was performed."
            )
            if replay is not None:
                note = (
                    f"Replay holds {original['controls_per_profile']} controls fixed. "
                    + note
                )
            save_figure(
                fig,
                results,
                f"grid_check_case_{case['case_index']}",
                "Grid refinement check",
                note=note,
            )


def plot_trajectory_overview(results, baseline, *, labels, filename, title, note):
    """Plot total data and all six cases' battery and power trajectories."""
    results, baseline = Path(results), Path(baseline)
    summary = json.loads((results / "summary.json").read_text())
    cases = [case["case_index"] for case in summary["cases"]]
    configure_style()
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 14), constrained_layout=True, sharex=True
    )
    with np.load(results / "controls.npz", allow_pickle=False) as candidate, np.load(
        baseline / "controls.npz", allow_pickle=False
    ) as start:
        for data, color, label in [
            (start, GRAY, labels[0]),
            (candidate, BLUE, labels[1]),
        ]:
            hours = data[f"case{cases[0]}_time_s"] / 3600
            total = sum(data[f"case{index}_Data"] for index in cases)
            axes[0].plot(hours, total, color=color, label=label)
        for axis, key, panel_title, units, bounds in zip(
            axes[1:],
            ["SOC", "I_bat", "P_comm"],
            ["Battery state of charge", "Battery current", "Communication power"],
            ["SOC", "Current [A]", "Power [W]"],
            [(0.2, 1.0), (-10, 5), (0, 25)],
        ):
            for index, color in zip(cases, COLORS + ["#8c564b"]):
                axis.plot(
                    candidate[f"case{index}_time_s"] / 3600,
                    candidate[f"case{index}_{key}"],
                    color=color,
                    label=f"Case {index}",
                )
            for position, bound in enumerate(bounds):
                axis.axhline(
                    bound,
                    color=RED,
                    linestyle=":",
                    label="Bounds" if position == 0 else None,
                )
            axis.set(title=panel_title, ylabel=units, xlim=(0, 12))
    axes[0].set(title="Total downloaded data", ylabel="CADRE data units", xlim=(0, 12))
    axes[-1].set_xlabel("Time [hours]")
    save_figure(
        fig,
        results,
        filename,
        title,
        note=note,
    )


def plot_checkpoint_candidate(results, baseline):
    """Compare an unconverged feasible checkpoint with its feasible start."""
    results, baseline = Path(results), Path(baseline)
    summary = json.loads((results / "summary.json").read_text())
    baseline_summary = json.loads((baseline / "summary.json").read_text())
    if not summary.get("feasibility_validated") or summary.get("converged"):
        raise ValueError(
            "The candidate plot requires a feasible, unconverged checkpoint"
        )
    if not baseline_summary.get("feasibility_validated"):
        raise ValueError("The comparison start must pass the feasibility checks")
    if summary["model_signature"] != baseline_summary["model_signature"]:
        raise ValueError("The checkpoint and start must use the same model and grid")
    plot_trajectory_overview(
        results,
        baseline,
        labels=("Feasible start", "Feasible checkpoint"),
        filename="candidate_overview",
        title="Six-case 600-node candidate",
        note="Feasible checkpoint; the optimizer has not converged.",
    )
    for report in sorted(results.glob("mesh_replay_*.json")):
        plot_fixed_control_replay(results, report)


def plot_reference_start(results, source):
    """Compare a reference-only fine-grid repair with its 600-node source."""
    results, source = Path(results), Path(source)
    summary = json.loads((results / "summary.json").read_text())
    original = json.loads((source / "summary.json").read_text())
    if not summary.get("reference_feasibility_validated") or summary.get(
        "native_model_checked"
    ):
        raise ValueError(
            "This plot requires a reference-checked start without a native check"
        )
    if not original.get("feasibility_validated"):
        raise ValueError("The source checkpoint must pass its feasibility checks")
    if (
        summary["cadre_source"] != original["cadre_source"]
        or summary["duration_hours"] != original["duration_hours"]
    ):
        raise ValueError(
            "The reference repair and source must use the same CADRE data and time window"
        )
    if [item["case_index"] for item in summary["cases"]] != [
        item["case_index"] for item in original["cases"]
    ]:
        raise ValueError("The reference repair and source must contain the same cases")
    plot_trajectory_overview(
        results,
        source,
        labels=("600-node candidate", "1,500-node reference repair"),
        filename="reference_start_overview",
        title="Six-case 1,500-node reference repair",
        note="Reference battery repair; no native 1,500-node check or CADRE optimization.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--checkpoint-baseline", type=Path)
    options.add_argument("--reference-source", type=Path)
    args = parser.parse_args()
    if args.checkpoint_baseline:
        plot_checkpoint_candidate(args.results, args.checkpoint_baseline)
    elif args.reference_source:
        plot_reference_start(args.results, args.reference_source)
    else:
        plot_saved_joint(args.results)


if __name__ == "__main__":
    main()
