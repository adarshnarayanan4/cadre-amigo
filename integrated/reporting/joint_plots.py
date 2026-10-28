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
            save_figure(
                fig,
                results,
                f"grid_check_case_{case['case_index']}",
                "Grid refinement check",
                note=f"Replay holds {original['controls_per_profile']} controls fixed. "
                f"The feasible start uses {summary['controls_per_profile']} refitted controls and a battery repair; "
                "no fine-grid optimization was performed.",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    plot_saved_joint(parser.parse_args().results)


if __name__ == "__main__":
    main()
