"""Plot saved CADRE trajectories and matched optimization comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualizations.style import (
    BLUE,
    TEAL,
    GOLD,
    GRAY,
    RED,
    configure_style,
    integer_ticks,
    save_figure,
)

LABELS = {
    "P_comm": "Communication power [W]",
    "SOC": "Battery state of charge",
    "I_bat": "Battery current [A]",
    "P_sol": "Solar power [W]",
    "Data": "Downloaded data [CADRE units]",
}


def find_reference(results, summary):
    """Load a validated comparison only when its grid and model options match."""
    design = summary.get("design_physics_restored", False)
    solar = summary.get("solar_voltage_coupled", False)

    if design:
        compare = summary.get("roll_optimized") or summary.get("allocation_optimized")
        name = "pcomm_design_reference"
    elif solar:
        compare = summary.get("solar_currents_optimized")
        name = "pcomm_coupled_voltage"
    else:
        compare = summary.get("initial_soc_optimized")
        name = "pcomm_constrained"

    case_index = summary.get("case_index", 0)

    if case_index:
        name += f"_case_{case_index}"

    reference = results.parent / name

    if not compare or not (reference / "summary.json").exists():
        return None

    previous = json.loads((reference / "summary.json").read_text())

    if previous.get("case_index", 0) != case_index:
        return None

    if not previous.get("validated") or any(
        previous[key] != summary[key] for key in ("nodes", "control_points", "dt_s")
    ):
        return None

    if solar:
        if not previous.get("solar_voltage_coupled"):
            return None

        if previous.get("initial_soc_optimized") != summary.get(
            "initial_soc_optimized"
        ):
            return None

        if design:
            if (
                not previous.get("design_physics_restored")
                or previous.get("roll_optimized")
                or previous.get("allocation_optimized")
                or previous.get("solar_currents_optimized")
                != summary.get("solar_currents_optimized")
            ):
                return None
        elif previous.get("solar_currents_optimized"):
            return None

    with np.load(reference / "trajectories.npz") as data:
        return previous, {key: data[key].copy() for key in data.files}


def comparison_metrics(summary, data, previous, reference):
    """Retain the existing comparison JSON fields for each formulation."""
    improvement = 100 * (summary["optimized_data"] / previous["optimized_data"] - 1)

    metrics = {
        "optimized_initial_SOC": summary["SOC_initial"],
        "feasible_benchmark_data": summary["baseline_data"],
        "improvement_over_benchmark_percent": summary["improvement_percent"],
    }

    if summary.get("solar_voltage_coupled"):
        fixed_energy = float(
            np.trapezoid(reference["P_sol"], reference["time_s"]) / 3600
        )

        solar_energy = float(np.trapezoid(data["P_sol"], data["time_s"]) / 3600)

        metrics.update(
            comparison_scope="Free vs fixed solar currents; voltage coupled in both",
            reference_SOC_initial=previous["SOC_initial"],
            fixed_solar_current_data=previous["optimized_data"],
            optimized_solar_current_data=summary["optimized_data"],
            improvement_over_fixed_solar_currents_percent=improvement,
            fixed_solar_energy_Wh=fixed_energy,
            optimized_solar_energy_Wh=solar_energy,
            solar_energy_improvement_percent=100 * (solar_energy / fixed_energy - 1),
        )

        if summary.get("design_physics_restored"):
            metrics["comparison_scope"] = (
                "Free roll/allocation vs fixed roll/allocation; restored lookup physics in both"
            )
            metrics["fixed_design_data"] = metrics.pop("fixed_solar_current_data")
            metrics["optimized_design_data"] = metrics.pop(
                "optimized_solar_current_data"
            )
            metrics["improvement_over_fixed_design_percent"] = metrics.pop(
                "improvement_over_fixed_solar_currents_percent"
            )
    else:
        metrics.update(
            comparison_scope="Free vs fixed initial SOC",
            fixed_initial_SOC=previous["SOC_initial"],
            fixed_initial_SOC_data=previous["optimized_data"],
            free_initial_SOC_data=summary["optimized_data"],
            improvement_over_fixed_initial_SOC_percent=improvement,
        )

    return metrics


def result_context(summary):
    status = "Validated result" if summary.get("validated") else "Unvalidated trial"
    nodes = summary["nodes"]
    count = summary.get("design_variable_count")
    controls = (
        f"{count:,} design variables"
        if count
        else f"{summary['control_points']:,} control points per profile"
    )
    duration = summary.get(
        "duration_hours", (nodes - 1) * summary.get("dt_s", 0) / 3600
    )
    return f"{status} · {duration:g} hours · {nodes:,} time nodes · {controls}"


def result_name(summary):
    if summary.get("roll_optimized") and summary.get("allocation_optimized"):
        return "Roll and surface-allocation optimization"

    if summary.get("roll_optimized"):
        return "Spacecraft roll optimization"

    if summary.get("allocation_optimized"):
        return "Surface-allocation optimization"

    if summary.get("design_physics_restored"):
        return "Fixed-design optimization reference"

    if summary.get("solar_currents_optimized"):
        return "Communication and solar-current optimization"

    return "Communication-power optimization"


def plot_trajectories(data, summary, output, reference=None):
    """Show every trajectory and constraint without clipping baseline curves."""
    configure_style()
    hours = data["time_s"] / 3600
    constrained = "periodicity_error" in summary
    solar = summary.get("solar_voltage_coupled", False)
    design = summary.get("design_physics_restored", False)
    keys = (
        ["P_comm", "SOC", "I_bat", "Data"] if constrained else ["P_comm", "SOC", "Data"]
    )

    if solar:
        keys.insert(-1, "P_sol")

    fig, axes = plt.subplots(
        len(keys),
        1,
        figsize=(14, 2.7 * len(keys) + 1.5),
        sharex=True,
        constrained_layout=True,
    )
    candidate = "Optimized" if summary.get("validated") else "Trial iterate"
    comparison = None

    if reference:
        previous, old = reference

        if summary.get("validated"):
            comparison = comparison_metrics(summary, data, previous, old)
            comparison_path = output / "comparison.json"

            if not comparison_path.exists():
                comparison_path.write_text(json.dumps(comparison, indent=2) + "\n")

        reference_label = (
            "Fixed-design optimum"
            if design
            else "Fixed-current optimum" if solar else "Fixed-SOC optimum"
        )

        for axis, key in zip(axes, keys):
            axis.plot(
                old["time_s"] / 3600,
                old[key],
                color=TEAL,
                linestyle="--",
                label=reference_label,
            )

    titles = {
        "P_comm": "Communication-power schedule",
        "SOC": "Battery state of charge",
        "I_bat": "Battery current",
        "P_sol": "Solar electrical generation",
        "Data": "Cumulative downloaded data",
    }

    for axis, key in zip(axes, keys):
        axis.plot(
            hours,
            data[f"baseline_{key}"],
            color=GRAY,
            label="Feasible baseline" if constrained else "Benchmark design",
        )
        axis.plot(hours, data[key], color=BLUE, label=candidate)
        axis.set(ylabel=LABELS[key], title=titles[key], xlim=(hours[0], hours[-1]))

        if key == "SOC":
            axis.axhline(0.2, color=RED, linestyle=":", label="SOC limits: 0.2 to 1.0")
            axis.axhline(1.0, color=RED, linestyle=":")
        elif key == "I_bat":
            axis.axhline(
                -10, color=RED, linestyle=":", label="Current limits: −10 to 5 A"
            )
            axis.axhline(5, color=RED, linestyle=":")

    axes[0].fill_between(
        hours,
        0,
        1,
        where=data["CommLOS"] > 0,
        transform=axes[0].get_xaxis_transform(),
        color=GOLD,
        alpha=0.18,
        label="Ground contact",
        zorder=0,
    )
    axes[-1].set_xlabel("Time [hours]")
    note = f"Downloaded data: baseline {summary['baseline_data']:,.2f}; {candidate.lower()} {summary['optimized_data']:,.2f} CADRE units."

    if summary.get("validated"):
        note += f" Improvement over baseline: {summary['improvement_percent']:.2f}%."
    else:
        note += " This trial has not passed convergence and independent validation."

    if comparison:
        gain = 100 * (summary["optimized_data"] / previous["optimized_data"] - 1)
        note += f" Gain over {reference_label.lower()}: {gain:.3f}%."

    if constrained:
        note += (
            f" Replay SOC start/end: {summary['SOC_initial']:.4f} / {summary['SOC_final']:.4f};"
            f" periodicity error: {summary['periodicity_error']:.2e}."
        )

    save_figure(
        fig, output, "optimization", result_name(summary), result_context(summary), note
    )


def plot_solar_currents(data, output, summary=None):
    """Use one legend and consistent scales for all twelve panel profiles."""
    configure_style()
    hours = data["time_s"] / 3600
    fig, axes = plt.subplots(
        3, 4, figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True
    )
    candidate = "Optimized" if summary and summary.get("validated") else "Trial iterate"

    if summary and not summary.get("solar_currents_optimized"):
        candidate = "Prescribed currents"

    low = min(0, float(np.min(data["baseline_Isetpt"])), float(np.min(data["Isetpt"])))
    high = max(
        0.4, float(np.max(data["baseline_Isetpt"])), float(np.max(data["Isetpt"]))
    )
    margin = 0.04 * max(high - low, 0.01)

    for panel, axis in enumerate(axes.ravel()):
        axis.plot(
            hours,
            data["baseline_Isetpt"][panel],
            color=GRAY,
            label="Benchmark currents",
        )
        axis.plot(hours, data["Isetpt"][panel], color=BLUE, label=candidate)
        axis.set(
            title=f"Panel {panel + 1:02d}",
            ylim=(low - margin, high + margin),
            xlim=(hours[0], hours[-1]),
        )
        axis.axhline(0.4, color=RED, linestyle=":", label="Control limit: 0.4 A")

    for axis in axes[-1]:
        axis.set_xlabel("Time [hours]")

    for axis in axes[:, 0]:
        axis.set_ylabel("Solar current [A]")

    save_figure(
        fig,
        output,
        "solar_currents",
        "Solar-current schedules by panel",
        (
            result_context(summary)
            if summary
            else "Twelve panels · Temperature-dependent voltage"
        ),
        "All panels use the same current scale. Curves show time-node currents reconstructed from the spline controls.",
        shared_legend=True,
    )


def plot_design_controls(data, summary, output):
    configure_style()
    hours = data["time_s"] / 3600
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), constrained_layout=True)
    axes[0].plot(
        hours, np.rad2deg(data["baseline_gamma"]), color=GRAY, label="Benchmark roll"
    )
    label = "Optimized roll" if summary.get("validated") else "Trial roll"

    if not summary.get("roll_optimized"):
        label = "Fixed roll"

    axes[0].plot(hours, np.rad2deg(data["gamma"]), color=BLUE, label=label)
    axes[0].set(
        xlabel="Time [hours]",
        ylabel="Roll angle [deg]",
        title="Spacecraft roll schedule",
        xlim=(hours[0], hours[-1]),
    )
    allocation = axes[1].imshow(
        data["cellInstd"], vmin=0, vmax=1, cmap="viridis", aspect="auto", origin="lower"
    )
    axes[1].set_xticks(range(12), range(1, 13))
    axes[1].set_yticks(range(7), range(1, 8))
    state = "Free" if summary.get("allocation_optimized") else "Fixed"
    axes[1].set(
        xlabel="Panel",
        ylabel="Cell location",
        title=f"{state} cell/radiator thermal allocation",
    )
    fig.colorbar(
        allocation,
        ax=axes[1],
        label="Allocation fraction",
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    save_figure(
        fig,
        output,
        "design_controls",
        "Roll schedule and surface allocation",
        result_context(summary),
        "Allocation interpolates thermal properties: 0 = radiator, 1 = solar cell. The same 7 × 12 allocation is shared across all time nodes.",
    )


def plot_convergence(history, summary, output):
    """Separate feasibility, stationarity, and complementarity diagnostics."""
    configure_style()

    if summary.get("warm_start"):
        history = [row for row in history if row["iteration"] > 0]

    iterations = np.array([row["iteration"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    specifications = [
        (
            "nlp_error",
            "Overall KKT optimality error",
            1e-8,
            "Requested tolerance: 1e-8",
        ),
        ("inf_pr", "Primal constraint violation", 1e-6, "Validation threshold: 1e-6"),
        ("inf_du", "Dual stationarity residual", None, None),
        ("compl", "Bound complementarity", None, None),
    ]

    for axis, (key, title, threshold, threshold_label) in zip(
        axes.ravel(), specifications
    ):
        values = np.array([row.get(key, np.nan) for row in history], dtype=float)
        valid = np.isfinite(values) & (values >= 0)
        plotted = np.where(valid, np.maximum(values, np.finfo(float).tiny), np.nan)
        axis.semilogy(iterations, plotted, color=BLUE, label="Recorded residual")

        if threshold is not None:
            axis.axhline(threshold, color=RED, linestyle="--", label=threshold_label)

        if not valid.any():
            axis.text(
                0.5,
                0.5,
                "Not recorded in this run",
                ha="center",
                transform=axis.transAxes,
            )

        axis.set(title=title, xlabel="Optimizer iteration", ylabel="Residual")
        integer_ticks(axis)

    note = "KKT error uses AMIGO scaling; dual and complementarity panels show the recorded raw metrics."

    if summary.get("warm_start"):
        note += " Warm-start iteration 0 is omitted because it precedes checkpoint restoration."

    save_figure(
        fig,
        output,
        "convergence",
        "Optimizer convergence diagnostics",
        result_context(summary),
        note,
        shared_legend=True,
    )


def plot_saved_case(results, output):
    """Refresh figures from saved arrays, keeping numerical results intact."""
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((results / "summary.json").read_text())
    reference = find_reference(results, summary)

    with np.load(results / "trajectories.npz") as data:
        plot_trajectories(data, summary, output, reference)

        if summary.get("solar_voltage_coupled"):
            plot_solar_currents(data, output, summary)

        if summary.get("design_physics_restored"):
            plot_design_controls(data, summary, output)

    history_path = results / "solver_history.json"

    if history_path.exists():
        plot_convergence(json.loads(history_path.read_text()), summary, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="?", default=Path("integrated/results/pcomm_solar")
    )
    parser.add_argument(
        "--output", type=Path, help="Save figures separately from the source result"
    )
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="Refresh every saved case in integrated/results; no optimization is run",
    )
    args = parser.parse_args(argv)
    cases = [args.results]

    if args.all_results:
        root = Path(__file__).resolve().parents[1] / "results"
        cases = [
            path.parent
            for path in sorted(root.glob("*/summary.json"))
            if (path.parent / "trajectories.npz").exists()
        ]

    for results in cases:
        output = args.output or results

        if args.all_results and args.output:
            output = args.output / results.name

        plot_saved_case(results, output)
        print(f"Figures saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
