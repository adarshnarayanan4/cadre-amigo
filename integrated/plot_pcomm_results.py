"""Plot saved CADRE trajectories and matched optimization comparisons."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

    reference = results.parent / name

    if not compare or not (reference / "summary.json").exists():
        return None

    previous = json.loads((reference / "summary.json").read_text())

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


def plot_trajectories(data, summary, output, reference=None):
    """Plot power, battery limits, and downloaded data on the same time axis."""
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
        figsize=(10, 13 if solar else 11 if constrained else 9),
        sharex=True,
        constrained_layout=True,
    )

    comparison = None

    if reference:
        previous, old = reference
        comparison = comparison_metrics(summary, data, previous, old)
        (output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

        reference_label = (
            "Fixed roll/allocation optimum"
            if design
            else "Fixed solar-current optimum" if solar else "Fixed initial SOC optimum"
        )

        for axis, key in zip(axes, keys):
            axis.plot(
                hours,
                old[key],
                color="#252525",
                linestyle="--",
                linewidth=1.3,
                label=reference_label,
            )

    for axis, key in zip(axes, keys):
        axis.plot(
            hours,
            data[f"baseline_{key}"],
            color="#6b7280",
            label="Feasible baseline" if constrained else "Benchmark design",
        )
        axis.plot(hours, data[key], color="#0866b2", label="Optimized")
        axis.set_ylabel(LABELS[key])
        axis.grid(alpha=0.2)

    power_top = (
        max(1.0, float(np.max(data["P_comm"])), float(np.max(data["baseline_P_comm"])))
        * 1.2
        if constrained
        else 25
    )

    axes[0].fill_between(
        hours,
        0,
        power_top,
        where=data["CommLOS"] > 0,
        color="#efb83c",
        alpha=0.25,
        label="Ground contact",
    )
    axes[0].set_ylim(-0.2 if constrained else -0.5, power_top)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    for limit in (0.2, 1.0):
        axes[1].axhline(limit, color="#b42e28", linestyle="--", linewidth=1)

    axes[1].set_ylim(0.15, 1.05)

    if constrained:
        for limit in (-10, 5):
            axes[2].axhline(limit, color="#b42e28", linestyle="--", linewidth=1)

        axes[2].set_ylim(-11, 6)

        axes[1].text(
            0.99,
            0.13,
            f"Optimized SOC start/end = {summary['SOC_initial']:.4f}",
            transform=axes[1].transAxes,
            ha="right",
            fontsize=9,
        )

        caption = f"Feasible baseline: {summary['baseline_data']:.2f}\nOptimized: {summary['optimized_data']:.2f}"

        if solar and comparison:
            gain_key = (
                "improvement_over_fixed_design_percent"
                if design
                else "improvement_over_fixed_solar_currents_percent"
            )
            gain_label = "fixed roll/allocation" if design else "fixed currents"
            caption += f"\nGain over {gain_label}: {comparison[gain_key]:.3f}%"

        axes[-1].text(
            0.99, 0.06, caption, transform=axes[-1].transAxes, ha="right", fontsize=9
        )

    axes[-1].set_xlabel("Time [hours]")

    scope = (
        "communication power and initial SOC"
        if summary.get("initial_soc_optimized")
        else "communication power"
    )

    if summary.get("solar_currents_optimized"):
        scope += ", and solar currents"

    if design:
        if summary.get("roll_optimized") and summary.get("allocation_optimized"):
            scope = "roll and cell/radiator allocation"
        elif summary.get("roll_optimized"):
            scope = "roll"
        elif summary.get("allocation_optimized"):
            scope = "cell/radiator allocation"
        else:
            scope = "fixed roll/allocation"

    status = "Validated" if summary["validated"] else "Unvalidated"
    count = summary.get("design_variable_count", summary["control_points"])
    fig.suptitle(
        f"{status} CADRE optimization: {scope}\n"
        f"{summary['nodes']} nodes · {count} design variables · "
        f"{summary['improvement_percent']:.2f}% more downloaded data",
        fontsize=12 if summary.get("solar_currents_optimized") else 14,
    )

    fig.savefig(output / "optimization.png", dpi=180)

    plt.close(fig)


def plot_solar_currents(data, output):
    """Plot prescribed and optimized current profiles for all 12 panels."""
    hours = data["time_s"] / 3600

    fig, axes = plt.subplots(
        3, 4, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True
    )

    for panel, axis in enumerate(axes.ravel()):
        axis.plot(
            hours,
            data["baseline_Isetpt"][panel],
            color="#6b7280",
            label="Benchmark currents",
        )
        axis.plot(hours, data["Isetpt"][panel], color="#0866b2", label="Optimized")
        axis.set_title(f"Panel {panel + 1}")
        axis.set_ylim(-0.01, 0.41)
        axis.grid(alpha=0.2)

    axes[0, 0].legend(fontsize=8)

    fig.supxlabel("Time [hours]")
    fig.supylabel("Solar current [A]")
    fig.suptitle("Solar-current profiles with temperature-dependent voltage")

    fig.savefig(output / "solar_currents.png", dpi=180)

    plt.close(fig)


def plot_design_controls(data, summary, output):
    """Plot roll and the shared 7-by-12 thermal allocation matrix."""
    hours = data["time_s"] / 3600

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(
        hours,
        np.rad2deg(data["baseline_gamma"]),
        color="#6b7280",
        label="Benchmark roll",
    )
    axes[0].plot(
        hours,
        np.rad2deg(data["gamma"]),
        color="#0866b2",
        label="Optimized roll" if summary.get("roll_optimized") else "Fixed roll",
    )
    axes[0].set_xlabel("Time [hours]")
    axes[0].set_ylabel("Roll angle [degrees]")
    axes[0].set_ylim(-2, 92)
    axes[0].grid(alpha=0.2)
    axes[0].legend()

    allocation = axes[1].imshow(
        data["cellInstd"], vmin=0, vmax=1, cmap="viridis", aspect="auto"
    )
    axes[1].set_xticks(range(12), range(1, 13))
    axes[1].set_yticks(range(7), range(1, 8))
    axes[1].set_xlabel("Panel")
    axes[1].set_ylabel("Surface location")
    axes[1].set_title(
        "Shared cell/radiator allocation (0 = radiator, 1 = cell thermal properties)"
    )

    fig.colorbar(allocation, ax=axes[1], label="Allocation")
    fig.suptitle("Roll profile and surface allocation")

    fig.savefig(output / "design_controls.png", dpi=180)

    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", type=Path, nargs="?", default=Path("integrated/results/pcomm_solar")
    )
    parser.add_argument(
        "--output", type=Path, help="Save figures separately from the source result"
    )

    args = parser.parse_args(argv)

    output = args.output or args.results
    output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.results / "summary.json").read_text())

    reference = find_reference(args.results, summary)

    with np.load(args.results / "trajectories.npz") as data:
        plot_trajectories(data, summary, output, reference)

        if summary.get("solar_voltage_coupled"):
            plot_solar_currents(data, output)

        if summary.get("design_physics_restored"):
            plot_design_controls(data, summary, output)


if __name__ == "__main__":
    main()
