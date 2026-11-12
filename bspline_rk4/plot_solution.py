"""
Figures of the CADRE solution
-----------------------------
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

matplotlib.use("Agg")

# Surface and ink of the reference palette, the VS Code blue of the full problem with two hues validated
# against it for color blind readers, and the sequential blues of the saved iterates
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {"initial": INK_SECONDARY, "baseline": "#a93226", "geometry": "#eb6834", "full": "#007acc"}
SEQUENTIAL = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab"]

LABELS = {
    "initial": "Initial",
    "baseline": "Baseline optimization",
    "geometry": "Geometric variables",
    "full": "Geometric and attitude variables",
}
PROBLEMS = ("baseline", "geometry", "full")
MONTHS = (1, 3, 5, 7, 9, 11)

# Twelve hour windows in a year, the paper's scale from a window to a year
WINDOWS_PER_YEAR = 365.0 * 24.0 / 12.0
# Data units of the model per gigabit
DATA_PER_GB = 1000.0

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 8,
        "text.color": INK,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

ROWS_1 = [
    ("Total data [Gb]", lambda s, pt: np.asarray(s[f"{pt}.data_t"]) / DATA_PER_GB),
    ("Comm. power [W]", lambda s, pt: np.asarray(s[f"{pt}.P_comm"])),
    ("Transmitter gain", lambda s, pt: np.asarray(s[f"{pt}.gain"])),
    ("Roll angle [deg]", lambda s, pt: np.degrees(np.asarray(s[f"{pt}.gamma"]))),
    ("State of charge [%]", lambda s, pt: 100.0 * np.asarray(s[f"{pt}.SOC"])),
]
ROWS_2 = [
    ("Solar power [W]", lambda s, pt: np.asarray(s[f"{pt}.P_sol"])),
    ("Max. panel curr. [A]", lambda s, pt: np.asarray(s[f"{pt}.Isetpt"]).max(axis=1)),
    ("Mean panel exp. area [1e-4 m2]", lambda s, pt: 1e4 * np.asarray(s[f"{pt}.A_exp"]).mean(axis=1)),
    ("Body temp. [K]", lambda s, pt: np.asarray(s[f"{pt}.T"])[:, 4]),
    ("Battery current [A]", lambda s, pt: np.asarray(s[f"{pt}.I_bat"])),
]


def load(src):
    """The saved solutions by name, whichever files exist."""
    out = {}
    for name in ("initial",) + PROBLEMS:
        path = src / f"cadre_{name}.npz"
        if path.exists():
            out[name] = np.load(path)
    return out


def hours(s):
    return np.asarray(s["t"]) / 3600.0


def gb_per_year(s):
    """The paper's annual estimate, the mean window scaled to a year."""
    return float(s["total_data"]) / 6.0 / DATA_PER_GB * WINDOWS_PER_YEAR


def time_axis(ax, last_row, month):
    ax.set_xlim(0.0, 12.0)
    ax.set_xticks([0, 3, 6, 9, 12])
    if last_row:
        ax.set_xlabel(f"Time [hr]\n(month {month})")
    else:
        ax.tick_params(labelbottom=False)


def legend(fig, names, y):
    handles = [Line2D([], [], color=SERIES[name], linewidth=1.6) for name in names]
    fig.legend(handles, [LABELS[name] for name in names], loc="lower center", ncol=len(names), bbox_to_anchor=(0.5, y))


def profiles(solutions, rows, path, title):
    """Six windows across, the line of sight strips on top, and one quantity per row for every solution."""
    names = [name for name in ("initial",) + PROBLEMS if name in solutions]
    reference = solutions[names[-1]]
    t = hours(reference)
    fig = plt.figure(figsize=(16, 1.4 + 1.6 * len(rows)))
    grid = fig.add_gridspec(len(rows) + 1, 6, height_ratios=[0.3] + [1.0] * len(rows), hspace=0.15, wspace=0.08)

    # The sun and ground station lines of sight of every window as two strips
    for c in range(6):
        pt = f"pt{c}"
        ax = fig.add_subplot(grid[0, c])
        ax.fill_between(t, 1.1, 1.9, where=np.asarray(reference[f"{pt}.LOS"]) > 0.5, color=GRID, step="mid", linewidth=0)
        ax.fill_between(t, 0.1, 0.9, where=np.asarray(reference[f"{pt}.CommLOS"]) > 0.5, color=INK_MUTED, step="mid", linewidth=0)
        ax.set_xlim(0.0, 12.0)
        ax.set_ylim(0.0, 2.0)
        ax.grid(False)
        ax.tick_params(labelbottom=False, bottom=False, left=False)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["GS LOS", "Sun LOS"] if c == 0 else [])
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)

    optimized = [name for name in names if name != "initial"]
    for r, (label, series) in enumerate(rows):
        first = None
        for c in range(6):
            pt = f"pt{c}"
            ax = fig.add_subplot(grid[r + 1, c], sharey=first)
            first = first or ax
            for name in names:
                s = solutions[name]
                ax.plot(hours(s), series(s, pt), color=SERIES[name], linewidth=1.0, solid_joinstyle="round")
            time_axis(ax, r == len(rows) - 1, MONTHS[c])
            if c == 0:
                ax.set_ylabel(label)
            else:
                ax.tick_params(labelleft=False)

        # The range follows the optimized solutions and the initial design may leave the frame
        if optimized:
            values = np.concatenate([series(solutions[name], f"pt{c}") for name in optimized for c in range(6)])
            low, high = float(values.min()), float(values.max())
            pad = 0.05 * (high - low) if high > low else 1.0
            first.set_ylim(low - pad, high + pad)

    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    legend(fig, names, 0.0)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def data_per_month(solutions, path):
    """The data of every window and their mean for the problems, grouped by month."""
    names = [name for name in PROBLEMS if name in solutions]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(7)
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        s = solutions[name]
        values = [float(s[f"pt{i}.data"]) / DATA_PER_GB for i in range(6)]
        values.append(float(np.mean(values)))
        ax.bar(x + (k - (len(names) - 1) / 2.0) * width, values, 0.9 * width, color=SERIES[name], label=LABELS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in MONTHS] + ["Avg."])
    ax.set_xlabel("Month")
    ax.set_ylabel("Data downloaded [Gb]")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig.suptitle("Division of the data downloaded over the six windows", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def convergence(s, path):
    """The history of the full solve and the data and charge of the saved iterates of every window."""
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(3, 6, height_ratios=[1.5, 1.0, 1.0], hspace=0.45, wspace=0.08)
    top = grid[0, :].subgridspec(1, 2, wspace=0.25)
    it = np.arange(len(s["hist_inf_pr"]))

    ax = fig.add_subplot(top[0, 0])
    ax.semilogy(it, np.asarray(s["hist_inf_du"]), color=SERIES["full"], linewidth=1.2, label="Optimality")
    ax.semilogy(it, np.asarray(s["hist_inf_pr"]), color=SERIES["geometry"], linewidth=1.2, label="Feasibility")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Convergence metric")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(top[0, 1])
    ax.plot(it, np.asarray(s["hist_objective"]) / DATA_PER_GB, color=SERIES["full"], linewidth=1.2)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Objective with the elastic penalty [Gb]")

    # The paper's iterates and the last saved one, light to dark
    saved = [int(k) for k in np.asarray(s["snap_iters"])]
    picks = [k for k in (0, 50, 100, 150, 200, 300) if k in saved]
    if saved[-1] not in picks:
        picks.append(saved[-1])
    colors = SEQUENTIAL[-len(picks) :]
    rows = (("Total data [Gb]", np.asarray(s["snap_data"]) / DATA_PER_GB), ("State of charge [%]", 100.0 * np.asarray(s["snap_SOC"])))
    t = hours(s)
    for r, (label, values) in enumerate(rows):
        first = None
        for c in range(6):
            ax = fig.add_subplot(grid[r + 1, c], sharey=first)
            first = first or ax
            for k, color in zip(picks, colors):
                ax.plot(t, values[saved.index(k), c], color=color, linewidth=1.0)
            time_axis(ax, r == 1, MONTHS[c])
            if c == 0:
                ax.set_ylabel(label)
            else:
                ax.tick_params(labelleft=False)
    handles = [Line2D([], [], color=color, linewidth=1.6) for color in colors]
    fig.legend(handles, [f"Iteration {k}" for k in picks], loc="lower center", ncol=len(picks), bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.97, bottom=0.12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(solutions):
    """The design and the annual data of the problems as the paper's table."""
    print(f"{'problem':10s} {'fin [deg]':>10s} {'antenna [deg]':>14s} {'data [Gb/yr]':>13s} {'iterations':>11s}")
    for name in PROBLEMS:
        if name in solutions:
            s = solutions[name]
            print(
                f"{name:10s} {float(s['fin_angle_deg']):10.1f} {float(s['antenna_angle_deg']):14.1f}"
                f" {gb_per_year(s):13.0f} {int(s['iterations']):11d}"
            )


if __name__ == "__main__":
    src = Path(__file__).resolve().parent / "results"
    solutions = load(src)
    if "full" in solutions and "snap_iters" in solutions["full"]:
        convergence(solutions["full"], src / "cadre_convergence.png")
        print(f"Saved {src / 'cadre_convergence.png'}")
    if any(name in solutions for name in PROBLEMS):
        data_per_month(solutions, src / "cadre_data_per_month.png")
        profiles(solutions, ROWS_1, src / "cadrefig1.png", "Operation of the CADRE design over the six windows")
        profiles(solutions, ROWS_2, src / "cadrefig2.png", "Power and thermal state of the CADRE design over the six windows")
        for name in ("cadre_data_per_month.png", "cadrefig1.png", "cadrefig2.png"):
            print(f"Saved {src / name}")
        table(solutions)
