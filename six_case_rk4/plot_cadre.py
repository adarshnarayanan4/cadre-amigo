"""
Plots of the CADRE solution
---------------------------
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "figures"

# Window of the operation figure, counted from zero
WINDOW = 3

# Colors of the curves, of the limits, and of the shaded passes and eclipses
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
LIMIT = "#3B2A7A"
PASS = "#C9C9C9"
ECLIPSE = "#EFEFEF"

# Twelve hour windows in a year and data units of the model per gigabit
WINDOWS_PER_YEAR = 730.0
DATA_PER_GB = 1000.0

# Month after launch of each window
MONTHS = (1, 3, 5, 7, 9, 11)

# The three problems of the reference with their names, colors, and markers
PROBLEMS = ("baseline", "geometry", "full")
NAMES = {
    "baseline": "baseline",
    "geometry": "geometry",
    "full": "geometry and attitude",
}
COLORS = {"baseline": GREEN, "geometry": ORANGE, "full": BLUE}
MARKERS = {"baseline": "^", "geometry": "s", "full": "o"}

# Fin angle, antenna angle, and annual data of the three problems in Table 4 of the reference
REFERENCE = {
    "baseline": (45.0, 0.0, 2122.0),
    "geometry": (63.8, -45.0, 2991.0),
    "full": (64.4, -45.0, 3758.0),
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.grid": False,
        "lines.linewidth": 1.3,
        "legend.frameon": True,
        "legend.edgecolor": "black",
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "savefig.bbox": "tight",
    }
)


def intervals_of(flag):
    """First and last index of every stretch of nodes where the flag is set"""
    on = flag > 0.5
    intervals = []
    first = None
    for k in range(len(on)):
        if on[k] and first is None:
            first = k
        if first is not None and (k == len(on) - 1 or not on[k + 1]):
            intervals.append((first, k))
            first = None
    return intervals


def shade_window(ax, hours, visible, sunlit):
    """Shade the eclipses of a window lightly and its ground station passes darker"""
    for interval in intervals_of(1.0 - sunlit):
        ax.axvspan(
            hours[interval[0]], hours[interval[1]], color=ECLIPSE, lw=0.0, zorder=0.0
        )
    for interval in intervals_of(visible):
        ax.axvspan(
            hours[interval[0]], hours[interval[1]], color=PASS, lw=0.0, zorder=0.5
        )


def draw_limit(ax, value, label, above):
    """Draw a limit as a dash dotted line with its name at the right edge"""
    ax.axhline(value, color=LIMIT, ls="-.", lw=1.2)
    ax.annotate(
        label,
        xy=(1.0, value),
        xycoords=ax.get_yaxis_transform(),
        xytext=(-4.0, 4.0 if above else -4.0),
        textcoords="offset points",
        ha="right",
        va="bottom" if above else "top",
        color=LIMIT,
        fontsize=9,
    )


def operation_figure(solution, window, target):
    """State of charge, power, and battery current of one window on a shared time axis"""
    point = f"pt{window}"
    hours = solution["t"] / 3600.0
    visible = solution[f"{point}.CommLOS"]
    sunlit = solution[f"{point}.LOS"]
    charge = solution[f"{point}.SOC"]
    data = solution[f"{point}.data_t"]
    solar = solution[f"{point}.P_sol"]
    comm = solution[f"{point}.P_comm"]
    current = solution[f"{point}.I_bat"]

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.5), sharex=True)
    labels = ("(a)", "(b)", "(c)")
    for k in range(3):
        axes[k].set_title(labels[k], loc="left", pad=9.0)
        axes[k].set_xlim(0.0, 12.0)
        axes[k].set_xticks(range(0, 13))
        shade_window(axes[k], hours, visible, sunlit)
    axes[2].set_xlabel("time (h)")

    # State of charge between its two limits
    axes[0].plot(hours, charge, color=BLUE)
    draw_limit(axes[0], 1.0, r"$\mathrm{SOC}_{\max}$", True)
    draw_limit(axes[0], 0.2, r"$\mathrm{SOC}_{\min}$", False)
    axes[0].set_ylim(0.0, 1.22)
    axes[0].set_yticks((0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    axes[0].set_ylabel("state of charge")

    # Power of the cells against the power sent to the transmitter
    axes[1].plot(hours, solar, color=GREEN, label="solar cells")
    axes[1].plot(hours, comm, color=BLUE, label="transmitter")
    axes[1].set_ylim(-0.3, 8.2)
    axes[1].set_ylabel("power (W)")
    axes[1].legend(loc="upper right", ncol=2)

    # Data that each pass downloads, written beside its burst
    for interval in intervals_of(visible):
        before = max(interval[0] - 1, 0)
        gained = (data[interval[1]] - data[before]) / DATA_PER_GB
        peak = comm[interval[0] : interval[1] + 1].max()
        axes[1].annotate(
            f"{gained:.2f} Gb",
            xy=(hours[interval[1]], peak),
            xytext=(4.0, -2.0),
            textcoords="offset points",
            ha="left",
            va="top",
            color=BLUE,
            fontsize=9,
        )

    # Battery current against its discharge limit
    axes[2].plot(hours, current, color=BLUE)
    draw_limit(axes[2], -10.0, r"$I_{\min}$", True)
    axes[2].set_ylim(-11.5, 2.5)
    axes[2].set_ylabel("battery current (A)")

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(target)
    plt.close(fig)


def log_axis(ax, values):
    """Span the values with whole decades and label every third decade from the lowest one they reach"""
    positive = values[values > 0.0]
    low = int(np.floor(np.log10(positive.min())))
    high = int(np.ceil(np.log10(positive.max())))

    # One unlabeled decade below the lowest label leaves room under the last iterate
    ax.set_ylim(10.0 ** (low - 1), 10.0**high)
    ax.set_yticks(10.0 ** np.arange(low, high + 1, 3))
    ax.set_yticks(10.0 ** np.arange(low - 1, high + 1), minor=True)
    ax.yaxis.set_minor_formatter(NullFormatter())


def convergence_figure(solution, target):
    """Constraint violation, dual infeasibility, and complementarity of every iteration"""
    violation = np.asarray(solution["hist_inf_pr"], dtype=float)
    dual = np.asarray(solution["hist_inf_du"], dtype=float)
    complementarity = np.asarray(solution["hist_compl"], dtype=float)
    barrier = np.asarray(solution["hist_barrier_param"], dtype=float)
    iterations = np.arange(len(violation))

    # The iteration axis runs a little past the last iteration so its values stay in view
    end = 1.03 * iterations[-1]

    # The panel of the complementarity is a little taller to hold its legend
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.4, 5.4),
        sharex=True,
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.25)},
    )
    labels = ("(a)", "(b)", "(c)")
    for k in range(3):
        axes[k].set_title(labels[k], loc="left", pad=9.0)
        axes[k].set_xlim(0, end)
    axes[2].set_xlabel("iteration")

    # Feasibility of the iterates
    axes[0].semilogy(iterations, violation, color=BLUE)
    axes[0].set_ylabel("constraint violation")
    log_axis(axes[0], violation)

    # Stationarity of the iterates
    axes[1].semilogy(iterations, dual, color=BLUE)
    axes[1].set_ylabel("dual infeasibility")
    log_axis(axes[1], dual)

    # Complementarity against the barrier parameter it follows
    axes[2].semilogy(iterations, complementarity, color=BLUE, label="complementarity")
    axes[2].semilogy(
        iterations, barrier, color=ORANGE, ls="--", label="barrier parameter"
    )
    axes[2].set_ylabel("complementarity")
    log_axis(axes[2], np.concatenate([complementarity, barrier]))

    # The legend sits in the empty corner under the early iterates
    axes[2].legend(loc="lower left")

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=1.6)
    fig.savefig(target)
    plt.close(fig)


def annual_data(data):
    """Annual data in Gb that the data of the six windows amounts to"""
    return data / 6.0 * WINDOWS_PER_YEAR / DATA_PER_GB


def problems_figure(solutions, target):
    """Annual data against the iteration and data of every window for the three problems"""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    axes[0].set_title("(a)", loc="left", pad=9.0)
    axes[1].set_title("(b)", loc="left", pad=9.0)

    longest = 0
    for problem in PROBLEMS:
        solution = solutions[problem]
        color = COLORS[problem]
        marker = MARKERS[problem]

        # Annual data of the iterates with the final value at the end of the curve
        history = annual_data(-solution["hist_objective"])
        iterations = np.arange(len(history))
        longest = max(longest, len(history))
        axes[0].plot(iterations, history, color=color, label=NAMES[problem])
        axes[0].plot(iterations[-1], history[-1], color=color, marker=marker, ms=4.0)
        axes[0].annotate(
            f"{history[-1]:.0f} Gb",
            xy=(iterations[-1], history[-1]),
            xytext=(6.0, 0.0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=9,
        )

        # Data of each window
        windows = np.zeros(6)
        for k in range(6):
            windows[k] = solution[f"pt{k}.data"] / DATA_PER_GB
        axes[1].plot(MONTHS, windows, color=color, marker=marker, ms=4.0)

    axes[0].set_xlim(0, 1.22 * longest)
    axes[0].set_ylim(0.0, 4700.0)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("annual data (Gb)")
    axes[0].legend(loc="lower right")

    axes[1].set_xticks(MONTHS)
    axes[1].set_xlim(0.2, 11.8)
    axes[1].set_ylim(0.0, 8.0)
    axes[1].set_xlabel("month of the window")
    axes[1].set_ylabel("data per window (Gb)")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(target)
    plt.close(fig)


def problems_table(solutions, target):
    """Design, annual data, iterations, and time of the three problems beside the reference as a LaTeX table"""
    lines = []
    lines.append(r"\begin{tabular}{lrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r" & \multicolumn{2}{c}{fin angle (deg)} & \multicolumn{2}{c}{antenna angle (deg)}"
        r" & \multicolumn{2}{c}{annual data (Gb)} & iterations & time (min) \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append(
        r"problem & Amigo & reference & Amigo & reference & Amigo & reference & & \\"
    )
    lines.append(r"\midrule")

    for problem in PROBLEMS:
        solution = solutions[problem]
        reference = REFERENCE[problem]
        fin = float(solution["fin_angle_deg"])
        antenna = int(round(float(solution["antenna_angle_deg"])))
        annual = annual_data(float(solution["total_data"]))
        minutes = float(solution["solve_time"]) / 60.0

        # The logged rows start at the initial point, so the iterations are one fewer
        iterations = len(solution["hist_objective"]) - 1

        # One row of the problem with the angles and the data of the reference beside its own
        row = f"{NAMES[problem]} & {fin:.1f} & {reference[0]:.1f}"
        row += f" & ${antenna}$ & ${int(reference[1])}$"
        row += f" & {annual:.0f} & {reference[2]:.0f}"
        row += f" & {iterations} & {minutes:.1f} " + r"\\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    text = "\n".join(lines) + "\n"
    target.write_text(text)
    print(text)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)

    # The saved solutions of the three problems
    solutions = {}
    for problem in PROBLEMS:
        solutions[problem] = np.load(
            RESULTS / f"cadre_{problem}.npz", allow_pickle=True
        )

    operation_figure(solutions["full"], WINDOW, OUT / "cadre_operation.pdf")
    convergence_figure(solutions["full"], OUT / "cadre_convergence.pdf")
    problems_figure(solutions, OUT / "cadre_problems.pdf")
    problems_table(solutions, OUT / "cadre_problems_table.tex")
    print(f"Saved the figures and the table in {OUT}")
