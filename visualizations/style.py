"""Common presentation for CADRE figures, without choosing a display backend."""

from pathlib import Path
import textwrap

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

BLUE = "#0866b2"
TEAL = "#237b7b"
GOLD = "#bd7b00"
GRAY = "#667085"
RED = "#b04238"
COLORS = [BLUE, TEAL, GOLD, "#8261a8", "#25364a"]


def configure_style():
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=COLORS),
            "axes.formatter.useoffset": False,
            "axes.formatter.limits": (-3, 4),
            "lines.linewidth": 1.6,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#dce2e8",
            "grid.linewidth": 0.7,
        }
    )


def format_axes(axis):
    """Keep subplot titles, ticks, and legends clear of the plotted data."""
    title = axis.get_title(loc="left") or axis.get_title()
    axis.set_title("")
    axis.set_title(title, loc="left", pad=12)

    if axis.name != "3d" and not axis.images:
        axis.grid(True, axis="both", which="major")
        axis.set_axisbelow(True)
        axis.margins(x=0.01, y=0.08)
    elif axis.images:
        axis.grid(False)

    handles, labels = axis.get_legend_handles_labels()
    entries = dict(zip(labels, handles))

    if entries:
        axis.legend(
            entries.values(),
            entries.keys(),
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
        )


def finish_figure(figure, title, subtitle="", note="", shared_legend=False):
    """Reserve a header and footer; let constrained layout fit the plot content."""
    height = figure.get_figheight()
    width = figure.get_figwidth()

    for axis in figure.axes:
        if axis.get_label() != "<colorbar>":
            format_axes(axis)

    header = 0.85 if subtitle else 0.55

    if shared_legend:
        entries = {}

        for axis in figure.axes:
            handles, labels = axis.get_legend_handles_labels()
            entries.update(zip(labels, handles))

            if axis.get_legend():
                axis.get_legend().remove()

        if entries:
            figure.legend(
                entries.values(),
                entries.keys(),
                loc="upper left",
                bbox_to_anchor=(0.06, 1 - (0.64 if subtitle else 0.34) / height),
                ncol=min(3, len(entries)),
                borderaxespad=0,
            )
            header += 0.25 * ((len(entries) + 2) // 3)

    wrapped_note = textwrap.fill(note, width=int(width * 13))
    footer = 0.22 + 0.16 * len(wrapped_note.splitlines()) if note else 0.15
    figure.set_layout_engine(
        "constrained",
        rect=(0.02, footer / height, 0.96, 1 - (header + footer) / height),
        h_pad=0.12,
        w_pad=0.12,
        hspace=0.08,
        wspace=0.08,
    )
    figure.text(0.06, 1 - 0.12 / height, title, va="top", fontsize=16, weight="bold")

    if subtitle:
        figure.text(
            0.06, 1 - 0.45 / height, subtitle, va="top", fontsize=10, color=GRAY
        )

    if note:
        figure.text(
            0.06,
            0.10 / height,
            wrapped_note,
            va="bottom",
            fontsize=9,
            color=GRAY,
        )


def save_figure(figure, output, name, title, subtitle="", note="", shared_legend=False):
    from matplotlib import pyplot as plt

    finish_figure(figure, title, subtitle, note, shared_legend)
    path = Path(output) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(figure)

    return path


def integer_ticks(axis):
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
