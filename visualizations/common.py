"""Shared data loading and figure styling for subsystem reports."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cadre_paths import cadre_path

BLUE = "#0866b2"
TEAL = "#237b7b"
GOLD = "#bd7b00"
GRAY = "#6b7280"
AXIS_COLORS = [BLUE, TEAL, GOLD]


@dataclass
class Case:
    data: dict
    seconds: np.ndarray

    @property
    def hours(self):
        return self.seconds / 3600

    @property
    def dt(self):
        return self.seconds[1] - self.seconds[0]

    @property
    def n(self):
        return len(self.seconds)

    def value(self, name):
        key = f"0:{name}" if f"0:{name}" in self.data else name

        return np.asarray(self.data[key], dtype=float)


def load_case():
    with open(cadre_path("test/data1346.pkl"), "rb") as source:
        data = pickle.load(source, encoding="latin1")

    n = len(data["0:P_comm"])

    return Case(data, np.linspace(0, 43200, n))


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(
                color=[*AXIS_COLORS, "#9266b2", "#071a2f"]
            ),
            "axes.formatter.useoffset": False,
            "legend.fontsize": 8,
            "legend.framealpha": 0.95,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def panels(rows, figsize=None):
    return plt.subplots(
        rows,
        1,
        figsize=figsize or (11, 2.6 * rows),
        sharex=True,
        constrained_layout=True,
    )


def time_axes(axes):
    for axis in np.atleast_1d(axes):
        axis.grid(alpha=0.2)
        axis.set_xlim(0, 12)

    np.atleast_1d(axes)[-1].set_xlabel("Time [hours]")


def eclipse(axis, case):
    axis.fill_between(
        case.hours,
        0,
        1,
        where=case.value("LOS") < 1 - 1e-8,
        transform=axis.get_xaxis_transform(),
        color="#d9e2e8",
        alpha=0.5,
        label="Eclipse / illumination transition",
        zorder=0,
    )


def compare(axis, case, source, reference, ylabel):
    axis.plot(case.hours, source, color=GRAY, linewidth=2.5, label="CADRE benchmark")
    axis.plot(
        case.hours, reference, color=BLUE, linestyle="--", label="Python reference"
    )
    axis.set_ylabel(ylabel)
    axis.legend(loc="best")


def components(axis, case, values, ylabel):
    for index, color in enumerate(AXIS_COLORS):
        axis.plot(case.hours, values[index], color=color, label="xyz"[index])

    axis.set_ylabel(ylabel)
    axis.legend(ncol=3)


def finish(figure, output, name, title):
    figure.suptitle(title, fontsize=16)
    path = Path(output) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)

    return path


def maximum_error(reference, source):
    difference = np.asarray(reference) - np.asarray(source)
    assert np.all(np.isfinite(difference)), "Nonfinite reference comparison"

    return float(np.max(np.abs(difference)))
