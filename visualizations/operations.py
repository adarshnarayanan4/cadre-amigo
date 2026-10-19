"""Energy balance, storage, communication opportunities, and spline controls."""

import matplotlib.pyplot as plt
import numpy as np

from battery.battery_reference import (
    propagate_soc,
    propagate_soc_trapezoid,
    battery_current,
)
from communication.communication_data_validation import (
    communication_bit_rate,
    propagate_data_rk4,
)
from parameters.parameters_reference import bspline_parameters
from power.power_amigo import solar_power, total_power
from visualizations.common import (
    BLUE,
    GOLD,
    GRAY,
    TEAL,
    compare,
    eclipse,
    finish,
    maximum_error,
    panels,
    time_axes,
)


def power(case, output):
    solar = solar_power(case.value("V_sol"), case.value("Isetpt"))
    battery = total_power(solar, case.value("P_comm"), case.value("P_RW"))
    communication_load = 5 * case.value("P_comm")
    wheel_load = case.value("P_RW").sum(axis=0)
    demand = communication_load + wheel_load + 2

    figure, axes = panels(3)
    axes[0].plot(case.hours, solar, color=TEAL, label="Solar supply")
    axes[0].plot(case.hours, demand, color=GOLD, label="Total electrical demand")
    axes[0].plot(
        case.hours,
        communication_load,
        color=BLUE,
        alpha=0.7,
        label="Communication electrical load (5 × P_comm)",
    )
    axes[0].set_ylabel("Power [W]")
    axes[0].legend()
    eclipse(axes[0], case)
    compare(axes[1], case, case.value("P_bat"), battery, "Net battery power [W]")
    axes[1].axhline(0, color=GRAY, linewidth=0.8)
    axes[1].set_title("Positive = charging; negative = battery supplies the shortfall")
    axes[2].plot(case.hours, battery - case.value("P_bat"), color=TEAL)
    axes[2].set_ylabel("Reconstruction error [W]")
    time_axes(axes)

    finish(
        figure,
        output,
        "power",
        "Power balance: solar supply must cover spacecraft demand",
    )

    return {
        "solar power max difference [W]": maximum_error(solar, case.value("P_sol")),
        "battery power max difference [W]": maximum_error(battery, case.value("P_bat")),
    }


def battery(case, output):
    inputs = (
        float(case.value("iSOC").ravel()[0]),
        case.value("P_bat"),
        case.value("temperature"),
        case.dt,
    )
    reference = propagate_soc(*inputs)
    trapezoid = propagate_soc_trapezoid(*inputs)
    current = battery_current(
        case.value("P_bat"), reference, case.value("temperature")[4]
    )

    figure, axes = panels(3)
    compare(axes[0], case, case.value("SOC").ravel(), reference, "Battery SOC")
    axes[0].plot(
        case.hours, trapezoid, color=GOLD, linestyle=":", label="Python trapezoid"
    )
    axes[0].axhline(0.2, color=GRAY, linestyle="--", linewidth=0.8)
    axes[0].axhline(1, color=GRAY, linestyle="--", linewidth=0.8)
    axes[0].legend()
    eclipse(axes[0], case)
    axes[1].plot(case.hours, current, color=BLUE)
    axes[1].axhline(
        -10, color=GOLD, linestyle="--", label="Integrated optimization limits"
    )
    axes[1].axhline(5, color=GOLD, linestyle="--")
    axes[1].axhline(0, color=GRAY, linewidth=0.8)
    axes[1].set_ylabel("Battery current [A]")
    axes[1].set_title(
        "Positive = charging; the original benchmark is not repaired for current limits"
    )
    axes[1].legend()
    axes[2].plot(case.hours, trapezoid - reference, color=GOLD)
    axes[2].set_ylabel("SOC difference")
    axes[2].set_title(
        "Trapezoidal minus RK4 integration, using the same prescribed power"
    )
    time_axes(axes)

    finish(
        figure,
        output,
        "battery",
        "Battery storage: charging and discharge accumulate into SOC",
    )

    return {
        "RK4 SOC max difference": maximum_error(reference, case.value("SOC").ravel()),
        "trapezoid vs RK4 SOC max difference": maximum_error(trapezoid, reference),
    }


def communication(case, output):
    rate = communication_bit_rate(
        case.value("P_comm"),
        case.value("gain"),
        case.value("GSdist"),
        case.value("CommLOS"),
    )
    downloaded = propagate_data_rk4(0, rate, case.dt)

    figure, axes = panels(4)
    axes[0].fill_between(case.hours, 0, case.value("CommLOS"), color=TEAL)
    axes[0].set_ylabel("Ground contact")
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(case.hours, case.value("GSdist"), color=BLUE)
    axes[1].set_ylabel("Station distance [km]")
    compare(axes[2], case, case.value("Dr"), rate, "Data rate [CADRE units/s]")
    compare(
        axes[3],
        case,
        case.value("Data").ravel(),
        downloaded,
        "Downloaded [CADRE units]",
    )
    time_axes(axes)

    finish(
        figure,
        output,
        "communication",
        "Communication: data arrives during brief ground-station contacts",
    )

    return {
        "data rate max difference [CADRE units/s]": maximum_error(
            rate, case.value("Dr")
        ),
        "downloaded data max difference [CADRE units]": maximum_error(
            downloaded, case.value("Data").ravel()
        ),
    }


def parameters(case, output):
    controls = [
        case.value("CP_P_comm"),
        case.value("CP_gamma"),
        case.value("CP_Isetpt")[0],
    ]
    power, roll, currents = bspline_parameters(
        controls[0], controls[1], case.value("CP_Isetpt"), case.n
    )
    signals = [power, roll, currents[0]]
    sources = [case.value("P_comm"), case.value("gamma"), case.value("Isetpt")[0]]
    labels = ["Communication power [W]", "Roll [rad]", "Panel 1 current [A]"]

    figure, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)

    for row, (control, signal, source, label) in enumerate(
        zip(controls, signals, sources, labels)
    ):
        axes[row, 0].plot(
            np.arange(len(control)), control, color=GOLD, marker=".", markersize=2
        )
        axes[row, 0].set_ylabel(label)
        axes[row, 0].set_xlabel("Control-point index")
        axes[row, 0].grid(alpha=0.2)
        compare(axes[row, 1], case, source, signal, label)
        time_axes([axes[row, 1]])

    axes[0, 0].set_title("Design weights: not samples of the trajectory")
    axes[0, 1].set_title(
        f"Cubic B-spline trajectories evaluated at {case.n:,} time nodes"
    )

    finish(
        figure,
        output,
        "parameters",
        "Spline controls: a compact set of weights produces smooth schedules",
    )

    return {
        "communication power max difference [W]": maximum_error(power, sources[0]),
        "roll max difference [rad]": maximum_error(roll, sources[1]),
        "panel current max difference [A]": maximum_error(
            currents, case.value("Isetpt")
        ),
    }
