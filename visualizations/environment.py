"""Illumination, panel exposure, and the spacecraft's thermal response."""

import numpy as np

from solar.solar_mbi_validation import build_solar_mbi, solar_exposed_area_mbi
from sun.sun_reference import (
    sun_position_eci,
    sun_line_of_sight,
    sun_position_body,
    sun_position_spherical,
)
from thermal.thermal_reference import propagate_temperature
from thermal.thermal_amigo import propagate_temperature_trapezoid
from visualizations.common import (
    BLUE,
    GOLD,
    TEAL,
    compare,
    eclipse,
    finish,
    maximum_error,
    panels,
    time_axes,
)


def sun(case, output):
    inertial = sun_position_eci(case.seconds, LD=float(case.value("LD").ravel()[0]))
    sunlight = sun_line_of_sight(case.value("r_e2b_I").T, inertial)
    body = sun_position_body(np.moveaxis(case.value("O_BI"), -1, 0), inertial)
    azimuth, elevation = sun_position_spherical(body)

    figure, axes = panels(3)
    compare(axes[0], case, case.value("LOS"), sunlight, "Sunlight fraction")
    axes[0].set_ylim(-0.05, 1.08)
    axes[0].set_title("Sunlight fraction: 0 = eclipse, 1 = full sunlight")
    axes[1].plot(case.hours, np.rad2deg(azimuth), label="Azimuth", color=BLUE)
    axes[1].plot(
        case.hours, np.rad2deg(elevation), label="Elevation (from +z)", color=GOLD
    )
    axes[1].set_ylabel("Body-frame angle [deg]")
    axes[1].set_title("Sun direction in the spacecraft body frame")
    axes[1].legend()
    eclipse(axes[1], case)
    axes[2].plot(case.hours, sunlight - case.value("LOS"), color=TEAL)
    axes[2].set_ylabel("Sunlight difference")
    axes[2].set_title("Sunlight reconstruction error: Python minus CADRE")
    time_axes(axes)

    finish(figure, output, "sun", "Solar illumination and Sun direction")

    return {"sunlight max difference": maximum_error(sunlight, case.value("LOS"))}


def solar(case, output):
    source = case.value("exposedArea")
    reference = solar_exposed_area_mbi(
        float(case.value("finAngle").ravel()[0]),
        case.value("azimuth"),
        case.value("elevation"),
        build_solar_mbi(),
    )
    geometric = source.sum(axis=0)
    illuminated = geometric * case.value("LOS")[None, :]

    figure, axes = panels(3, (13, 10))
    maximum = max(float(geometric.max()), float(illuminated.max()))

    for axis, values, title in zip(
        axes[:2],
        [geometric, illuminated],
        [
            "Geometric exposed area by panel",
            "Illuminated area by panel",
        ],
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=[0, 12, 0.5, 12.5],
            cmap="viridis",
            vmin=0,
            vmax=maximum,
        )
        axis.set_yticks(np.arange(1, 13))
        axis.set_ylabel("Panel")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, label="Area [m²]")

    axes[2].plot(
        case.hours, np.max(np.abs(reference - source), axis=(0, 1)), color=TEAL
    )
    axes[2].set_ylabel("Max cell difference [m²]")
    axes[2].set_title("Exposed-area lookup error: largest cell difference")
    time_axes(axes)

    finish(
        figure,
        output,
        "solar",
        "Solar-panel exposure and illumination",
        "Each panel contains seven cells. Both heatmaps share the same area scale; illuminated area equals geometric area × sunlight fraction.",
    )

    return {"exposed-area max difference [m²]": maximum_error(reference, source)}


def thermal(case, output):
    inputs = (
        np.full(5, 273.0),
        case.value("exposedArea"),
        case.value("LOS"),
        case.value("P_comm"),
        case.value("cellInstd"),
        case.dt,
    )
    reference = propagate_temperature(*inputs)
    trapezoid, _ = propagate_temperature_trapezoid(*inputs)
    source = case.value("temperature")

    figure, axes = panels(3)
    labels = ["Fin 1", "Fin 2", "Fin 3", "Fin 4", "Body"]

    for index, label in enumerate(labels):
        axes[0].plot(case.hours, reference[index], label=label)

    axes[0].set_ylabel("Temperature [K]")
    axes[0].set_title("Five thermal states: four fins and the body")
    axes[0].legend(ncol=5)
    eclipse(axes[0], case)
    axes[1].plot(case.hours, np.max(np.abs(reference - source), axis=0), color=TEAL)
    axes[1].set_ylabel("Max state difference [K]")
    axes[1].set_title("RK4 reconstruction error: largest state difference")
    axes[2].plot(case.hours, np.max(np.abs(trapezoid - reference), axis=0), color=GOLD)
    axes[2].set_ylabel("Max state difference [K]")
    axes[2].set_title("Trapezoidal versus RK4: largest state difference")
    time_axes(axes)

    finish(
        figure,
        output,
        "thermal",
        "Spacecraft temperatures and integration accuracy",
    )

    return {
        "RK4 temperature max difference [K]": maximum_error(reference, source),
        "trapezoid vs RK4 temperature max difference [K]": maximum_error(
            trapezoid, reference
        ),
    }
