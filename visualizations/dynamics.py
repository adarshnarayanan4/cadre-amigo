"""Orbit geometry, spacecraft attitude, and reaction-wheel response."""

import matplotlib.pyplot as plt
import numpy as np

from attitude.attitude_reference import (
    attitude_from_orbit,
    attitude_roll,
    combine_rotation_matrices,
    rotation_matrix_rates,
    angular_velocity,
    angular_acceleration,
    attitude_torque,
)
from orbit.orbit_reference import RE, rk4_step
from reaction_wheel.reaction_wheel_reference import (
    reaction_wheel_torque,
    propagate_reaction_wheel,
    propagate_reaction_wheel_trapezoid,
    reaction_wheel_power,
)
from visualizations.common import (
    BLUE,
    TEAL,
    GOLD,
    compare,
    components,
    finish,
    maximum_error,
    panels,
    time_axes,
)


def orbit(case, output):
    source = case.value("r_e2b_I").T
    reference = np.empty_like(source)
    reference[0] = source[0]

    for node in range(case.n - 1):
        reference[node + 1] = rk4_step(reference[node], case.dt)

    figure = plt.figure(figsize=(13, 7), constrained_layout=True)
    layout = figure.add_gridspec(2, 2, width_ratios=[1.2, 1])
    geometry = figure.add_subplot(layout[:, 0], projection="3d")
    altitude = figure.add_subplot(layout[0, 1])
    error_axis = figure.add_subplot(layout[1, 1], sharex=altitude)

    longitude = np.linspace(0, 2 * np.pi, 60)
    colatitude = np.linspace(0, np.pi, 30)
    geometry.plot_surface(
        RE * np.outer(np.cos(longitude), np.sin(colatitude)),
        RE * np.outer(np.sin(longitude), np.sin(colatitude)),
        RE * np.outer(np.ones_like(longitude), np.cos(colatitude)),
        color="#80b9cb",
        alpha=0.35,
        linewidth=0,
    )
    geometry.plot(*source[:, :3].T, color=BLUE, linewidth=1, label="CADRE trajectory")
    geometry.scatter(*source[0, :3], color=GOLD, s=35, label="Start")
    geometry.set(xlabel="ECI x [km]", ylabel="ECI y [km]", zlabel="ECI z [km]")
    radius = np.max(np.linalg.norm(source[:, :3], axis=1))
    geometry.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    geometry.set_box_aspect((1, 1, 1))
    geometry.legend(loc="upper left")

    compare(
        altitude,
        case,
        np.linalg.norm(source[:, :3], axis=1) - RE,
        np.linalg.norm(reference[:, :3], axis=1) - RE,
        "Altitude [km]",
    )
    position_error = np.linalg.norm(reference[:, :3] - source[:, :3], axis=1)
    error_axis.plot(case.hours, position_error, color=TEAL)
    error_axis.set_ylabel("Position difference [km]")
    time_axes([altitude, error_axis])

    finish(figure, output, "orbit", "Orbit: the trajectory and its reconstruction")

    return {"RK4 position difference max [km]": float(position_error.max())}


def attitude(case, output):
    orientation = combine_rotation_matrices(
        attitude_roll(case.value("gamma")), attitude_from_orbit(case.value("r_e2b_I").T)
    )
    rate = rotation_matrix_rates(orientation, case.dt)
    velocity = angular_velocity(orientation, rate)
    acceleration = angular_acceleration(velocity, case.dt)
    torque = attitude_torque(velocity, acceleration)

    figure, axes = panels(4)
    axes[0].plot(case.hours, np.rad2deg(case.value("gamma")), color=BLUE)
    axes[0].set_ylabel("Prescribed roll [deg]")
    components(axes[1], case, velocity.T, "Angular velocity [1/s]")
    components(axes[2], case, torque.T, "Required torque [N m]")
    axes[3].plot(case.hours, np.max(np.abs(velocity.T - case.value("w_B")), axis=0))
    axes[3].set_ylabel("Max axis difference [1/s]")
    axes[3].set_title("Python angular velocity minus the CADRE benchmark")
    time_axes(axes)

    finish(
        figure,
        output,
        "attitude",
        "Attitude: roll sets the motion the wheels must support",
    )

    return {
        "angular velocity max difference [1/s]": maximum_error(
            velocity.T, case.value("w_B")
        ),
        "torque max difference [N m]": maximum_error(torque.T, case.value("T_tot")),
    }


def reaction_wheel(case, output):
    torque = reaction_wheel_torque(case.value("T_tot"))
    speed = propagate_reaction_wheel(np.zeros(3), case.value("w_B"), torque, case.dt)
    trapezoid, _ = propagate_reaction_wheel_trapezoid(
        np.zeros(3), case.value("w_B"), torque, case.dt
    )
    power = reaction_wheel_power(speed, torque)

    figure, axes = panels(4)
    components(axes[0], case, torque, "Wheel torque [N m]")
    components(axes[1], case, speed, "Wheel speed [1/s]")
    compare(
        axes[2],
        case,
        case.value("P_RW").sum(axis=0),
        power.sum(axis=0),
        "Total wheel power [W]",
    )
    axes[2].set_ylim(
        0, 1.1 * max(case.value("P_RW").sum(axis=0).max(), power.sum(axis=0).max())
    )
    axes[2].set_title("The total wheel load is nearly constant for this benchmark")

    axes[3].plot(case.hours, np.linalg.norm(trapezoid - speed, axis=0), color=GOLD)
    axes[3].set_ylabel("Speed difference [1/s]")
    axes[3].set_title("Trapezoidal vs RK4 integration: vector difference magnitude")
    time_axes(axes)

    finish(
        figure,
        output,
        "reaction_wheel",
        "Reaction wheels: attitude control consumes electrical power",
    )

    return {
        "RK4 speed max difference [1/s]": maximum_error(speed, case.value("w_RW")),
        "wheel power max difference [W]": maximum_error(power, case.value("P_RW")),
        "trapezoid vs RK4 speed max difference [1/s]": maximum_error(trapezoid, speed),
    }
