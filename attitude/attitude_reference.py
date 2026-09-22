import numpy as np
import matplotlib.pyplot as plt

from orbit.orbit_reference import (
    orbital_elements_to_state,
    rk4_step,
    calculate_orbital_period,
)

# Spacecraft inertia matrix from original CADRE Attitude_Torque
J = np.array(
    [
        [0.018, 0.0, 0.0],
        [0.0, 0.018, 0.0],
        [0.0, 0.0, 0.006],
    ]
)


# Build orbit trajectory
def build_orbit_trajectory(q0, dt, num_time_steps):
    """
    Propagate the spacecraft orbit with RK4.

    Returns
    -------
    states : (num_time_steps + 1, 6) [x, y, z, vx, vy, vz] at each time point.
    """

    states = np.zeros((num_time_steps + 1, 6))
    states[0] = q0

    for i in range(num_time_steps):
        states[i + 1] = rk4_step(states[i], dt)

    return states


# Cadre Attitute_Attitude
def attitude_from_orbit(states):
    """
    Create the base spacecraft orientation from orbital
    position and velocity.

    This recreates CADRE Attitude_Attitude.
    """

    n = states.shape[0]

    O_RI = np.zeros((n, 3, 3))

    for i in range(n):
        r = states[i, 0:3].copy()
        v = states[i, 3:6].copy()

        norm_r = np.linalg.norm(r)
        norm_v = np.linalg.norm(v)

        # Same protection used by CADRE
        if norm_r < 1.0e-10:
            norm_r = 1.0e-10

        if norm_v < 1.0e-10:
            norm_v = 1.0e-10

        r = r / norm_r
        v = v / norm_v

        # Cross-product matrix for velocity
        vx = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )

        iB = vx @ r
        jB = -(vx @ iB)

        O_RI[i, 0, :] = iB
        O_RI[i, 1, :] = jB
        O_RI[i, 2, :] = -v

    return O_RI


# CADRE Attitude_Roll
def attitude_roll(gamma):
    """
    Convert spacecraft roll angle into a rotation matrix.

    This recreates CADRE Attitude_Roll.
    """

    n = len(gamma)

    O_BR = np.zeros((n, 3, 3))

    for i in range(n):
        O_BR[i, 0, 0] = np.cos(gamma[i])
        O_BR[i, 0, 1] = np.sin(gamma[i])

        O_BR[i, 1, 0] = -np.sin(gamma[i])
        O_BR[i, 1, 1] = np.cos(gamma[i])

        O_BR[i, 2, 2] = 1.0

    return O_BR


# CADRE Attitude_RotationMtx
def combine_rotation_matrices(O_BR, O_RI):
    """
    Combine the base orbital orientation with spacecraft roll.

    O_BI = O_BR @ O_RI
    """

    n = O_RI.shape[0]

    O_BI = np.zeros((n, 3, 3))

    for i in range(n):
        O_BI[i] = O_BR[i] @ O_RI[i]

    return O_BI


# CADRE Attitude_RotationMtxRates
def rotation_matrix_rates(O_BI, dt):
    """
    Calculate time derivative of orientation matrix using
    the same finite differences as CADRE.
    """

    n = O_BI.shape[0]

    Odot_BI = np.zeros_like(O_BI)

    # Forward difference at first point
    Odot_BI[0] = (O_BI[1] - O_BI[0]) / dt

    # Central difference at interior points
    Odot_BI[1:-1] = (O_BI[2:] - O_BI[:-2]) / (2.0 * dt)

    # Backward difference at last point
    Odot_BI[-1] = (O_BI[-1] - O_BI[-2]) / dt

    return Odot_BI


# CADRE Attitude_Angular
def angular_velocity(O_BI, Odot_BI):
    """
    Calculate body-frame angular velocity.

    This recreates CADRE Attitude_Angular.
    """

    n = O_BI.shape[0]

    w_B = np.zeros((n, 3))

    for i in range(n):
        w_B[i, 0] = np.dot(Odot_BI[i, 2, :], O_BI[i, 1, :])
        w_B[i, 1] = np.dot(Odot_BI[i, 0, :], O_BI[i, 2, :])
        w_B[i, 2] = np.dot(Odot_BI[i, 1, :], O_BI[i, 0, :])

    return w_B


# CADRE Attitude_AngularRates
def angular_acceleration(w_B, dt):
    """
    Calculate angular acceleration using CADRE's
    finite-difference scheme.
    """

    wdot_B = np.zeros_like(w_B)

    # Forward difference
    wdot_B[0] = (w_B[1] - w_B[0]) / dt

    # Central difference
    wdot_B[1:-1] = (w_B[2:] - w_B[:-2]) / (2.0 * dt)

    # Backward difference
    wdot_B[-1] = (w_B[-1] - w_B[-2]) / dt

    return wdot_B


# CADRE Attitude_Sideslip
def body_frame_velocity(states, O_BI):
    """
    Transform inertial velocity into spacecraft body frame.

    This recreates CADRE's computepositionrotd operation.
    """

    n = states.shape[0]

    v_B = np.zeros((n, 3))

    for i in range(n):
        v_I = states[i, 3:6]
        v_B[i] = O_BI[i] @ v_I

    return v_B


# CADRE Attitude_Torque
def attitude_torque(w_B, wdot_B):
    """
    Calculate required spacecraft torque.

    T = J * wdot + w x (J * w)
    """

    n = w_B.shape[0]

    T_tot = np.zeros((n, 3))

    for i in range(n):
        w = w_B[i]
        wx = np.array(
            [
                [0.0, -w[2], w[1]],
                [w[2], 0.0, -w[0]],
                [-w[1], w[0], 0.0],
            ]
        )
        T_tot[i] = J @ wdot_B[i] + wx @ (J @ w)

    return T_tot


# Main
def main():
    # Initial orbit
    r0, v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    q0 = np.concatenate((r0, v0))

    # Time grid
    orbital_period = calculate_orbital_period(
        alt_perigee=500.0, alt_apogee=500.0
    )

    num_time_steps = 568

    dt = orbital_period / num_time_steps

    times = np.linspace(0.0, orbital_period, num_time_steps + 1)

    # Orbit trajectory
    states = build_orbit_trajectory(q0, dt, num_time_steps)

    # Roll command
    # Start with zero commanded roll everywhere.
    gamma = np.zeros(num_time_steps + 1)

    # Attitude calculations
    O_RI = attitude_from_orbit(states)
    O_BR = attitude_roll(gamma)
    O_BI = combine_rotation_matrices(O_BR, O_RI)
    Odot_BI = rotation_matrix_rates(O_BI, dt)

    w_B = angular_velocity(O_BI, Odot_BI)
    wdot_B = angular_acceleration(w_B, dt)

    v_B = body_frame_velocity(states, O_BI)

    T_tot = attitude_torque(w_B, wdot_B)

    # Sanity checks
    max_orthogonality_error = 0.0

    for i in range(len(O_BI)):
        error = np.max(np.abs(O_BI[i] @ O_BI[i].T - np.eye(3)))

        max_orthogonality_error = max(max_orthogonality_error, error)

    print()
    print("=" * 70)
    print("CADRE ATTITUDE REFERENCE")
    print("=" * 70)

    print()
    print("Number of time points:")
    print(num_time_steps + 1)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Maximum rotation-matrix orthogonality error:")
    print(max_orthogonality_error)

    print()
    print("Maximum angular velocity magnitude [1/s]:")
    print(np.max(np.linalg.norm(w_B, axis=1)))

    print()
    print("Maximum angular acceleration magnitude [1/s^2]:")
    print(np.max(np.linalg.norm(wdot_B, axis=1)))

    print()
    print("Maximum torque magnitude [N*m]:")
    print(np.max(np.linalg.norm(T_tot, axis=1)))

    print()
    print("Maximum body-frame lateral velocity:")
    print(np.max(np.linalg.norm(v_B[:, 0:2], axis=1)))

    # Angular velocity plot
    plt.figure(figsize=(9, 5))

    plt.plot(times / 60.0, w_B[:, 0], label="wx")
    plt.plot(times / 60.0, w_B[:, 1], label="wy")
    plt.plot(times / 60.0, w_B[:, 2], label="wz")

    plt.xlabel("Time [min]")
    plt.ylabel("Angular Velocity [1/s]")

    plt.title("CADRE Attitude Angular Velocity")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Angular acceleration plot
    plt.figure(figsize=(9, 5))

    plt.plot(times / 60.0, wdot_B[:, 0], label="wx dot")
    plt.plot(times / 60.0, wdot_B[:, 1], label="wy dot")
    plt.plot(times / 60.0, wdot_B[:, 2], label="wz dot")

    plt.xlabel("Time [min]")
    plt.ylabel("Angular Acceleration [1/s^2]")

    plt.title("CADRE Attitude Angular Acceleration")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    # Torque plot
    plt.figure(figsize=(9, 5))

    plt.plot(times / 60.0, T_tot[:, 0], label="Tx")
    plt.plot(times / 60.0, T_tot[:, 1], label="Ty")
    plt.plot(times / 60.0, T_tot[:, 2], label="Tz")

    plt.xlabel("Time [min]")
    plt.ylabel("Torque [N m]")

    plt.title("CADRE Required Attitude Torque")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
