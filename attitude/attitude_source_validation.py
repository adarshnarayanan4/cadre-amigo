import numpy as np

from orbit.orbit_reference import (
    orbital_elements_to_state,
    rk4_step,
    calculate_orbital_period,
)

from attitude.attitude_reference import (
    attitude_from_orbit,
    attitude_roll,
    combine_rotation_matrices,
    rotation_matrix_rates,
    angular_velocity,
    angular_acceleration,
    body_frame_velocity,
    attitude_torque,
)

# Spacecraft inertia matrix from original CADRE
J = np.array(
    [
        [0.018, 0.0, 0.0],
        [0.0, 0.018, 0.0],
        [0.0, 0.0, 0.006],
    ]
)


# Build orbit trajectory
def build_orbit_trajectory(q0, dt, num_time_steps):
    states = np.zeros((num_time_steps + 1, 6))
    states[0] = q0

    for i in range(num_time_steps):
        states[i + 1] = rk4_step(states[i], dt)

    return states


# Original CADRE Attitude_Attitude equations
def cadre_attitude_from_orbit(states):
    n = states.shape[0]

    O_RI = np.zeros((n, 3, 3))

    for i in range(n):
        r = states[i, 0:3].copy()
        v = states[i, 3:6].copy()

        norm_r = np.sqrt(np.dot(r, r))
        norm_v = np.sqrt(np.dot(v, v))

        if norm_r < 1e-10:
            norm_r = 1e-10

        if norm_v < 1e-10:
            norm_v = 1e-10

        r = r / norm_r
        v = v / norm_v

        vx = np.zeros((3, 3))
        vx[0, :] = (0.0, -v[2], v[1])
        vx[1, :] = (v[2], 0.0, -v[0])
        vx[2, :] = (-v[1], v[0], 0.0)

        iB = np.dot(vx, r)
        jB = -np.dot(vx, iB)

        O_RI[i, 0, :] = iB
        O_RI[i, 1, :] = jB
        O_RI[i, 2, :] = -v

    return O_RI


# Orginial CADRE Attitude_Roll equations
def cadre_attitude_roll(gamma):
    n = len(gamma)

    O_BR = np.zeros((n, 3, 3))
    O_BR[:, 0, 0] = np.cos(gamma)
    O_BR[:, 0, 1] = np.sin(gamma)
    O_BR[:, 1, 0] = -O_BR[:, 0, 1]
    O_BR[:, 1, 1] = O_BR[:, 0, 0]
    O_BR[:, 2, 2] = 1.0

    return O_BR


# Original CADRE Attitude_RotationMtx equations
def cadre_rotation_matrix(O_BR, O_RI):
    n = O_RI.shape[0]

    O_BI = np.zeros((n, 3, 3))

    for i in range(n):
        O_BI[i] = np.dot(O_BR[i], O_RI[i])

    return O_BI


# Original CADRE Attitude_RotationMtxRates equations
def cadre_rotation_matrix_rates(O_BI, dt):
    Odot_BI = np.zeros_like(O_BI)
    Odot_BI[0] = (O_BI[1] - O_BI[0]) / dt
    Odot_BI[1:-1] = (O_BI[2:] - O_BI[:-2]) / (2.0 * dt)
    Odot_BI[-1] = (O_BI[-1] - O_BI[-2]) / dt

    return Odot_BI


# Original CADRE Attitude_Angular equations
def cadre_angular_velocity(O_BI, Odot_BI):
    n = O_BI.shape[0]

    w_B = np.zeros((n, 3))

    for i in range(n):
        w_B[i, 0] = np.dot(Odot_BI[i, 2, :], O_BI[i, 1, :])
        w_B[i, 1] = np.dot(Odot_BI[i, 0, :], O_BI[i, 2, :])
        w_B[i, 2] = np.dot(Odot_BI[i, 1, :], O_BI[i, 0, :])

    return w_B


# Original CADRE Attitude_AngularRates equations
def cadre_angular_acceleration(w_B, dt):
    wdot_B = np.zeros_like(w_B)
    wdot_B[0] = (w_B[1] - w_B[0]) / dt
    wdot_B[1:-1] = (w_B[2:] - w_B[:-2]) / (2.0 * dt)
    wdot_B[-1] = (w_B[-1] - w_B[-2]) / dt

    return wdot_B


# Original CADRE Attitude_Sideslip equations
def cadre_body_velocity(states, O_BI):
    n = states.shape[0]

    v_B = np.zeros((n, 3))

    for i in range(n):
        v_I = states[i, 3:6]
        v_B[i] = np.dot(
            O_BI[i],
            v_I,
        )

    return v_B


# Original CADRE Attitude_Torque equations
def cadre_attitude_torque(w_B, wdot_B):
    n = w_B.shape[0]

    T_tot = np.zeros((n, 3))

    for i in range(n):
        w = w_B[i]
        wx = np.zeros((3, 3))
        wx[0, :] = (0.0, -w[2], w[1])
        wx[1, :] = (w[2], 0.0, -w[0])
        wx[2, :] = (-w[1], w[0], 0.0)

        T_tot[i] = np.dot(J, wdot_B[i]) + np.dot(wx, np.dot(J, w))

    return T_tot


# Main validation
def main():

    print()
    print("=" * 75)
    print("CADRE ATTITUDE SOURCE VALIDATION")
    print("=" * 75)

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

    # Orbit trajectory
    states = build_orbit_trajectory(q0, dt, num_time_steps)

    # Roll command
    gamma = np.zeros(num_time_steps + 1)

    # ORIGINAL CADRE SOURCE CALCULATIONS
    cadre_O_RI = cadre_attitude_from_orbit(states)
    cadre_O_BR = cadre_attitude_roll(gamma)
    cadre_O_BI = cadre_rotation_matrix(cadre_O_BR, cadre_O_RI)
    cadre_Odot_BI = cadre_rotation_matrix_rates(cadre_O_BI, dt)
    cadre_w_B = cadre_angular_velocity(cadre_O_BI, cadre_Odot_BI)
    cadre_wdot_B = cadre_angular_acceleration(cadre_w_B, dt)
    cadre_v_B = cadre_body_velocity(states, cadre_O_BI)
    cadre_T_tot = cadre_attitude_torque(cadre_w_B, cadre_wdot_B)

    # OUR REFERENCE CALCULATIONS
    our_O_RI = attitude_from_orbit(states)
    our_O_BR = attitude_roll(gamma)
    our_O_BI = combine_rotation_matrices(our_O_BR, our_O_RI)
    our_Odot_BI = rotation_matrix_rates(our_O_BI, dt)
    our_w_B = angular_velocity(our_O_BI, our_Odot_BI)
    our_wdot_B = angular_acceleration(our_w_B, dt)
    our_v_B = body_frame_velocity(states, our_O_BI)
    our_T_tot = attitude_torque(
        our_w_B,
        our_wdot_B,
    )

    # Differences
    O_RI_difference = np.max(np.abs(cadre_O_RI - our_O_RI))
    O_BR_difference = np.max(np.abs(cadre_O_BR - our_O_BR))
    O_BI_difference = np.max(np.abs(cadre_O_BI - our_O_BI))
    Odot_difference = np.max(np.abs(cadre_Odot_BI - our_Odot_BI))
    w_difference = np.max(np.abs(cadre_w_B - our_w_B))
    wdot_difference = np.max(np.abs(cadre_wdot_B - our_wdot_B))
    velocity_difference = np.max(np.abs(cadre_v_B - our_v_B))
    torque_difference = np.max(np.abs(cadre_T_tot - our_T_tot))

    # Print results
    print()
    print("1. BASE ORIENTATION MATRIX")
    print("--------------------------")
    print("Maximum difference:")
    print(O_RI_difference)

    print()
    print("2. ROLL ROTATION MATRIX")
    print("-----------------------")
    print("Maximum difference:")
    print(O_BR_difference)

    print()
    print("3. FINAL ORIENTATION MATRIX")
    print("---------------------------")
    print("Maximum difference:")
    print(O_BI_difference)

    print()
    print("4. ORIENTATION MATRIX RATE")
    print("--------------------------")
    print("Maximum difference:")
    print(Odot_difference)

    print()
    print("5. ANGULAR VELOCITY")
    print("-------------------")
    print("Maximum difference:")
    print(w_difference)

    print()
    print("6. ANGULAR ACCELERATION")
    print("-----------------------")
    print("Maximum difference:")
    print(wdot_difference)

    print()
    print("7. BODY-FRAME VELOCITY")
    print("----------------------")
    print("Maximum difference:")
    print(velocity_difference)

    print()
    print("8. REQUIRED TORQUE")
    print("------------------")
    print("Maximum difference:")
    print(torque_difference)

    # Pass/fail
    tolerance = 1.0e-12

    differences = [
        O_RI_difference,
        O_BR_difference,
        O_BI_difference,
        Odot_difference,
        w_difference,
        wdot_difference,
        velocity_difference,
        torque_difference,
    ]

    passed = all(difference < tolerance for difference in differences)

    print()
    print("=" * 75)
    print("RESULT")
    print("=" * 75)

    print("Attitude equations match:", passed)

    if passed:
        print()
        print(
            "PASS: attitude_reference.py numerically matches the original "
            + "CADRE attitude equations."
        )

    else:
        print()
        print(
            "CHECK NEEDED: at least one comparison is larger than the "
            + "tolerance."
        )


if __name__ == "__main__":
    main()
