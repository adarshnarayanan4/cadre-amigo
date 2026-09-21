import numpy as np
from math import sqrt
from orbit.orbit_reference import (orbital_elements_to_state, orbit_dynamics, rk4_step, calculate_orbital_period,)

# Original CADRE constants
MU = 398600.44
RE = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

C1 = -MU
C2 = -1.5 * MU * J2 * RE**2
C3 = -2.5 * MU * J3 * RE**3
C4 = 1.875 * MU * J4 * RE**4

# Original CADRE Orbit_Initial equations
def cadre_initial_source(alt_perigee, alt_apogee, raan, inclination, arg_perigee, true_anomaly,):
    """Standalone version of the original CADRE Orbit_Initial equations."""
    def S(v):
        matrix = np.zeros((3, 3))
        matrix[0, :] = [0.0, -v[2], v[1]]
        matrix[1, :] = [v[2], 0.0, -v[0]]
        matrix[2, :] = [-v[1], v[0], 0.0]
        return matrix

    def get_rotation(axis, angle):
        return (np.eye(3) + S(axis) * np.sin(angle) + (1.0 - np.cos(angle)) * (np.outer(axis, axis) - np.eye(3)))

    d2r = np.pi / 180.0

    r_perigee = RE + alt_perigee
    r_apogee = RE + alt_apogee

    e = ((r_apogee - r_perigee) / (r_apogee + r_perigee))

    a = (r_perigee + r_apogee) / 2.0
    p = a * (1.0 - e**2)

    rmag0 = p / (1.0 + e * np.cos(d2r * true_anomaly))

    r0_P = np.array([rmag0 * np.cos(d2r * true_anomaly), rmag0 * np.sin(d2r * true_anomaly), 0.0])
    v0_P = np.array([-np.sqrt(MU / p) * np.sin(d2r * true_anomaly), np.sqrt(MU / p) * (e + np.cos(d2r * true_anomaly)), 0.0])

    O_IP = np.eye(3)

    O_IP = np.dot(O_IP, get_rotation(np.array([0.0, 0.0, 1.0]), raan * d2r))
    O_IP = np.dot(O_IP, get_rotation(np.array([1.0, 0.0, 0.0]), inclination * d2r))
    O_IP = np.dot(O_IP, get_rotation(np.array([0.0, 0.0, 1.0]), arg_perigee * d2r))

    r0_ECI = np.dot(O_IP, r0_P)
    v0_ECI = np.dot(O_IP, v0_P)

    return r0_ECI, v0_ECI

# Original CADRE Orbit_Dynamics.f_dot equations
def cadre_dynamics_source(state):
    """Standalone version of original CADRE Orbit_Dynamics.f_dot."""
    x = state[0]
    y = state[1]

    # This tiny-z workaround is present in the original CADRE source.
    z = (state[2] if abs(state[2]) > 1e-15 else 1e-5)

    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z

    r = sqrt(x * x + y * y + z2)

    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r
    r5 = r4 * r
    r7 = r5 * r * r

    T2 = 1.0 - 5.0 * z2 / r2
    T3 = (3.0 * z - 7.0 * z3 / r2)
    T4 = (1.0 - 14.0 * z2 / r2 + 21.0 * z4 / r4)

    T3z = (3.0 * z - 0.6 * r2 / z)
    T4z = (4.0 - (28.0 / 3.0) * z2 / r2)

    f_dot = np.zeros(6)
    f_dot[0:3] = state[3:]

    common = (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 + C4 / r7 * T4)

    f_dot[3:] = state[0:3] * common
    f_dot[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)

    return f_dot

# Original CADRE RK4 equations
def cadre_rk4_step_source(state, dt):
    """One RK4 step written in the same form as CADRE rk4.py."""
    a = cadre_dynamics_source(state)
    b = cadre_dynamics_source(state + dt / 2.0 * a)
    c = cadre_dynamics_source(state + dt / 2.0 * b)
    d = cadre_dynamics_source(state + dt * c)

    return state + dt / 6.0 * (a + 2.0 * (b + c) + d)

# Validation
def main():
    print()
    print("=" * 70)
    print("CADRE SOURCE VALIDATION")
    print("=" * 70)

    # Compare initial state
    cadre_r0, cadre_v0 = cadre_initial_source(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    our_r0, our_v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    cadre_q0 = np.concatenate((cadre_r0, cadre_v0))
    our_q0 = np.concatenate((our_r0, our_v0))

    initial_difference = np.max(np.abs(cadre_q0 - our_q0))

    print()
    print("1. INITIAL STATE CHECK")
    print("----------------------")

    print("Maximum difference:")
    print(initial_difference)

    # Compare dynamics over one orbit
    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)

    num_time_steps = 568
    dt = orbital_period / num_time_steps

    state = our_q0.copy()

    max_dynamics_difference = 0.0
    max_rk4_difference = 0.0

    for i in range(num_time_steps):
        # Compare derivative at this state
        cadre_rate = cadre_dynamics_source(state)
        our_rate = orbit_dynamics(state)

        dynamics_difference = np.max(np.abs(cadre_rate - our_rate))
        max_dynamics_difference = max(max_dynamics_difference, dynamics_difference)

        # Compare one RK4 step
        cadre_next = cadre_rk4_step_source(state, dt)
        our_next = rk4_step(state, dt)

        rk4_difference = np.max(np.abs(cadre_next - our_next))
        max_rk4_difference = max(max_rk4_difference, rk4_difference)

        # Advance using our reference trajectory
        state = our_next


    print()
    print("2. DYNAMICS CHECK")
    print("-----------------")

    print("Maximum derivative difference over one orbit:")
    print(max_dynamics_difference)

    print()
    print("3. RK4 STEP CHECK")
    print("-----------------")

    print("Maximum one-step RK4 difference over one orbit:")
    print(max_rk4_difference)

    # Final result
    tolerance = 1e-10

    initial_pass = (initial_difference < tolerance)
    dynamics_pass = (max_dynamics_difference < tolerance)
    rk4_pass = (max_rk4_difference < tolerance)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print("Initial-state equations match:", initial_pass)
    print("Dynamics equations match:", dynamics_pass)
    print("RK4 equations match:", rk4_pass)

    if (initial_pass and dynamics_pass and rk4_pass):
        print()
        print("PASS: orbit_reference.py numerically matches the original CADRE orbit equations.")
    else:
        print()
        print("CHECK NEEDED: at least one comparison was larger than the tolerance.")

if __name__ == "__main__":
    main()