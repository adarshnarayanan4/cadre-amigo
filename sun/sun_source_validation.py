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
)

from sun.sun_reference import (
    sun_position_eci,
    sun_line_of_sight,
    sun_position_body,
    sun_position_spherical,
)


RE = 6378.137
R1 = RE * 0.85
R2 = RE
D2R = np.pi / 180.0


# Build orbit trajectory
def build_orbit_trajectory(q0, dt, num_time_steps):
    states = np.zeros((num_time_steps + 1, 6))
    states[0] = q0

    for i in range(num_time_steps):
        states[i + 1] = rk4_step(states[i], dt)

    return states


# Original CADRE Sun_PositionECI equations
def cadre_sun_position_eci(times, LD=0.0):
    n = len(times)
    r_e2s_I = np.zeros((n, 3))

    T = LD + times / 3600.0 / 24.0

    for i in range(n):
        L = D2R * 280.460 + D2R * 0.9856474 * T[i]
        g = D2R * 357.528 + D2R * 0.9856003 * T[i]

        Lambda = L + D2R * 1.914666 * np.sin(g) + D2R * 0.01999464 * np.sin(2.0 * g)
        eps = D2R * 23.439 - D2R * 3.56e-7 * T[i]

        r_e2s_I[i, 0] = np.cos(Lambda)
        r_e2s_I[i, 1] = np.sin(Lambda) * np.cos(eps)
        r_e2s_I[i, 2] = np.sin(Lambda) * np.sin(eps)

    return r_e2s_I


# Original CADRE Sun_LOS equations
def cadre_sun_line_of_sight(states, r_e2s_I):
    n = states.shape[0]
    LOS = np.zeros(n)

    for i in range(n):
        r_b = states[i, 0:3]
        r_s = r_e2s_I[i]

        dot = np.dot(r_b, r_s)
        cross = np.cross(r_b, r_s)
        dist = np.sqrt(np.dot(cross, cross))

        if dot >= 0.0:
            LOS[i] = 1.0
        elif dist <= R1:
            LOS[i] = 0.0
        elif dist >= R2:
            LOS[i] = 1.0
        else:
            x = (dist - R1) / (R2 - R1)
            LOS[i] = 3.0 * x**2 - 2.0 * x**3

    return LOS


# Original CADRE Sun_PositionBody equations
def cadre_sun_position_body(O_BI, r_e2s_I):
    n = r_e2s_I.shape[0]
    r_e2s_B = np.zeros((n, 3))

    for i in range(n):
        r_e2s_B[i] = np.dot(O_BI[i], r_e2s_I[i])

    return r_e2s_B


# Original CADRE custom arctan
def cadre_arctan(x, y):
    if x == 0.0:
        if y > 0.0:
            return np.pi / 2.0
        elif y < 0.0:
            return 3.0 * np.pi / 2.0
        else:
            return 0.0

    elif y == 0.0:
        if x > 0.0:
            return 0.0
        elif x < 0.0:
            return np.pi
        else:
            return 0.0

    elif x < 0.0:
        return np.arctan(y / x) + np.pi

    elif y < 0.0:
        return np.arctan(y / x) + 2.0 * np.pi

    elif y > 0.0:
        return np.arctan(y / x)

    return 0.0


# Original CADRE Sun_PositionSpherical equations
def cadre_sun_position_spherical(r_e2s_B):
    n = r_e2s_B.shape[0]

    azimuth = np.zeros(n)
    elevation = np.zeros(n)

    r = np.sqrt(np.sum(r_e2s_B * r_e2s_B, axis=1))

    for i in range(n):
        x = r_e2s_B[i, 0]
        y = r_e2s_B[i, 1]

        if r[i] < 1e-15:
            r[i] = 1e-5

        azimuth[i] = cadre_arctan(x, y)

    elevation = np.arccos(r_e2s_B[:, 2] / r)

    return azimuth, elevation


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
        alt_perigee=500.0,
        alt_apogee=500.0,
    )

    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps
    times = np.linspace(0.0, orbital_period, num_nodes)

    # Orbit
    states = build_orbit_trajectory(q0, dt, num_time_steps)

    # Attitude
    gamma = np.zeros(num_nodes)
    O_RI = attitude_from_orbit(states)
    O_BR = attitude_roll(gamma)
    O_BI = combine_rotation_matrices(O_BR, O_RI)

    # Original CADRE calculations
    cadre_r_e2s_I = cadre_sun_position_eci(times)
    cadre_LOS = cadre_sun_line_of_sight(states, cadre_r_e2s_I)
    cadre_r_e2s_B = cadre_sun_position_body(O_BI, cadre_r_e2s_I)
    cadre_azimuth, cadre_elevation = cadre_sun_position_spherical(cadre_r_e2s_B)

    # Reference calculations
    our_r_e2s_I = sun_position_eci(times)
    our_LOS = sun_line_of_sight(states, our_r_e2s_I)
    our_r_e2s_B = sun_position_body(O_BI, our_r_e2s_I)
    our_azimuth, our_elevation = sun_position_spherical(our_r_e2s_B)

    # Differences
    eci_difference = np.max(np.abs(cadre_r_e2s_I - our_r_e2s_I))
    LOS_difference = np.max(np.abs(cadre_LOS - our_LOS))
    body_difference = np.max(np.abs(cadre_r_e2s_B - our_r_e2s_B))
    azimuth_difference = np.max(np.abs(cadre_azimuth - our_azimuth))
    elevation_difference = np.max(np.abs(cadre_elevation - our_elevation))

    print()
    print("=" * 70)
    print("CADRE SUN SOURCE VALIDATION")
    print("=" * 70)

    print()
    print("Sun_PositionECI maximum difference:")
    print(eci_difference)

    print()
    print("Sun_LOS maximum difference:")
    print(LOS_difference)

    print()
    print("Sun_PositionBody maximum difference:")
    print(body_difference)

    print()
    print("Sun azimuth maximum difference:")
    print(azimuth_difference)

    print()
    print("Sun elevation maximum difference:")
    print(elevation_difference)

    differences = [
        eci_difference,
        LOS_difference,
        body_difference,
        azimuth_difference,
        elevation_difference,
    ]

    passed = all(difference < 1e-12 for difference in differences)

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print()
    print("Sun equations match:")
    print(passed)


if __name__ == "__main__":
    main()