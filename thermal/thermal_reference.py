import numpy as np

from orbit.orbit_reference import orbital_elements_to_state, calculate_orbital_period
from orbit.orbit_amigo import build_reference_trajectory

from attitude.attitude_reference import attitude_from_orbit, attitude_roll, combine_rotation_matrices

from sun.sun_reference import sun_position_eci, sun_position_body, sun_position_spherical, sun_line_of_sight

from solar.solar_reference import load_solar_data, build_interpolators, solar_exposed_area


m_f = 0.4
m_b = 2.0

cp_f = 0.6e3
cp_b = 2.0e3

A_T = 2.66e-3

alpha_c = 0.9
alpha_r = 0.2

eps_c = 0.87
eps_r = 0.88

q_sol = 1360.0
K = 5.67051e-8


def thermal_dynamics(state, exposed_area, LOS, P_comm, cellInstd):
    alpha = alpha_c * cellInstd + alpha_r - alpha_r * cellInstd
    eps = eps_c * cellInstd + eps_r - eps_r * cellInstd

    f = np.zeros(5)

    for p in range(12):
        if p < 4:
            f_i = 4
            mass = m_b
            cp = cp_b
        else:
            f_i = (p + 1) % 4
            mass = m_f
            cp = cp_f

        fact1 = q_sol * LOS / (mass * cp)
        fact2 = K * A_T * state[f_i]**4 / (mass * cp)

        f[f_i] += np.sum(alpha[:, p] * exposed_area[:, p]) * fact1
        f[f_i] -= np.sum(eps[:, p]) * fact2

    f[4] += 4.0 * P_comm / m_b / cp_b

    return f


def propagate_temperature(T0, exposed_area, LOS, P_comm, cellInstd, dt):
    num_nodes = LOS.size

    temperature = np.zeros((5, num_nodes))
    temperature[:, 0] = T0

    for k in range(num_nodes - 1):
        state = temperature[:, k]

        area_k = exposed_area[:, :, k]
        LOS_k = LOS[k]
        P_comm_k = P_comm[k]

        a = thermal_dynamics(state, area_k, LOS_k, P_comm_k, cellInstd)
        b = thermal_dynamics(state + dt / 2.0 * a, area_k, LOS_k, P_comm_k, cellInstd)
        c = thermal_dynamics(state + dt / 2.0 * b, area_k, LOS_k, P_comm_k, cellInstd)
        d = thermal_dynamics(state + dt * c, area_k, LOS_k, P_comm_k, cellInstd)

        temperature[:, k + 1] = state + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d)

    return temperature


def main():
    r0, v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    q0 = np.concatenate((r0, v0))

    orbital_period = calculate_orbital_period(
        alt_perigee=500.0,
        alt_apogee=500.0,
    )

    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps
    times = np.linspace(0.0, orbital_period, num_nodes)

    print()
    print("=" * 70)
    print("CADRE THERMAL REFERENCE")
    print("=" * 70)

    # Orbit
    reference_states, reference_rates = build_reference_trajectory(
        q0,
        dt,
        num_time_steps,
    )

    # Attitude
    gamma_reference = np.zeros(num_nodes)

    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(
        O_BR_reference,
        O_RI_reference,
    )

    # Sun
    r_e2s_I_reference = sun_position_eci(times)

    r_e2s_B_reference = sun_position_body(
        O_BI_reference,
        r_e2s_I_reference,
    )

    azimuth_reference, elevation_reference = sun_position_spherical(
        r_e2s_B_reference,
    )

    LOS_reference = sun_line_of_sight(
        reference_states,
        r_e2s_I_reference,
    )

    # Solar exposed area
    angle, azimuth_grid, elevation_grid, data = load_solar_data()

    interpolators = build_interpolators(
        angle,
        azimuth_grid,
        elevation_grid,
        data,
    )

    fin_angle = 0.0

    exposed_area = solar_exposed_area(
        fin_angle,
        azimuth_reference,
        elevation_reference,
        interpolators,
    )

    # Thermal inputs
    T0 = 273.0 * np.ones(5)
    cellInstd = np.ones((7, 12))
    P_comm = 0.1 * np.ones(num_nodes)

    # Thermal propagation
    temperature = propagate_temperature(
        T0,
        exposed_area,
        LOS_reference,
        P_comm,
        cellInstd,
        dt,
    )

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Temperature shape:")
    print(temperature.shape)

    print()
    print("Initial temperatures [K]:")
    print(temperature[:, 0])

    print()
    print("Final temperatures [K]:")
    print(temperature[:, -1])

    print()
    print("Minimum temperature [K]:")
    print(np.min(temperature))

    print()
    print("Maximum temperature [K]:")
    print(np.max(temperature))

    print()
    print("Minimum temperature by state [K]:")
    print(np.min(temperature, axis=1))

    print()
    print("Maximum temperature by state [K]:")
    print(np.max(temperature, axis=1))


if __name__ == "__main__":
    main()