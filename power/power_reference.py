import numpy as np
from scipy.interpolate import RegularGridInterpolator

from orbit.orbit_reference import orbital_elements_to_state, calculate_orbital_period
from orbit.orbit_amigo import build_reference_trajectory
from attitude.attitude_reference import attitude_from_orbit, attitude_roll, combine_rotation_matrices
from sun.sun_reference import sun_position_eci, sun_position_body, sun_position_spherical, sun_line_of_sight
from solar.solar_reference import load_solar_data, build_interpolators, solar_exposed_area
from thermal.thermal_reference import propagate_temperature


def load_power_data():
    path = "../../../CADRE/src/CADRE/data/Power/curve.dat"
    dat = np.loadtxt(path)

    nT = int(dat[0])
    nA = int(dat[1])
    nI = int(dat[2])

    index = 3

    T = dat[index:index + nT]
    index += nT

    A = dat[index:index + nA]
    index += nA

    I = dat[index:index + nI]
    index += nI

    V = dat[index:].reshape((nT, nA, nI), order="F")

    return T, A, I, V


def build_power_interpolator(T, A, I, V):
    return RegularGridInterpolator((T, A, I), V, method="linear", bounds_error=False, fill_value=None)


def power_cell_voltage(LOS, temperature, exposedArea, Isetpt, interpolator):
    num_nodes = LOS.size
    V_sol = np.zeros((12, num_nodes))

    for p in range(12):
        temp_index = 4 if p < 4 else p % 4

        for c in range(7):
            effective_area = LOS * exposedArea[c, p, :]
            points = np.column_stack((temperature[temp_index, :], effective_area, Isetpt[p, :]))
            V_sol[p, :] += interpolator(points)

    return V_sol


def solar_power(V_sol, Isetpt):
    return np.sum(V_sol * Isetpt, axis=0)


def main():
    r0, v0 = orbital_elements_to_state(alt_perigee=500.0, alt_apogee=500.0, raan=66.279, inclination=82.072, arg_perigee=0.0, true_anomaly=337.987)
    q0 = np.concatenate((r0, v0))

    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)
    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps
    times = np.linspace(0.0, orbital_period, num_nodes)

    print()
    print("=" * 70)
    print("CADRE POWER REFERENCE")
    print("=" * 70)

    # Orbit
    reference_states, reference_rates = build_reference_trajectory(q0, dt, num_time_steps)

    # Attitude
    gamma_reference = np.zeros(num_nodes)
    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(O_BR_reference, O_RI_reference)

    # Sun
    r_e2s_I_reference = sun_position_eci(times)
    r_e2s_B_reference = sun_position_body(O_BI_reference, r_e2s_I_reference)
    azimuth_reference, elevation_reference = sun_position_spherical(r_e2s_B_reference)
    LOS_reference = sun_line_of_sight(reference_states, r_e2s_I_reference)

    # Solar
    angle, azimuth_grid, elevation_grid, solar_data = load_solar_data()
    solar_interpolators = build_interpolators(angle, azimuth_grid, elevation_grid, solar_data)
    exposed_area = solar_exposed_area(0.0, azimuth_reference, elevation_reference, solar_interpolators)

    # Thermal
    T0 = 273.0 * np.ones(5)
    P_comm = 0.1 * np.ones(num_nodes)
    cellInstd = np.ones((7, 12))
    temperature = propagate_temperature(T0, exposed_area, LOS_reference, P_comm, cellInstd, dt)

    # Panel current
    Isetpt = 0.2 * np.ones((12, num_nodes))

    # Power lookup table
    T_grid, A_grid, I_grid, V_data = load_power_data()
    power_interpolator = build_power_interpolator(T_grid, A_grid, I_grid, V_data)

    # Cell/panel voltage
    V_sol = power_cell_voltage(LOS_reference, temperature, exposed_area, Isetpt, power_interpolator)

    # Total solar power
    P_sol = solar_power(V_sol, Isetpt)

    print()
    print("Power table temperature range [K]:")
    print(T_grid[0], T_grid[-1])

    print()
    print("Power table area range [m^2]:")
    print(A_grid[0], A_grid[-1])

    print()
    print("Power table current range [A]:")
    print(I_grid[0], I_grid[-1])

    print()
    print("Panel-voltage shape:")
    print(V_sol.shape)

    print()
    print("Minimum panel voltage [V]:")
    print(np.min(V_sol))

    print()
    print("Maximum panel voltage [V]:")
    print(np.max(V_sol))

    print()
    print("Minimum total solar power [W]:")
    print(np.min(P_sol))

    print()
    print("Maximum total solar power [W]:")
    print(np.max(P_sol))

    print()
    print("Mean total solar power [W]:")
    print(np.mean(P_sol))

    print()
    print("Initial panel voltages [V]:")
    print(V_sol[:, 0])

    print()
    print("Initial total solar power [W]:")
    print(P_sol[0])

    print()
    print("Final panel voltages [V]:")
    print(V_sol[:, -1])

    print()
    print("Final total solar power [W]:")
    print(P_sol[-1])

    print()
    print("Full eclipse nodes:")
    print(np.sum(LOS_reference == 0.0))

    print()
    print("Full sunlight nodes:")
    print(np.sum(LOS_reference == 1.0))


if __name__ == "__main__":
    main()