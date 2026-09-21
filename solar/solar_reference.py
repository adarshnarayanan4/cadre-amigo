import numpy as np
from scipy.interpolate import RegularGridInterpolator

from orbit.orbit_reference import orbital_elements_to_state, calculate_orbital_period
from orbit.orbit_amigo import build_reference_trajectory
from attitude.attitude_reference import attitude_from_orbit, attitude_roll, combine_rotation_matrices
from sun.sun_reference import sun_position_eci, sun_position_body, sun_position_spherical, sun_line_of_sight

def fixangles(azimuth, elevation):
    azimuth = np.mod(azimuth, 2.0 * np.pi)
    elevation = np.mod(elevation, 2.0 * np.pi)

    mask = elevation > np.pi

    elevation[mask] = 2.0 * np.pi - elevation[mask]
    azimuth[mask] = np.mod(np.pi + azimuth[mask], 2.0 * np.pi)

    return azimuth, elevation


def load_solar_data():
    raw1 = np.genfromtxt("../../../CADRE/src/CADRE/data/Solar/Area10.txt")
    raw2 = np.loadtxt("../../../CADRE/src/CADRE/data/Solar/Area_all.txt")

    na = 10
    nz = 73
    ne = 37
    npanels = 12
    ncells = 7

    angle = raw1[0:na].copy()
    azimuth = raw1[na:na + nz].copy()

    elevation_start = na + nz - 1
    elevation = raw1[elevation_start:elevation_start + ne].copy()

    angle[0] = 0.0
    angle[-1] = np.pi / 2.0

    azimuth[0] = 0.0
    azimuth[-1] = 2.0 * np.pi

    elevation[0] = 0.0
    elevation[-1] = np.pi

    data = np.zeros((na, nz, ne, ncells, npanels))

    flat_size = na * nz * ne
    counter = 0

    for p in range(npanels):
        for c in range(ncells):
            values = raw2[7 * p + c, 119:119 + flat_size]
            data[:, :, :, c, p] = values.reshape((na, nz, ne))
            counter += 1

    return angle, azimuth, elevation, data


def build_interpolators(angle, azimuth, elevation, data):
    interpolators = []

    for c in range(7):
        row = []

        for p in range(12):
            interpolator = RegularGridInterpolator(
                (angle, azimuth, elevation),
                data[:, :, :, c, p],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            row.append(interpolator)

        interpolators.append(row)

    return interpolators


def solar_exposed_area(fin_angle, azimuth, elevation, interpolators):
    azimuth, elevation = fixangles(azimuth.copy(), elevation.copy())

    n = len(azimuth)

    points = np.zeros((n, 3))
    points[:, 0] = fin_angle
    points[:, 1] = azimuth
    points[:, 2] = elevation

    exposed_area = np.zeros((7, 12, n))

    for c in range(7):
        for p in range(12):
            exposed_area[c, p, :] = interpolators[c][p](points)

    return exposed_area


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
    print("CADRE SOLAR REFERENCE")
    print("=" * 70)

    reference_states, reference_rates = build_reference_trajectory(
        q0,
        dt,
        num_time_steps,
    )

    gamma_reference = np.zeros(num_nodes)

    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(
        O_BR_reference,
        O_RI_reference,
    )

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

    exposed_area_with_LOS = exposed_area * LOS_reference[np.newaxis, np.newaxis, :]

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Exposed-area shape:")
    print(exposed_area.shape)

    print()
    print("Minimum exposed area:")
    print(np.min(exposed_area))

    print()
    print("Maximum exposed area:")
    print(np.max(exposed_area))

    print()
    print("Minimum exposed area with LOS:")
    print(np.min(exposed_area_with_LOS))

    print()
    print("Maximum exposed area with LOS:")
    print(np.max(exposed_area_with_LOS))

    print()
    print("Maximum total exposed area at one time:")
    print(np.max(np.sum(exposed_area_with_LOS, axis=(0, 1))))

    print()
    print("Minimum total exposed area at one time:")
    print(np.min(np.sum(exposed_area_with_LOS, axis=(0, 1))))

    print()
    print("Full eclipse nodes:")
    print(np.sum(LOS_reference == 0.0))

    print()
    print("Full sunlight nodes:")
    print(np.sum(LOS_reference == 1.0))


if __name__ == "__main__":
    main()