import pickle

import numpy as np

from cadre_paths import cadre_path


RE = 6378.137


def get_case_value(data, name):
    case_key = f"0:{name}"

    if case_key in data:
        return data[case_key]

    return data[name]


def scalar_value(value):
    return float(
        np.asarray(value).reshape(-1)[0]
    )


def antenna_rotation(ant_angle, n):
    q_A = np.zeros((4, n))

    rt2 = np.sqrt(2.0)

    q_A[0, :] = np.cos(ant_angle / 2.0)
    q_A[1, :] = np.sin(ant_angle / 2.0) / rt2
    q_A[2, :] = -np.sin(ant_angle / 2.0) / rt2
    q_A[3, :] = 0.0

    return q_A


def earth_spin(t):
    q_E = np.zeros((4, len(t)))

    fact = np.pi / 3600.0 / 24.0
    theta = fact * t

    q_E[0, :] = np.cos(theta)
    q_E[3, :] = -np.sin(theta)

    return q_E


def quaternion_rotation_matrix(q):
    n = q.shape[1]

    O = np.zeros((3, 3, n))

    A = np.zeros((4, 3))
    B = np.zeros((4, 3))

    for i in range(n):
        A[0, :] = [
            q[0, i],
            -q[3, i],
            q[2, i],
        ]

        A[1, :] = [
            q[3, i],
            q[0, i],
            -q[1, i],
        ]

        A[2, :] = [
            -q[2, i],
            q[1, i],
            q[0, i],
        ]

        A[3, :] = [
            q[1, i],
            q[2, i],
            q[3, i],
        ]

        B[0, :] = [
            q[0, i],
            q[3, i],
            -q[2, i],
        ]

        B[1, :] = [
            -q[3, i],
            q[0, i],
            q[1, i],
        ]

        B[2, :] = [
            q[2, i],
            -q[1, i],
            q[0, i],
        ]

        B[3, :] = [
            q[1, i],
            q[2, i],
            q[3, i],
        ]

        O[:, :, i] = A.T @ B

    return O


def ground_station_position_earth(lon, lat, alt, n):
    d2r = np.pi / 180.0

    cos_lat = np.cos(d2r * lat)
    sin_lat = np.sin(d2r * lat)

    cos_lon = np.cos(d2r * lon)
    sin_lon = np.sin(d2r * lon)

    radius = RE + alt

    r_e2g_E = np.zeros((3, n))

    r_e2g_E[0, :] = (
        radius * cos_lat * cos_lon
    )

    r_e2g_E[1, :] = (
        radius * cos_lat * sin_lon
    )

    r_e2g_E[2, :] = (
        radius * sin_lat
    )

    return r_e2g_E


def rotate_vectors(matrix, vectors):
    n = vectors.shape[1]

    result = np.zeros_like(vectors)

    for i in range(n):
        result[:, i] = (
            matrix[:, :, i] @ vectors[:, i]
        )

    return result


def vector_eci(r_e2g_I, r_e2b_I):
    return r_e2g_I - r_e2b_I[:3, :]


def communication_los(r_b2g_I, r_e2g_I):
    n = r_b2g_I.shape[1]

    CommLOS = np.zeros(n)

    Rb = 100.0

    for i in range(n):
        proj = (
            np.dot(
                r_b2g_I[:, i],
                r_e2g_I[:, i],
            )
            / RE
        )

        if proj > 0.0:
            CommLOS[i] = 0.0

        elif proj < -Rb:
            CommLOS[i] = 1.0

        else:
            x = proj / -Rb

            CommLOS[i] = (
                3.0 * x**2
                - 2.0 * x**3
            )

    return CommLOS


def ground_station_distance(r_b2g_A):
    return np.sqrt(
        np.sum(
            r_b2g_A * r_b2g_A,
            axis=0,
        )
    )


def cadre_arctan(x, y):
    if x == 0.0:
        if y > 0.0:
            return np.pi / 2.0

        if y < 0.0:
            return 3.0 * np.pi / 2.0

        return 0.0

    if y == 0.0:
        if x > 0.0:
            return 0.0

        if x < 0.0:
            return np.pi

        return 0.0

    if x < 0.0:
        return np.arctan(y / x) + np.pi

    if y < 0.0:
        return np.arctan(y / x) + 2.0 * np.pi

    if y > 0.0:
        return np.arctan(y / x)

    return 0.0


def vector_spherical(r_b2g_A):
    n = r_b2g_A.shape[1]

    azimuth = np.zeros(n)
    elevation = np.zeros(n)

    radius = np.sqrt(
        np.sum(
            r_b2g_A * r_b2g_A,
            axis=0,
        )
    )

    for i in range(n):
        x = r_b2g_A[0, i]
        y = r_b2g_A[1, i]

        if radius[i] < 1.0e-15:
            radius[i] = 1.0e-5

        azimuth[i] = cadre_arctan(
            x,
            y,
        )

    elevation = np.arccos(
        r_b2g_A[2, :] / radius
    )

    return azimuth, elevation


def print_difference(name, reference, source):
    difference = np.max(
        np.abs(
            reference - source
        )
    )

    print()
    print(name)
    print(difference)


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(data_path, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1",
        )

    # Inputs
    t = get_case_value(
        data,
        "t",
    )

    ant_angle = scalar_value(
        get_case_value(
            data,
            "antAngle",
        )
    )

    lon = scalar_value(
        get_case_value(
            data,
            "lon",
        )
    )

    lat = scalar_value(
        get_case_value(
            data,
            "lat",
        )
    )

    alt = scalar_value(
        get_case_value(
            data,
            "alt",
        )
    )

    r_e2b_I = get_case_value(
        data,
        "r_e2b_I",
    )

    O_BI = get_case_value(
        data,
        "O_BI",
    )

    n = len(t)

    # Antenna orientation
    q_A = antenna_rotation(
        ant_angle,
        n,
    )

    O_AB = quaternion_rotation_matrix(
        q_A,
    )

    # Earth rotation
    q_E = earth_spin(
        t,
    )

    O_IE = quaternion_rotation_matrix(
        q_E,
    )

    # Ground-station position
    r_e2g_E = ground_station_position_earth(
        lon,
        lat,
        alt,
        n,
    )

    r_e2g_I = rotate_vectors(
        O_IE,
        r_e2g_E,
    )

    # Satellite to ground station vector
    r_b2g_I = vector_eci(
        r_e2g_I,
        r_e2b_I,
    )

    # Communication LOS
    CommLOS = communication_los(
        r_b2g_I,
        r_e2g_I,
    )

    # Convert into spacecraft body frame
    r_b2g_B = rotate_vectors(
        O_BI,
        r_b2g_I,
    )

    # Convert into antenna frame
    r_b2g_A = rotate_vectors(
        O_AB,
        r_b2g_B,
    )

    # Distance
    GSdist = ground_station_distance(
        r_b2g_A,
    )

    # Azimuth and elevation
    azimuthGS, elevationGS = (
        vector_spherical(
            r_b2g_A,
        )
    )

    print()
    print("=" * 70)
    print("CADRE COMMUNICATION GEOMETRY REFERENCE")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Ground station longitude [deg]:")
    print(lon)

    print()
    print("Ground station latitude [deg]:")
    print(lat)

    print()
    print("Ground station altitude [km]:")
    print(alt)

    print()
    print("=" * 70)
    print("SOURCE VALIDATION")
    print("=" * 70)

    print_difference(
        "Maximum q_A difference:",
        q_A,
        get_case_value(data, "q_A"),
    )

    print_difference(
        "Maximum O_AB difference:",
        O_AB,
        get_case_value(data, "O_AB"),
    )

    print_difference(
        "Maximum q_E difference:",
        q_E,
        get_case_value(data, "q_E"),
    )

    print_difference(
        "Maximum O_IE difference:",
        O_IE,
        get_case_value(data, "O_IE"),
    )

    print_difference(
        "Maximum r_e2g_E difference [km]:",
        r_e2g_E,
        get_case_value(data, "r_e2g_E"),
    )

    print_difference(
        "Maximum r_e2g_I difference [km]:",
        r_e2g_I,
        get_case_value(data, "r_e2g_I"),
    )

    print_difference(
        "Maximum r_b2g_I difference [km]:",
        r_b2g_I,
        get_case_value(data, "r_b2g_I"),
    )

    print_difference(
        "Maximum CommLOS difference:",
        CommLOS,
        get_case_value(data, "CommLOS"),
    )

    print_difference(
        "Maximum r_b2g_B difference [km]:",
        r_b2g_B,
        get_case_value(data, "r_b2g_B"),
    )

    print_difference(
        "Maximum r_b2g_A difference [km]:",
        r_b2g_A,
        get_case_value(data, "r_b2g_A"),
    )

    print_difference(
        "Maximum GSdist difference [km]:",
        GSdist,
        get_case_value(data, "GSdist"),
    )

    print_difference(
        "Maximum azimuthGS difference [rad]:",
        azimuthGS,
        get_case_value(data, "azimuthGS"),
    )

    print_difference(
        "Maximum elevationGS difference [rad]:",
        elevationGS,
        get_case_value(data, "elevationGS"),
    )

    print()
    print("=" * 70)
    print("COMMUNICATION GEOMETRY SUMMARY")
    print("=" * 70)

    print()
    print("Minimum ground-station distance [km]:")
    print(np.min(GSdist))

    print()
    print("Maximum ground-station distance [km]:")
    print(np.max(GSdist))

    print()
    print("Minimum CommLOS:")
    print(np.min(CommLOS))

    print()
    print("Maximum CommLOS:")
    print(np.max(CommLOS))

    print()
    print("Full line-of-sight nodes:")
    print(np.sum(CommLOS == 1.0))

    print()
    print("Blocked nodes:")
    print(np.sum(CommLOS == 0.0))


if __name__ == "__main__":
    main()