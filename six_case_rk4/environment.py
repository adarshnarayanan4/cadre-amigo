"""
Orbit environment for CADRE
---------------------------
"""

import numpy as np
from scipy.integrate import solve_ivp

# Gravity of orbit.py with the J2, J3, and J4 zonal terms
MU = 398600.44
RE = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6
C1 = -MU
C2 = -1.5 * MU * J2 * RE**2
C3 = -2.5 * MU * J3 * RE**3
C4 = 1.875 * MU * J4 * RE**4

# Ann Arbor ground station of CADRE_group.py in degrees and km
LAT = 42.2708
LON = -83.7264
ALT = 0.256


def orbit_rate(state):
    """Rate of the position and velocity in km and km/s as Orbit_Dynamics.f_dot."""
    x = state[0]
    y = state[1]
    z = state[2] if abs(state[2]) > 1e-15 else 1e-5

    z2 = z * z
    z3 = z2 * z
    z4 = z3 * z
    r = np.sqrt(x * x + y * y + z2)
    r2 = r * r
    r3 = r2 * r
    r4 = r3 * r
    r5 = r4 * r
    r7 = r5 * r * r

    T2 = 1 - 5 * z2 / r2
    T3 = 3 * z - 7 * z3 / r2
    T4 = 1 - 14 * z2 / r2 + 21 * z4 / r4
    T3z = 3 * z - 0.6 * r2 / z
    T4z = 4 - 28.0 / 3.0 * z2 / r2

    f = np.zeros(6)
    f[0:3] = state[3:]
    f[3:] = state[0:3] * (C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 + C4 / r7 * T4)
    f[5] += z * (2.0 * C2 / r5 + C3 / r7 * T3z + C4 / r7 * T4z)
    return f


def orbit_rk4(r0, n, h):
    """Orbit on the CADRE grid by the RK4 steps of rk4.py, one column per node."""
    y = np.zeros((n, 6))
    y[0] = r0
    for k in range(n - 1):
        a = orbit_rate(y[k])
        b = orbit_rate(y[k] + h / 2.0 * a)
        c = orbit_rate(y[k] + h / 2.0 * b)
        d = orbit_rate(y[k] + h * c)
        y[k + 1] = y[k] + h / 6.0 * (a + 2 * (b + c) + d)
    return y.T


def orbit_dense(r0, t_end, margin=60.0):
    """Continuous orbit from a tight integration that also covers a margin past both ends."""
    rate = lambda s, y: orbit_rate(y)
    opts = dict(method="DOP853", rtol=1e-13, atol=1e-10, dense_output=True)
    ahead = solve_ivp(rate, (0.0, t_end + margin), r0, **opts)
    behind = solve_ivp(rate, (0.0, -margin), r0, **opts)

    def orbit(t):
        t = np.atleast_1d(np.asarray(t, dtype=float))
        out = np.empty((6, len(t)))
        past = t < 0.0
        out[:, ~past] = ahead.sol(t[~past])
        if np.any(past):
            out[:, past] = behind.sol(t[past])
        return out

    return orbit


def rolled_frame(r_e2b_I):
    """Rotation from the inertial frame to the rolled frame as Attitude_Attitude."""
    r = r_e2b_I[0:3]
    v = r_e2b_I[3:]
    normr = np.sqrt(np.sum(r * r, axis=0))
    normv = np.sqrt(np.sum(v * v, axis=0))
    normr = np.where(normr < 1e-10, 1e-10, normr)
    normv = np.where(normv < 1e-10, 1e-10, normv)
    r = r / normr
    v = v / normv

    # The cross product matrix of v applied to r and then to iB
    iB = np.array([-v[2] * r[1] + v[1] * r[2], v[2] * r[0] - v[0] * r[2], -v[1] * r[0] + v[0] * r[1]])
    jB = -np.array([-v[2] * iB[1] + v[1] * iB[2], v[2] * iB[0] - v[0] * iB[2], -v[1] * iB[0] + v[0] * iB[1]])
    return np.stack([iB, jB, -v])


def sun_direction(LD, t):
    """Earth to sun direction in the inertial frame as Sun_PositionECI."""
    d2r = np.pi / 180.0
    T = LD + t / 3600.0 / 24.0
    L = d2r * 280.460 + d2r * 0.9856474 * T
    g = d2r * 357.528 + d2r * 0.9856003 * T
    Lambda = L + d2r * 1.914666 * np.sin(g) + d2r * 0.01999464 * np.sin(2 * g)
    eps = d2r * 23.439 - d2r * 3.56e-7 * T
    return np.array([np.cos(Lambda), np.sin(Lambda) * np.cos(eps), np.sin(Lambda) * np.sin(eps)])


def sun_visibility(r_e2b_I, r_e2s_I):
    """Satellite to sun line of sight with the smooth penumbra of Sun_LOS."""
    r1 = 6378.137 * 0.85
    r2 = 6378.137
    rb = r_e2b_I[:3]
    dot = np.sum(rb * r_e2s_I, axis=0)
    cross = np.cross(rb.T, r_e2s_I.T).T
    dist = np.sqrt(np.sum(cross * cross, axis=0))
    x = (dist - r1) / (r2 - r1)
    shade = np.where(dist <= r1, 0.0, np.where(dist >= r2, 1.0, 3 * x**2 - 2 * x**3))
    return np.where(dot >= 0.0, 1.0, shade)


def quaternion_matrix(q):
    """Rotation matrices of quaternions as Comm_EarthsSpinMtx and Comm_AntRotationMtx."""
    A = np.array([[q[0], -q[3], q[2]], [q[3], q[0], -q[1]], [-q[2], q[1], q[0]], [q[1], q[2], q[3]]])
    B = np.array([[q[0], q[3], -q[2]], [-q[3], q[0], q[1]], [q[2], -q[1], q[0]], [q[1], q[2], q[3]]])
    return np.einsum("ru...,rv...->uv...", A, B)


def ground_station(t):
    """Earth to ground station vector in the inertial frame as Comm_GSposEarth and Comm_GSposECI."""
    theta = np.pi / 3600.0 / 24.0 * t
    zero = np.zeros_like(theta)
    O_IE = quaternion_matrix(np.array([np.cos(theta), zero, zero, -np.sin(theta)]))

    d2r = np.pi / 180.0
    cos_lat = np.cos(d2r * LAT)
    r_GS = RE + ALT
    r_e2g_E = np.array(
        [r_GS * cos_lat * np.cos(d2r * LON), r_GS * cos_lat * np.sin(d2r * LON), r_GS * np.sin(d2r * LAT)]
    )
    return np.einsum("uvn,v->un", O_IE, r_e2g_E)


def ground_visibility(r_b2g_I, r_e2g_I):
    """Satellite to ground station line of sight with the smooth horizon of Comm_LOS."""
    Rb = 100.0
    proj = np.sum(r_b2g_I * r_e2g_I, axis=0) / RE
    x = (proj - 0) / (-Rb - 0)
    return np.where(proj > 0, 0.0, np.where(proj < -Rb, 1.0, 3 * x**2 - 2 * x**3))


def geometry(r_e2b_I, LD, t):
    """Frame, sun, and ground station data at the time points of one case.

    The roll is a rotation about the third axis of the rolled frame, so the
    sun elevation, the ground station distance, and both lines of sight do not
    depend on the design. The sun and ground station vectors are kept in the
    rolled frame, where the components rotate them by the roll angle.
    """
    O_RI = rolled_frame(r_e2b_I)
    r_e2s_I = sun_direction(LD, t)
    r_e2g_I = ground_station(t)
    r_b2g_I = r_e2g_I - r_e2b_I[:3]

    sun = np.einsum("uvn,vn->un", O_RI, r_e2s_I)
    ground = np.einsum("uvn,vn->un", O_RI, r_b2g_I)
    norm = np.sqrt(np.sum(sun * sun, axis=0))
    norm = np.where(norm < 1e-15, 1e-5, norm)
    return {
        "O_RI": O_RI,
        "sun": sun,
        "elevation": np.arccos(sun[2] / norm),
        "LOS": sun_visibility(r_e2b_I, r_e2s_I),
        "ground": ground,
        "dist": np.sqrt(np.sum(ground * ground, axis=0)),
        "CommLOS": ground_visibility(r_b2g_I, r_e2g_I),
    }


def frame_rates(orbit, t, delta=10.0):
    """Rolled frame with its first and second time derivatives by fourth order central differences."""
    O = [rolled_frame(orbit(t + k * delta)) for k in (-2, -1, 0, 1, 2)]
    O1 = (O[0] - 8.0 * O[1] + 8.0 * O[3] - O[4]) / (12.0 * delta)
    O2 = (-O[0] + 16.0 * O[1] - 30.0 * O[2] + 16.0 * O[3] - O[4]) / (12.0 * delta**2)
    return O[2], O1, O2
