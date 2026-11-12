"""
CADRE CubeSat design
--------------------

    maximize    Data_0 + Data_1 + ... + Data_5
    subject to  0.2 <= SOC <= 1,  -10 <= I_bat <= 5
                SOC(tf) = SOC(0) in each case

- Six orbit cases of a CubeSat share the solar cells, the fin angle, and the antenna angle
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import amigo as am

import environment as env
import mbi

# Problem sizes of the CADRE paper
num_nodes = 1500
num_ctrl = 300
num_cases = 6

# The 12 hour window in seconds
horizon = 43200.0

# Thermal constants of thermal_temperature.py in SI units
m_f, m_b = 0.4, 2.0
cp_f, cp_b = 0.6e3, 2.0e3
A_T = 2.66e-3
alpha_c, alpha_r = 0.9, 0.2
eps_c, eps_r = 0.87, 0.88
q_sol = 1360.0
K_sb = 5.67051e-8

# Battery constants of battery.py with the capacity in coulombs
sigma = 1e-10
eta = 0.99
Cp = 2900.0 * 0.001 * 3600.0
IR = 0.9
T0 = 293.0
alpha_b = np.log(1 / 1.1**5)

# Reaction wheel constants of reactionwheel.py
V_rw, a_rw, b_rw, I0_rw = 4.0, 4.9e-4, 4.5e2, 0.017
J_RW = 2.8e-5

# Spacecraft inertia of Attitude_Torque
J_sc = (0.018, 0.018, 0.006)

# Units of the body rates and wheel torques that keep the wheel steps of order one
rate_unit = 1e-2
torque_unit = 1e-6

# Link budget of Comm_BitRate
alpha_comm = (
    299792458**2 * 10 ** (12.9 / 10.0) * 10 ** (-2.0 / 10.0)
    / 16.0 / (2 * np.arccos(0.0)) ** 2 / 437e6**2 / 1.3806503e-23
    / 10 ** (5.0 / 10.0) / 500.0 / 1e6
)


def panel_side(p):
    """Temperature that heats panel p and the mass and heat capacity it heats."""
    if p < 4:
        return 4, m_b * cp_b
    return (p + 1) % 4, m_f * cp_f


def voltage_side(p):
    """Temperature that sets the cell voltage of panel p."""
    return 4 if p < 4 else p % 4


def rotate_roll(gamma, v):
    """Components of a rolled frame vector in the body frame."""
    c = am.cos(gamma)
    s = am.sin(gamma)
    return [c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]]


def azimuth_of(x, y):
    """Azimuth on [0, 2 pi) with the branch of kinematics.arctan."""
    return np.pi + am.atan2(-y, -x)


class ControlPoints(am.Component):
    """Control points of the three profiles of one case."""

    def __init__(self):
        super().__init__()

        # Control point bounds and starting values of examples/mdp.py
        self.add_input("CP_P_comm", lower=0.0, upper=25.0, value=0.1)
        self.add_input("CP_gamma", lower=0.0, upper=np.pi / 2.0, value=np.pi / 4.0)
        self.add_input("CP_Isetpt", shape=12, lower=0.0, upper=0.4, value=0.2)


class ControlSpline(am.Component):
    """Profile values at one node from the four control points of its span."""

    def __init__(self):
        super().__init__()

        # Weights of the four nonzero basis functions at this node
        self.add_data("N", shape=4)

        self.add_input("CP_P_comm", shape=4)
        self.add_input("CP_gamma", shape=4)
        self.add_input("CP_Isetpt", shape=(4, 12))

        self.add_input("P_comm", label="communication power")
        self.add_input("gamma", label="roll angle")
        self.add_input("Isetpt", shape=12, label="panel currents")

        self.add_constraint("res", shape=14)

    def compute(self):
        N = self.data["N"]
        CP_P_comm = self.inputs["CP_P_comm"]
        CP_gamma = self.inputs["CP_gamma"]
        CP_Isetpt = self.inputs["CP_Isetpt"]
        Isetpt = self.inputs["Isetpt"]

        P_comm = 0.0
        gamma = 0.0
        for j in range(4):
            P_comm = P_comm + N[j] * CP_P_comm[j]
            gamma = gamma + N[j] * CP_gamma[j]

        res = [self.inputs["P_comm"] - P_comm, self.inputs["gamma"] - gamma]
        for p in range(12):
            current = 0.0
            for j in range(4):
                current = current + N[j] * CP_Isetpt[j, p]
            res.append(Isetpt[p] - current)

        self.constraints["res"] = res


class BodyRate(am.Component):
    """Body rates from central differences of the attitude as Attitude_RotationMtxRates and Attitude_Angular."""

    def __init__(self):
        super().__init__()

        # Rolled frames at the previous, this, and the next node
        self.add_data("O_RI", shape=(3, 9))
        self.add_data("w_prev")
        self.add_data("w_next")
        self.add_data("inv_h")

        self.add_input("gamma", shape=3)
        self.add_input("w_B", shape=3, label="body rates in rate_unit")

        self.add_constraint("res", shape=3)

    def compute(self):
        R = self.data["O_RI"]
        gamma = self.inputs["gamma"]
        w_B = self.inputs["w_B"]

        # Body frame rows at the three nodes
        O = []
        for s in range(3):
            c = am.cos(gamma[s])
            sn = am.sin(gamma[s])
            rows = [[c * R[s, j] + sn * R[s, 3 + j] for j in range(3)]]
            rows.append([-sn * R[s, j] + c * R[s, 3 + j] for j in range(3)])
            rows.append([R[s, 6 + j] for j in range(3)])
            O.append(rows)

        # Time derivative of the body frame
        inv_h = self.data["inv_h"]
        dO = [
            [(self.data["w_next"] * O[2][u][v] - self.data["w_prev"] * O[0][u][v]) * inv_h for v in range(3)]
            for u in range(3)
        ]

        rate = [0.0, 0.0, 0.0]
        for v in range(3):
            rate[0] = rate[0] + dO[2][v] * O[1][1][v]
            rate[1] = rate[1] + dO[0][v] * O[1][2][v]
            rate[2] = rate[2] + dO[1][v] * O[1][0][v]

        self.constraints["res"] = [w_B[k] - rate[k] / rate_unit for k in range(3)]


class Torque(am.Component):
    """Wheel torque from central differences of the body rates as Attitude_AngularRates and Attitude_Torque."""

    def __init__(self):
        super().__init__()

        self.add_data("w_prev")
        self.add_data("w_next")
        self.add_data("inv_h")

        self.add_input("w_B", shape=(3, 3), label="body rates at the three nodes in rate_unit")
        self.add_input("T_RW", shape=3, label="wheel torques in torque_unit")

        self.add_constraint("res", shape=3)

    def compute(self):
        T_RW = self.inputs["T_RW"]
        inv_h = self.data["inv_h"]

        # Body rates at the three nodes in rad/s
        w = [[rate_unit * self.inputs["w_B"][s, k] for k in range(3)] for s in range(3)]
        wdot = [(self.data["w_next"] * w[2][k] - self.data["w_prev"] * w[0][k]) * inv_h for k in range(3)]

        # Angular momentum balance of the body
        Jw = [J_sc[k] * w[1][k] for k in range(3)]
        torque = [
            J_sc[0] * wdot[0] + (w[1][1] * Jw[2] - w[1][2] * Jw[1]),
            J_sc[1] * wdot[1] + (w[1][2] * Jw[0] - w[1][0] * Jw[2]),
            J_sc[2] * wdot[2] + (w[1][0] * Jw[1] - w[1][1] * Jw[0]),
        ]

        self.constraints["res"] = [T_RW[k] - torque[k] / torque_unit for k in range(3)]


class Panel(am.Component):
    """Exposed areas, absorbed sunlight, and electric power of one solar panel at one node."""

    def __init__(self):
        super().__init__()

        # Exposed area spline over azimuth of the seven cells with the elevation axis contracted
        self.add_data("area", shape=(10, 28))

        # Sun direction in the rolled frame and the sun line of sight
        self.add_data("sun", shape=3)
        self.add_data("LOS")

        # Mass times specific heat of the fin or the body that the panel heats
        self.add_data("mcp")

        # Coordinate splines of the fin, azimuth, temperature, area, and current axes
        self.add_data("cx_fin", shape=4)
        self.add_data("cx_az", shape=10)
        self.add_data("cx_T", shape=6)
        self.add_data("cx_A", shape=6)
        self.add_data("cx_I", shape=15)

        # Cell voltage spline over current with one column per temperature and area pair
        self.add_data("voltage", shape=(15, 36))

        self.add_input("gamma")
        self.add_input("finAngle")
        self.add_input("cellInstd", shape=7)
        self.add_input("T", label="temperature that sets the cell voltage")
        self.add_input("Isetpt", label="panel current")
        self.add_input("A_exp", label="exposed area of the panel")

        # Shares of the heating of the fin or the body and of the battery power
        self.add_constraint("heat")
        self.add_constraint("power")
        self.add_constraint("exposed")

    def compute(self):
        LOS = self.data["LOS"]
        ci = self.inputs["cellInstd"]
        Isetpt = self.inputs["Isetpt"]

        # Sun azimuth in the body frame and the exposed area of each cell
        sun = rotate_roll(self.inputs["gamma"], self.data["sun"])
        t_az = am.bspline_inverse(self.data["cx_az"], azimuth_of(sun[0], sun[1]), 4)
        t_fin = am.bspline_inverse(self.data["cx_fin"], self.inputs["finAngle"], 4)
        fin = [am.bspline_basis(t_fin, a, 4, 4) for a in range(4)]

        area = []
        for c in range(7):
            value = 0.0
            for a in range(4):
                value = value + fin[a] * am.bspline(self.data["area"], t_az, 4, column=7 * a + c)
            area.append(value)

        # Exposed area of the panel before the line of sight
        total_area = area[0]
        for c in range(1, 7):
            total_area = total_area + area[c]

        self.constraints["exposed"] = self.inputs["A_exp"] - total_area

        # Sunlight absorbed by the cells and the radiators
        absorbed = 0.0
        for c in range(7):
            alpha = alpha_c * ci[c] + alpha_r - alpha_r * ci[c]
            absorbed = absorbed + alpha * area[c]

        self.constraints["heat"] = -(absorbed * (q_sol * LOS) / self.data["mcp"])

        # Temperature and current axes of the voltage spline contracted once for the panel
        tT = am.bspline_inverse(self.data["cx_T"], self.inputs["T"], 3)
        tI = am.bspline_inverse(self.data["cx_I"], Isetpt, 3)

        W = []
        for b in range(6):
            value = 0.0
            for a in range(6):
                spline = am.bspline(self.data["voltage"], tI, 3, column=6 * a + b)
                value = value + am.bspline_basis(tT, a, 6, 3) * spline
            W.append(value)

        # Panel voltage as the sum of its seven cell voltages
        V = 0.0
        for c in range(7):
            tA = am.bspline_inverse(self.data["cx_A"], LOS * area[c], 3)
            cell = 0.0
            for b in range(6):
                cell = cell + am.bspline_basis(tA, b, 6, 3) * W[b]
            V = V + cell

        self.constraints["power"] = -(V * Isetpt)


class Heat(am.Component):
    """Absorbed solar heating of the four fins and the body that the panels add up."""

    def __init__(self):
        super().__init__()

        self.add_input("Q", shape=5, label="absorbed solar heating")

        self.add_constraint("res", shape=5)

    def compute(self):
        Q = self.inputs["Q"]
        self.constraints["res"] = [Q[i] for i in range(5)]


class Battery(am.Component):
    """Battery power after the loads and the battery current at one node."""

    def __init__(self):
        super().__init__()

        self.add_input("T", shape=5, label="temperatures")
        self.add_input("w_RW", shape=3)
        self.add_input("T_RW", shape=3, label="wheel torques in torque_unit")
        self.add_input("SOC")
        self.add_input("P_comm")
        self.add_input("P_bat", label="battery power")
        self.add_input("I_bat", label="battery current")

        # Battery power with the loads that the twelve panel powers complete
        self.add_constraint("power")
        self.add_constraint("current")

    def compute(self):
        T = self.inputs["T"]
        w = self.inputs["w_RW"]
        T_RW = self.inputs["T_RW"]
        P_bat = self.inputs["P_bat"]

        # Transmitter, wheels, and the 2 W bus load
        P_RW = [V_rw * (a_rw * w[k] + b_rw * torque_unit * T_RW[k]) ** 2 + V_rw * I0_rw for k in range(3)]
        load = 5.0 * self.inputs["P_comm"] + (P_RW[0] + P_RW[1] + P_RW[2]) + 2.0

        self.constraints["power"] = P_bat + load

        # Battery current at the open circuit voltage derated by the body temperature
        voc = 3.0 + (am.exp(self.inputs["SOC"]) - 1.0) / (np.e - 1)
        V_bat = IR * voc * (2.0 - am.exp(alpha_b * (T[4] - T0) / T0))

        self.constraints["current"] = self.inputs["I_bat"] - P_bat / V_bat


class Comm(am.Component):
    """Antenna gain and download rate as Comm_VectorAnt, Comm_GainPattern, and Comm_BitRate."""

    def __init__(self):
        super().__init__()

        # Ground station vector in the rolled frame, its distance, and line of sight
        self.add_data("ground", shape=3)
        self.add_data("dist")
        self.add_data("CommLOS")

        # Gain spline over azimuth with one column per elevation control point
        self.add_data("cx_az", shape=15)
        self.add_data("cx_el", shape=15)
        self.add_data("gain", shape=(15, 15))

        self.add_input("gamma")
        self.add_input("antAngle")
        self.add_input("P_comm")
        self.add_input("Gt", label="transmitter gain")
        self.add_input("Dr", label="download rate")

        self.add_constraint("res", shape=2)

    def compute(self):
        r_B = rotate_roll(self.inputs["gamma"], self.data["ground"])

        # Antenna rotation about the diagonal axis of the body frame
        half = 0.5 * self.inputs["antAngle"]
        q = [am.cos(half), am.sin(half) / np.sqrt(2), -am.sin(half) / np.sqrt(2), 0.0]
        A = [[q[0], -q[3], q[2]], [q[3], q[0], -q[1]], [-q[2], q[1], q[0]], [q[1], q[2], q[3]]]
        B = [[q[0], q[3], -q[2]], [-q[3], q[0], q[1]], [q[2], -q[1], q[0]], [q[1], q[2], q[3]]]

        r_A = []
        for u in range(3):
            value = 0.0
            for v in range(3):
                O_uv = A[0][u] * B[0][v] + A[1][u] * B[1][v] + A[2][u] * B[2][v] + A[3][u] * B[3][v]
                value = value + O_uv * r_B[v]
            r_A.append(value)

        # Gain at the azimuth and elevation of the station in the antenna frame
        dist = self.data["dist"]
        t_az = am.bspline_inverse(self.data["cx_az"], azimuth_of(r_A[0], r_A[1]), 4)
        t_el = am.bspline_inverse(self.data["cx_el"], am.acos(r_A[2] / dist), 4)

        gain = 0.0
        for j in range(15):
            gain = gain + am.bspline_basis(t_el, j, 15, 4) * am.bspline(self.data["gain"], t_az, 4, column=j)

        # Download rate of the link budget
        rate = alpha_comm * self.inputs["P_comm"] * self.inputs["Gt"] * self.data["CommLOS"] / (dist * 1e3) ** 2

        self.constraints["res"] = [self.inputs["Gt"] - gain, self.inputs["Dr"] - rate]


def thermal_rate(T, Q, E, P_comm):
    """Temperature rates of the fins and the body."""
    f = [Q[i] - K_sb * A_T * E[i] * T[i] ** 4 for i in range(5)]
    f[4] = f[4] + 4.0 * P_comm / m_b / cp_b
    return f


def wheel_rate(w, w_B, T_RW):
    """Wheel speed rates in the rotating body frame."""
    cross = [
        -w_B[2] * w[1] + w_B[1] * w[2],
        w_B[2] * w[0] - w_B[0] * w[2],
        -w_B[1] * w[0] + w_B[0] * w[1],
    ]
    return [-T_RW[k] / J_RW - cross[k] for k in range(3)]


def charge_rate(SOC, P_bat, T_body):
    """Charge rate from the battery power at the derated cell voltage."""
    voc = 3 + (am.exp(SOC) - 1.0) / (np.e - 1)
    V = IR * voc * (2.0 - am.exp(alpha_b * (T_body - T0) / T0))
    return -sigma / 24 * SOC + eta / Cp * (P_bat / V)


def rk4_step(rate, y, h):
    """Combination of the four RK4 stages of rk4.py."""
    n = len(y)
    a = rate(y)
    b = rate([y[i] + h / 2.0 * a[i] for i in range(n)])
    c = rate([y[i] + h / 2.0 * b[i] for i in range(n)])
    d = rate([y[i] + h * c[i] for i in range(n)])
    return [y[i] + h / 6.0 * (a[i] + 2 * (b[i] + c[i]) + d[i]) for i in range(n)]


class ThermalStep(am.Component):
    """One RK4 step of the five temperatures."""

    def __init__(self):
        super().__init__()

        self.add_data("h")

        self.add_input("T1", shape=5)
        self.add_input("T2", shape=5)
        self.add_input("Q", shape=5)
        self.add_input("E", shape=5)
        self.add_input("P_comm")

        self.add_constraint("res", shape=5)

    def compute(self):
        T1 = self.inputs["T1"]
        T2 = self.inputs["T2"]
        Q = self.inputs["Q"]
        E = self.inputs["E"]
        P_comm = self.inputs["P_comm"]

        # The heating and transmitter power hold their start values over the step
        rate = lambda T: thermal_rate(T, Q, E, P_comm)
        end = rk4_step(rate, [T1[i] for i in range(5)], self.data["h"])

        self.constraints["res"] = [T2[i] - end[i] for i in range(5)]


class WheelStep(am.Component):
    """One RK4 step of the three wheel speeds."""

    def __init__(self):
        super().__init__()

        self.add_data("h")

        self.add_input("w1", shape=3)
        self.add_input("w2", shape=3)
        self.add_input("w_B", shape=3)
        self.add_input("T_RW", shape=3)

        self.add_constraint("res", shape=3)

    def compute(self):
        w1 = self.inputs["w1"]
        w2 = self.inputs["w2"]

        # Body rates and wheel torques in rad/s and N m
        w_B = [rate_unit * self.inputs["w_B"][k] for k in range(3)]
        T_RW = [torque_unit * self.inputs["T_RW"][k] for k in range(3)]

        rate = lambda w: wheel_rate(w, w_B, T_RW)
        end = rk4_step(rate, [w1[k] for k in range(3)], self.data["h"])

        self.constraints["res"] = [w2[k] - end[k] for k in range(3)]


class ChargeStep(am.Component):
    """One RK4 step of the state of charge."""

    def __init__(self):
        super().__init__()

        self.add_data("h")

        self.add_input("SOC1")
        self.add_input("SOC2")
        self.add_input("P_bat")
        self.add_input("T_body")

        self.add_constraint("res")

    def compute(self):
        rate = lambda s: [charge_rate(s[0], self.inputs["P_bat"], self.inputs["T_body"])]
        end = rk4_step(rate, [self.inputs["SOC1"]], self.data["h"])

        self.constraints["res"] = self.inputs["SOC2"] - end[0]


class DataStep(am.Component):
    """One RK4 step of the downloaded data."""

    def __init__(self):
        super().__init__()

        self.add_data("h")

        self.add_input("Data1")
        self.add_input("Data2")
        self.add_input("Dr")

        self.add_constraint("res")

    def compute(self):
        rate = lambda s: [self.inputs["Dr"]]
        end = rk4_step(rate, [self.inputs["Data1"]], self.data["h"])

        self.constraints["res"] = self.inputs["Data2"] - end[0]


class InitialConditions(am.Component):
    """Temperatures of 273 K, wheels at rest, and no data at the start of every case."""

    def __init__(self):
        super().__init__()

        self.add_input("T", shape=5)
        self.add_input("w_RW", shape=3)
        self.add_input("Data")

        self.add_constraint("res", shape=9)

    def compute(self):
        T = self.inputs["T"]
        w = self.inputs["w_RW"]

        self.constraints["res"] = [T[i] - 273.0 for i in range(5)] + [w[k] for k in range(3)] + [
            self.inputs["Data"]
        ]


class Periodicity(am.Component):
    """The battery ends each case at its initial charge."""

    def __init__(self):
        super().__init__()

        self.add_input("SOCi")
        self.add_input("SOCf")

        self.add_constraint("res")

    def compute(self):
        self.constraints["res"] = self.inputs["SOCi"] - self.inputs["SOCf"]


class Shared(am.Component):
    """Design shared by all cases and the radiating emittance of the fins and the body."""

    def __init__(self):
        super().__init__()

        self.add_input("cellInstd", shape=(7, 12), lower=0.0, upper=1.0, value=1.0)
        self.add_input("finAngle", lower=0.0, upper=np.pi / 2.0, value=np.pi / 4.0)
        self.add_input("antAngle", lower=-np.pi / 4.0, upper=np.pi / 4.0, value=0.0)
        self.add_input("E", shape=5, label="emittance over heat capacity")

        self.add_constraint("res", shape=5)

    def compute(self):
        ci = self.inputs["cellInstd"]
        E = self.inputs["E"]

        total = [0.0] * 5
        for p in range(12):
            emittance = 0.0
            for c in range(7):
                emittance = emittance + (eps_c * ci[c, p] + eps_r - eps_r * ci[c, p])
            i, mcp = panel_side(p)
            total[i] = total[i] + emittance / mcp

        self.constraints["res"] = [E[i] - total[i] for i in range(5)]


class Objective(am.Component):
    """The data downloaded over all cases, maximized."""

    def __init__(self, num_cases):
        super().__init__()
        self.num_cases = num_cases

        self.add_input("Data", shape=num_cases, label="final downloaded data")

        self.add_objective("obj")

    def compute(self):
        Data = self.inputs["Data"]

        total = 0.0
        for i in range(self.num_cases):
            total = total - Data[i]

        self.objective["obj"] = total


def read_tables(data_dir):
    """The exposed area, cell voltage, and antenna gain fits of the CADRE data files."""
    data_dir = Path(data_dir)

    # Solar_ExposedArea of solar.py
    raw1 = np.genfromtxt(data_dir / "Area10.txt")
    raw2 = np.loadtxt(data_dir / "Area_all.txt")
    na, nz, ne = 10, 73, 37

    angle = raw1[:na].copy()
    azimuth = raw1[na : na + nz].copy()
    elevation = raw1[na + nz - 1 : na + nz - 1 + ne].copy()
    azimuth[nz - 1] = 2.0 * np.pi
    angle[0], angle[-1] = 0.0, np.pi / 2.0
    azimuth[0], azimuth[-1] = 0.0, 2 * np.pi
    elevation[0], elevation[-1] = 0.0, np.pi

    cells = np.zeros((na, nz, ne, 84))
    flat_size = na * nz * ne
    for p in range(12):
        for c in range(7):
            cells[:, :, :, 7 * p + c] = raw2[7 * p + c][119 : 119 + flat_size].reshape((na, nz, ne))
    solar = mbi.MBI(cells, [angle, azimuth, elevation], [4, 10, 8], [4, 4, 4])

    # Power_CellVoltage of power.py
    dat = np.genfromtxt(data_dir / "curve.dat")
    nT, nA, nI = (int(v) for v in dat[:3])
    T = dat[3 : 3 + nT]
    A = dat[3 + nT : 3 + nT + nA]
    I = dat[3 + nT + nA : 3 + nT + nA + nI]
    V = dat[3 + nT + nA + nI :].reshape((nT, nA, nI), order="F")
    voltage = mbi.MBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

    # Comm_GainPattern of comm.py
    rawGdata = np.genfromtxt(data_dir / "Gain.txt")
    rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order="F")
    az = np.linspace(0, 2 * np.pi, 361)
    el = np.linspace(0, 2 * np.pi, 361)
    gain = mbi.MBI(rawG, [az, el], [15, 15], [4, 4])

    return {
        "solar": solar.coefficients(),
        "solar_cx": [np.asarray(cx, dtype=float) for cx in solar.Cx],
        "voltage": voltage.coefficients()[..., 0],
        "voltage_cx": [np.asarray(cx, dtype=float) for cx in voltage.Cx],
        "gain": gain.coefficients()[..., 0],
        "gain_cx": [np.asarray(cx, dtype=float) for cx in gain.Cx],
    }


def load_tables(data_dir, cache):
    """Read the table fits saved by an earlier run or fit them from the CADRE data files."""
    if cache.exists():
        with np.load(cache, allow_pickle=False) as f:
            out = {name: f[name] for name in ("solar", "voltage", "gain")}
            for name in ("solar", "voltage", "gain"):
                out[f"{name}_cx"] = [f[f"{name}_cx{i}"] for i in range(3 if name != "gain" else 2)]
        return out

    print("Fitting the CADRE tables with the MBI port")
    tables = read_tables(data_dir)

    flat = {name: tables[name] for name in ("solar", "voltage", "gain")}
    for name in ("solar", "voltage", "gain"):
        for i, cx in enumerate(tables[f"{name}_cx"]):
            flat[f"{name}_cx{i}"] = cx
    np.savez(cache, **flat)

    return load_tables(data_dir, cache)


def control_weights(t_param):
    """Weights and first indices of the four nonzero control basis functions at spline parameters."""
    knots = mbi.knotopen(4, num_ctrl)
    weights = np.zeros((len(t_param), 4))
    first = np.zeros(len(t_param), dtype=int)
    for i, t in enumerate(t_param):
        weights[i], first[i] = mbi.basis(4, t, knots)
    return weights, first


def area_tables(tables, elevation):
    """Exposed area table of every panel at every node with its elevation axis contracted."""
    C = tables["solar"]
    t_el = mbi.inverse(tables["solar_cx"][2], 4, elevation)
    B = mbi.basis_values(4, 8, t_el)

    # Axes of the fin angle, azimuth, elevation, and cell with the cells of panel p at 7 p to 7 p + 6
    D = np.einsum("abcq,nc->nbaq", C, B).reshape(len(elevation), 10, 4, 12, 7)
    return D.transpose(0, 3, 1, 2, 4).reshape(12 * len(elevation), 10, 28)


def case_data(tables, r_e2b_I, LD, t):
    """Data of the Panel and Comm components of one case at the time points t."""
    geo = env.geometry(r_e2b_I, LD, t)
    n = len(t)
    panels = 12 * n

    # Voltage axes of temperature, area, and current with the current along the rows
    V = tables["voltage"]

    return geo, {
        "panel.area": area_tables(tables, geo["elevation"]),
        "panel.sun": np.repeat(geo["sun"].T, 12, axis=0),
        "panel.LOS": np.repeat(geo["LOS"], 12),
        "panel.mcp": np.tile([panel_side(p)[1] for p in range(12)], n),
        "panel.cx_fin": np.tile(tables["solar_cx"][0], (panels, 1)),
        "panel.cx_az": np.tile(tables["solar_cx"][1], (panels, 1)),
        "panel.cx_T": np.tile(tables["voltage_cx"][0], (panels, 1)),
        "panel.cx_A": np.tile(tables["voltage_cx"][1], (panels, 1)),
        "panel.cx_I": np.tile(tables["voltage_cx"][2], (panels, 1)),
        "panel.voltage": np.tile(V.transpose(2, 0, 1).reshape(15, 36), (panels, 1, 1)),
        "comm.ground": geo["ground"].T,
        "comm.dist": geo["dist"],
        "comm.CommLOS": geo["CommLOS"],
        "comm.cx_az": np.tile(tables["gain_cx"][0], (n, 1)),
        "comm.cx_el": np.tile(tables["gain_cx"][1], (n, 1)),
        "comm.gain": np.tile(tables["gain"], (n, 1, 1)),
    }


def link_controls(model, group, first, n):
    """Link the control points of the four nonzero basis functions of every node."""
    for j in range(4):
        idx = first + j
        model.link("ctrl.CP_P_comm", f"{group}.CP_P_comm[:, {j}]", src_indices=idx)
        model.link("ctrl.CP_gamma", f"{group}.CP_gamma[:, {j}]", src_indices=idx)
        model.link("ctrl.CP_Isetpt", f"{group}.CP_Isetpt[:, {j}, :]", src_indices=idx)


def link_panels(case):
    """Link every panel to its controls and temperature and add its shares to the heating and the battery power."""
    for p in range(12):
        panel = f"[{p}::12]"
        case.link("spline.gamma", f"panel.gamma{panel}")
        case.link(f"spline.Isetpt[:, {p}]", f"panel.Isetpt{panel}")
        case.link(f"battery.T[:, {voltage_side(p)}]", f"panel.T{panel}")
        case.link(f"panel.heat{panel}", f"heat.res[:, {panel_side(p)[0]}]")
        case.link(f"panel.power{panel}", "battery.power")


def build_case(first):
    """One CADRE case on its node grid with the RK4 steps."""
    N = num_nodes
    K = N - 1

    case = am.Model("case")
    case.add_component("ctrl", num_ctrl, ControlPoints())
    case.add_component("spline", N, ControlSpline())
    case.add_component("rate", N, BodyRate())
    case.add_component("torque", N, Torque())
    case.add_component("panel", 12 * N, Panel())
    case.add_component("heat", N, Heat())
    case.add_component("battery", N, Battery())
    case.add_component("comm", N, Comm())
    case.add_component("thermal", K, ThermalStep())
    case.add_component("wheel", K, WheelStep())
    case.add_component("charge", K, ChargeStep())
    case.add_component("data", K, DataStep())
    case.add_component("ic", 1, InitialConditions())
    case.add_component("periodic", 1, Periodicity())

    link_controls(case, "spline", first, N)

    # Central differences with one sided ends as in attitude.py
    nodes = np.arange(N)
    prev = np.maximum(nodes - 1, 0)
    nxt = np.minimum(nodes + 1, N - 1)
    for s, idx in enumerate((prev, nodes, nxt)):
        case.link("spline.gamma", f"rate.gamma[:, {s}]", src_indices=idx)
        case.link("rate.w_B", f"torque.w_B[:, {s}, :]", src_indices=idx)

    # Controls and states shared by the node components
    link_panels(case)
    case.link("spline.gamma", "comm.gamma")
    case.link("spline.P_comm", "battery.P_comm")
    case.link("spline.P_comm", "comm.P_comm")
    case.link("torque.T_RW", "battery.T_RW")

    # RK4 steps with the inputs held at the start of each step
    case.link("battery.T[:-1, :]", "thermal.T1")
    case.link("battery.T[1:, :]", "thermal.T2")
    case.link("heat.Q[:-1, :]", "thermal.Q")
    case.link("spline.P_comm[:-1]", "thermal.P_comm")
    case.link("battery.w_RW[:-1, :]", "wheel.w1")
    case.link("battery.w_RW[1:, :]", "wheel.w2")
    case.link("rate.w_B[:-1, :]", "wheel.w_B")
    case.link("torque.T_RW[:-1, :]", "wheel.T_RW")
    case.link("battery.SOC[:-1]", "charge.SOC1")
    case.link("battery.SOC[1:]", "charge.SOC2")
    case.link("battery.P_bat[:-1]", "charge.P_bat")
    case.link("battery.T[:-1, 4]", "charge.T_body")
    case.link("data.Data2[:-1]", "data.Data1[1:]")
    case.link("comm.Dr[:-1]", "data.Dr")

    # Initial states and the periodic charge
    case.link("battery.T[0, :]", "ic.T[0, :]")
    case.link("battery.w_RW[0, :]", "ic.w_RW[0, :]")
    case.link("data.Data1[0]", "ic.Data[0]")
    case.link("battery.SOC[0]", "periodic.SOCi[0]")
    case.link(f"battery.SOC[{N - 1}]", "periodic.SOCf[0]")

    return case


def build_model(case):
    """The six cases linked to the shared design and the total data objective."""
    model = am.Model("cadre_rk4")
    for i in range(num_cases):
        model.add_model(f"pt{i}", case)
    model.add_component("shared", 1, Shared())
    model.add_component("obj", 1, Objective(num_cases))

    # Shared design and emittance at every node of every case
    N = num_nodes
    cells = np.tile(np.arange(7), (N, 1))
    for i in range(num_cases):
        pt = f"pt{i}"
        for p in range(12):
            model.link(f"shared.cellInstd[0, :, {p}]", f"{pt}.panel.cellInstd[{p}::12, :]", src_indices=cells)
        model.link("shared.finAngle[0]", f"{pt}.panel.finAngle")
        model.link("shared.antAngle[0]", f"{pt}.comm.antAngle")
        model.link("shared.E", f"{pt}.thermal.E", src_indices=np.zeros(N - 1, dtype=int))
        model.link(f"{pt}.data.Data2[{N - 2}]", f"obj.Data[0, {i}]")

    # One copy of the tables that do not change from node to node
    panel_tables = ("panel.cx_fin", "panel.cx_az", "panel.cx_T", "panel.cx_A", "panel.cx_I", "panel.voltage")
    comm_tables = ("comm.cx_az", "comm.cx_el", "comm.gain")
    for names, count in ((panel_tables, 12 * N), (comm_tables, N)):
        for name in names:
            for i in range(num_cases):
                model.link(f"pt0.{name}", f"pt{i}.{name}", src_indices=np.zeros(count, dtype=int))

    return model


def launch_cases(data_dir):
    """Initial orbit and launch day of each case from launch1.dat as CADRE_mdp.py reads them."""
    launch = np.loadtxt(Path(data_dir) / "launch1.dat")
    return launch[1::2, 1:], launch[1::2, 0] - 2451545


def layout(tables, data_dir):
    """Grid, control weights, and node data of every case."""
    N = num_nodes
    h = horizon / (N - 1)
    t = np.arange(N) * h
    weights, first = control_weights(mbi.paramuni(4, num_ctrl, N))

    # Central difference weights with one sided ends
    nodes = np.arange(N)
    prev = np.maximum(nodes - 1, 0)
    nxt = np.minimum(nodes + 1, N - 1)
    w_side = np.where((nodes == 0) | (nodes == N - 1), 1.0, 0.5)

    r0s, LDs = launch_cases(data_dir)
    cases = []
    for i in range(num_cases):
        r_e2b_I = env.orbit_rk4(r0s[i], N, h)
        geo, data = case_data(tables, r_e2b_I, LDs[i], t)

        O = geo["O_RI"].reshape(9, N).T
        data["spline.N"] = weights
        data["rate.O_RI"] = np.stack([O[prev], O, O[nxt]], axis=1)
        for group in ("rate", "torque"):
            data[f"{group}.w_prev"] = w_side
            data[f"{group}.w_next"] = w_side
            data[f"{group}.inv_h"] = np.full(N, 1.0 / h)
        for group in ("thermal", "wheel", "charge", "data"):
            data[f"{group}.h"] = np.full(N - 1, h)

        cases.append((geo, data))

    return t, first, cases


def set_guess(model):
    """CADRE's starting design with resting states and the matching emittance."""
    ci = np.ones((7, 12))
    E = np.zeros(5)
    for p in range(12):
        i, mcp = panel_side(p)
        E[i] += np.sum(eps_c * ci[:, p] + eps_r - eps_r * ci[:, p]) / mcp
    model.set_meta("value", "shared.E", E)

    for i in range(num_cases):
        pt = f"pt{i}"
        model.set_meta("value", f"{pt}.spline.P_comm", 0.1)
        model.set_meta("value", f"{pt}.spline.gamma", np.pi / 4.0)
        model.set_meta("value", f"{pt}.spline.Isetpt", 0.2)
        model.set_meta("value", f"{pt}.battery.T", 273.0)
        model.set_meta("value", f"{pt}.battery.SOC", 0.5)


def set_bounds(model):
    """The battery limits of the paper as bounds on the charge and the current at every node."""
    for i in range(num_cases):
        pt = f"pt{i}"
        model.set_meta("lower", f"{pt}.battery.SOC", 0.2)
        model.set_meta("upper", f"{pt}.battery.SOC", 1.0)
        model.set_meta("lower", f"{pt}.battery.I_bat", -10.0)
        model.set_meta("upper", f"{pt}.battery.I_bat", 5.0)


def fix_problem(model, problem):
    """Equal bounds that hold the design the paper's baseline and geometry problems leave out."""
    if problem == "full":
        return

    for i in range(num_cases):
        model.set_meta("lower", f"pt{i}.ctrl.CP_gamma", np.pi / 4.0)
        model.set_meta("upper", f"pt{i}.ctrl.CP_gamma", np.pi / 4.0)

    if problem == "baseline":
        model.set_meta("lower", "shared.finAngle", np.pi / 4.0)
        model.set_meta("upper", "shared.finAngle", np.pi / 4.0)
        model.set_meta("lower", "shared.antAngle", 0.0)
        model.set_meta("upper", "shared.antAngle", 0.0)


def design_names():
    """Design variables of the CADRE MDP."""
    names = []
    for i in range(num_cases):
        pt = f"pt{i}"
        names += [f"{pt}.ctrl.CP_P_comm", f"{pt}.ctrl.CP_gamma", f"{pt}.ctrl.CP_Isetpt", f"{pt}.battery.SOC[0]"]
    return names + ["shared.cellInstd", "shared.finAngle", "shared.antAngle"]


def periodic_names():
    """Periodic charge rows that the optimizer enforces and the analysis leaves out"""
    names = []
    for i in range(num_cases):
        names.append(f"pt{i}.periodic.res")
    return names


def simulate(model, x, tol=1e-9, max_iter=20):
    """States of the design in x by Newton's method on the state rows with the design held."""
    n = model.num_variables
    cons = np.asarray(model.constraint_indices)
    design = np.ravel(model.get_indices(design_names()))
    periodic = np.ravel(model.get_indices(periodic_names()))

    # The design holds its values and every row but the periodic charge defines a state
    states = np.setdiff1d(np.setdiff1d(np.arange(n), cons), design)
    rows = np.setdiff1d(cons, periodic)
    if rows.size != states.size:
        raise ValueError(f"{rows.size} state equations for {states.size} states")

    xv = x.get_vector().get_array()
    xv[cons] = 0.0
    g = model.create_vector()
    mat = model.create_matrix()

    # The model reads its data on the device when it runs there
    model.get_data_vector().copy_host_to_device()

    for it in range(max_iter):
        # Residuals of the state rows at x
        x.copy_host_to_device()
        model.eval_gradient(x, g, zero_fixed_rows=False)
        g.copy_device_to_host()
        r = np.array(g.get_vector().get_array()[rows])
        if np.max(np.abs(r)) < tol:
            return

        model.eval_hessian(x, mat, zero_fixed_rows_and_columns=False)
        mat.copy_data_device_to_host()
        _, _, _, rowp, cols = mat.get_nonzero_structure()
        K = sp.csr_matrix((np.array(mat.get_data()), np.asarray(cols), np.asarray(rowp)), shape=(n, n))
        xv[states] -= spla.splu(K[rows][:, states].tocsc()).solve(r)

    raise RuntimeError(f"The simulation of the design did not converge in {max_iter} iterations")


def data_trajectory(x, pt):
    """Downloaded data of a case at every node, zero at the start."""
    return np.concatenate([[0.0], np.asarray(x[f"{pt}.data.Data2"])])


def solar_power(x, pt):
    """Solar power of a case as the battery power plus the transmitter, wheel, and bus loads."""
    w = np.asarray(x[f"{pt}.battery.w_RW"])
    T_RW = np.asarray(x[f"{pt}.torque.T_RW"])
    P_RW = V_rw * (a_rw * w + b_rw * torque_unit * T_RW) ** 2 + V_rw * I0_rw
    return np.asarray(x[f"{pt}.battery.P_bat"]) + 5.0 * np.asarray(x[f"{pt}.spline.P_comm"]) + P_RW.sum(axis=1) + 2.0


class Snapshots:
    """Data and charge trajectories of every case at every fiftieth iterate."""

    def __init__(self, every=50):
        self.every = every
        self.iters = []
        self.data = []
        self.SOC = []

    def __call__(self, k, x):
        if k % self.every:
            return

        x.copy_device_to_host()
        self.iters.append(k)
        self.data.append([data_trajectory(x, f"pt{i}") for i in range(num_cases)])
        self.SOC.append([np.asarray(x[f"pt{i}.battery.SOC"]) for i in range(num_cases)])


def solution_of(model, x, t, cases, problem, opt_data=None, solve_time=0.0, snapshots=None):
    """The design, the trajectories of every case, and the history of the solve at a point."""
    x.copy_device_to_host()

    solution = {
        "problem": problem,
        "t": t,
        "fin_angle_deg": np.degrees(float(x["shared.finAngle"][0])),
        "antenna_angle_deg": np.degrees(float(x["shared.antAngle"][0])),
        "cellInstd": np.asarray(x["shared.cellInstd"]).reshape(7, 12),
        "iterations": 0 if opt_data is None else len(opt_data["iterations"]),
        "solve_time": solve_time,
        "converged": False if opt_data is None else bool(opt_data["converged"]),
    }

    total = 0.0
    for i, (geo, _) in enumerate(cases):
        pt = f"pt{i}"
        data = data_trajectory(x, pt)
        total += data[-1]

        solution[f"{pt}.data"] = data[-1]
        solution[f"{pt}.data_t"] = data
        solution[f"{pt}.P_sol"] = solar_power(x, pt)
        solution[f"{pt}.LOS"] = np.asarray(geo["LOS"])
        solution[f"{pt}.CommLOS"] = np.asarray(geo["CommLOS"])
        for name in ("T", "w_RW", "SOC", "I_bat"):
            solution[f"{pt}.{name}"] = np.asarray(x[f"{pt}.battery.{name}"])
        for name in ("P_comm", "gamma", "Isetpt"):
            solution[f"{pt}.{name}"] = np.asarray(x[f"{pt}.spline.{name}"])
        solution[f"{pt}.gain"] = np.asarray(x[f"{pt}.comm.Gt"])
        solution[f"{pt}.A_exp"] = np.asarray(x[f"{pt}.panel.A_exp"]).reshape(num_nodes, 12)

    solution["total_data"] = total

    if opt_data is not None:
        for name in ("nlp_error", "inf_pr", "inf_du", "compl", "barrier_param", "objective"):
            solution[f"hist_{name}"] = np.array([it[name] for it in opt_data["iterations"]])

    if snapshots is not None and snapshots.iters:
        solution["snap_iters"] = np.array(snapshots.iters)
        solution["snap_data"] = np.array(snapshots.data)
        solution["snap_SOC"] = np.array(snapshots.SOC)

    return solution


def parse_value(text):
    """Option value as a bool, an int, a float, or a string."""
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


if __name__ == "__main__":
    src = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="The six case CADRE problem of Hwang et al. 2014")
    parser.add_argument("--build", action="store_true", help="compile the model before solving")
    parser.add_argument("--problem", choices=("full", "geometry", "baseline"), default="full", help="the paper's problem")
    parser.add_argument("--bound", action="append", default=[], metavar="NAME=LOWER,UPPER", help="bounds for a study")
    parser.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE", help="solver option")
    args = parser.parse_args()

    data_dir = src / "data"
    build = src / "build"
    results = src / "results"
    results.mkdir(exist_ok=True)
    sys.path.insert(0, str(build))

    # The model with the data of every case
    tables = load_tables(data_dir, data_dir / "cadre_tables.npz")
    t, first, cases = layout(tables, data_dir)

    model = build_model(build_case(first))
    for i, (geo, data) in enumerate(cases):
        for name, value in data.items():
            model.set_data(f"pt{i}.{name}", value)

    set_guess(model)
    set_bounds(model)
    fix_problem(model, args.problem)
    for spec in args.bound:
        name, _, values = spec.partition("=")
        lower, upper = (float(v) for v in values.split(","))
        model.set_meta("lower", name, lower)
        model.set_meta("upper", name, upper)

    if args.build:
        model.build_module(source_dir=build, build_dir=build / "cmake")
    model.initialize()
    print(f"{model.num_variables} variables and {model.num_constraints} constraints")

    # The settings the paper's problem solves with
    options = {
        "max_iterations": 600,
        "initial_barrier_param": 0.1,
        "fraction_to_boundary": 0.995,
        "hessian_perturbation_indices": "free",
        "objective_tolerance": 1e-6,
    }
    for kv in args.opt:
        key, _, val = kv.partition("=")
        options[key] = parse_value(val)

    snapshots = Snapshots()
    options["iteration_callback"] = snapshots

    # CADRE's start, the default design with the states its run computes
    x = model.create_vector()
    x.copy(model.get_initial_point())
    t0 = time.perf_counter()
    simulate(model, x)
    print(f"Simulated the default design in {time.perf_counter() - t0:.1f} s")
    np.savez(results / "cadre_initial.npz", **solution_of(model, x, t, cases, "initial"))

    # The solve
    t0 = time.perf_counter()
    opt = am.Optimizer(model, x)
    opt_data = opt.optimize(options)
    solve_time = time.perf_counter() - t0

    x = opt.get_optimized_point()
    solution = solution_of(model, x, t, cases, args.problem, opt_data, solve_time, snapshots)

    # The result
    for i in range(num_cases):
        pt = f"pt{i}"
        SOC, I_bat = solution[f"{pt}.SOC"], solution[f"{pt}.I_bat"]
        print(f"Case {i}: data {solution[f'{pt}.data']:.3f}, SOC {SOC.min():.3f} to {SOC.max():.3f}, I_bat {I_bat.min():.2f} to {I_bat.max():.2f} A")

    print(f"Total data {solution['total_data']:.3f}")
    print(f"Fin {solution['fin_angle_deg']:.2f} deg, antenna {solution['antenna_angle_deg']:.2f} deg")
    print(f"Converged {solution['converged']} in {solution['iterations']} iterations and {solve_time:.0f} s")

    # A study with its own bounds or options leaves the solutions of the paper's problems in place
    name = "cadre_study.npz" if args.bound or args.opt else f"cadre_{args.problem}.npz"
    np.savez(results / name, **solution)
    print(f"Saved {results / name}")
