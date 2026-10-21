"""Roll-dependent exposure, antenna gain, and voltage lookup references."""

import numpy as np
from scipy.interpolate import BSpline

from attitude.attitude_reference import (
    angular_acceleration,
    angular_velocity,
    attitude_roll,
    attitude_torque,
    combine_rotation_matrices,
    rotation_matrix_rates,
)
from communication.communication_gain_validation import ModernMBI2D, load_gain_data
from power.mbi_modern import knotopen
from power.voltage_spline import VoltageSpline
from solar.solar_mbi_validation import build_solar_mbi
from thermal.thermal_amigo import cp_b, cp_f, m_b, m_f
from integrated.reference.benchmark import select_case


class PhysicalSplineAxis:
    def __init__(self, mbi, axis):
        self.knots = knotopen(int(mbi.ks[axis]), int(mbi.ms[axis]))
        self.degree = int(mbi.ks[axis]) - 1

        self.coordinate = BSpline(self.knots, mbi.Cx[axis], self.degree)
        self.basis = BSpline(self.knots, np.eye(int(mbi.ms[axis])), self.degree)

    def evaluate(self, values):
        values = np.asarray(values)
        lo, hi = self.coordinate(0), self.coordinate(1)

        clipped = np.clip(values, lo, hi)
        t = np.clip((clipped - lo) / (hi - lo), 0, 1)

        for _ in range(35):
            t = np.clip(
                t - (self.coordinate(t) - clipped) / self.coordinate(t, 1), 0, 1
            )

            if np.max(np.abs(self.coordinate(t) - clipped)) < 1e-12:
                break

        first = 1 / self.coordinate(t, 1)
        second = -self.coordinate(t, 2) * first**3

        outside = (values < lo) | (values > hi)

        first = np.where(outside, 0, first)
        second = np.where(outside, 0, second)

        b = self.basis(t)
        d = self.basis(t, 1) * first[..., None]
        dd = (
            self.basis(t, 2) * first[..., None] ** 2
            + self.basis(t, 1) * second[..., None]
        )

        return b, d, dd


class DesignLookups:
    def __init__(self, data, n, voltage=None, case_index=0):
        data = select_case(data, case_index)
        self.n = n
        self.voltage = voltage or VoltageSpline()
        self.voltage_axes = [PhysicalSplineAxis(self.voltage.mbi, i) for i in range(3)]

        self.base_orientation = np.asarray(data["O_RI"])[:, :, :n].transpose(2, 0, 1)
        sun = np.asarray(data["r_e2s_I"])[:, :n].T
        sun_base = np.einsum("nij,nj->ni", self.base_orientation, sun)

        self.sun_azimuth = np.arctan2(sun_base[:, 1], sun_base[:, 0])
        self.sun_elevation = np.arccos(
            np.clip(sun_base[:, 2] / np.linalg.norm(sun_base, axis=1), -1, 1)
        )

        solar = build_solar_mbi()
        self.solar_reference = solar
        axes = [PhysicalSplineAxis(solar, i) for i in range(3)]
        fin = float(np.asarray(data["finAngle"]).ravel()[0])
        self.fin_angle = fin
        self.sun_inertial = sun

        bf = axes[0].evaluate(np.full(n, fin))[0]

        be = axes[2].evaluate(self.sun_elevation)[0]
        # Solar output order is cell-fast (Fortran); store cell-major like thermal.
        tensor = solar.C.reshape((4, 10, 8, 7, 12), order="F")
        self.solar_axes = axes
        self.solar_tensor = tensor
        self.solar_elevation_basis = be
        self.area_coeff = np.einsum("na,nc,abcij->nbij", bf, be, tensor)
        self.area_axis = axes[1]

        ga, ge, gvalues = load_gain_data()
        gain = ModernMBI2D(gvalues, [ga, ge], [15, 15], [4, 4])
        self.gain_reference = gain
        self.gain_axes = [PhysicalSplineAxis(gain, i) for i in range(2)]
        self.gain_coeff = gain.C.reshape((15, 15), order="F")

        ground = np.asarray(data["r_b2g_I"])[:, :n].T
        self.ground_inertial = ground
        self.ground_base = np.einsum("nij,nj->ni", self.base_orientation, ground)
        self.antenna = np.asarray(data["O_AB"])[:, :, :n].transpose(2, 0, 1)

        self.LOS = np.asarray(data["LOS"])[:n]

        self.panel_temperature = np.array([4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3])
        self.thermal_temperature = np.array(
            [4 if p < 4 else (p + 1) % 4 for p in range(12)]
        )
        self.mass_cp = np.where(self.thermal_temperature == 4, m_b * cp_b, m_f * cp_f)

    def exposure(self, gamma):
        az = np.mod(self.sun_azimuth - gamma, 2 * np.pi)
        b, d, dd = self.area_axis.evaluate(az)

        return tuple(np.einsum("nb,nbij->nij", v, self.area_coeff) for v in (b, -d, dd))

    def gain(self, gamma):
        c, s = np.cos(gamma), np.sin(gamma)
        x, y, z = self.ground_base.T

        v = np.column_stack((c * x + s * y, -s * x + c * y, z))
        dv = np.column_stack((-s * x + c * y, -c * x - s * y, np.zeros(self.n)))
        ddv = np.column_stack((-c * x - s * y, s * x - c * y, np.zeros(self.n)))

        v, dv, ddv = [np.einsum("nij,nj->ni", self.antenna, a) for a in (v, dv, ddv)]

        x, y, z = v.T

        dx, dy, dz = dv.T

        ddx, ddy, ddz = ddv.T

        radius2 = np.sum(v * v, axis=1)
        planar = x * x + y * y

        if (
            np.min(planar / radius2) < 1e-12
            or np.min((radius2 - z * z) / radius2) < 1e-12
        ):
            raise ValueError("Antenna spherical coordinates are singular")

        az = np.mod(np.arctan2(y, x), 2 * np.pi)
        el = np.arccos(np.clip(z / np.sqrt(radius2), -1, 1))

        da = (x * dy - y * dx) / planar
        dda = (x * ddy - y * ddx) / planar - da * 2 * (x * dx + y * dy) / planar

        root = np.sqrt(radius2 - z * z)
        de = -dz / root
        dde = -ddz / root - z * dz * dz / root**3

        ba, dba, ddba = self.gain_axes[0].evaluate(az)
        be, dbe, ddbe = self.gain_axes[1].evaluate(el)

        ev = lambda a, b: np.einsum("na,ab,nb->n", a, self.gain_coeff, b)

        value = ev(ba, be)

        first = ev(dba, be) * da + ev(ba, dbe) * de
        second = (
            ev(ddba, be) * da * da
            + 2 * ev(dba, dbe) * da * de
            + ev(ba, ddbe) * de * de
            + ev(dba, be) * dda
            + ev(ba, dbe) * dde
        )

        return value, first, second

    def panel_voltage(self, temperature, currents, area, area_d, area_dd):
        # n, panel, cell; shared temperature and current for seven cells.
        temp = temperature[:, self.panel_temperature]
        effective = area.transpose(0, 2, 1) * self.LOS[:, None, None]
        ea_d = area_d.transpose(0, 2, 1) * self.LOS[:, None, None]
        ea_dd = area_dd.transpose(0, 2, 1) * self.LOS[:, None, None]

        bt, dt, ddt = self.voltage_axes[0].evaluate(temp)
        bi, di, ddi = self.voltage_axes[2].evaluate(currents)
        ba, da, dda = self.voltage_axes[1].evaluate(effective)

        a = ba.sum(axis=2)
        ag = (da * ea_d[..., None]).sum(axis=2)
        agg = (dda * ea_d[..., None] ** 2 + da * ea_dd[..., None]).sum(axis=2)

        ev = lambda t, a, i: np.einsum(
            "npa,npb,npc,abc->np", t, a, i, self.voltage.coefficients
        )

        value = ev(bt, a, bi)
        grad = np.stack((ev(bt, ag, bi), ev(dt, a, bi), ev(bt, a, di)), axis=-1)

        hess = np.empty((self.n, 12, 3, 3))
        hess[:, :, 0, 0] = ev(bt, agg, bi)
        hess[:, :, 1, 1] = ev(ddt, a, bi)
        hess[:, :, 2, 2] = ev(bt, a, ddi)
        hess[:, :, 0, 1] = hess[:, :, 1, 0] = ev(dt, ag, bi)
        hess[:, :, 0, 2] = hess[:, :, 2, 0] = ev(bt, ag, di)
        hess[:, :, 1, 2] = hess[:, :, 2, 1] = ev(dt, a, di)

        return value, grad, hess

    def geometry(self, gamma, reference=False):
        area = self.exposure(gamma)[0].transpose(1, 2, 0)
        gain = self.gain(gamma)[0]

        orientation = combine_rotation_matrices(
            attitude_roll(gamma), self.base_orientation
        )

        if reference:
            from sun.sun_reference import sun_position_spherical
            from communication.communication_reference import vector_spherical
            from solar.solar_mbi_validation import solar_exposed_area_mbi

            sun_body = np.einsum("nij,nj->ni", orientation, self.sun_inertial)
            az, el = sun_position_spherical(sun_body)
            area = solar_exposed_area_mbi(self.fin_angle, az, el, self.solar_reference)

            ground_body = np.einsum("nij,nj->ni", orientation, self.ground_inertial)
            ground_antenna = np.einsum("nij,nj->ni", self.antenna, ground_body)
            az, el = vector_spherical(ground_antenna.T)
            gain = self.gain_reference.evaluate(np.column_stack((az, el)))

        rate = rotation_matrix_rates(orientation, self.dt)
        w = angular_velocity(orientation, rate)
        wdot = angular_acceleration(w, self.dt)
        torque = attitude_torque(w, wdot)

        return area, gain, w.T, torque.T, orientation, rate, wdot
