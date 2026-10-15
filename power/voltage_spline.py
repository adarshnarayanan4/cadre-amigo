"""Exact piecewise-quadratic representation of the modern CADRE voltage MBI.

The mapped temperature/current coordinates are explicit AMIGO states. Their
coordinate-map equations and voltage equation are differentiated by AMIGO,
including second derivatives, rather than treating the voltage lookup as fixed.
"""

import numpy as np
from scipy.interpolate import BSpline, PPoly

from cadre_paths import cadre_path
from power.mbi_modern import ModernMBI, knotopen


def quadratic_transform(m):
    knots = knotopen(3, m)
    breaks = np.unique(knots)[1:-1]
    transform = np.zeros((m, m))
    for column in range(m):
        spline = BSpline(knots, np.eye(m)[column], 2)
        polynomial = PPoly.from_spline(spline)
        transform[:3, column] = [spline(0), spline(0, 1), spline(0, 2) / 2]
        for j, point in enumerate(breaks):
            left = np.searchsorted(polynomial.x, point, side="left") - 1
            right = np.searchsorted(polynomial.x, point, side="right") - 1
            transform[3 + j, column] = polynomial.c[0, right] - polynomial.c[0, left]
    return breaks, transform


def hinge_basis(t, breaks):
    t = np.asarray(t)
    return np.stack(
        [np.ones_like(t), t, t * t, *[np.maximum(t - knot, 0) ** 2 for knot in breaks]],
        axis=-1,
    )


class VoltageSpline:
    def __init__(self):
        raw = np.loadtxt(cadre_path("data/Power/curve.dat"))
        sizes = raw[:3].astype(int)
        cursor = 3
        axes = []
        for size in sizes:
            axes.append(raw[cursor : cursor + size])
            cursor += size
        values = raw[cursor:].reshape(tuple(sizes), order="F")
        self.mbi = ModernMBI(values, axes, [6, 6, 15], [3, 3, 3])
        self.maps = [
            BSpline(knotopen(3, m), cx, 2) for m, cx in zip(self.mbi.ms, self.mbi.Cx)
        ]
        for axis in (0, 2):
            coordinate = self.maps[axis]
            # Increasing controls ensure monotonicity inside the table. The
            # endpoint curvatures ensure its quadratic extension is monotone
            # too, so bounded physical inputs cannot select another root.
            if not (
                np.all(np.diff(self.mbi.Cx[axis]) > 0)
                and coordinate(0, 2) <= 1e-8
                and coordinate(1, 2) >= -1e-8
            ):
                raise ValueError("Voltage coordinate map is not globally monotone")
        self.breaks_T, self.transform_T = quadratic_transform(6)
        self.breaks_I, self.transform_I = quadratic_transform(15)
        self.map_T = self.transform_T @ self.mbi.Cx[0]
        self.map_I = self.transform_I @ self.mbi.Cx[2]
        self.coefficients = self.mbi.C.reshape((6, 6, 15), order="F")

    def inverse(self, values, axis):
        values = np.asarray(values)
        coordinate = self.maps[axis]
        lo, hi = coordinate(0), coordinate(1)
        clipped = np.clip(values, lo, hi)
        t = np.clip((clipped - lo) / (hi - lo), 0, 1)
        for _ in range(30):
            step = (coordinate(t) - clipped) / coordinate(t, 1)
            t = np.clip(t - step, 0, 1)
            if np.max(np.abs(coordinate(t) - clipped)) < 1e-12:
                break
        return t

    def panel_coefficients(self, exposed_area, los):
        # Sum seven cells at their prescribed effective illuminated areas.
        area = (exposed_area * los[None, None, :]).transpose(2, 1, 0)
        mapped = self.inverse(area.ravel(), 1)
        area_basis = BSpline.design_matrix(mapped, knotopen(3, 6), 2).toarray()
        area_basis = area_basis.reshape((*area.shape, 6)).sum(axis=2)
        contracted = np.einsum("npb,abc->npac", area_basis, self.coefficients)
        return np.einsum(
            "ia,npab,jb->npij", self.transform_T, contracted, self.transform_I
        )

    def evaluate_panels(self, temperature, currents, coefficients):
        panel_temperature = temperature[[4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3]].T
        tT = self.inverse(panel_temperature, 0)
        tI = self.inverse(currents.T, 2)
        hT = hinge_basis(tT, self.breaks_T)
        hI = hinge_basis(tI, self.breaks_I)
        voltage = np.einsum("npi,npij,npj->np", hT, coefficients, hI).T
        return voltage, tT, tI

    def reference_panels(self, temperature, currents, exposed_area, los):
        """Replay with the original tensor B-splines, without hinge transforms."""
        area = (exposed_area * los[None, None, :]).transpose(2, 1, 0)
        area_basis = BSpline.design_matrix(
            self.inverse(area.ravel(), 1), knotopen(3, 6), 2
        ).toarray()
        area_basis = area_basis.reshape((*area.shape, 6)).sum(axis=2)
        temps = temperature[[4, 4, 4, 4, 0, 1, 2, 3, 0, 1, 2, 3]].T
        t_basis = BSpline.design_matrix(
            self.inverse(temps.ravel(), 0), knotopen(3, 6), 2
        ).toarray()
        i_basis = BSpline.design_matrix(
            self.inverse(currents.T.ravel(), 2), knotopen(3, 15), 2
        ).toarray()
        return np.einsum(
            "npa,npb,npc,abc->np",
            t_basis.reshape((*temps.shape, 6)),
            area_basis,
            i_basis.reshape((*temps.shape, 15)),
            self.coefficients,
        ).T


def make_voltage_component(spline):
    import amigo as am

    class CoupledPanelVoltage(am.Component):
        def __init__(self):
            super().__init__()
            self.add_input("temperature")
            self.add_input("current")
            self.add_input("tT")
            self.add_input("tI")
            self.add_input("voltage")
            self.add_data("coefficients", shape=(90,))
            self.add_constraint("temperature_map")
            self.add_constraint("current_map")
            self.add_constraint("voltage_res")

        def compute(self):
            tT, tI = self.inputs["tT"], self.inputs["tI"]
            hT = [1.0, tT, tT * tT]
            hI = [1.0, tI, tI * tI]
            for knot in spline.breaks_T:
                delta = tT - float(knot)
                positive = 0.5 * (delta + am.fabs(delta))
                hT.append(positive * positive)
            for knot in spline.breaks_I:
                delta = tI - float(knot)
                positive = 0.5 * (delta + am.fabs(delta))
                hI.append(positive * positive)
            temperature = sum(float(c) * h for c, h in zip(spline.map_T, hT))
            current = sum(float(c) * h for c, h in zip(spline.map_I, hI))
            coefficients = self.data["coefficients"]
            voltage = sum(
                coefficients[15 * a + b] * hT[a] * hI[b]
                for a in range(6)
                for b in range(15)
            )
            self.constraints["temperature_map"] = (
                temperature - self.inputs["temperature"]
            )
            self.constraints["current_map"] = current - self.inputs["current"]
            self.constraints["voltage_res"] = voltage - self.inputs["voltage"]

    return CoupledPanelVoltage()
