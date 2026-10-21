"""Analytic first and second lookup derivatives for shared CADRE angles."""

import numpy as np

from integrated.reference.benchmark import select_case
from integrated.reference.design import DesignLookups


def antenna_matrices(angle):
    """CADRE's quaternion rotation and its first two angle derivatives."""
    axis = np.array([1, -1, 0]) / np.sqrt(2)
    outer = np.outer(axis, axis)
    skew = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    identity = np.eye(3)
    cosine, sine = np.cos(angle), np.sin(angle)
    return (
        cosine * identity + (1 - cosine) * outer - sine * skew,
        -sine * identity + sine * outer - cosine * skew,
        -cosine * identity + cosine * outer + sine * skew,
    )


def _spherical_partials(direction, vector_gradient, vector_hessian):
    """Convert a rotated vector and its derivatives to azimuth and elevation."""
    x, y, z = direction.T
    dx, dy, dz = vector_gradient[:, 0], vector_gradient[:, 1], vector_gradient[:, 2]
    radius_squared = np.sum(direction * direction, axis=1)
    planar_radius_squared = x * x + y * y

    if np.min(planar_radius_squared / radius_squared) < 1e-12:
        raise ValueError("Antenna spherical coordinates are singular")

    planar_radius = np.sqrt(planar_radius_squared)
    azimuth = np.mod(np.arctan2(y, x), 2 * np.pi)
    elevation = np.arccos(np.clip(z / np.sqrt(radius_squared), -1, 1))
    numerator = x[:, None] * dy - y[:, None] * dx
    azimuth_gradient = numerator / planar_radius_squared[:, None]
    # Rotations preserve radius, so only z contributes to the elevation gradient.
    elevation_gradient = -dz / planar_radius[:, None]
    azimuth_hessian = np.empty((len(direction), 2, 2))
    elevation_hessian = np.empty_like(azimuth_hessian)

    for a in range(2):
        for b in range(2):
            azimuth_hessian[:, a, b] = (
                dx[:, a] * dy[:, b]
                + x * vector_hessian[:, 1, a, b]
                - dy[:, a] * dx[:, b]
                - y * vector_hessian[:, 0, a, b]
            ) / planar_radius_squared - numerator[:, b] * 2 * (
                x * dx[:, a] + y * dy[:, a]
            ) / planar_radius_squared**2
            elevation_hessian[:, a, b] = (
                -vector_hessian[:, 2, a, b] / planar_radius
                - z * dz[:, a] * dz[:, b] / planar_radius**3
            )

    return (
        azimuth,
        elevation,
        azimuth_gradient,
        elevation_gradient,
        azimuth_hessian,
        elevation_hessian,
    )


class SharedAngleLookups(DesignLookups):
    def __init__(self, data, n, voltage=None, case_index=0):
        super().__init__(data, n, voltage, case_index)
        self._fin_cache = None
        self._fin_coefficients = None
        selected = select_case(data, case_index)
        self.set_angles(self.fin_angle, float(np.asarray(selected["antAngle"]).item()))

    def set_angles(self, fin, antenna):
        self.fin_angle = float(fin)
        self.antenna_angle = float(antenna)
        self.antenna = np.broadcast_to(antenna_matrices(antenna)[0], (self.n, 3, 3))

    def exposure_partials(self, gamma):
        """Area derivatives in (roll, fin), in node/cell/panel order."""
        azimuth = np.mod(self.sun_azimuth - gamma, 2 * np.pi)
        azimuth_basis = self.area_axis.evaluate(azimuth)
        fin = self.solar_axes[0].evaluate(np.array([self.fin_angle]))

        if self._fin_cache != self.fin_angle:
            self._fin_coefficients = [
                np.einsum(
                    "a,nc,abcij->nbij",
                    fin_derivative[0],
                    self.solar_elevation_basis,
                    self.solar_tensor,
                )
                for fin_derivative in fin
            ]
            self._fin_cache = self.fin_angle

        coefficients = self._fin_coefficients

        def evaluate(roll_order, fin_order):
            return (-1) ** roll_order * np.einsum(
                "nb,nbij->nij", azimuth_basis[roll_order], coefficients[fin_order]
            )

        area = evaluate(0, 0)
        gradient = np.stack([evaluate(1, 0), evaluate(0, 1)], axis=-1)
        hessian = np.empty(area.shape + (2, 2))
        hessian[..., 0, 0] = evaluate(2, 0)
        hessian[..., 0, 1] = hessian[..., 1, 0] = evaluate(1, 1)
        hessian[..., 1, 1] = evaluate(0, 2)
        return area, gradient, hessian

    def exposure(self, gamma):
        area, gradient, hessian = self.exposure_partials(gamma)
        return area, gradient[..., 0], hessian[..., 0, 0]

    def _antenna_direction_partials(self, gamma):
        """Rotate the ground-station vector through roll and antenna mounting."""
        cosine, sine = np.cos(gamma), np.sin(gamma)
        x, y, z = self.ground_base.T
        rolled = np.column_stack((cosine * x + sine * y, -sine * x + cosine * y, z))
        first = np.column_stack(
            (-sine * x + cosine * y, -cosine * x - sine * y, np.zeros(self.n))
        )
        second = np.column_stack(
            (-cosine * x - sine * y, sine * x - cosine * y, np.zeros(self.n))
        )
        rotation, rotation_derivative, rotation_second_derivative = antenna_matrices(
            self.antenna_angle
        )
        value = rolled @ rotation.T
        # The last derivative axis is (roll, antenna angle).
        vector_gradient = np.stack(
            [first @ rotation.T, rolled @ rotation_derivative.T], axis=-1
        )
        vector_hessian = np.empty(value.shape + (2, 2))
        vector_hessian[..., 0, 0] = second @ rotation.T
        vector_hessian[..., 0, 1] = vector_hessian[..., 1, 0] = (
            first @ rotation_derivative.T
        )
        vector_hessian[..., 1, 1] = rolled @ rotation_second_derivative.T
        return value, vector_gradient, vector_hessian

    def gain_partials(self, gamma):
        """Differentiate antenna gain with respect to roll and mounting angle."""
        direction, vector_gradient, vector_hessian = self._antenna_direction_partials(
            gamma
        )
        (
            azimuth,
            elevation,
            azimuth_gradient,
            elevation_gradient,
            azimuth_hessian,
            elevation_hessian,
        ) = _spherical_partials(direction, vector_gradient, vector_hessian)

        azimuth_basis, azimuth_first, azimuth_second = self.gain_axes[0].evaluate(
            azimuth
        )
        elevation_basis, elevation_first, elevation_second = self.gain_axes[1].evaluate(
            elevation
        )

        def evaluate_spline(azimuth_basis, elevation_basis):
            return np.einsum(
                "na,ab,nb->n", azimuth_basis, self.gain_coeff, elevation_basis
            )

        gain_azimuth = evaluate_spline(azimuth_first, elevation_basis)
        gain_elevation = evaluate_spline(azimuth_basis, elevation_first)
        gradient = (
            gain_azimuth[:, None] * azimuth_gradient
            + gain_elevation[:, None] * elevation_gradient
        )
        hessian = np.empty_like(azimuth_hessian)

        for a in range(2):
            for b in range(2):
                hessian[:, a, b] = (
                    evaluate_spline(azimuth_second, elevation_basis)
                    * azimuth_gradient[:, a]
                    * azimuth_gradient[:, b]
                    + evaluate_spline(azimuth_first, elevation_first)
                    * (
                        azimuth_gradient[:, a] * elevation_gradient[:, b]
                        + elevation_gradient[:, a] * azimuth_gradient[:, b]
                    )
                    + evaluate_spline(azimuth_basis, elevation_second)
                    * elevation_gradient[:, a]
                    * elevation_gradient[:, b]
                    + gain_azimuth * azimuth_hessian[:, a, b]
                    + gain_elevation * elevation_hessian[:, a, b]
                )

        return evaluate_spline(azimuth_basis, elevation_basis), gradient, hessian

    def gain(self, gamma):
        value, gradient, hessian = self.gain_partials(gamma)
        return value, gradient[:, 0], hessian[:, 0, 0]

    def voltage_fin_partials(self, temperature, currents, area, gradient, hessian):
        """Return fin, roll/fin, temperature/fin, current/fin, and fin/fin partials.

        Area inputs use node/cell/panel order; outputs use node/panel order.
        """
        effective = area.transpose(0, 2, 1) * self.LOS[:, None, None]
        area_roll = gradient[..., 0].transpose(0, 2, 1) * self.LOS[:, None, None]
        area_fin = gradient[..., 1].transpose(0, 2, 1) * self.LOS[:, None, None]
        area_roll_fin = hessian[..., 0, 1].transpose(0, 2, 1) * self.LOS[:, None, None]
        area_fin_fin = hessian[..., 1, 1].transpose(0, 2, 1) * self.LOS[:, None, None]
        temperature_basis, temperature_derivative, _ = self.voltage_axes[0].evaluate(
            temperature[:, self.panel_temperature]
        )
        current_basis, current_derivative, _ = self.voltage_axes[2].evaluate(currents)
        _, area_derivative, area_second_derivative = self.voltage_axes[1].evaluate(
            effective
        )
        fin_derivative = (area_derivative * area_fin[..., None]).sum(axis=2)
        roll_fin_derivative = (
            area_second_derivative * area_roll[..., None] * area_fin[..., None]
            + area_derivative * area_roll_fin[..., None]
        ).sum(axis=2)
        fin_second_derivative = (
            area_second_derivative * area_fin[..., None] ** 2
            + area_derivative * area_fin_fin[..., None]
        ).sum(axis=2)

        def evaluate_spline(temperature_basis, area_basis, current_basis):
            return np.einsum(
                "npa,npb,npc,abc->np",
                temperature_basis,
                area_basis,
                current_basis,
                self.voltage.coefficients,
            )

        return (
            evaluate_spline(temperature_basis, fin_derivative, current_basis),
            evaluate_spline(temperature_basis, roll_fin_derivative, current_basis),
            evaluate_spline(temperature_derivative, fin_derivative, current_basis),
            evaluate_spline(temperature_basis, fin_derivative, current_derivative),
            evaluate_spline(temperature_basis, fin_second_derivative, current_basis),
        )
