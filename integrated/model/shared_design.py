"""External CADRE physics with shared fin and antenna design variables."""

import amigo as am
import numpy as np
from scipy.sparse import coo_matrix

from integrated.model.design import DesignPhysics
from thermal.thermal_amigo import alpha_c, alpha_r, q_sol


class DesignAngles(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("finAngle")
        self.add_input("antAngle")


class SharedDesignPhysics:
    """Extend each case's physics with shared fin and antenna angle derivatives.

    Inputs follow DesignPhysics, then fin and antenna angles. Residual rows
    contain five temperatures, twelve panel voltages, and antenna gain per node.
    """

    def __init__(self, lookups):
        self.lookups = lookups
        self.base = DesignPhysics(lookups)
        self.n = lookups.n
        self.nvars = self.base.nvars + 2
        self.ncon = self.base.ncon
        self.fin, self.antenna = self.base.nvars, self.base.nvars + 1
        self.row_columns = []
        base_positions = []
        fin_positions, antenna_positions = [], []
        offset = 0

        for row, columns in enumerate(self.base.row_columns):
            added = self.fin if row < 17 * self.n else self.antenna
            expanded = np.r_[columns, added]
            self.row_columns.append(expanded)
            base_positions.extend(offset + np.arange(len(columns)))
            added_position = offset + len(columns)

            if added == self.fin:
                fin_positions.append(added_position)
            else:
                antenna_positions.append(added_position)

            offset += len(expanded)

        self.base_positions = np.asarray(base_positions)
        self.fin_positions = np.asarray(fin_positions)
        self.antenna_positions = np.asarray(antenna_positions)
        self.rowp = np.r_[0, np.cumsum([len(c) for c in self.row_columns])]
        self.cols = np.concatenate(self.row_columns)
        self.point_provider = None
        self.cache_point = self.last = None

    def get_constraint_jacobian_csr(self):
        return self.ncon, self.nvars, self.rowp, self.cols

    def register(self, model):
        self.base.set_expressions()
        model.add_component("angles", 1, DesignAngles())

        for name, value, lower, upper in [
            ("finAngle", self.lookups.fin_angle, 0, np.pi / 2),
            ("antAngle", self.lookups.antenna_angle, -np.pi / 4, np.pi / 4),
        ]:
            model.set_meta("value", f"angles.{name}[:]", value)
            model.set_meta("lower", f"angles.{name}[:]", lower)
            model.set_meta("upper", f"angles.{name}[:]", upper)

        self.inputs = self.base.inputs + ["angles.finAngle[:]", "angles.antAngle[:]"]
        self.constraints = self.base.constraints.copy()
        model.add_external_component(
            "design_physics", self, self.inputs, self.constraints
        )
        self.bind(model)

    def bind(self, model, prefix=""):
        if prefix:
            self.inputs = [prefix + name for name in self.inputs]
            self.constraints = [prefix + name for name in self.constraints]

        self.model = self.base.model = model
        self.base.inputs = self.inputs[:-2]
        self.base.constraints = self.constraints

    def compute(self, point):
        base = self.base
        lookups = self.lookups
        node_count = self.n
        lookups.set_angles(point[self.fin], point[self.antenna])
        base.compute(point[: base.nvars])
        self.residual = base.residual
        area, gradient, hessian = lookups.exposure_partials(point[base.g])
        self.voltage_fin = lookups.voltage_fin_partials(
            point[base.t], point[base.i], area, gradient, hessian
        )
        self.gain_partials = lookups.gain_partials(point[base.g])
        cell = point[base.cell]
        alpha = alpha_r + (alpha_c - alpha_r) * cell
        solar = q_sol * lookups.LOS[:, None] / lookups.mass_cp[None, :]
        thermal_fin, thermal_roll_fin, thermal_fin_fin = [
            np.zeros((node_count, 5)) for _ in range(3)
        ]

        for panel, temperature_index in enumerate(lookups.thermal_temperature):
            for target, values in [
                (thermal_fin, gradient[..., 1]),
                (thermal_roll_fin, hessian[..., 0, 1]),
                (thermal_fin_fin, hessian[..., 1, 1]),
            ]:
                target[:, temperature_index] += (
                    values[:, :, panel] * alpha[None, :, panel]
                ).sum(axis=1) * solar[:, panel]

        self.thermal_cross = thermal_roll_fin, thermal_fin_fin, gradient[..., 1], solar
        self.jacobian = np.zeros(len(self.cols))
        self.jacobian[self.base_positions] = base.jacobian
        self.jacobian[self.fin_positions] = -np.r_[
            thermal_fin.T.ravel(), self.voltage_fin[0].T.ravel()
        ]
        self.jacobian[self.antenna_positions] = -self.gain_partials[1][:, 1]

    def evaluate(self, x, con, grad, jac):
        self.last = np.array(x, copy=True)
        point = x[: self.nvars]

        if self.cache_point is None or not np.array_equal(point, self.cache_point):
            self.compute(point)
            self.cache_point = point.copy()

        con[:] = self.residual
        grad[:] = 0
        jac[:] = self.jacobian

    def hessian_correction(self, shape):
        if self.point_provider is not None:
            full = np.asarray(self.point_provider())
            inputs = full[self.model.get_indices(self.inputs)]
            multipliers = full[self.model.get_indices(self.constraints)]
            self.last = np.r_[inputs, multipliers]

            if self.cache_point is None or not np.array_equal(inputs, self.cache_point):
                self.compute(inputs)
                self.cache_point = inputs.copy()

        if self.last is None:
            return coo_matrix(shape).tocsr()

        base = self.base
        node_count = self.n
        multipliers = self.last[self.nvars :]
        base.last = np.r_[self.last[: base.nvars], multipliers]
        result = base.hessian_correction(shape)
        # Residuals are grouped by temperature, voltage, then gain.
        thermal_multipliers = multipliers[: 5 * node_count].reshape(5, node_count).T
        voltage_multipliers = (
            multipliers[5 * node_count : 17 * node_count].reshape(12, node_count).T
        )
        gain_multipliers = multipliers[17 * node_count :]
        thermal_roll_fin, thermal_fin_fin, area_fin, solar = self.thermal_cross
        (
            _,
            voltage_roll_fin,
            voltage_temperature_fin,
            voltage_current_fin,
            voltage_fin_fin,
        ) = self.voltage_fin
        gain_hessian = self.gain_partials[2]
        rows, cols, values = [], [], []

        def append(row, column, value, symmetric=False):
            row, column, value = np.broadcast_arrays(row, column, value)
            rows.extend(row.ravel())
            cols.extend(column.ravel())
            values.extend(value.ravel())

            if symmetric:
                rows.extend(column.ravel())
                cols.extend(row.ravel())
                values.extend(value.ravel())

        append(
            self.fin,
            self.fin,
            -np.sum(thermal_multipliers * thermal_fin_fin)
            - np.sum(voltage_multipliers * voltage_fin_fin),
        )
        append(
            base.g,
            self.fin,
            -np.sum(thermal_multipliers * thermal_roll_fin, axis=1)
            - np.sum(voltage_multipliers * voltage_roll_fin, axis=1),
            True,
        )
        append(
            self.antenna,
            self.antenna,
            -np.sum(gain_multipliers * gain_hessian[:, 1, 1]),
        )
        append(base.g, self.antenna, -gain_multipliers * gain_hessian[:, 0, 1], True)

        for panel in range(12):
            temperature_index = self.lookups.panel_temperature[panel]
            append(
                base.t[:, temperature_index],
                self.fin,
                -voltage_multipliers[:, panel] * voltage_temperature_fin[:, panel],
                True,
            )
            append(
                base.i[:, panel],
                self.fin,
                -voltage_multipliers[:, panel] * voltage_current_fin[:, panel],
                True,
            )

        for cell in range(84):
            panel = cell % 12
            temperature_index = self.lookups.thermal_temperature[panel]
            cross = (
                -thermal_multipliers[:, temperature_index]
                * (alpha_c - alpha_r)
                * area_fin[:, cell // 12, panel]
                * solar[:, panel]
            )
            append(base.cell.ravel()[cell], self.fin, np.sum(cross), True)

        indices = self.model.get_indices(self.inputs)
        correction = coo_matrix(
            (
                values,
                (
                    indices[np.asarray(rows, dtype=int)],
                    indices[np.asarray(cols, dtype=int)],
                ),
            ),
            shape=shape,
        )
        return result + correction.tocsr()
