"""AMIGO roll components and analytic external lookup derivatives."""

import amigo as am
import numpy as np
from scipy.sparse import coo_matrix

from thermal.thermal_amigo import (
    A_T,
    K,
    alpha_c,
    alpha_r,
    cp_b,
    eps_c,
    eps_r,
    m_b,
    q_sol,
)


class ThermalDesignState(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("temperature", shape=(5,))
        self.add_input("temperature_dot", shape=(5,))
        self.add_input("P_comm")

        self.add_constraint("res", shape=(5,))


class SurfaceAllocation(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("cellInstd", shape=(84,))


class LookupResiduals(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constraint("voltage", shape=(12,))
        self.add_constraint("gain")


class RolledOrientation(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("gamma")
        self.add_input("O_BI", shape=(9,))

        self.add_data("O_RI", shape=(9,))

        self.add_constraint("res", shape=(9,))

    def compute(self):
        g = self.inputs["gamma"]
        o = self.inputs["O_BI"]

        base = self.data["O_RI"]

        c, s = am.cos(g), am.sin(g)

        calc = [c * base[j] + s * base[j + 3] for j in range(3)]
        calc += [-s * base[j] + c * base[j + 3] for j in range(3)]
        calc += [base[j] for j in range(6, 9)]

        self.constraints["res"] = [o[j] - calc[j] for j in range(9)]


class DesignPhysics:
    """External residuals with analytic Jacobian and multiplier-weighted Hessian."""

    def __init__(self, lookups):
        self.lookups = lookups
        self.n = lookups.n

        n = self.n
        self.nvars = 37 * n + 84
        self.ncon = 18 * n

        self.g = np.arange(n)
        self.t = np.arange(n, 6 * n).reshape(5, n).T
        self.p = np.arange(6 * n, 7 * n)
        self.i = np.arange(7 * n, 19 * n).reshape(12, n).T

        self.v = np.arange(19 * n, 31 * n).reshape(12, n).T
        self.gain = np.arange(31 * n, 32 * n)
        self.tdot = np.arange(32 * n, 37 * n).reshape(5, n).T
        self.cell = np.arange(37 * n, 37 * n + 84).reshape(7, 12)

        self.row_columns = []

        # Constraint order is temperature component-major, panel-major, gain.
        for k in range(5):
            panels = np.flatnonzero(lookups.thermal_temperature == k)
            cells = self.cell[:, panels].ravel()

            for node in range(n):
                cols = [self.g[node], self.t[node, k], self.tdot[node, k]]

                if k == 4:
                    cols.append(self.p[node])

                cols.extend(cells)

                self.row_columns.append(np.array(sorted(cols), dtype=int))

        for panel in range(12):
            k = lookups.panel_temperature[panel]

            for node in range(n):
                self.row_columns.append(
                    np.array(
                        [
                            self.g[node],
                            self.t[node, k],
                            self.i[node, panel],
                            self.v[node, panel],
                        ]
                    )
                )

        for node in range(n):
            self.row_columns.append(np.array([self.g[node], self.gain[node]]))

        self.rowp = np.concatenate(([0], np.cumsum([len(c) for c in self.row_columns])))
        self.cols = np.concatenate(self.row_columns)
        self.rows = np.repeat(np.arange(self.ncon), np.diff(self.rowp))

        self.last = None
        self.cache_point = None
        self.point_provider = None
        self.fixed_global_indices = np.array([], dtype=int)

    def get_constraint_jacobian_csr(self):
        return self.ncon, self.nvars, self.rowp, self.cols

    def set_expressions(self):
        self.inputs = ["values.Gamma[:]"] + [
            f"thermal.temperature[:, {k}]" for k in range(5)
        ]
        self.inputs += ["thermal.P_comm[:]"] + [
            f"values.Isetpt[:, {p}]" for p in range(12)
        ]
        self.inputs += [f"power_solar.V_sol[:, {p}]" for p in range(12)]
        self.inputs += ["comm_rate.gain[:]"] + [
            f"thermal.temperature_dot[:, {k}]" for k in range(5)
        ]
        self.inputs += ["surface.cellInstd[0, :]"]

        self.constraints = [f"thermal.res[:, {k}]" for k in range(5)]
        self.constraints += [f"design_lookup.voltage[:, {p}]" for p in range(12)] + [
            "design_lookup.gain[:]"
        ]

    def register(self, model):
        self.set_expressions()
        model.add_external_component(
            "design_physics", self, self.inputs, self.constraints
        )

        self.model = model

    def evaluate(self, x, con, grad, jac):
        self.last = np.array(x, copy=True)
        point = x[: self.nvars]

        if self.cache_point is None or not np.array_equal(point, self.cache_point):
            self.cache_point = np.array(point, copy=True)
            self.compute(point)

        con[:] = self.residual
        grad[:] = 0
        jac[:] = self.jacobian

    def compute(self, x):
        l = self.lookups
        n = self.n

        g = x[self.g]
        t = x[self.t]
        p = x[self.p]
        current = x[self.i]
        cell = x[self.cell]

        area, ad, add = l.exposure(g)
        voltage, vg, vh = l.panel_voltage(t, current, area, ad, add)
        gain, gd, gdd = l.gain(g)

        alpha = alpha_r + (alpha_c - alpha_r) * cell
        eps = eps_r + (eps_c - eps_r) * cell

        solar = q_sol * l.LOS[:, None] / l.mass_cp[None, :]
        radiation = K * A_T / l.mass_cp

        heating = np.sum(area * alpha[None, :, :], axis=1) * solar
        cooling = (
            eps.sum(axis=0)[None, :]
            * radiation[None, :]
            * t[:, l.thermal_temperature] ** 4
        )

        rhs = np.zeros((n, 5))
        rg = np.zeros((n, 5))
        rgg = np.zeros((n, 5))
        rt = np.zeros((n, 5))
        rtt = np.zeros((n, 5))
        cell_jac = np.zeros((n, 84))
        cell_g = np.zeros((n, 84))
        cell_t = np.zeros((n, 84))

        for panel, k in enumerate(l.thermal_temperature):
            rhs[:, k] += heating[:, panel] - cooling[:, panel]
            rg[:, k] += (ad[:, :, panel] * alpha[None, :, panel]).sum(axis=1) * solar[
                :, panel
            ]
            rgg[:, k] += (add[:, :, panel] * alpha[None, :, panel]).sum(axis=1) * solar[
                :, panel
            ]

            rt[:, k] -= 4 * eps[:, panel].sum() * radiation[panel] * t[:, k] ** 3
            rtt[:, k] -= 12 * eps[:, panel].sum() * radiation[panel] * t[:, k] ** 2

            ids = np.arange(7) * 12 + panel
            cell_jac[:, ids] = (alpha_c - alpha_r) * area[:, :, panel] * solar[
                :, panel, None
            ] - (eps_c - eps_r) * radiation[panel] * t[:, k, None] ** 4
            cell_g[:, ids] = (
                (alpha_c - alpha_r) * ad[:, :, panel] * solar[:, panel, None]
            )
            cell_t[:, ids] = (
                -4 * (eps_c - eps_r) * radiation[panel] * t[:, k, None] ** 3
            )

        rhs[:, 4] += 4 * p / (m_b * cp_b)

        self.residual = np.concatenate(
            (
                (x[self.tdot] - rhs).T.ravel(),
                (x[self.v] - voltage).T.ravel(),
                x[self.gain] - gain,
            )
        )
        # Sparse rows are short; fill each known derivative position in CSR order.
        vals = []

        for k in range(5):
            panels = np.flatnonzero(l.thermal_temperature == k)
            cell_ids = self.cell[:, panels].ravel() - 37 * n

            for node in range(n):
                derivatives = {
                    self.g[node]: -rg[node, k],
                    self.t[node, k]: -rt[node, k],
                    self.tdot[node, k]: 1.0,
                }

                if k == 4:
                    derivatives[self.p[node]] = -4 / (m_b * cp_b)

                derivatives.update(
                    {37 * n + int(c): -cell_jac[node, c] for c in cell_ids}
                )

                vals.extend(derivatives[c] for c in self.row_columns[k * n + node])

        for panel in range(12):
            for node in range(n):
                vals.extend(
                    [-vg[node, panel, 0], -vg[node, panel, 1], -vg[node, panel, 2], 1.0]
                )

        for node in range(n):
            vals.extend([-gd[node], 1.0])

        self.jacobian = np.array(vals)
        self.hessian_terms = (rgg, rtt, cell_g, cell_t, vh, gdd)

    def hessian_correction(self, shape):
        if self.point_provider is not None:
            full = np.asarray(self.point_provider())
            inputs = full[self.model.get_indices(self.inputs)]
            multipliers = full[self.model.get_indices(self.constraints)]
            self.last = np.concatenate((inputs, multipliers))

            if self.cache_point is None or not np.array_equal(inputs, self.cache_point):
                self.cache_point = inputs.copy()
                self.compute(inputs)

        if self.last is None:
            return coo_matrix(shape).tocsr()

        n = self.n
        l = self.lookups

        multipliers = self.last[self.nvars :]

        lt = multipliers[: 5 * n].reshape(5, n).T

        lv = multipliers[5 * n : 17 * n].reshape(12, n).T

        lg = multipliers[17 * n :]

        rgg, rtt, cg, ct, vh, gdd = self.hessian_terms

        rows = []
        cols = []
        vals = []

        def append(a, b, value):
            rows.extend(np.asarray(a).ravel())

            cols.extend(np.asarray(b).ravel())

            vals.extend(np.asarray(value).ravel())

        append(self.g, self.g, -np.sum(lt * rgg, axis=1) - lg * gdd)

        for k in range(5):
            append(self.t[:, k], self.t[:, k], -lt[:, k] * rtt[:, k])

        for c in range(84):
            panel = c % 12
            k = l.thermal_temperature[panel]
            global_col = np.full(n, 37 * n + c)

            for local, val in (
                (self.g, -lt[:, k] * cg[:, c]),
                (self.t[:, k], -lt[:, k] * ct[:, c]),
            ):
                append(local, global_col, val)
                append(global_col, local, val)

        for panel in range(12):
            ids = (self.g, self.t[:, l.panel_temperature[panel]], self.i[:, panel])

            for a in range(3):
                for b in range(3):
                    append(ids[a], ids[b], -lv[:, panel] * vh[:, panel, a, b])

        local = coo_matrix((vals, (rows, cols)), shape=(self.nvars, self.nvars)).tocoo()
        indices = self.model.get_indices(self.inputs)
        global_rows, global_cols = indices[local.row], indices[local.col]

        keep = ~(
            np.isin(global_rows, self.fixed_global_indices)
            | np.isin(global_cols, self.fixed_global_indices)
        )

        return coo_matrix(
            (local.data[keep], (global_rows[keep], global_cols[keep])), shape=shape
        ).tocsr()


def add_roll_attitude(model, lookups, geometry, dt):
    from attitude.attitude_amigo import (
        ForwardMatrixRate,
        CentralMatrixRate,
        BackwardMatrixRate,
        AttitudeAngular,
        ForwardAngularRate,
        CentralAngularRate,
        BackwardAngularRate,
        AttitudeTorque,
    )

    n = lookups.n
    _, _, w, torque, o, odot, wdot = geometry

    components = [
        ("roll_attitude", n, RolledOrientation()),
        ("roll_rate_forward", 1, ForwardMatrixRate(dt)),
        ("roll_rate_central", n - 2, CentralMatrixRate(dt)),
        ("roll_rate_backward", 1, BackwardMatrixRate(dt)),
        ("roll_angular", n, AttitudeAngular()),
        ("roll_acc_forward", 1, ForwardAngularRate(dt)),
        ("roll_acc_central", n - 2, CentralAngularRate(dt)),
        ("roll_acc_backward", 1, BackwardAngularRate(dt)),
        ("roll_torque", n, AttitudeTorque()),
    ]

    for name, size, comp in components:
        model.add_component(name, size, comp)

    model.set_data("roll_attitude.O_RI", lookups.base_orientation.reshape(n, 9))
    model.link("values.Gamma[:]", "roll_attitude.gamma[:]")

    model.link("roll_attitude.O_BI[0, :]", "roll_rate_forward.O0[0, :]")
    model.link("roll_attitude.O_BI[1, :]", "roll_rate_forward.O1[0, :]")

    model.link(f"roll_attitude.O_BI[:{n-2}, :]", "roll_rate_central.O_prev")
    model.link("roll_attitude.O_BI[2:, :]", "roll_rate_central.O_next")

    model.link(f"roll_attitude.O_BI[{n-2}, :]", "roll_rate_backward.O_prev[0, :]")
    model.link(f"roll_attitude.O_BI[{n-1}, :]", "roll_rate_backward.O_last[0, :]")

    model.link("roll_attitude.O_BI", "roll_angular.O_BI")
    model.link("roll_rate_forward.Odot[0, :]", "roll_angular.Odot_BI[0, :]")
    model.link("roll_rate_central.Odot", f"roll_angular.Odot_BI[1:{n-1}, :]")
    model.link("roll_rate_backward.Odot[0, :]", f"roll_angular.Odot_BI[{n-1}, :]")

    model.link("roll_angular.w_B[0, :]", "roll_acc_forward.w0[0, :]")
    model.link("roll_angular.w_B[1, :]", "roll_acc_forward.w1[0, :]")

    model.link(f"roll_angular.w_B[:{n-2}, :]", "roll_acc_central.w_prev")
    model.link("roll_angular.w_B[2:, :]", "roll_acc_central.w_next")

    model.link(f"roll_angular.w_B[{n-2}, :]", "roll_acc_backward.w_prev[0, :]")
    model.link(f"roll_angular.w_B[{n-1}, :]", "roll_acc_backward.w_last[0, :]")

    model.link("roll_angular.w_B", "roll_torque.w_B")
    model.link("roll_acc_forward.wdot[0, :]", "roll_torque.wdot_B[0, :]")
    model.link("roll_acc_central.wdot", f"roll_torque.wdot_B[1:{n-1}, :]")
    model.link("roll_acc_backward.wdot[0, :]", f"roll_torque.wdot_B[{n-1}, :]")

    model.link("roll_torque.T_tot", "rw_torque.T_tot")

    model.link("roll_angular.w_B", "rw_dynamics.w_B")

    for name, values in [
        ("roll_attitude.O_BI", o.reshape(n, 9)),
        ("roll_angular.Odot_BI", odot.reshape(n, 9)),
        ("roll_angular.w_B", w.T),
        ("roll_torque.wdot_B", wdot),
        ("roll_torque.T_tot", torque.T),
    ]:
        for col in range(values.shape[1]):
            model.set_meta("value", f"{name}[:, {col}]", values[:, col])
