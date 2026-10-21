"""Check analytic lookup Jacobians/Hessians and source equation agreement."""

import json, pickle
import argparse
from pathlib import Path

import amigo as am
import numpy as np
from scipy.sparse import csr_matrix

from cadre_paths import cadre_path
from communication.communication_amigo import CommunicationBitRate
from integrated.reporting.console import print_validation
from integrated.model.design import (
    DesignPhysics,
    LookupResiduals,
    SurfaceAllocation,
    ThermalDesignState,
)
from integrated.reference.design import DesignLookups
from parameters.parameters_amigo import ParameterValues
from power.power_amigo import PowerSolar
from thermal.thermal_amigo import thermal_rhs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, choices=range(6), default=0)
    args = parser.parse_args(argv)

    with open(cadre_path("test/data1346.pkl"), "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Cover illumination changes across the orbit, including the region
    # where the full-size solve showed large voltage residuals.
    full_n = len(data[f"{args.case_index}:P_comm"])
    nodes = np.unique(np.r_[np.linspace(0, full_n - 1, 16, dtype=int), 812, 815, 927])

    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim and value.shape[-1] == full_n:
            data[key] = value[..., nodes]

    n = len(nodes)
    l = DesignLookups(data, n, case_index=args.case_index)
    l.dt = 43200 / 1499
    physics = DesignPhysics(l)
    rng = np.random.default_rng(295)
    x = np.zeros(physics.nvars + physics.ncon)
    x[physics.g] = rng.uniform(0.2, 1.3, n)
    x[physics.t] = rng.uniform(265, 310, (n, 5))
    x[physics.i] = rng.uniform(0.02, 0.25, (n, 12))
    x[physics.cell] = rng.uniform(0.2, 0.8, (7, 12))
    x[physics.p] = rng.uniform(0, 5, n)
    x[physics.nvars :] = rng.normal(size=physics.ncon)
    residual = np.zeros(physics.ncon)
    grad = np.zeros(physics.nvars)
    jac = np.zeros(physics.cols.size)

    def evaluate(point):
        physics.evaluate(point, residual, grad, jac)

        return residual.copy(), csr_matrix(
            (jac.copy(), physics.cols, physics.rowp),
            shape=(physics.ncon, physics.nvars),
        )

    value, j = evaluate(x)
    direction = rng.normal(size=physics.nvars)
    direction[physics.g] *= 0.1
    direction[physics.i] *= 0.01
    direction[physics.cell] *= 0.1
    h = 2e-5
    xp = x.copy()
    xm = x.copy()
    xp[: physics.nvars] += h * direction
    xm[: physics.nvars] -= h * direction
    vp, jp = evaluate(xp)
    vm, jm = evaluate(xm)
    fd = (vp - vm) / (2 * h)
    analytic = j @ direction
    jac_error = np.max(np.abs(analytic - fd) / np.maximum(1, np.abs(fd)))

    class IdentityModel:
        def get_indices(self, expr):
            return np.arange(physics.nvars)

    physics.model = IdentityModel()
    physics.inputs = []
    evaluate(x)

    hh = physics.hessian_correction((physics.nvars, physics.nvars))
    fd_h = (jp.T @ x[physics.nvars :] - jm.T @ x[physics.nvars :]) / (2 * h)
    h_error = np.max(np.abs(hh @ direction - fd_h) / np.maximum(1, np.abs(fd_h)))
    area = l.exposure(x[physics.g])[0]
    rhs = np.stack(
        [
            thermal_rhs(
                x[physics.t][k], area[k], l.LOS[k], x[physics.p][k], x[physics.cell]
            )
            for k in range(n)
        ]
    )
    thermal_error = np.max(abs(value[: 5 * n].reshape(5, n).T + rhs))
    reference = l.voltage.reference_panels(
        x[physics.t].T, x[physics.i].T, area.transpose(1, 2, 0), l.LOS
    )
    voltage_error = np.max(abs(value[5 * n : 17 * n].reshape(12, n) + reference))
    contracted = l.geometry(x[physics.g])
    original = l.geometry(x[physics.g], reference=True)
    area_error = np.max(abs(contracted[0] - original[0]))
    gain_error = np.max(abs(contracted[1] - original[1]))
    result = {
        "jacobian_direction_relative_error": float(jac_error),
        "lagrangian_hessian_direction_relative_error": float(h_error),
        "thermal_rhs_reference_max_error": float(thermal_error),
        "voltage_reference_max_error": float(voltage_error),
        "exposure_original_tensor_max_error": float(area_error),
        "gain_original_tensor_max_error": float(gain_error),
    }

    for fixed_cell in (False, True):
        model = am.Model("cadre_optimize_design")
        model.add_component("values", n, ParameterValues())
        model.add_component("thermal", n, ThermalDesignState())
        model.add_component("surface", 1, SurfaceAllocation())
        model.add_component("design_lookup", n, LookupResiduals())
        model.add_component("power_solar", n, PowerSolar())
        model.add_component("comm_rate", n, CommunicationBitRate())

        assembled = DesignPhysics(l)
        assembled.register(model)

        if fixed_cell:
            model.add_fixed("surface.cellInstd[0, :]")

        model._built = True
        model.initialize(order_type=am.OrderingType.NATURAL)

        state = model.get_initial_point()
        gradient = model.create_vector()

        assembled.point_provider = lambda: state.get_vector().get_array()

        if fixed_cell:
            assembled.fixed_global_indices = model.get_indices(
                "surface.cellInstd[0, :]"
            )

        state.get_vector().get_array()[model.get_indices(assembled.inputs)] = x[
            : physics.nvars
        ]
        state["comm_rate.GSdist[:]"] = 1000.0
        state["comm_rate.CommLOS[:]"] = 0.5
        state["comm_rate.P_comm[:]"] = 2.0
        names = model.get_names()[1]

        for name in names:
            ids = model.get_indices(name)
            state.get_vector().get_array()[ids] = rng.normal(size=ids.shape)

        model.get_problem().update(state.get_vector())

        model.eval_gradient(state, gradient, alpha=0)

        matrix = model.create_matrix()
        model.eval_hessian(state, matrix, alpha=0)

        nr, nc, _, rp, columns = matrix.get_nonzero_structure()
        native = csr_matrix(
            (np.array(matrix.get_data(), copy=True), columns, rp),
            shape=(nr, nc),
            copy=True,
        )
        complete = native + assembled.hessian_correction((nr, nc))
        global_direction = np.zeros(nr)
        global_direction[model.get_indices(assembled.inputs)] = direction
        global_direction[assembled.fixed_global_indices] = 0.0
        original_state = state.get_vector().get_array().copy()
        state.get_vector().get_array()[:] = original_state + h * global_direction
        model.get_problem().update(state.get_vector())

        model.eval_gradient(state, gradient, alpha=0)

        plus = gradient.get_vector().get_array().copy()
        state.get_vector().get_array()[:] = original_state - h * global_direction
        model.get_problem().update(state.get_vector())

        model.eval_gradient(state, gradient, alpha=0)

        minus = gradient.get_vector().get_array().copy()
        finite = (plus - minus) / (2 * h)
        assembly_error = np.max(abs(complete @ global_direction - finite)) / max(
            1, np.max(abs(finite))
        )
        result[
            (
                "amigo_fixed_allocation_kkt_direction_relative_error"
                if fixed_cell
                else "amigo_assembled_kkt_direction_relative_error"
            )
        ] = float(assembly_error)

        if assembly_error > 1e-6:
            raise AssertionError("Assembled AMIGO lookup Hessian failed")

    print_validation("DESIGN PHYSICS VALIDATION", result)

    if (
        jac_error > 1e-6
        or h_error > 1e-5
        or thermal_error > 1e-10
        or voltage_error > 1e-10
        or area_error > 1e-10
        or gain_error > 1e-8
    ):
        raise AssertionError("Design physics derivatives/reference failed")

    Path("integrated/results/validation").mkdir(parents=True, exist_ok=True)

    suffix = f"_case_{args.case_index}" if args.case_index else ""
    Path(
        f"integrated/results/validation/design_derivative_validation{suffix}.json"
    ).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
