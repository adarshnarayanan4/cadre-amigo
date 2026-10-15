"""Check voltage values and native AMIGO first/second derivatives.

Run after the coupled model has compiled; reuse its voltage component without
rewriting generated sources or CMake configuration.
"""

import json
from pathlib import Path

import amigo as am
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csr_matrix

from power.mbi_modern import knotopen
from power.voltage_spline import VoltageSpline, make_voltage_component


def main():
    spline = VoltageSpline()
    rng = np.random.default_rng(846)
    size = 8
    coefficient = rng.normal(size=(size, 6, 15))
    # Actual tensor coefficient transformation (the reference stays untransformed).
    transformed = np.einsum(
        "ia,nab,jb->nij", spline.transform_T, coefficient, spline.transform_I
    )
    tT = np.linspace(0.13, 0.87, size)
    tI = np.linspace(0.14, 0.88, size)
    source_T = spline.maps[0](tT)
    source_I = spline.maps[2](tI)
    basis_T = BSpline.design_matrix(tT, knotopen(3, 6), 2).toarray()
    basis_I = BSpline.design_matrix(tI, knotopen(3, 15), 2).toarray()
    voltage = np.einsum("na,nab,nb->n", basis_T, coefficient, basis_I)
    model = am.Model("cadre_optimize_pcomm")
    model.add_component("v", size, make_voltage_component(spline))
    model.set_data("v.coefficients", transformed.reshape(size, 90))
    for name, values in [
        ("temperature", source_T),
        ("current", source_I),
        ("tT", tT),
        ("tI", tI),
        ("voltage", voltage),
    ]:
        model.set_meta("value", f"v.{name}[:]", values)
    model._built = True  # Existing compiled class; never rewrite the live model.
    model.initialize(order_type=am.OrderingType.NATURAL)
    x = model.get_initial_point()
    g = model.create_vector()
    model.eval_gradient(x, g, alpha=0)
    constraints = ["temperature_map", "current_map", "voltage_res"]
    value_error = max(float(np.max(np.abs(g[f"v.{name}[:]"]))) for name in constraints)
    for name in constraints:
        x[f"v.{name}[:]"] = rng.normal(size=size)
    model.eval_gradient(x, g, alpha=0)

    # Reference residuals and finite differences in every local primal variable.
    def reference():
        bt = BSpline.design_matrix(x["v.tT[:]"], knotopen(3, 6), 2).toarray()
        bi = BSpline.design_matrix(x["v.tI[:]"], knotopen(3, 15), 2).toarray()
        return {
            "temperature_map": spline.maps[0](x["v.tT[:]"]) - x["v.temperature[:]"],
            "current_map": spline.maps[2](x["v.tI[:]"]) - x["v.current[:]"],
            "voltage_res": np.einsum("na,nab,nb->n", bt, coefficient, bi)
            - x["v.voltage[:]"],
        }

    step = 1e-6
    gradient_error = 0.0
    for name in ["temperature", "current", "tT", "tI", "voltage"]:
        for node in range(size):
            key = f"v.{name}[{node}]"
            value = float(x[key])
            x[key] = value + step
            plus = reference()
            x[key] = value - step
            minus = reference()
            x[key] = value
            fd = sum(
                float(np.dot(x[f"v.{c}[:]"], (plus[c] - minus[c]) / (2 * step)))
                for c in constraints
            )
            gradient_error = max(
                gradient_error, abs(fd - float(g[key])) / max(1, abs(fd))
            )
    hessian = model.create_matrix()
    model.eval_hessian(x, hessian, alpha=0)
    rows, cols, _, rowp, column = hessian.get_nonzero_structure()
    h = csr_matrix(
        (np.array(hessian.get_data(), copy=True), column, rowp),
        shape=(rows, cols),
        copy=True,
    )
    direction = np.zeros(rows)
    for name in ["temperature", "current", "tT", "tI", "voltage"]:
        direction[model.get_indices(f"v.{name}[:]")] = rng.normal(size=size)
    original = x.get_vector().get_array().copy()
    x.get_vector().get_array()[:] = original + step * direction
    model.eval_gradient(x, g, alpha=0)
    plus = g.get_vector().get_array().copy()
    x.get_vector().get_array()[:] = original - step * direction
    model.eval_gradient(x, g, alpha=0)
    minus = g.get_vector().get_array().copy()
    x.get_vector().get_array()[:] = original
    fd = (plus - minus) / (2 * step)
    expected = h @ direction
    hessian_error = float(np.max(np.abs(expected - fd)) / max(1, np.max(np.abs(fd))))
    result = {
        "voltage_and_map_value_max_error": value_error,
        "gradient_max_relative_error": gradient_error,
        "hessian_product_relative_error": hessian_error,
    }
    print(result)
    assert value_error < 1e-8 and gradient_error < 1e-6 and hessian_error < 1e-6
    Path("integrated/results").mkdir(exist_ok=True)
    Path("integrated/results/voltage_derivative_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
