import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path
from parameters.parameters_reference import bspline_parameters
from power.mbi_modern import basis, knotopen, paramuni


class ParameterControlPoints(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("CP_P_comm")
        self.add_input("CP_gamma")
        self.add_input("CP_Isetpt", shape=(12,))


class ParameterValues(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("P_comm")
        self.add_input("Gamma")
        self.add_input("Isetpt", shape=(12,))


class ParameterSplineKernel(am.Component):
    def __init__(self):
        super().__init__()

        # Cubic B-spline has four active basis functions per point
        self.add_data("N", shape=(4,))

        self.add_input("CP_P_comm", shape=(4,))
        self.add_input("CP_gamma", shape=(4,))
        self.add_input("CP_Isetpt", shape=(48,))

        self.add_input("P_comm")
        self.add_input("Gamma")
        self.add_input("Isetpt", shape=(12,))

        self.add_constraint("res_P_comm")
        self.add_constraint("res_Gamma")
        self.add_constraint("res_Isetpt", shape=(12,))

    def compute(self):
        N = self.data["N"]

        CP_P_comm = self.inputs["CP_P_comm"]
        CP_gamma = self.inputs["CP_gamma"]
        CP_Isetpt = self.inputs["CP_Isetpt"]

        P_comm = self.inputs["P_comm"]
        Gamma = self.inputs["Gamma"]
        Isetpt = self.inputs["Isetpt"]

        P_comm_value = 0.0
        Gamma_value = 0.0

        for j in range(4):
            P_comm_value += N[j] * CP_P_comm[j]
            Gamma_value += N[j] * CP_gamma[j]

        self.constraints["res_P_comm"] = (
            P_comm - P_comm_value
        )

        self.constraints["res_Gamma"] = (
            Gamma - Gamma_value
        )

        res_Isetpt = 12 * [None]

        for panel in range(12):
            current = 0.0

            for j in range(4):
                index = 4 * panel + j
                current += N[j] * CP_Isetpt[index]

            res_Isetpt[panel] = Isetpt[panel] - current

        self.constraints["res_Isetpt"] = res_Isetpt


def build_sparse_basis(n, m):
    k = 4

    interp_points = paramuni(
        k,
        m,
        n,
    )

    knots = knotopen(
        k,
        m,
    )

    weights = np.zeros((n, k))
    indices = np.zeros((n, k), dtype=int)

    for i in range(n):
        B, i0 = basis(
            k,
            interp_points[i],
            knots,
        )

        weights[i, :] = B

        for j in range(k):
            indices[i, j] = i0 + j

    return weights, indices


def extract_Isetpt(x, n):
    Isetpt = np.zeros((12, n))

    for panel in range(12):
        Isetpt[panel, :] = x[
            f"values.Isetpt[:, {panel}]"
        ]

    return Isetpt


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(data_path, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1",
        )

    # CADRE control points
    CP_P_comm = data["0:CP_P_comm"]
    CP_gamma = data["0:CP_gamma"]
    CP_Isetpt = data["0:CP_Isetpt"]

    # Original CADRE outputs
    P_comm_source = data["0:P_comm"]
    gamma_source = data["0:gamma"]
    Isetpt_source = data["0:Isetpt"]

    n = len(P_comm_source)
    m = len(CP_P_comm)

    print()
    print("=" * 70)
    print("CADRE BSPLINE PARAMETERS AMIGO")
    print("=" * 70)

    print()
    print("Number of time nodes:")
    print(n)

    print()
    print("Number of control points:")
    print(m)

    # Python reference
    P_comm_reference, gamma_reference, Isetpt_reference = (
        bspline_parameters(
            CP_P_comm,
            CP_gamma,
            CP_Isetpt,
            n,
        )
    )

    # Sparse B-spline structure
    weights, indices = build_sparse_basis(
        n,
        m,
    )

    print()
    print("=" * 70)
    print("PYTHON PRECHECK")
    print("=" * 70)

    print()
    print("Maximum basis row-sum error:")
    print(
        np.max(
            np.abs(
                np.sum(weights, axis=1) - 1.0
            )
        )
    )

    print()
    print("Python P_comm vs CADRE [W]:")
    print(
        np.max(
            np.abs(
                P_comm_reference - P_comm_source
            )
        )
    )

    print()
    print("Python Gamma vs CADRE [rad]:")
    print(
        np.max(
            np.abs(
                gamma_reference - gamma_source
            )
        )
    )

    print()
    print("Python Isetpt vs CADRE [A]:")
    print(
        np.max(
            np.abs(
                Isetpt_reference - Isetpt_source
            )
        )
    )

    # Components
    control_points = ParameterControlPoints()
    values = ParameterValues()
    kernel = ParameterSplineKernel()

    # Model
    model = am.Model(
        "cadre_parameters"
    )

    model.add_component(
        "control_points",
        m,
        control_points,
    )

    model.add_component(
        "values",
        n,
        values,
    )

    model.add_component(
        "kernel",
        n,
        kernel,
    )

    # Each spline point depends on four control points
    for i in range(n):
        for j in range(4):
            control_index = indices[i, j]

            model.link(
                f"control_points.CP_P_comm[{control_index}]",
                f"kernel.CP_P_comm[{i}, {j}]",
            )

            model.link(
                f"control_points.CP_gamma[{control_index}]",
                f"kernel.CP_gamma[{i}, {j}]",
            )

            for panel in range(12):
                local_index = 4 * panel + j

                model.link(
                    (
                        "control_points."
                        f"CP_Isetpt[{control_index}, {panel}]"
                    ),
                    (
                        "kernel."
                        f"CP_Isetpt[{i}, {local_index}]"
                    ),
                )

    # Link spline outputs
    model.link(
        "kernel.P_comm",
        "values.P_comm",
    )

    model.link(
        "kernel.Gamma",
        "values.Gamma",
    )

    model.link(
        "kernel.Isetpt",
        "values.Isetpt",
    )

    # Store the four B-spline weights at every time node
    model.set_data(
        "kernel.N",
        weights,
    )

    # Fix CADRE control points for validation
    model.set_meta(
        "value",
        "control_points.CP_P_comm[:]",
        CP_P_comm,
    )
    model.set_meta(
        "lower",
        "control_points.CP_P_comm[:]",
        CP_P_comm,
    )
    model.set_meta(
        "upper",
        "control_points.CP_P_comm[:]",
        CP_P_comm,
    )

    model.set_meta(
        "value",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )
    model.set_meta(
        "lower",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )
    model.set_meta(
        "upper",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )

    for panel in range(12):
        values_panel = CP_Isetpt[panel, :]

        model.set_meta(
            "value",
            f"control_points.CP_Isetpt[:, {panel}]",
            values_panel,
        )
        model.set_meta(
            "lower",
            f"control_points.CP_Isetpt[:, {panel}]",
            values_panel,
        )
        model.set_meta(
            "upper",
            f"control_points.CP_Isetpt[:, {panel}]",
            values_panel,
        )

    # Initial output guesses
    model.set_meta(
        "value",
        "values.P_comm[:]",
        P_comm_reference,
    )

    model.set_meta(
        "value",
        "values.Gamma[:]",
        gamma_reference,
    )

    for panel in range(12):
        model.set_meta(
            "value",
            f"values.Isetpt[:, {panel}]",
            Isetpt_reference[panel, :],
        )

    print()
    print("Building AMIGO parameter model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

    x = model.create_vector()
    opt = am.Optimizer(
        model,
        x,
    )

    opt_options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print("Solving AMIGO parameter model...")

    opt.optimize(
        opt_options
    )

    print()
    print("AMIGO solve complete.")

    # Extract AMIGO results
    P_comm_amigo = np.array(
        x["values.P_comm[:]"]
    )

    gamma_amigo = np.array(
        x["values.Gamma[:]"]
    )

    Isetpt_amigo = extract_Isetpt(
        x,
        n,
    )

    print()
    print("=" * 70)
    print("PARAMETERS AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("AMIGO vs Python P_comm difference [W]:")
    print(
        np.max(
            np.abs(
                P_comm_amigo - P_comm_reference
            )
        )
    )

    print()
    print("AMIGO vs Python Gamma difference [rad]:")
    print(
        np.max(
            np.abs(
                gamma_amigo - gamma_reference
            )
        )
    )

    print()
    print("AMIGO vs Python Isetpt difference [A]:")
    print(
        np.max(
            np.abs(
                Isetpt_amigo - Isetpt_reference
            )
        )
    )

    print()
    print("=" * 70)
    print("AMIGO VS ORIGINAL CADRE")
    print("=" * 70)

    print()
    print("Maximum P_comm difference [W]:")
    print(
        np.max(
            np.abs(
                P_comm_amigo - P_comm_source
            )
        )
    )

    print()
    print("Maximum Gamma difference [rad]:")
    print(
        np.max(
            np.abs(
                gamma_amigo - gamma_source
            )
        )
    )

    print()
    print("Maximum Isetpt difference [A]:")
    print(
        np.max(
            np.abs(
                Isetpt_amigo - Isetpt_source
            )
        )
    )


if __name__ == "__main__":
    main()