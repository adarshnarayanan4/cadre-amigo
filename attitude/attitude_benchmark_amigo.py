import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_reference import (
    attitude_from_orbit,
    attitude_roll,
    combine_rotation_matrices,
    rotation_matrix_rates,
    angular_velocity,
    angular_acceleration,
    attitude_torque,
)

from attitude.attitude_amigo import (
    AttitudeOrientation,
    ForwardMatrixRate,
    CentralMatrixRate,
    BackwardMatrixRate,
    AttitudeAngular,
    ForwardAngularRate,
    CentralAngularRate,
    BackwardAngularRate,
    AttitudeTorque,
    extract_matrix,
)

from attitude.attitude_benchmark_validation import (
    get_case_value,
    matrix_history,
    vector_history,
)


class DummyObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("value")
        self.add_objective("obj")

    def compute(self):
        value = self.inputs["value"]

        self.objective["obj"] = (
            1.0e-12 * value * value
        )


def extract_vector(
    x,
    variable_name,
    n,
    size,
):
    values = np.zeros(
        (n, size)
    )

    for i in range(size):
        values[:, i] = (
            x[
                f"{variable_name}[:, {i}]"
            ]
        )

    return values


def report_difference(
    name,
    amigo_value,
    reference_value,
):
    difference = np.abs(
        amigo_value
        - reference_value
    )

    print()
    print(
        f"Maximum {name} difference:"
    )
    print(
        np.max(difference)
    )

    print(
        f"Mean {name} difference:"
    )
    print(
        np.mean(difference)
    )


def main():
    # --------------------------------------------------------
    # Load original CADRE benchmark
    # --------------------------------------------------------

    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(
        data_path,
        "rb",
    ) as file:
        data = pickle.load(
            file,
            encoding="latin1",
        )

    orbit_source = get_case_value(
        data,
        "r_e2b_I",
    ).T

    gamma_source = get_case_value(
        data,
        "gamma",
        "Gamma",
    ).reshape(-1)

    n = orbit_source.shape[0]

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    print()
    print("=" * 70)
    print(
        "CADRE ATTITUDE AMIGO BENCHMARK"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    # --------------------------------------------------------
    # Python reference
    # --------------------------------------------------------

    O_RI_reference = (
        attitude_from_orbit(
            orbit_source
        )
    )

    O_BR_reference = attitude_roll(
        gamma_source
    )

    O_BI_reference = (
        combine_rotation_matrices(
            O_BR_reference,
            O_RI_reference,
        )
    )

    Odot_BI_reference = (
        rotation_matrix_rates(
            O_BI_reference,
            dt,
        )
    )

    w_B_reference = angular_velocity(
        O_BI_reference,
        Odot_BI_reference,
    )

    wdot_B_reference = (
        angular_acceleration(
            w_B_reference,
            dt,
        )
    )

    T_tot_reference = attitude_torque(
        w_B_reference,
        wdot_B_reference,
    )

    # Flatten matrix histories for AMIGO
    O_RI_flat = O_RI_reference.reshape(
        n,
        9,
    )

    O_BR_flat = O_BR_reference.reshape(
        n,
        9,
    )

    O_BI_flat = O_BI_reference.reshape(
        n,
        9,
    )

    Odot_BI_flat = (
        Odot_BI_reference.reshape(
            n,
            9,
        )
    )

    # --------------------------------------------------------
    # AMIGO components
    # --------------------------------------------------------

    attitude = AttitudeOrientation()

    rate_forward = ForwardMatrixRate(
        dt
    )

    rate_central = CentralMatrixRate(
        dt
    )

    rate_backward = BackwardMatrixRate(
        dt
    )

    angular = AttitudeAngular()

    angular_rate_forward = (
        ForwardAngularRate(dt)
    )

    angular_rate_central = (
        CentralAngularRate(dt)
    )

    angular_rate_backward = (
        BackwardAngularRate(dt)
    )

    torque = AttitudeTorque()

    objective = DummyObjective()

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = am.Model(
        "cadre_attitude_benchmark"
    )

    model.add_component(
        "attitude",
        n,
        attitude,
    )

    model.add_component(
        "rate_forward",
        1,
        rate_forward,
    )

    model.add_component(
        "rate_central",
        n - 2,
        rate_central,
    )

    model.add_component(
        "rate_backward",
        1,
        rate_backward,
    )

    model.add_component(
        "angular",
        n,
        angular,
    )

    model.add_component(
        "angular_rate_forward",
        1,
        angular_rate_forward,
    )

    model.add_component(
        "angular_rate_central",
        n - 2,
        angular_rate_central,
    )

    model.add_component(
        "angular_rate_backward",
        1,
        angular_rate_backward,
    )

    model.add_component(
        "torque",
        n,
        torque,
    )

    model.add_component(
        "objective",
        1,
        objective,
    )

    # --------------------------------------------------------
    # O_BI finite differences
    # --------------------------------------------------------

    model.link(
        "attitude.O_BI[0, :]",
        "rate_forward.O0[0, :]",
    )

    model.link(
        "attitude.O_BI[1, :]",
        "rate_forward.O1[0, :]",
    )

    model.link(
        f"attitude.O_BI[:{n - 2}, :]",
        "rate_central.O_prev",
    )

    model.link(
        "attitude.O_BI[2:, :]",
        "rate_central.O_next",
    )

    model.link(
        f"attitude.O_BI[{n - 2}, :]",
        "rate_backward.O_prev[0, :]",
    )

    model.link(
        f"attitude.O_BI[{n - 1}, :]",
        "rate_backward.O_last[0, :]",
    )

    # --------------------------------------------------------
    # O_BI and Odot_BI -> angular velocity
    # --------------------------------------------------------

    model.link(
        "attitude.O_BI",
        "angular.O_BI",
    )

    model.link(
        "rate_forward.Odot[0, :]",
        "angular.Odot_BI[0, :]",
    )

    model.link(
        "rate_central.Odot",
        f"angular.Odot_BI[1:{n - 1}, :]",
    )

    model.link(
        "rate_backward.Odot[0, :]",
        f"angular.Odot_BI[{n - 1}, :]",
    )

    # --------------------------------------------------------
    # w_B finite differences
    # --------------------------------------------------------

    model.link(
        "angular.w_B[0, :]",
        "angular_rate_forward.w0[0, :]",
    )

    model.link(
        "angular.w_B[1, :]",
        "angular_rate_forward.w1[0, :]",
    )

    model.link(
        f"angular.w_B[:{n - 2}, :]",
        "angular_rate_central.w_prev",
    )

    model.link(
        "angular.w_B[2:, :]",
        "angular_rate_central.w_next",
    )

    model.link(
        f"angular.w_B[{n - 2}, :]",
        "angular_rate_backward.w_prev[0, :]",
    )

    model.link(
        f"angular.w_B[{n - 1}, :]",
        "angular_rate_backward.w_last[0, :]",
    )

    # --------------------------------------------------------
    # Angular velocity -> torque
    # --------------------------------------------------------

    model.link(
        "angular.w_B",
        "torque.w_B",
    )

    model.link(
        "angular_rate_forward.wdot[0, :]",
        "torque.wdot_B[0, :]",
    )

    model.link(
        "angular_rate_central.wdot",
        f"torque.wdot_B[1:{n - 1}, :]",
    )

    model.link(
        "angular_rate_backward.wdot[0, :]",
        f"torque.wdot_B[{n - 1}, :]",
    )

    # Small dummy objective
    model.link(
        "torque.T_tot[0, 0]",
        "objective.value[0]",
    )

    # --------------------------------------------------------
    # Fix benchmark inputs:
    # CADRE orbit and CADRE Gamma
    # --------------------------------------------------------

    for i in range(6):
        variable = (
            f"attitude.q[:, {i}]"
        )

        model.set_meta(
            "value",
            variable,
            orbit_source[:, i],
        )

        model.set_meta(
            "lower",
            variable,
            orbit_source[:, i],
        )

        model.set_meta(
            "upper",
            variable,
            orbit_source[:, i],
        )

    model.set_meta(
        "value",
        "attitude.gamma[:]",
        gamma_source,
    )

    model.set_meta(
        "lower",
        "attitude.gamma[:]",
        gamma_source,
    )

    model.set_meta(
        "upper",
        "attitude.gamma[:]",
        gamma_source,
    )

    # --------------------------------------------------------
    # Initial guesses
    # --------------------------------------------------------

    for i in range(9):
        model.set_meta(
            "value",
            f"attitude.O_RI[:, {i}]",
            O_RI_flat[:, i],
        )

        model.set_meta(
            "value",
            f"attitude.O_BR[:, {i}]",
            O_BR_flat[:, i],
        )

        model.set_meta(
            "value",
            f"attitude.O_BI[:, {i}]",
            O_BI_flat[:, i],
        )

    # Orientation-rate guesses
    for i in range(9):
        model.set_meta(
            "value",
            f"rate_forward.Odot[:, {i}]",
            Odot_BI_flat[0:1, i],
        )

        model.set_meta(
            "value",
            f"rate_central.Odot[:, {i}]",
            Odot_BI_flat[1:-1, i],
        )

        model.set_meta(
            "value",
            f"rate_backward.Odot[:, {i}]",
            Odot_BI_flat[-1:, i],
        )

    # Angular velocity
    for i in range(3):
        model.set_meta(
            "value",
            f"angular.w_B[:, {i}]",
            w_B_reference[:, i],
        )

    # Angular acceleration
    for i in range(3):
        model.set_meta(
            "value",
            f"angular_rate_forward.wdot[:, {i}]",
            wdot_B_reference[0:1, i],
        )

        model.set_meta(
            "value",
            f"angular_rate_central.wdot[:, {i}]",
            wdot_B_reference[1:-1, i],
        )

        model.set_meta(
            "value",
            f"angular_rate_backward.wdot[:, {i}]",
            wdot_B_reference[-1:, i],
        )

        model.set_meta(
            "value",
            f"torque.T_tot[:, {i}]",
            T_tot_reference[:, i],
        )

    # --------------------------------------------------------
    # Build and solve
    # --------------------------------------------------------

    print()
    print(
        "Building AMIGO attitude benchmark..."
    )

    model.build_module()

    print("Build successful.")

    model.initialize()

    print(
        "Initialization successful."
    )

    x = model.create_vector()

    optimizer = am.Optimizer(
        model,
        x,
    )

    options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print(
        "Solving AMIGO attitude benchmark..."
    )

    optimizer.optimize(
        options
    )

    print()
    print(
        "AMIGO solve complete."
    )

    # --------------------------------------------------------
    # Extract AMIGO results
    # --------------------------------------------------------

    O_RI_amigo = extract_matrix(
        x,
        "attitude.O_RI",
        n,
    )

    O_BR_amigo = extract_matrix(
        x,
        "attitude.O_BR",
        n,
    )

    O_BI_amigo = extract_matrix(
        x,
        "attitude.O_BI",
        n,
    )

    Odot_BI_amigo = np.zeros(
        (n, 3, 3)
    )

    Odot_flat = np.zeros(
        (n, 9)
    )

    for i in range(9):
        Odot_flat[0, i] = x[
            f"rate_forward.Odot[0, {i}]"
        ]

        Odot_flat[1:-1, i] = x[
            f"rate_central.Odot[:, {i}]"
        ]

        Odot_flat[-1, i] = x[
            f"rate_backward.Odot[0, {i}]"
        ]

    Odot_BI_amigo[:] = (
        Odot_flat.reshape(
            n,
            3,
            3,
        )
    )

    w_B_amigo = extract_vector(
        x,
        "angular.w_B",
        n,
        3,
    )

    wdot_B_amigo = np.zeros(
        (n, 3)
    )

    for i in range(3):
        wdot_B_amigo[0, i] = x[
            f"angular_rate_forward.wdot[0, {i}]"
        ]

        wdot_B_amigo[1:-1, i] = x[
            f"angular_rate_central.wdot[:, {i}]"
        ]

        wdot_B_amigo[-1, i] = x[
            f"angular_rate_backward.wdot[0, {i}]"
        ]

    T_tot_amigo = extract_vector(
        x,
        "torque.T_tot",
        n,
        3,
    )

    # --------------------------------------------------------
    # AMIGO vs Python reference
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AMIGO VS PYTHON ATTITUDE"
    )
    print("=" * 70)

    report_difference(
        "O_RI",
        O_RI_amigo,
        O_RI_reference,
    )

    report_difference(
        "O_BR",
        O_BR_amigo,
        O_BR_reference,
    )

    report_difference(
        "O_BI",
        O_BI_amigo,
        O_BI_reference,
    )

    report_difference(
        "Odot_BI",
        Odot_BI_amigo,
        Odot_BI_reference,
    )

    report_difference(
        "angular velocity",
        w_B_amigo,
        w_B_reference,
    )

    report_difference(
        "angular acceleration",
        wdot_B_amigo,
        wdot_B_reference,
    )

    report_difference(
        "torque",
        T_tot_amigo,
        T_tot_reference,
    )

    # --------------------------------------------------------
    # AMIGO vs original CADRE
    # --------------------------------------------------------

    O_BI_source = matrix_history(
        get_case_value(
            data,
            "O_BI",
        )
    )

    w_B_source = vector_history(
        get_case_value(
            data,
            "w_B",
        )
    )

    wdot_B_source = vector_history(
        get_case_value(
            data,
            "wdot_B",
        )
    )

    T_tot_source = vector_history(
        get_case_value(
            data,
            "T_tot",
        )
    )

    print()
    print("=" * 70)
    print(
        "AMIGO VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "O_BI",
        O_BI_amigo,
        O_BI_source,
    )

    report_difference(
        "angular velocity",
        w_B_amigo,
        w_B_source,
    )

    report_difference(
        "angular acceleration",
        wdot_B_amigo,
        wdot_B_source,
    )

    report_difference(
        "torque",
        T_tot_amigo,
        T_tot_source,
    )


if __name__ == "__main__":
    main()