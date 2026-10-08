import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_benchmark_validation import (
    get_case_value,
)

from thermal.thermal_reference import (
    propagate_temperature,
)

from thermal.thermal_amigo import (
    ThermalDynamics,
    TrapezoidRule,
    InitialConditions,
    extract_vector,
    propagate_temperature_trapezoid,
    thermal_rhs,
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

    # --------------------------------------------------------
    # CADRE benchmark inputs
    # --------------------------------------------------------

    exposed_area = get_case_value(
        data,
        "exposedArea",
    )

    LOS = get_case_value(
        data,
        "LOS",
    ).reshape(-1)

    P_comm = get_case_value(
        data,
        "P_comm",
    ).reshape(-1)

    cellInstd = get_case_value(
        data,
        "cellInstd",
    )

    temperature_source = get_case_value(
        data,
        "temperature",
    )

    n = LOS.size
    num_intervals = n - 1

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / num_intervals

    T0 = 273.0 * np.ones(5)

    print()
    print("=" * 70)
    print(
        "CADRE THERMAL AMIGO BENCHMARK"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    # --------------------------------------------------------
    # Python RK4 reference
    # --------------------------------------------------------

    temperature_rk4 = (
        propagate_temperature(
            T0,
            exposed_area,
            LOS,
            P_comm,
            cellInstd,
            dt,
        )
    )

    # --------------------------------------------------------
    # Python trapezoid reference
    # --------------------------------------------------------

    (
        temperature_trap,
        temperature_dot_trap,
    ) = propagate_temperature_trapezoid(
        T0,
        exposed_area,
        LOS,
        P_comm,
        cellInstd,
        dt,
    )

    # --------------------------------------------------------
    # Python prechecks
    # --------------------------------------------------------

    max_dynamics_residual = 0.0

    for k in range(n):
        f = thermal_rhs(
            temperature_trap[:, k],
            exposed_area[:, :, k],
            LOS[k],
            P_comm[k],
            cellInstd,
        )

        residual = (
            f
            - temperature_dot_trap[k]
        )

        max_dynamics_residual = max(
            max_dynamics_residual,
            np.max(
                np.abs(residual)
            ),
        )

    max_trapezoid_residual = 0.0

    for k in range(
        num_intervals
    ):
        residual = (
            temperature_trap[:, k + 1]
            - temperature_trap[:, k]
            - 0.5
            * dt
            * (
                temperature_dot_trap[k]
                + temperature_dot_trap[k + 1]
            )
        )

        max_trapezoid_residual = max(
            max_trapezoid_residual,
            np.max(
                np.abs(residual)
            ),
        )

    rk4_trap_difference = np.max(
        np.abs(
            temperature_rk4
            - temperature_trap
        )
    )

    print()
    print("=" * 70)
    print(
        "PYTHON PRECHECK"
    )
    print("=" * 70)

    print()
    print(
        "Python dynamics residual:"
    )
    print(
        max_dynamics_residual
    )

    print()
    print(
        "Python trapezoid residual:"
    )
    print(
        max_trapezoid_residual
    )

    print()
    print(
        "RK4 vs trapezoid maximum "
        "temperature difference [K]:"
    )
    print(
        rk4_trap_difference
    )

    # --------------------------------------------------------
    # Flatten CADRE inputs for AMIGO
    # --------------------------------------------------------

    exposed_area_flat = np.zeros(
        (n, 84)
    )

    for k in range(n):
        exposed_area_flat[k] = (
            exposed_area[:, :, k]
            .reshape(84)
        )

    cell_flat = (
        cellInstd.reshape(84)
    )

    # --------------------------------------------------------
    # AMIGO components
    # --------------------------------------------------------

    thermal = ThermalDynamics()

    trapezoid = TrapezoidRule(
        dt
    )

    initial = InitialConditions()

    model = am.Model(
        "cadre_thermal_benchmark"
    )

    model.add_component(
        "thermal",
        n,
        thermal,
    )

    model.add_component(
        "trapezoid",
        num_intervals,
        trapezoid,
    )

    model.add_component(
        "initial",
        1,
        initial,
    )

    # --------------------------------------------------------
    # Links
    # --------------------------------------------------------

    model.link(
        "thermal.temperature[:-1]",
        "trapezoid.temperature0",
    )

    model.link(
        "thermal.temperature[1:]",
        "trapezoid.temperature1",
    )

    model.link(
        "thermal.temperature_dot[:-1]",
        "trapezoid.temperature_dot0",
    )

    model.link(
        "thermal.temperature_dot[1:]",
        "trapezoid.temperature_dot1",
    )

    model.link(
        "thermal.temperature[0]",
        "initial.temperature[0]",
    )

    # --------------------------------------------------------
    # Initial guesses
    # --------------------------------------------------------

    for i in range(5):
        model.set_meta(
            "value",
            f"thermal.temperature[:, {i}]",
            temperature_trap[i],
        )

        model.set_meta(
            "value",
            f"thermal.temperature_dot[:, {i}]",
            temperature_dot_trap[:, i],
        )

    # --------------------------------------------------------
    # Fix CADRE exposed area
    # --------------------------------------------------------

    for i in range(84):
        variable = (
            f"thermal.exposedArea[:, {i}]"
        )

        values = (
            exposed_area_flat[:, i]
        )

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # --------------------------------------------------------
    # Fix CADRE LOS
    # --------------------------------------------------------

    model.set_meta(
        "value",
        "thermal.LOS[:]",
        LOS,
    )

    model.set_meta(
        "lower",
        "thermal.LOS[:]",
        LOS,
    )

    model.set_meta(
        "upper",
        "thermal.LOS[:]",
        LOS,
    )

    # --------------------------------------------------------
    # Fix CADRE communication power
    # --------------------------------------------------------

    model.set_meta(
        "value",
        "thermal.P_comm[:]",
        P_comm,
    )

    model.set_meta(
        "lower",
        "thermal.P_comm[:]",
        P_comm,
    )

    model.set_meta(
        "upper",
        "thermal.P_comm[:]",
        P_comm,
    )

    # --------------------------------------------------------
    # Fix CADRE cell installation map
    # --------------------------------------------------------

    for i in range(84):
        values = (
            cell_flat[i]
            * np.ones(n)
        )

        variable = (
            f"thermal.cellInstd[:, {i}]"
        )

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # --------------------------------------------------------
    # Build and solve
    # --------------------------------------------------------

    print()
    print(
        "Building AMIGO thermal benchmark..."
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
        "Solving AMIGO thermal benchmark..."
    )

    optimizer.optimize(
        options
    )

    print()
    print(
        "AMIGO solve complete."
    )

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    temperature_amigo = (
        extract_vector(
            x,
            "thermal.temperature",
            n,
            5,
        ).T
    )

    temperature_dot_amigo = (
        extract_vector(
            x,
            "thermal.temperature_dot",
            n,
            5,
        )
    )

    # --------------------------------------------------------
    # AMIGO vs Python trapezoid
    # --------------------------------------------------------

    temperature_difference = (
        np.max(
            np.abs(
                temperature_amigo
                - temperature_trap
            )
        )
    )

    rate_difference = np.max(
        np.abs(
            temperature_dot_amigo
            - temperature_dot_trap
        )
    )

    amigo_trapezoid_residual = 0.0

    for k in range(
        num_intervals
    ):
        residual = (
            temperature_amigo[:, k + 1]
            - temperature_amigo[:, k]
            - 0.5
            * dt
            * (
                temperature_dot_amigo[k]
                + temperature_dot_amigo[k + 1]
            )
        )

        amigo_trapezoid_residual = max(
            amigo_trapezoid_residual,
            np.max(
                np.abs(residual)
            ),
        )

    print()
    print("=" * 70)
    print(
        "AMIGO VS PYTHON TRAPEZOID"
    )
    print("=" * 70)

    print()
    print(
        "Maximum temperature difference [K]:"
    )
    print(
        temperature_difference
    )

    print()
    print(
        "Maximum temperature-rate difference:"
    )
    print(
        rate_difference
    )

    print()
    print(
        "AMIGO trapezoid residual:"
    )
    print(
        amigo_trapezoid_residual
    )

    # --------------------------------------------------------
    # AMIGO vs original CADRE
    # --------------------------------------------------------

    cadre_difference = np.max(
        np.abs(
            temperature_amigo
            - temperature_source
        )
    )

    final_difference = np.max(
        np.abs(
            temperature_amigo[:, -1]
            - temperature_source[:, -1]
        )
    )

    print()
    print("=" * 70)
    print(
        "AMIGO VS ORIGINAL CADRE"
    )
    print("=" * 70)

    print()
    print(
        "Maximum temperature difference [K]:"
    )
    print(
        cadre_difference
    )

    print()
    print(
        "Final temperature difference [K]:"
    )
    print(
        final_difference
    )

    print()
    print(
        "Original CADRE final temperatures [K]:"
    )
    print(
        temperature_source[:, -1]
    )

    print()
    print(
        "AMIGO final temperatures [K]:"
    )
    print(
        temperature_amigo[:, -1]
    )

    print()
    print(
        "AMIGO minimum temperature [K]:"
    )
    print(
        np.min(
            temperature_amigo
        )
    )

    print()
    print(
        "AMIGO maximum temperature [K]:"
    )
    print(
        np.max(
            temperature_amigo
        )
    )


if __name__ == "__main__":
    main()