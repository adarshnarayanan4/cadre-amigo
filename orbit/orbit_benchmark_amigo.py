import pickle

import amigo as am
import numpy as np
from scipy.optimize import root

from cadre_paths import cadre_path
from orbit.orbit_amigo import (
    InitialConditions,
    OrbitDynamics,
    TrapezoidRule,
)
from orbit.orbit_reference import (
    orbit_dynamics,
    rk4_step,
)


def propagate_trapezoid(initial_state, dt, n):
    """Propagate the orbit using implicit trapezoidal integration."""

    states = np.zeros((n, 6))
    states[0] = initial_state

    for k in range(n - 1):
        current_state = states[k]

        current_rate = orbit_dynamics(
            current_state
        )

        # RK4 gives a very good initial guess
        next_guess = rk4_step(
            current_state,
            dt,
        )

        def residual(next_state):
            next_rate = orbit_dynamics(
                next_state
            )

            return (
                next_state
                - current_state
                - 0.5
                * dt
                * (
                    current_rate
                    + next_rate
                )
            )

        solution = root(
            residual,
            next_guess,
            tol=1e-12,
        )

        if not solution.success:
            if np.max(np.abs(solution.fun)) > 1e-9:
                raise RuntimeError(
                    "Trapezoid solve failed "
                    f"at interval {k}"
                )

        states[k + 1] = solution.x

    rates = np.zeros_like(states)

    for k in range(n):
        rates[k] = orbit_dynamics(
            states[k]
        )

    return states, rates


def maximum_trapezoid_residual(
    states,
    rates,
    dt,
):
    residual = (
        states[1:]
        - states[:-1]
        - 0.5
        * dt
        * (
            rates[:-1]
            + rates[1:]
        )
    )

    return np.max(
        np.abs(residual)
    )


def build_model(
    initial_state,
    states,
    rates,
    dt,
):
    n = states.shape[0]
    num_intervals = n - 1

    orbit = OrbitDynamics()
    trap = TrapezoidRule(dt)
    initial = InitialConditions(
        initial_state
    )

    model = am.Model(
        "cadre_orbit_benchmark"
    )

    model.add_component(
        "orbit",
        n,
        orbit,
    )

    model.add_component(
        "trap",
        6 * num_intervals,
        trap,
    )

    model.add_component(
        "initial",
        1,
        initial,
    )

    # Connect each state coordinate
    # to the trapezoid constraints.
    for i in range(6):
        start = i * num_intervals
        end = (i + 1) * num_intervals

        model.link(
            f"orbit.q[:-1, {i}]",
            f"trap.q1[{start}:{end}]",
        )

        model.link(
            f"orbit.q[1:, {i}]",
            f"trap.q2[{start}:{end}]",
        )

        model.link(
            f"orbit.qdot[:-1, {i}]",
            f"trap.q1dot[{start}:{end}]",
        )

        model.link(
            f"orbit.qdot[1:, {i}]",
            f"trap.q2dot[{start}:{end}]",
        )

    # Initial condition
    model.link(
        "orbit.q[0, :]",
        "initial.q[0, :]",
    )

    # Use the Python trapezoid solution
    # as AMIGO's starting point.
    for i in range(6):
        model.set_meta(
            "value",
            f"orbit.q[:, {i}]",
            states[:, i],
        )

        model.set_meta(
            "value",
            f"orbit.qdot[:, {i}]",
            rates[:, i],
        )

    print()
    print(
        "Building AMIGO orbit benchmark..."
    )

    model.build_module()

    print("Build successful.")

    model.initialize()

    print(
        "Initialization successful."
    )

    return model


def main():
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

    # Original CADRE trajectory
    source_states = np.asarray(
        data["0:r_e2b_I"],
        dtype=float,
    ).T

    n = source_states.shape[0]

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    initial_state = (
        source_states[0].copy()
    )

    print()
    print("=" * 70)
    print(
        "CADRE ORBIT AMIGO BENCHMARK"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    # --------------------------------------------------------
    # Original CADRE-style RK4 trajectory
    # --------------------------------------------------------

    rk4_states = np.zeros_like(
        source_states
    )

    rk4_states[0] = initial_state

    for k in range(n - 1):
        rk4_states[k + 1] = (
            rk4_step(
                rk4_states[k],
                dt,
            )
        )

    # --------------------------------------------------------
    # Python trapezoid trajectory
    # --------------------------------------------------------

    print()
    print(
        "Building Python trapezoid reference..."
    )

    (
        trapezoid_states,
        trapezoid_rates,
    ) = propagate_trapezoid(
        initial_state,
        dt,
        n,
    )

    trap_residual = (
        maximum_trapezoid_residual(
            trapezoid_states,
            trapezoid_rates,
            dt,
        )
    )

    print()
    print("=" * 70)
    print(
        "PYTHON TRAPEZOID PRECHECK"
    )
    print("=" * 70)

    print()
    print(
        "Maximum trapezoid residual:"
    )
    print(trap_residual)

    # --------------------------------------------------------
    # RK4 vs trapezoid
    # --------------------------------------------------------

    position_difference = (
        np.linalg.norm(
            trapezoid_states[:, 0:3]
            - rk4_states[:, 0:3],
            axis=1,
        )
    )

    velocity_difference = (
        np.linalg.norm(
            trapezoid_states[:, 3:6]
            - rk4_states[:, 3:6],
            axis=1,
        )
    )

    print()
    print(
        "RK4 vs trapezoid maximum "
        "position difference [km]:"
    )
    print(
        np.max(position_difference)
    )

    print()
    print(
        "RK4 vs trapezoid final "
        "position difference [km]:"
    )
    print(
        position_difference[-1]
    )

    print()
    print(
        "RK4 vs trapezoid maximum "
        "velocity difference [km/s]:"
    )
    print(
        np.max(velocity_difference)
    )

    # --------------------------------------------------------
    # Build AMIGO
    # --------------------------------------------------------

    model = build_model(
        initial_state,
        trapezoid_states,
        trapezoid_rates,
        dt,
    )

    x = model.create_vector()

    optimizer = am.Optimizer(
        model,
        x,
    )

    options = {
        "max_iterations": 200,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print(
        "Solving AMIGO orbit benchmark..."
    )

    optimizer.optimize(
        options
    )

    print()
    print(
        "AMIGO solve complete."
    )

    # --------------------------------------------------------
    # Extract AMIGO trajectory
    # --------------------------------------------------------

    amigo_states = np.zeros(
        (n, 6)
    )

    amigo_rates = np.zeros(
        (n, 6)
    )

    for i in range(6):
        amigo_states[:, i] = (
            x[f"orbit.q[:, {i}]"]
        )

        amigo_rates[:, i] = (
            x[f"orbit.qdot[:, {i}]"]
        )

    # --------------------------------------------------------
    # AMIGO vs Python trapezoid
    # --------------------------------------------------------

    amigo_position_difference = (
        np.linalg.norm(
            amigo_states[:, 0:3]
            - trapezoid_states[:, 0:3],
            axis=1,
        )
    )

    amigo_velocity_difference = (
        np.linalg.norm(
            amigo_states[:, 3:6]
            - trapezoid_states[:, 3:6],
            axis=1,
        )
    )

    amigo_rate_difference = (
        np.max(
            np.abs(
                amigo_rates
                - trapezoid_rates
            )
        )
    )

    amigo_trap_residual = (
        maximum_trapezoid_residual(
            amigo_states,
            amigo_rates,
            dt,
        )
    )

    print()
    print("=" * 70)
    print(
        "AMIGO VS PYTHON TRAPEZOID"
    )
    print("=" * 70)

    print()
    print(
        "Maximum position difference [km]:"
    )
    print(
        np.max(
            amigo_position_difference
        )
    )

    print()
    print(
        "Maximum velocity difference [km/s]:"
    )
    print(
        np.max(
            amigo_velocity_difference
        )
    )

    print()
    print(
        "Maximum state-rate difference:"
    )
    print(
        amigo_rate_difference
    )

    print()
    print(
        "AMIGO trapezoid residual:"
    )
    print(
        amigo_trap_residual
    )

    # --------------------------------------------------------
    # AMIGO vs original CADRE
    # --------------------------------------------------------

    cadre_position_difference = (
        np.linalg.norm(
            amigo_states[:, 0:3]
            - source_states[:, 0:3],
            axis=1,
        )
    )

    cadre_velocity_difference = (
        np.linalg.norm(
            amigo_states[:, 3:6]
            - source_states[:, 3:6],
            axis=1,
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
        "Maximum position difference [km]:"
    )
    print(
        np.max(
            cadre_position_difference
        )
    )

    print()
    print(
        "Final position difference [km]:"
    )
    print(
        cadre_position_difference[-1]
    )

    print()
    print(
        "Maximum velocity difference [km/s]:"
    )
    print(
        np.max(
            cadre_velocity_difference
        )
    )

    print()
    print(
        "Final velocity difference [km/s]:"
    )
    print(
        cadre_velocity_difference[-1]
    )


if __name__ == "__main__":
    main()