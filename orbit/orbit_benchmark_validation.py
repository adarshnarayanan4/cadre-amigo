import pickle

import numpy as np

from cadre_paths import cadre_path
from orbit.orbit_reference import orbit_dynamics, rk4_step


def main():
    # Load the original CADRE benchmark results.
    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    # Original CADRE orbit trajectory.
    # CADRE stores it as (6, n), so transpose to (n, 6).
    source_states = np.asarray(
        data["0:r_e2b_I"],
        dtype=float,
    ).T

    n = source_states.shape[0]

    # Original CADRE benchmark runs for 12 hours.
    t1 = 0.0
    t2 = 43200.0

    dt = (t2 - t1) / (n - 1)

    print()
    print("=" * 70)
    print("CADRE ORBIT RK4 BENCHMARK VALIDATION")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    # Use the actual initial state stored in the CADRE benchmark.
    #
    # This is better than using the old default orbital elements because
    # the full CADRE benchmark gets its initial orbit from launch data.
    initial_state = source_states[0].copy()

    print()
    print("Initial CADRE state:")
    print(initial_state)

    # Propagate using our RK4 implementation.
    reference_states = np.zeros_like(source_states)
    reference_states[0] = initial_state

    for k in range(n - 1):
        reference_states[k + 1] = rk4_step(
            reference_states[k],
            dt,
        )

    # State derivatives for later AMIGO work.
    reference_rates = np.zeros_like(reference_states)

    for k in range(n):
        reference_rates[k] = orbit_dynamics(
            reference_states[k]
        )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    state_difference = (
        reference_states - source_states
    )

    position_difference = np.linalg.norm(
        state_difference[:, 0:3],
        axis=1,
    )

    velocity_difference = np.linalg.norm(
        state_difference[:, 3:6],
        axis=1,
    )

    print()
    print("=" * 70)
    print("PYTHON RK4 VS ORIGINAL CADRE")
    print("=" * 70)

    print()
    print("Maximum individual state difference:")
    print(
        np.max(
            np.abs(state_difference)
        )
    )

    print()
    print("Maximum position difference [km]:")
    print(
        np.max(position_difference)
    )

    print()
    print("Mean position difference [km]:")
    print(
        np.mean(position_difference)
    )

    print()
    print("Final position difference [km]:")
    print(
        position_difference[-1]
    )

    print()
    print("Maximum velocity difference [km/s]:")
    print(
        np.max(velocity_difference)
    )

    print()
    print("Mean velocity difference [km/s]:")
    print(
        np.mean(velocity_difference)
    )

    print()
    print("Final velocity difference [km/s]:")
    print(
        velocity_difference[-1]
    )

    print()
    print("Final CADRE state:")
    print(
        source_states[-1]
    )

    print()
    print("Final Python RK4 state:")
    print(
        reference_states[-1]
    )

    # ------------------------------------------------------------
    # Trapezoid residual of the RK4 trajectory
    #
    # This tells us how different the RK4 trajectory is from the
    # discretization that AMIGO will use next.
    # ------------------------------------------------------------

    trapezoid_residual = (
        reference_states[1:]
        - reference_states[:-1]
        - 0.5
        * dt
        * (
            reference_rates[:-1]
            + reference_rates[1:]
        )
    )

    print()
    print("=" * 70)
    print("RK4 TRAJECTORY VS TRAPEZOID RULE")
    print("=" * 70)

    print()
    print("Maximum trapezoid residual:")
    print(
        np.max(
            np.abs(trapezoid_residual)
        )
    )


if __name__ == "__main__":
    main()