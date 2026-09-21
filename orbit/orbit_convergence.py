import numpy as np
import matplotlib.pyplot as plt
from orbit.orbit_reference import (orbital_elements_to_state, calculate_orbital_period)
from orbit.orbit_amigo import (build_reference_trajectory, build_model, solve_model)

def run_case(num_time_steps):
    """Run one AMIGO/RK4 comparison for a specified time grid."""
    # Initial spacecraft state
    r0, v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    q0 = np.concatenate((r0, v0))

    # Time discretization
    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)

    dt = orbital_period / num_time_steps

    print()
    print("=" * 60)
    print(f"Running case with {num_time_steps} intervals")
    print(f"Time step: {dt:.6f} s")
    print("=" * 60)

    # RK4 reference trajectory
    reference_states, reference_rates = build_reference_trajectory(q0, dt, num_time_steps)

    # Build AMIGO model
    model = build_model(q0, dt, num_time_steps, reference_states, reference_rates, model_name=f"cadre_orbit_{num_time_steps}")

    # Solve AMIGO model
    x = model.create_vector()

    solve_model(model, x)

    # Extract AMIGO trajectory
    amigo_states = np.zeros((num_time_steps + 1, 6))
    for i in range(6):
        amigo_states[:, i] = x[f"orbit.q[:, {i}]"]

    # Calculate AMIGO vs RK4 differences
    position_difference = np.linalg.norm(amigo_states[:, 0:3] - reference_states[:, 0:3], axis=1,)
    velocity_difference = np.linalg.norm(amigo_states[:, 3:6] - reference_states[:, 3:6], axis=1,)

    # Store results
    results = {
        "num_time_steps": num_time_steps,
        "dt": dt,
        "max_position_error": np.max(position_difference),
        "final_position_error": position_difference[-1],
        "max_velocity_error": np.max(velocity_difference),
        "final_velocity_error": velocity_difference[-1],
    }

    print()
    print("Maximum position difference [km]:")
    print(results["max_position_error"])

    print()
    print("Final position difference [km]:")
    print(results["final_position_error"])

    print()
    print("Maximum velocity difference [km/s]:")
    print(results["max_velocity_error"])

    print()
    print("Final velocity difference [km/s]:")
    print(results["final_velocity_error"])

    return results

def calculate_observed_order(coarse, fine, error_key):
    """Calculate observed numerical convergence order."""
    error_coarse = coarse[error_key]
    error_fine = fine[error_key]

    dt_coarse = coarse["dt"]
    dt_fine = fine["dt"]

    order = (np.log(error_coarse / error_fine) / np.log(dt_coarse / dt_fine))

    return order

def main():
    # Grid sizes
    grid_sizes = [284, 568, 1136]
    results = []

    # Run convergence cases
    for num_time_steps in grid_sizes:
        result = run_case(num_time_steps)

        results.append(result)

    # Print summary table
    print()
    print()
    print("=" * 90)
    print("GRID CONVERGENCE SUMMARY")
    print("=" * 90)

    print(
        f"{'Intervals':>10}"
        f"{'dt [s]':>14}"
        f"{'Max Pos Err [km]':>20}"
        f"{'Final Pos Err [km]':>22}"
        f"{'Max Vel Err [km/s]':>22}"
    )

    for result in results:
        print(
            f"{result['num_time_steps']:>10d}"
            f"{result['dt']:>14.6f}"
            f"{result['max_position_error']:>20.9f}"
            f"{result['final_position_error']:>22.9f}"
            f"{result['max_velocity_error']:>22.9e}"
        )

    # Calculated observed convergence order
    print()
    print("Observed convergence order:")

    for i in range(len(results) - 1):
        coarse = results[i]
        fine = results[i + 1]
        position_order = calculate_observed_order(coarse, fine, "max_position_error")
        velocity_order = calculate_observed_order(coarse, fine, "max_velocity_error")

        print()
        print(f"{coarse['num_time_steps']} -> "f"{fine['num_time_steps']} intervals")

        print("Position order:", position_order)
        print("Velocity order:", velocity_order)

    # Convergence plot
    dt_values = np.array([result["dt"] for result in results])

    position_errors = np.array([result["max_position_error"] for result in results])
    velocity_errors = np.array([result["max_velocity_error"] for result in results])

    plt.figure(figsize=(8, 6))

    plt.loglog(dt_values, position_errors, marker="o", label="Position Difference",)

    plt.xlabel("Time Step [s]")
    plt.ylabel("Maximum Position Difference [km]")

    plt.title("AMIGO Grid Convergence")

    plt.grid(True, which="both",)

    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()