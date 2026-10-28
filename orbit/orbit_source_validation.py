import numpy as np
from cadre_paths import source_provenance
from cadre_source import source_class
from orbit.orbit_reference import (
    orbital_elements_to_state,
    orbit_dynamics,
    rk4_step,
    calculate_orbital_period,
)


# These wrappers execute the equations in OpenMDAO/CADRE, rather than a
# second hand-copied implementation that could drift from that repository.
def cadre_initial_source(
    alt_perigee, alt_apogee, raan, inclination, arg_perigee, true_anomaly
):
    component = source_class("orbit", "Orbit_Initial")()
    position, velocity = component.compute_rv(
        alt_perigee, alt_apogee, raan, inclination, arg_perigee, true_anomaly
    )
    return position.real, velocity.real


def cadre_dynamics_source(state):
    component = source_class("orbit", "Orbit_Dynamics")(2, 1.0)
    return component.f_dot(None, state)


def cadre_rk4_step_source(state, dt):
    # Match the RK4 stages used by CADRE while evaluating its actual force law.
    a = cadre_dynamics_source(state)
    b = cadre_dynamics_source(state + dt / 2.0 * a)
    c = cadre_dynamics_source(state + dt / 2.0 * b)
    d = cadre_dynamics_source(state + dt * c)
    return state + dt / 6.0 * (a + 2.0 * (b + c) + d)


# Validation
def main():
    source_provenance()
    print()
    print("=" * 70)
    print("CADRE SOURCE VALIDATION")
    print("=" * 70)

    # Compare initial state
    cadre_r0, cadre_v0 = cadre_initial_source(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    our_r0, our_v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    cadre_q0 = np.concatenate((cadre_r0, cadre_v0))
    our_q0 = np.concatenate((our_r0, our_v0))

    initial_difference = np.max(np.abs(cadre_q0 - our_q0))

    print()
    print("1. INITIAL STATE CHECK")
    print("----------------------")

    print("Maximum difference:")
    print(initial_difference)

    # Compare dynamics over one orbit
    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)

    num_time_steps = 568
    dt = orbital_period / num_time_steps

    state = our_q0.copy()

    max_dynamics_difference = 0.0
    max_rk4_difference = 0.0

    for i in range(num_time_steps):
        # Compare derivative at this state
        cadre_rate = cadre_dynamics_source(state)
        our_rate = orbit_dynamics(state)

        dynamics_difference = np.max(np.abs(cadre_rate - our_rate))
        max_dynamics_difference = max(max_dynamics_difference, dynamics_difference)

        # Compare one RK4 step
        cadre_next = cadre_rk4_step_source(state, dt)
        our_next = rk4_step(state, dt)

        rk4_difference = np.max(np.abs(cadre_next - our_next))
        max_rk4_difference = max(max_rk4_difference, rk4_difference)

        # Advance using our reference trajectory
        state = our_next

    print()
    print("2. DYNAMICS CHECK")
    print("-----------------")

    print("Maximum derivative difference over one orbit:")
    print(max_dynamics_difference)

    print()
    print("3. RK4 STEP CHECK")
    print("-----------------")

    print("Maximum one-step RK4 difference over one orbit:")
    print(max_rk4_difference)

    # Final result
    tolerance = 1e-10

    initial_pass = initial_difference < tolerance
    dynamics_pass = max_dynamics_difference < tolerance
    rk4_pass = max_rk4_difference < tolerance

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print("Initial-state equations match:", initial_pass)
    print("Dynamics equations match:", dynamics_pass)
    print("RK4 equations match:", rk4_pass)

    if initial_pass and dynamics_pass and rk4_pass:
        print()
        print(
            "PASS: orbit_reference.py numerically matches the original "
            + "CADRE orbit equations."
        )
    else:
        print()
        print(
            "CHECK NEEDED: at least one comparison was larger than the " + "tolerance."
        )
        raise AssertionError("Orbit equations differ from OpenMDAO/CADRE")


if __name__ == "__main__":
    main()
