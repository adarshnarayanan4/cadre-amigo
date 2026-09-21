import amigo as am
import numpy as np
import matplotlib.pyplot as plt

from orbit.orbit_reference import (
    orbital_elements_to_state,
    orbit_dynamics,
    rk4_step,
    calculate_orbital_period,
)


class OrbitDynamics(am.Component):
    def __init__(self):
        super().__init__()

        # Earth constants
        self.add_constant("mu", value=398600.44)
        self.add_constant("Re", value=6378.137)

        self.add_constant("J2", value=1.08264e-3)
        self.add_constant("J3", value=-2.51e-6)
        self.add_constant("J4", value=-1.60e-6)

        # Spacecraft state
        self.add_input("q", shape=(6,), label="state")

        # Time derivative of spacecraft state
        self.add_input("qdot", shape=(6,), label="rate")

        # Dynamics residual
        self.add_constraint("res", shape=(6,), label="residual")

    def compute(self):
        # Earth constants
        mu = self.constants["mu"]
        Re = self.constants["Re"]

        J2 = self.constants["J2"]
        J3 = self.constants["J3"]
        J4 = self.constants["J4"]

        # State and state derivative
        q = self.inputs["q"]
        qdot = self.inputs["qdot"]

        x = q[0]
        y = q[1]
        z = q[2]

        vx = q[3]
        vy = q[4]
        vz = q[5]

        # Gravity model constants
        C1 = -mu
        C2 = -1.5 * mu * J2 * Re**2
        C3 = -2.5 * mu * J3 * Re**3
        C4 = 1.875 * mu * J4 * Re**4

        # Position quantities
        r2 = x * x + y * y + z * z
        r = r2**0.5

        z2 = z * z
        z3 = z2 * z
        z4 = z3 * z

        r3 = r2 * r
        r4 = r3 * r
        r5 = r4 * r
        r7 = r5 * r2

        # J2, J3, J4 correction terms
        T2 = (1.0 - 5.0 * z2 / r2)
        T3 = (3.0 * z - 7.0 * z3 / r2)
        T4 = (1.0 - 14.0 * z2 / r2 + 21.0 * z4 / r4)

        common = C1 / r3 + C2 / r5 * T2 + C3 / r7 * T3 + C4 / r7 * T4

        # Acceleration
        ax = common * x
        ay = common * y
        az = common * z

        az += 2.0 * C2 * z / r5

        az += C3 / r7 * (3.0 * z2 - 0.6 * r2)

        az += C4 / r7 * (4.0 * z - (28.0 / 3.0) * z3 / r2)

        # Dynamics residual
        res = 6 * [None]

        res[0] = vx - qdot[0]
        res[1] = vy - qdot[1]
        res[2] = vz - qdot[2]

        res[3] = ax - qdot[3]
        res[4] = ay - qdot[4]
        res[5] = az - qdot[5]

        self.constraints["res"] = res


class TrapezoidRule(am.Component):
    def __init__(self, dt):
        super().__init__()

        # Time step
        self.add_constant("dt", value=dt)

        # State at beginning and end of interval
        self.add_input("q1")
        self.add_input("q2")

        # State derivative at beginning and end of interval
        self.add_input("q1dot")
        self.add_input("q2dot")

        # Integration residual
        self.add_constraint("res")

    def compute(self):
        dt = self.constants["dt"]

        q1 = self.inputs["q1"]
        q2 = self.inputs["q2"]

        q1dot = self.inputs["q1dot"]
        q2dot = self.inputs["q2dot"]

        self.constraints["res"] = q2 - q1 - 0.5 * dt * (q1dot + q2dot)


class InitialConditions(am.Component):
    def __init__(self, q0):
        super().__init__()

        # Initial-state constants
        self.add_constant("x0", value=float(q0[0]))
        self.add_constant("y0", value=float(q0[1]))
        self.add_constant("z0", value=float(q0[2]))

        self.add_constant("vx0", value=float(q0[3]))
        self.add_constant("vy0", value=float(q0[4]))
        self.add_constant("vz0", value=float(q0[5]))

        # Initial state
        self.add_input("q", shape=(6,))

        # Initial-condition residual
        self.add_constraint("res", shape=(6,))

        # Dummy objective for feasibility solve
        self.add_objective("obj")

    def compute(self):
        q = self.inputs["q"]

        x0 = self.constants["x0"]
        y0 = self.constants["y0"]
        z0 = self.constants["z0"]

        vx0 = self.constants["vx0"]
        vy0 = self.constants["vy0"]
        vz0 = self.constants["vz0"]

        res = 6 * [None]

        res[0] = q[0] - x0
        res[1] = q[1] - y0
        res[2] = q[2] - z0

        res[3] = q[3] - vx0
        res[4] = q[4] - vy0
        res[5] = q[5] - vz0

        self.constraints["res"] = res

        self.objective["obj"] = 1.0e-12 * q[0] * q[0]


def build_reference_trajectory(q0, dt, num_time_steps):
    """Propagate the RK4 reference trajectory and its state derivatives."""
    reference_states = np.zeros((num_time_steps + 1, 6))
    reference_states[0] = q0

    for i in range(num_time_steps):
        reference_states[i + 1] = rk4_step(reference_states[i], dt)

    reference_rates = np.zeros((num_time_steps + 1, 6))
    for i in range(num_time_steps + 1):
        reference_rates[i] = orbit_dynamics(reference_states[i])

    return reference_states, reference_rates


def check_trapezoid_guess_quality(reference_states, reference_rates, dt):
    """Report how well the RK4 trajectory satisfies the trapezoid rule."""
    residual = (
        reference_states[1:]
        - reference_states[:-1]
        - 0.5 * dt * (reference_rates[:-1] + reference_rates[1:])
    )
    max_residual = np.max(np.abs(residual))

    print()
    print("Maximum trapezoid residual in RK4 initial guess:")
    print(max_residual)


def build_model(q0, dt, num_time_steps, reference_states, reference_rates, model_name="cadre_orbit"):
    """Create, link, initialize, and build the AMIGO trajectory model."""
    orbit = OrbitDynamics()
    trap = TrapezoidRule(dt)
    ic = InitialConditions(q0)

    model = am.Model(model_name)

    model.add_component("orbit", num_time_steps + 1, orbit)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("ic", 1, ic)

    # Link trajectory variables to trapezoid constraints
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        model.link(f"orbit.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]")
        model.link(f"orbit.q[1:, {i}]", f"trap.q2[{start}:{end}]")
        model.link(f"orbit.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"orbit.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Link initial condition
    model.link("orbit.q[0, :]", "ic.q[0, :]")

    # Stage RK4 trajectory as AMIGO's actual initial point
    for i in range(6):
        model.set_meta("value", f"orbit.q[:, {i}]", reference_states[:, i])
        model.set_meta("value", f"orbit.qdot[:, {i}]", reference_rates[:, i])

    # Build and initialize
    print()
    print("Building AMIGO orbit model...")

    model.build_module()
    print("Build successful.")

    model.initialize()
    print("Initialization successful.")

    initial_point = model.get_initial_point()

    print()
    print("AMIGO stored initial spacecraft state:")
    print(initial_point["orbit.q[0, :]"])

    print()
    print("Expected initial spacecraft state:")
    print(q0)

    print()
    print("Maximum difference between AMIGO stored guess and RK4 guess:")

    max_initial_difference = 0.0

    for i in range(6):
        difference = np.max(np.abs(initial_point[f"orbit.q[:, {i}]"] - reference_states[:, i]))
        max_initial_difference = max(max_initial_difference, difference)

    print(max_initial_difference)

    return model

def report_setup_diagnostics(model):
    """Check AMIGO's stored initial point and constraint bounds."""

    initial_point = model.get_initial_point()
    lower = model.get_lower()
    upper = model.get_upper()

    print()
    print("NaN values in AMIGO stored initial point:")
    print(np.isnan(np.asarray(initial_point[:])).sum())

    print()
    print("AMIGO stored initial point contains NaNs:")
    print(np.any(np.isnan(np.asarray(initial_point[:]))))

    print()
    print("AMIGO stored initial point contains infinities:")
    print(np.any(np.isinf(np.asarray(initial_point[:]))))

    print()
    print("Orbit dynamics constraint bounds:")
    print(lower["orbit.res"][0], upper["orbit.res"][0])

    print()
    print("Trapezoid constraint bounds:")
    print(lower["trap.res"][0], upper["trap.res"][0])

    print()
    print("Initial-condition constraint bounds:")
    print(lower["ic.res"][0], upper["ic.res"][0])

def solve_model(model, x):
    opt = am.Optimizer(model, x)

    opt_options = {
        "max_iterations": 200,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print("Solving AMIGO orbit...")

    opt_data = opt.optimize(opt_options)

    print()
    print("AMIGO solve complete.")

    return opt_data


def report_solution_comparison(x, reference_states, orbital_period, num_time_steps):
    amigo_states = np.zeros((num_time_steps + 1, 6))
    for i in range(6):
        amigo_states[:, i] = x[f"orbit.q[:, {i}]"]

    times = np.linspace(0.0, orbital_period, num_time_steps + 1)

    position_difference = np.linalg.norm(
        amigo_states[:, 0:3] - reference_states[:, 0:3], axis=1
    )
    velocity_difference = np.linalg.norm(
        amigo_states[:, 3:6] - reference_states[:, 3:6], axis=1
    )

    print()
    print("Maximum position difference [km]:")
    print(np.max(position_difference))

    print()
    print("Final position difference [km]:")
    print(position_difference[-1])

    print()
    print("Maximum velocity difference [km/s]:")
    print(np.max(velocity_difference))

    print()
    print("Final velocity difference [km/s]:")
    print(velocity_difference[-1])

    return amigo_states, times

def plot_solution_comparison(amigo_states, reference_states, times):
    """Plot AMIGO and RK4 trajectories and their differences."""
    # Extract positions
    amigo_x = amigo_states[:, 0]
    amigo_y = amigo_states[:, 1]
    amigo_z = amigo_states[:, 2]

    rk4_x = reference_states[:, 0]
    rk4_y = reference_states[:, 1]
    rk4_z = reference_states[:, 2]

    # 3D trajectory comparison
    fig = plt.figure(figsize=(9, 8))

    ax = fig.add_subplot(111, projection="3d")

    ax.plot(rk4_x, rk4_y, rk4_z, label="RK4 Reference")
    ax.plot(amigo_x, amigo_y, amigo_z, linestyle="--", label="AMIGO")

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")

    ax.set_title("AMIGO vs RK4 Orbit")

    ax.legend()

    plt.tight_layout()

    # Position difference versus time
    position_difference = np.linalg.norm(amigo_states[:, 0:3] - reference_states[:, 0:3], axis=1)

    plt.figure(figsize=(9, 5))

    plt.plot(times / 60.0, position_difference)

    plt.xlabel("Time [min]")
    plt.ylabel("Position Difference [km]")

    plt.title("AMIGO vs RK4 Position Difference")

    plt.grid()

    plt.tight_layout()

    # Velocity difference versus time
    velocity_difference = np.linalg.norm(amigo_states[:, 3:6] - reference_states[:, 3:6], axis=1)

    plt.figure(figsize=(9, 5))

    plt.plot(times / 60.0, velocity_difference)

    plt.xlabel("Time [min]")
    plt.ylabel("Velocity Difference [km/s]")

    plt.title("AMIGO vs RK4 Velocity Difference")

    plt.grid()

    plt.tight_layout()
    plt.show()

def main():
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
    num_time_steps = 568
    dt = orbital_period / num_time_steps

    print()
    print("Number of time intervals:")
    print(num_time_steps)

    print()
    print("Number of time nodes:")
    print(num_time_steps + 1)

    print()
    print("AMIGO time step [s]:")
    print(dt)

    # Reference trajectory and initial-guess quality check
    reference_states, reference_rates = build_reference_trajectory(q0, dt, num_time_steps)
    check_trapezoid_guess_quality(reference_states, reference_rates, dt)

    # Build and initialize AMIGO model
    model = build_model(q0, dt, num_time_steps, reference_states, reference_rates)

    report_setup_diagnostics(model)

    x = model.create_vector()

    # Solve
    solve_model(model, x)

    # Compare against RK4 reference
    amigo_states, times = report_solution_comparison(x, reference_states, orbital_period, num_time_steps)

    plot_solution_comparison(amigo_states, reference_states, times)

if __name__ == "__main__":
    main()