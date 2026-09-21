import amigo as am
import numpy as np

from orbit.orbit_reference import orbital_elements_to_state, calculate_orbital_period
from orbit.orbit_amigo import build_reference_trajectory
from attitude.attitude_reference import attitude_from_orbit, attitude_roll, combine_rotation_matrices
from sun.sun_reference import sun_position_eci, sun_position_body, sun_position_spherical, sun_line_of_sight
from solar.solar_reference import load_solar_data, build_interpolators, solar_exposed_area
from thermal.thermal_reference import propagate_temperature


m_f = 0.4
m_b = 2.0
cp_f = 0.6e3
cp_b = 2.0e3
A_T = 2.66e-3
alpha_c = 0.9
alpha_r = 0.2
eps_c = 0.87
eps_r = 0.88
q_sol = 1360.0
K = 5.67051e-8


def thermal_rhs(state, exposed_area, LOS, P_comm, cellInstd):
    f = np.zeros(5)

    for p in range(12):
        if p < 4:
            f_i = 4
            mass = m_b
            cp = cp_b
        else:
            f_i = (p + 1) % 4
            mass = m_f
            cp = cp_f

        fact1 = q_sol * LOS / (mass * cp)
        fact2 = K * A_T * state[f_i]**4 / (mass * cp)

        heating = 0.0
        cooling = 0.0

        for c in range(7):
            alpha = alpha_c * cellInstd[c, p] + alpha_r - alpha_r * cellInstd[c, p]
            eps = eps_c * cellInstd[c, p] + eps_r - eps_r * cellInstd[c, p]
            heating += alpha * exposed_area[c, p]
            cooling += eps

        f[f_i] += heating * fact1
        f[f_i] -= cooling * fact2

    f[4] += 4.0 * P_comm / m_b / cp_b

    return f


def propagate_temperature_trapezoid(T0, exposed_area, LOS, P_comm, cellInstd, dt):
    num_nodes = LOS.size
    temperature = np.zeros((5, num_nodes))
    temperature_dot = np.zeros((num_nodes, 5))
    temperature[:, 0] = T0

    for k in range(num_nodes - 1):
        y0 = temperature[:, k]
        f0 = thermal_rhs(y0, exposed_area[:, :, k], LOS[k], P_comm[k], cellInstd)
        y1 = y0 + dt * f0

        for _ in range(100):
            f1 = thermal_rhs(y1, exposed_area[:, :, k + 1], LOS[k + 1], P_comm[k + 1], cellInstd)
            y1_new = y0 + 0.5 * dt * (f0 + f1)

            if np.max(np.abs(y1_new - y1)) < 1e-13:
                y1 = y1_new
                break

            y1 = y1_new

        temperature[:, k + 1] = y1

    for k in range(num_nodes):
        temperature_dot[k, :] = thermal_rhs(temperature[:, k], exposed_area[:, :, k], LOS[k], P_comm[k], cellInstd)

    return temperature, temperature_dot


class ThermalDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("temperature", shape=(5,))
        self.add_input("temperature_dot", shape=(5,))
        self.add_input("exposedArea", shape=(84,))
        self.add_input("LOS")
        self.add_input("P_comm")
        self.add_input("cellInstd", shape=(84,))
        self.add_constraint("res", shape=(5,))

    def compute(self):
        temperature = self.inputs["temperature"]
        temperature_dot = self.inputs["temperature_dot"]
        exposedArea = self.inputs["exposedArea"]
        LOS = self.inputs["LOS"]
        P_comm = self.inputs["P_comm"]
        cellInstd = self.inputs["cellInstd"]

        f = 5 * [0.0]

        for p in range(12):
            if p < 4:
                f_i = 4
                mass = m_b
                cp = cp_b
            else:
                f_i = (p + 1) % 4
                mass = m_f
                cp = cp_f

            fact1 = q_sol * LOS / (mass * cp)
            fact2 = K * A_T * temperature[f_i]**4 / (mass * cp)

            heating = 0.0
            cooling = 0.0

            for c in range(7):
                index = c * 12 + p
                alpha = alpha_c * cellInstd[index] + alpha_r - alpha_r * cellInstd[index]
                eps = eps_c * cellInstd[index] + eps_r - eps_r * cellInstd[index]
                heating += alpha * exposedArea[index]
                cooling += eps

            f[f_i] += heating * fact1
            f[f_i] -= cooling * fact2

        f[4] += 4.0 * P_comm / m_b / cp_b

        res = 5 * [None]

        for i in range(5):
            res[i] = f[i] - temperature_dot[i]

        self.constraints["res"] = res


class TrapezoidRule(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("temperature0", shape=(5,))
        self.add_input("temperature1", shape=(5,))
        self.add_input("temperature_dot0", shape=(5,))
        self.add_input("temperature_dot1", shape=(5,))
        self.add_constraint("res", shape=(5,))

    def compute(self):
        dt = self.constants["dt"]
        temperature0 = self.inputs["temperature0"]
        temperature1 = self.inputs["temperature1"]
        temperature_dot0 = self.inputs["temperature_dot0"]
        temperature_dot1 = self.inputs["temperature_dot1"]

        res = 5 * [None]

        for i in range(5):
            res[i] = temperature1[i] - temperature0[i] - 0.5 * dt * (temperature_dot0[i] + temperature_dot1[i])

        self.constraints["res"] = res


class InitialConditions(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("temperature", shape=(5,))
        self.add_constraint("res", shape=(5,))

    def compute(self):
        temperature = self.inputs["temperature"]
        self.constraints["res"] = [temperature[0] - 273.0, temperature[1] - 273.0, temperature[2] - 273.0, temperature[3] - 273.0, temperature[4] - 273.0]


def extract_vector(x, variable_name, n, width):
    values = np.zeros((n, width))

    for i in range(width):
        values[:, i] = x[f"{variable_name}[:, {i}]"]

    return values


def main():
    r0, v0 = orbital_elements_to_state(alt_perigee=500.0, alt_apogee=500.0, raan=66.279, inclination=82.072, arg_perigee=0.0, true_anomaly=337.987)
    q0 = np.concatenate((r0, v0))

    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)
    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps
    times = np.linspace(0.0, orbital_period, num_nodes)

    print()
    print("=" * 70)
    print("CADRE THERMAL AMIGO")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Time step [s]:")
    print(dt)

    # Orbit
    reference_states, reference_rates = build_reference_trajectory(q0, dt, num_time_steps)

    # Attitude
    gamma_reference = np.zeros(num_nodes)
    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(O_BR_reference, O_RI_reference)

    # Sun
    r_e2s_I_reference = sun_position_eci(times)
    r_e2s_B_reference = sun_position_body(O_BI_reference, r_e2s_I_reference)
    azimuth_reference, elevation_reference = sun_position_spherical(r_e2s_B_reference)
    LOS_reference = sun_line_of_sight(reference_states, r_e2s_I_reference)

    # Solar
    angle, azimuth_grid, elevation_grid, data = load_solar_data()
    interpolators = build_interpolators(angle, azimuth_grid, elevation_grid, data)
    exposed_area = solar_exposed_area(0.0, azimuth_reference, elevation_reference, interpolators)

    # Thermal inputs
    T0 = 273.0 * np.ones(5)
    P_comm_reference = 0.1 * np.ones(num_nodes)
    cellInstd_reference = np.ones((7, 12))

    # RK4 reference
    temperature_reference = propagate_temperature(T0, exposed_area, LOS_reference, P_comm_reference, cellInstd_reference, dt)

    # Trapezoid reference
    temperature_trapezoid, temperature_dot_trapezoid = propagate_temperature_trapezoid(T0, exposed_area, LOS_reference, P_comm_reference, cellInstd_reference, dt)

    # Python checks
    max_dynamics_residual = 0.0
    max_trapezoid_residual = 0.0

    for k in range(num_nodes):
        f = thermal_rhs(temperature_trapezoid[:, k], exposed_area[:, :, k], LOS_reference[k], P_comm_reference[k], cellInstd_reference)
        max_dynamics_residual = max(max_dynamics_residual, np.max(np.abs(f - temperature_dot_trapezoid[k, :])))

    for k in range(num_time_steps):
        res = temperature_trapezoid[:, k + 1] - temperature_trapezoid[:, k] - 0.5 * dt * (temperature_dot_trapezoid[k, :] + temperature_dot_trapezoid[k + 1, :])
        max_trapezoid_residual = max(max_trapezoid_residual, np.max(np.abs(res)))

    rk4_trapezoid_difference = np.max(np.abs(temperature_reference - temperature_trapezoid))

    print()
    print("=" * 70)
    print("PYTHON PRECHECK")
    print("=" * 70)

    print()
    print("Python dynamics residual:")
    print(max_dynamics_residual)

    print()
    print("Python trapezoid residual:")
    print(max_trapezoid_residual)

    print()
    print("RK4 vs trapezoid temperature difference [K]:")
    print(rk4_trapezoid_difference)

    # Flatten solar area
    exposed_area_flat = np.zeros((num_nodes, 84))

    for k in range(num_nodes):
        exposed_area_flat[k, :] = exposed_area[:, :, k].reshape(84)

    cellInstd_flat = cellInstd_reference.reshape(84)

    # Components
    dynamics = ThermalDynamics()
    trapezoid = TrapezoidRule(dt)
    initial = InitialConditions()

    # Model
    model = am.Model("cadre_thermal")

    model.add_component("thermal", num_nodes, dynamics)
    model.add_component("trapezoid", num_time_steps, trapezoid)
    model.add_component("initial", 1, initial)

    # Links
    model.link("thermal.temperature[:-1]", "trapezoid.temperature0")
    model.link("thermal.temperature[1:]", "trapezoid.temperature1")
    model.link("thermal.temperature_dot[:-1]", "trapezoid.temperature_dot0")
    model.link("thermal.temperature_dot[1:]", "trapezoid.temperature_dot1")
    model.link("thermal.temperature[0]", "initial.temperature[0]")

    # Initial guesses
    for i in range(5):
        model.set_meta("value", f"thermal.temperature[:, {i}]", temperature_trapezoid[i, :])
        model.set_meta("value", f"thermal.temperature_dot[:, {i}]", temperature_dot_trapezoid[:, i])

    # Fix exposed area
    for i in range(84):
        model.set_meta("value", f"thermal.exposedArea[:, {i}]", exposed_area_flat[:, i])
        model.set_meta("lower", f"thermal.exposedArea[:, {i}]", exposed_area_flat[:, i])
        model.set_meta("upper", f"thermal.exposedArea[:, {i}]", exposed_area_flat[:, i])

    # Fix LOS
    model.set_meta("value", "thermal.LOS[:]", LOS_reference)
    model.set_meta("lower", "thermal.LOS[:]", LOS_reference)
    model.set_meta("upper", "thermal.LOS[:]", LOS_reference)

    # Fix communication power
    model.set_meta("value", "thermal.P_comm[:]", P_comm_reference)
    model.set_meta("lower", "thermal.P_comm[:]", P_comm_reference)
    model.set_meta("upper", "thermal.P_comm[:]", P_comm_reference)

    # Fix cell map
    for i in range(84):
        values = cellInstd_flat[i] * np.ones(num_nodes)
        model.set_meta("value", f"thermal.cellInstd[:, {i}]", values)
        model.set_meta("lower", f"thermal.cellInstd[:, {i}]", values)
        model.set_meta("upper", f"thermal.cellInstd[:, {i}]", values)

    print()
    print("Building AMIGO thermal model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

    x = model.create_vector()
    opt = am.Optimizer(model, x)

    opt_options = {"max_iterations": 100, "convergence_tolerance": 1e-10, "initial_barrier_param": 0.1, "barrier_strategy": "heuristic", "max_line_search_iterations": 4}

    print()
    print("Solving AMIGO thermal model...")

    opt.optimize(opt_options)

    print()
    print("AMIGO solve complete.")

    # Extract AMIGO results
    temperature_amigo = extract_vector(x, "thermal.temperature", num_nodes, 5).T
    temperature_dot_amigo = extract_vector(x, "thermal.temperature_dot", num_nodes, 5)

    # Differences
    amigo_trapezoid_difference = np.max(np.abs(temperature_amigo - temperature_trapezoid))
    amigo_rk4_difference = np.max(np.abs(temperature_amigo - temperature_reference))
    final_rk4_difference = np.max(np.abs(temperature_amigo[:, -1] - temperature_reference[:, -1]))

    # AMIGO dynamics residual
    max_amigo_dynamics_residual = 0.0

    for k in range(num_nodes):
        f = thermal_rhs(temperature_amigo[:, k], exposed_area[:, :, k], LOS_reference[k], P_comm_reference[k], cellInstd_reference)
        max_amigo_dynamics_residual = max(max_amigo_dynamics_residual, np.max(np.abs(f - temperature_dot_amigo[k, :])))

    # AMIGO trapezoid residual
    max_amigo_trapezoid_residual = 0.0

    for k in range(num_time_steps):
        res = temperature_amigo[:, k + 1] - temperature_amigo[:, k] - 0.5 * dt * (temperature_dot_amigo[k, :] + temperature_dot_amigo[k + 1, :])
        max_amigo_trapezoid_residual = max(max_amigo_trapezoid_residual, np.max(np.abs(res)))

    print()
    print("=" * 70)
    print("THERMAL AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("Reference RK4 final temperatures [K]:")
    print(temperature_reference[:, -1])

    print()
    print("Python trapezoid final temperatures [K]:")
    print(temperature_trapezoid[:, -1])

    print()
    print("AMIGO final temperatures [K]:")
    print(temperature_amigo[:, -1])

    print()
    print("AMIGO vs Python trapezoid difference [K]:")
    print(amigo_trapezoid_difference)

    print()
    print("AMIGO vs CADRE RK4 maximum difference [K]:")
    print(amigo_rk4_difference)

    print()
    print("AMIGO vs CADRE RK4 final difference [K]:")
    print(final_rk4_difference)

    print()
    print("AMIGO dynamics residual:")
    print(max_amigo_dynamics_residual)

    print()
    print("AMIGO trapezoid residual:")
    print(max_amigo_trapezoid_residual)

    print()
    print("AMIGO minimum temperature [K]:")
    print(np.min(temperature_amigo))

    print()
    print("AMIGO maximum temperature [K]:")
    print(np.max(temperature_amigo))


if __name__ == "__main__":
    main()