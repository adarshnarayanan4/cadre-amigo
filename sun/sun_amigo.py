import amigo as am
import numpy as np

from orbit.orbit_reference import (orbital_elements_to_state, calculate_orbital_period)

from orbit.orbit_amigo import (build_reference_trajectory)

from attitude.attitude_reference import (attitude_from_orbit, attitude_roll, combine_rotation_matrices)

from sun.sun_reference import (sun_position_eci, sun_position_body, sun_position_spherical, sun_line_of_sight)

# CADRE Sun_PositionECI
class SunPositionECI(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("d2r", value=np.pi / 180.0)
        self.add_constant("LD", value=0.0)
        self.add_input("t")
        self.add_input("r_e2s_I", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        d2r = self.constants["d2r"]
        LD = self.constants["LD"]
        t = self.inputs["t"]
        r = self.inputs["r_e2s_I"]

        T = LD + t / 3600.0 / 24.0
        L = d2r * 280.460 + d2r * 0.9856474 * T
        g = d2r * 357.528 + d2r * 0.9856003 * T
        Lambda = L + d2r * 1.914666 * am.sin(g) + d2r * 0.01999464 * am.sin(2.0 * g)
        eps = d2r * 23.439 - d2r * 3.56e-7 * T

        x = am.cos(Lambda)
        y = am.sin(Lambda) * am.cos(eps)
        z = am.sin(Lambda) * am.sin(eps)

        self.constraints["res"] = [r[0] - x, r[1] - y, r[2] - z]


# CADRE Sun_PositionBody
class SunPositionBody(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("O_BI", shape=(9,))
        self.add_input("r_e2s_I", shape=(3,))
        self.add_input("r_e2s_B", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        O = self.inputs["O_BI"]
        r_I = self.inputs["r_e2s_I"]
        r_B = self.inputs["r_e2s_B"]

        x = O[0] * r_I[0] + O[1] * r_I[1] + O[2] * r_I[2]
        y = O[3] * r_I[0] + O[4] * r_I[1] + O[5] * r_I[2]
        z = O[6] * r_I[0] + O[7] * r_I[1] + O[8] * r_I[2]

        self.constraints["res"] = [r_B[0] - x, r_B[1] - y, r_B[2] - z]


# Extract AMIGO vector
def extract_vector(x, variable_name, n, width):
    values = np.zeros((n, width))

    for i in range(width):
        values[:, i] = x[f"{variable_name}[:, {i}]"]

    return values

# CADRE Sun_PositionSpherical
class SunPositionSpherical(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("r_e2s_B", shape=(3,))
        self.add_input("azimuth")
        self.add_input("elevation")
        self.add_constraint("res", shape=(2,))

    def compute(self):
        r = self.inputs["r_e2s_B"]
        azimuth = self.inputs["azimuth"]
        elevation = self.inputs["elevation"]

        x = r[0]
        y = r[1]
        z = r[2]

        radius = (x*x + y*y + z*z)**0.5
        horizontal = (x*x + y*y)**0.5

        azimuth_res = am.sin(azimuth) * x - am.cos(azimuth) * y
        elevation_res = am.cos(elevation) - z / radius

        self.constraints["res"] = [azimuth_res, elevation_res]
# Full sunlight
class SunLOSVisible(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("LOS")
        self.add_constraint("res")

    def compute(self):
        LOS = self.inputs["LOS"]
        self.constraints["res"] = LOS - 1.0


# Full eclipse
class SunLOSEclipse(am.Component):
    def __init__(self):
        super().__init__()
        self.add_input("LOS")
        self.add_constraint("res")

    def compute(self):
        LOS = self.inputs["LOS"]
        self.constraints["res"] = LOS


# Eclipse transition
class SunLOSTransition(am.Component):
    def __init__(self):
        super().__init__()
        self.add_constant("r1", value=6378.137 * 0.85)
        self.add_constant("r2", value=6378.137)
        self.add_input("r_b", shape=(3,))
        self.add_input("r_s", shape=(3,))
        self.add_input("LOS")
        self.add_constraint("res")

    def compute(self):
        r1 = self.constants["r1"]
        r2 = self.constants["r2"]
        r_b = self.inputs["r_b"]
        r_s = self.inputs["r_s"]
        LOS = self.inputs["LOS"]

        cross_x = r_b[1] * r_s[2] - r_b[2] * r_s[1]
        cross_y = r_b[2] * r_s[0] - r_b[0] * r_s[2]
        cross_z = r_b[0] * r_s[1] - r_b[1] * r_s[0]
        dist = (cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)**0.5
        x = (dist - r1) / (r2 - r1)
        LOS_calc = 3.0 * x**2 - 2.0 * x**3

        self.constraints["res"] = LOS - LOS_calc

def main():
    # Initial orbit state
    r0, v0 = orbital_elements_to_state(
        alt_perigee=500.0,
        alt_apogee=500.0,
        raan=66.279,
        inclination=82.072,
        arg_perigee=0.0,
        true_anomaly=337.987,
    )

    q0 = np.concatenate((r0, v0))

    # Time grid
    orbital_period = calculate_orbital_period(alt_perigee=500.0, alt_apogee=500.0)
    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps
    times = np.linspace(0.0, orbital_period, num_nodes)

    print()
    print("=" * 70)
    print("CADRE SUN AMIGO STAGE 3")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Time step [s]:")
    print(dt)

    # Reference orbit
    reference_states, reference_rates = build_reference_trajectory(q0, dt, num_time_steps)

    # Reference attitude
    gamma_reference = np.zeros(num_nodes)
    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(O_BR_reference, O_RI_reference)
    O_BI_flat = O_BI_reference.reshape(num_nodes, 9)

    # Reference Sun
    r_e2s_I_reference = sun_position_eci(times)
    r_e2s_B_reference = sun_position_body(O_BI_reference, r_e2s_I_reference)
    azimuth_reference, elevation_reference = sun_position_spherical(r_e2s_B_reference)
    LOS_reference = sun_line_of_sight(reference_states, r_e2s_I_reference)

    # Classify LOS regions
    r1 = 6378.137 * 0.85
    r2 = 6378.137

    visible_indices = []
    eclipse_indices = []
    transition_indices = []

    for i in range(num_nodes):
        r_b = reference_states[i, 0:3]
        r_s = r_e2s_I_reference[i]
        dot = np.dot(r_b, r_s)
        dist = np.linalg.norm(np.cross(r_b, r_s))

        if dot >= 0.0:
            visible_indices.append(i)
        elif dist <= r1:
            eclipse_indices.append(i)
        elif dist >= r2:
            visible_indices.append(i)
        else:
            transition_indices.append(i)

    visible_indices = np.array(visible_indices, dtype=int)
    eclipse_indices = np.array(eclipse_indices, dtype=int)
    transition_indices = np.array(transition_indices, dtype=int)

    print()
    print("Full sunlight nodes:")
    print(len(visible_indices))

    print()
    print("Full eclipse nodes:")
    print(len(eclipse_indices))

    print()
    print("Transition nodes:")
    print(len(transition_indices))

    # Components
    sun_eci = SunPositionECI()
    sun_body = SunPositionBody()
    sun_spherical = SunPositionSpherical()
    los_visible = SunLOSVisible()
    los_eclipse = SunLOSEclipse()
    los_transition = SunLOSTransition()

    # Build model
    model = am.Model("cadre_sun_stage3")

    model.add_component("sun_eci", num_nodes, sun_eci)
    model.add_component("sun_body", num_nodes, sun_body)
    model.add_component("sun_spherical", num_nodes, sun_spherical)
    model.add_component("los_visible", len(visible_indices), los_visible)
    model.add_component("los_eclipse", len(eclipse_indices), los_eclipse)
    model.add_component("los_transition", len(transition_indices), los_transition)

    # Sun ECI to Sun body
    model.link("sun_eci.r_e2s_I", "sun_body.r_e2s_I")

    # Sun body to spherical angles
    model.link("sun_body.r_e2s_B", "sun_spherical.r_e2s_B")

    # Fix time values
    model.set_meta("value", "sun_eci.t[:]", times)
    model.set_meta("lower", "sun_eci.t[:]", times)
    model.set_meta("upper", "sun_eci.t[:]", times)

    # Fix attitude values
    for i in range(9):
        model.set_meta("value", f"sun_body.O_BI[:, {i}]", O_BI_flat[:, i])
        model.set_meta("lower", f"sun_body.O_BI[:, {i}]", O_BI_flat[:, i])
        model.set_meta("upper", f"sun_body.O_BI[:, {i}]", O_BI_flat[:, i])

    # Sun initial guesses
    for i in range(3):
        model.set_meta("value", f"sun_eci.r_e2s_I[:, {i}]", r_e2s_I_reference[:, i])
        model.set_meta("value", f"sun_body.r_e2s_B[:, {i}]", r_e2s_B_reference[:, i])

    # Angle initial guesses
    model.set_meta("value", "sun_spherical.azimuth[:]", azimuth_reference)
    model.set_meta("value", "sun_spherical.elevation[:]", elevation_reference)

    # LOS initial guesses
    model.set_meta("value", "los_visible.LOS[:]", LOS_reference[visible_indices])
    model.set_meta("value", "los_eclipse.LOS[:]", LOS_reference[eclipse_indices])
    model.set_meta("value", "los_transition.LOS[:]", LOS_reference[transition_indices])

    # Fix transition orbit and Sun vectors
    transition_r_b = reference_states[transition_indices, 0:3]
    transition_r_s = r_e2s_I_reference[transition_indices]

    for i in range(3):
        model.set_meta("value", f"los_transition.r_b[:, {i}]", transition_r_b[:, i])
        model.set_meta("lower", f"los_transition.r_b[:, {i}]", transition_r_b[:, i])
        model.set_meta("upper", f"los_transition.r_b[:, {i}]", transition_r_b[:, i])
        model.set_meta("value", f"los_transition.r_s[:, {i}]", transition_r_s[:, i])
        model.set_meta("lower", f"los_transition.r_s[:, {i}]", transition_r_s[:, i])
        model.set_meta("upper", f"los_transition.r_s[:, {i}]", transition_r_s[:, i])

    # Build model
    print()
    print("Building AMIGO Sun model...")

    model.build_module()
    print("Build successful.")

    model.initialize()
    print("Initialization successful.")

    # Solve model
    x = model.create_vector()
    opt = am.Optimizer(model, x)

    opt_options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print("Solving AMIGO Sun model...")

    opt.optimize(opt_options)

    print()
    print("AMIGO solve complete.")

    # Extract Sun results
    r_e2s_I_amigo = extract_vector(x, "sun_eci.r_e2s_I", num_nodes, 3)
    r_e2s_B_amigo = extract_vector(x, "sun_body.r_e2s_B", num_nodes, 3)
    azimuth_amigo = x["sun_spherical.azimuth[:]"]
    elevation_amigo = x["sun_spherical.elevation[:]"]

    # Reconstruct LOS
    LOS_amigo = np.zeros(num_nodes)
    LOS_amigo[visible_indices] = x["los_visible.LOS[:]"]
    LOS_amigo[eclipse_indices] = x["los_eclipse.LOS[:]"]
    LOS_amigo[transition_indices] = x["los_transition.LOS[:]"]

    # Differences
    eci_difference = np.max(np.abs(r_e2s_I_amigo - r_e2s_I_reference))
    body_difference = np.max(np.abs(r_e2s_B_amigo - r_e2s_B_reference))
    azimuth_difference = np.max(np.abs(azimuth_amigo - azimuth_reference))
    elevation_difference = np.max(np.abs(elevation_amigo - elevation_reference))
    LOS_difference = np.max(np.abs(LOS_amigo - LOS_reference))

    # Fixed input checks
    time_amigo = x["sun_eci.t[:]"]
    O_BI_amigo = extract_vector(x, "sun_body.O_BI", num_nodes, 9)

    time_difference = np.max(np.abs(time_amigo - times))
    attitude_difference = np.max(np.abs(O_BI_amigo - O_BI_flat))

    # Print validation
    print()
    print("=" * 70)
    print("SUN AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("Maximum time difference:")
    print(time_difference)

    print()
    print("Maximum fixed O_BI difference:")
    print(attitude_difference)

    print()
    print("Maximum r_e2s_I difference:")
    print(eci_difference)

    print()
    print("Maximum r_e2s_B difference:")
    print(body_difference)

    print()
    print("Maximum azimuth difference:")
    print(azimuth_difference)

    print()
    print("Maximum elevation difference:")
    print(elevation_difference)

    print()
    print("Maximum LOS difference:")
    print(LOS_difference)


if __name__ == "__main__":
    main()