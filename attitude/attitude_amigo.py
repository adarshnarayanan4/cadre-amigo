import amigo as am
import numpy as np

from orbit.orbit_reference import (
    orbital_elements_to_state,
    calculate_orbital_period,
)

from orbit.orbit_amigo import (
    OrbitDynamics,
    TrapezoidRule,
    InitialConditions,
    build_reference_trajectory,
    solve_model,
)

from attitude.attitude_reference import (
    attitude_from_orbit,
    attitude_roll,
    combine_rotation_matrices,
    rotation_matrix_rates,
    angular_velocity,
    angular_acceleration,
    attitude_torque,
    body_frame_velocity,
)


# AMIGO attitude orientation component
class AttitudeOrientation(am.Component):
    def __init__(self):
        super().__init__()

        # Orbit state:
        # [x, y, z, vx, vy, vz]
        self.add_input("q", shape=(6,))

        # Roll angle
        self.add_input("gamma")

        # Base orientation matrix
        # Stored as 9 values
        self.add_input("O_RI", shape=(9,))

        # Roll rotation matrix
        self.add_input("O_BR", shape=(9,))

        # Final body orientation matrix
        self.add_input("O_BI", shape=(9,))

        # 9 equations for O_RI
        # 9 equations for O_BR
        # 9 equations for O_BI
        self.add_constraint("res", shape=(27,))

    def compute(self):
        q = self.inputs["q"]
        gamma = self.inputs["gamma"]

        O_RI = self.inputs["O_RI"]
        O_BR = self.inputs["O_BR"]
        O_BI = self.inputs["O_BI"]

        # Orbit position and velocity
        x = q[0]
        y = q[1]
        z = q[2]

        vx = q[3]
        vy = q[4]
        vz = q[5]

        # Normalize position and velocity
        norm_r = (x * x + y * y + z * z) ** 0.5
        norm_v = (vx * vx + vy * vy + vz * vz) ** 0.5

        rx = x / norm_r
        ry = y / norm_r
        rz = z / norm_r

        vxn = vx / norm_v
        vyn = vy / norm_v
        vzn = vz / norm_v

        # CADRE Attitude_Attitude
        # iB = v x r
        # jB = -v x iB
        iB0 = vyn * rz - vzn * ry
        iB1 = vzn * rx - vxn * rz
        iB2 = vxn * ry - vyn * rx

        jB0 = -vyn * iB2 + vzn * iB1
        jB1 = -vzn * iB0 + vxn * iB2
        jB2 = -vxn * iB1 + vyn * iB0

        O_RI_calc = [iB0, iB1, iB2, jB0, jB1, jB2, -vxn, -vyn, -vzn]

        # CADRE Attitude_Roll
        c = am.cos(gamma)
        s = am.sin(gamma)

        O_BR_calc = [c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0]

        # CADRE Attitude_RotationMtx
        # O_BI = O_BR @ O_RI
        O_BI_calc = 9 * [None]

        O_BI_calc[0] = (
            O_BR[0] * O_RI[0] + O_BR[1] * O_RI[3] + O_BR[2] * O_RI[6]
        )
        O_BI_calc[1] = (
            O_BR[0] * O_RI[1] + O_BR[1] * O_RI[4] + O_BR[2] * O_RI[7]
        )
        O_BI_calc[2] = (
            O_BR[0] * O_RI[2] + O_BR[1] * O_RI[5] + O_BR[2] * O_RI[8]
        )
        O_BI_calc[3] = (
            O_BR[3] * O_RI[0] + O_BR[4] * O_RI[3] + O_BR[5] * O_RI[6]
        )
        O_BI_calc[4] = (
            O_BR[3] * O_RI[1] + O_BR[4] * O_RI[4] + O_BR[5] * O_RI[7]
        )
        O_BI_calc[5] = (
            O_BR[3] * O_RI[2] + O_BR[4] * O_RI[5] + O_BR[5] * O_RI[8]
        )
        O_BI_calc[6] = (
            O_BR[6] * O_RI[0] + O_BR[7] * O_RI[3] + O_BR[8] * O_RI[6]
        )
        O_BI_calc[7] = (
            O_BR[6] * O_RI[1] + O_BR[7] * O_RI[4] + O_BR[8] * O_RI[7]
        )
        O_BI_calc[8] = (
            O_BR[6] * O_RI[2] + O_BR[7] * O_RI[5] + O_BR[8] * O_RI[8]
        )

        # Residual equations
        res = 27 * [None]

        # O_RI equations
        for i in range(9):
            res[i] = O_RI[i] - O_RI_calc[i]

        # O_BR equations
        for i in range(9):
            res[9 + i] = O_BR[i] - O_BR_calc[i]

        # O_BI equations
        for i in range(9):
            res[18 + i] = O_BI[i] - O_BI_calc[i]

        self.constraints["res"] = res


# Zero-roll baseline
class ZeroRoll(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("gamma")
        self.add_constraint("res")

    def compute(self):
        gamma = self.inputs["gamma"]

        # gamma = 0
        self.constraints["res"] = gamma


# Helper: extract a flattened AMIGO matrix
def extract_matrix(x, variable_name, n):
    matrix_flat = np.zeros((n, 9))

    for i in range(9):
        matrix_flat[:, i] = x[f"{variable_name}[:, {i}]"]

    return matrix_flat.reshape(n, 3, 3)


# Forward finite difference
class ForwardMatrixRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("O0", shape=(9,))
        self.add_input("O1", shape=(9,))
        self.add_input("Odot", shape=(9,))
        self.add_constraint("res", shape=(9,))

    def compute(self):
        dt = self.constants["dt"]
        O0 = self.inputs["O0"]
        O1 = self.inputs["O1"]
        Odot = self.inputs["Odot"]

        res = 9 * [None]

        for i in range(9):
            res[i] = Odot[i] - (O1[i] - O0[i]) / dt

        self.constraints["res"] = res


# Central finite difference
class CentralMatrixRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("O_prev", shape=(9,))
        self.add_input("O_next", shape=(9,))
        self.add_input("Odot", shape=(9,))
        self.add_constraint("res", shape=(9,))

    def compute(self):
        dt = self.constants["dt"]
        O_prev = self.inputs["O_prev"]
        O_next = self.inputs["O_next"]
        Odot = self.inputs["Odot"]

        res = 9 * [None]

        for i in range(9):
            res[i] = Odot[i] - (O_next[i] - O_prev[i]) / (2.0 * dt)

        self.constraints["res"] = res


# Backward finite difference
class BackwardMatrixRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("O_prev", shape=(9,))
        self.add_input("O_last", shape=(9,))
        self.add_input("Odot", shape=(9,))
        self.add_constraint("res", shape=(9,))

    def compute(self):
        dt = self.constants["dt"]
        O_prev = self.inputs["O_prev"]
        O_last = self.inputs["O_last"]
        Odot = self.inputs["Odot"]

        res = 9 * [None]

        for i in range(9):
            res[i] = Odot[i] - (O_last[i] - O_prev[i]) / dt

        self.constraints["res"] = res


# CADRE Attitude_Angular
class AttitudeAngular(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("O_BI", shape=(9,))
        self.add_input("Odot_BI", shape=(9,))
        self.add_input("w_B", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        rotation = self.inputs["O_BI"]
        Odot = self.inputs["Odot_BI"]
        w = self.inputs["w_B"]

        # Flattened matrix indexing:
        # [0 1 2]
        # [3 4 5]
        # [6 7 8]

        wx = (
            Odot[6] * rotation[3]
            + Odot[7] * rotation[4]
            + Odot[8] * rotation[5]
        )
        wy = (
            Odot[0] * rotation[6]
            + Odot[1] * rotation[7]
            + Odot[2] * rotation[8]
        )
        wz = (
            Odot[3] * rotation[0]
            + Odot[4] * rotation[1]
            + Odot[5] * rotation[2]
        )

        self.constraints["res"] = [w[0] - wx, w[1] - wy, w[2] - wz]


# Forward angular acceleration
class ForwardAngularRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("w0", shape=(3,))
        self.add_input("w1", shape=(3,))
        self.add_input("wdot", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        dt = self.constants["dt"]
        w0 = self.inputs["w0"]
        w1 = self.inputs["w1"]
        wdot = self.inputs["wdot"]

        res = 3 * [None]

        for i in range(3):
            res[i] = wdot[i] - (w1[i] - w0[i]) / dt

        self.constraints["res"] = res


# Central angular acceleration
class CentralAngularRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("w_prev", shape=(3,))
        self.add_input("w_next", shape=(3,))
        self.add_input("wdot", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        dt = self.constants["dt"]
        w_prev = self.inputs["w_prev"]
        w_next = self.inputs["w_next"]
        wdot = self.inputs["wdot"]

        res = 3 * [None]

        for i in range(3):
            res[i] = wdot[i] - (w_next[i] - w_prev[i]) / (2.0 * dt)

        self.constraints["res"] = res


# Backward angular acceleration
class BackwardAngularRate(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)
        self.add_input("w_prev", shape=(3,))
        self.add_input("w_last", shape=(3,))
        self.add_input("wdot", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        dt = self.constants["dt"]
        w_prev = self.inputs["w_prev"]
        w_last = self.inputs["w_last"]
        wdot = self.inputs["wdot"]

        res = 3 * [None]

        for i in range(3):
            res[i] = wdot[i] - (w_last[i] - w_prev[i]) / dt

        self.constraints["res"] = res


# CADRE Attitude_Torque
class AttitudeTorque(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("w_B", shape=(3,))
        self.add_input("wdot_B", shape=(3,))
        self.add_input("T_tot", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        w = self.inputs["w_B"]
        wdot = self.inputs["wdot_B"]
        T = self.inputs["T_tot"]

        Jx = 0.018
        Jy = 0.018
        Jz = 0.006

        Jw0 = Jx * w[0]
        Jw1 = Jy * w[1]
        Jw2 = Jz * w[2]

        Tx = Jx * wdot[0] + w[1] * Jw2 - w[2] * Jw1
        Ty = Jy * wdot[1] + w[2] * Jw0 - w[0] * Jw2
        Tz = Jz * wdot[2] + w[0] * Jw1 - w[1] * Jw0

        self.constraints["res"] = [T[0] - Tx, T[1] - Ty, T[2] - Tz]


# CADRE Attitude_Sideslip
class AttitudeSideslip(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("q", shape=(6,))
        self.add_input("O_BI", shape=(9,))
        self.add_input("v_B", shape=(3,))
        self.add_constraint("res", shape=(3,))

    def compute(self):
        q = self.inputs["q"]
        rotation = self.inputs["O_BI"]
        v_B = self.inputs["v_B"]

        vx = q[3]
        vy = q[4]
        vz = q[5]

        v0 = rotation[0] * vx + rotation[1] * vy + rotation[2] * vz
        v1 = rotation[3] * vx + rotation[4] * vy + rotation[5] * vz
        v2 = rotation[6] * vx + rotation[7] * vy + rotation[8] * vz

        self.constraints["res"] = [v_B[0] - v0, v_B[1] - v1, v_B[2] - v2]


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
    orbital_period = calculate_orbital_period(
        alt_perigee=500.0, alt_apogee=500.0
    )
    num_time_steps = 568
    num_nodes = num_time_steps + 1
    dt = orbital_period / num_time_steps

    print()
    print("=" * 70)
    print("CADRE COUPLED ORBIT + ATTITUDE")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Time step [s]:")
    print(dt)

    # Reference orbit
    reference_states, reference_rates = build_reference_trajectory(
        q0, dt, num_time_steps
    )

    # Reference attitude
    gamma_reference = np.zeros(num_nodes)

    O_RI_reference = attitude_from_orbit(reference_states)
    O_BR_reference = attitude_roll(gamma_reference)
    O_BI_reference = combine_rotation_matrices(O_BR_reference, O_RI_reference)
    reference_Odot_BI = rotation_matrix_rates(O_BI_reference, dt)
    reference_w_B = angular_velocity(O_BI_reference, reference_Odot_BI)
    reference_wdot_B = angular_acceleration(reference_w_B, dt)
    reference_T_tot = attitude_torque(reference_w_B, reference_wdot_B)
    reference_v_B = body_frame_velocity(reference_states, O_BI_reference)

    # Flatten reference matrices
    O_RI_flat = O_RI_reference.reshape(num_nodes, 9)
    O_BR_flat = O_BR_reference.reshape(num_nodes, 9)
    O_BI_flat = O_BI_reference.reshape(num_nodes, 9)

    # Components
    orbit = OrbitDynamics()
    trap = TrapezoidRule(dt)
    ic = InitialConditions(q0)
    attitude = AttitudeOrientation()
    roll_zero = ZeroRoll()
    rate_forward = ForwardMatrixRate(dt)
    rate_central = CentralMatrixRate(dt)
    rate_backward = BackwardMatrixRate(dt)
    angular = AttitudeAngular()
    angular_rate_forward = ForwardAngularRate(dt)
    angular_rate_central = CentralAngularRate(dt)
    angular_rate_backward = BackwardAngularRate(dt)
    torque = AttitudeTorque()
    sideslip = AttitudeSideslip()

    # Build model
    model = am.Model("cadre_orbit_attitude_stage4")

    model.add_component("orbit", num_nodes, orbit)
    model.add_component("trap", 6 * num_time_steps, trap)
    model.add_component("ic", 1, ic)
    model.add_component("attitude", num_nodes, attitude)
    model.add_component("roll_zero", num_nodes, roll_zero)
    model.add_component("rate_forward", 1, rate_forward)
    model.add_component("rate_central", num_nodes - 2, rate_central)
    model.add_component("rate_backward", 1, rate_backward)
    model.add_component("angular", num_nodes, angular)
    model.add_component("angular_rate_forward", 1, angular_rate_forward)
    model.add_component(
        "angular_rate_central", num_nodes - 2, angular_rate_central
    )
    model.add_component("angular_rate_backward", 1, angular_rate_backward)
    model.add_component("torque", num_nodes, torque)
    model.add_component("sideslip", num_nodes, sideslip)

    # Orbit trapezoid links
    for i in range(6):
        start = i * num_time_steps
        end = (i + 1) * num_time_steps

        model.link(
            f"orbit.q[:{num_time_steps}, {i}]", f"trap.q1[{start}:{end}]"
        )
        model.link(f"orbit.q[1:, {i}]", f"trap.q2[{start}:{end}]")
        model.link(f"orbit.qdot[:-1, {i}]", f"trap.q1dot[{start}:{end}]")
        model.link(f"orbit.qdot[1:, {i}]", f"trap.q2dot[{start}:{end}]")

    # Initial orbit condition
    model.link("orbit.q[0, :]", "ic.q[0, :]")

    # Orbit to attitude
    model.link("orbit.q", "attitude.q")

    # Zero roll
    model.link("attitude.gamma", "roll_zero.gamma")

    # First orientation-rate point
    model.link("attitude.O_BI[0, :]", "rate_forward.O0[0, :]")
    model.link("attitude.O_BI[1, :]", "rate_forward.O1[0, :]")

    # Interior orientation-rate points
    model.link(f"attitude.O_BI[:{num_nodes - 2}, :]", "rate_central.O_prev")
    model.link("attitude.O_BI[2:, :]", "rate_central.O_next")

    # Final orientation-rate point
    model.link(
        f"attitude.O_BI[{num_nodes - 2}, :]", "rate_backward.O_prev[0, :]"
    )
    model.link(
        f"attitude.O_BI[{num_nodes - 1}, :]", "rate_backward.O_last[0, :]"
    )

    # Orientation to angular velocity
    model.link("attitude.O_BI", "angular.O_BI")

    # Orientation rate to angular velocity
    model.link("rate_forward.Odot[0, :]", "angular.Odot_BI[0, :]")
    model.link("rate_central.Odot", f"angular.Odot_BI[1:{num_nodes - 1}, :]")
    model.link(
        "rate_backward.Odot[0, :]", f"angular.Odot_BI[{num_nodes - 1}, :]"
    )

    # First angular-acceleration point
    model.link("angular.w_B[0, :]", "angular_rate_forward.w0[0, :]")
    model.link("angular.w_B[1, :]", "angular_rate_forward.w1[0, :]")

    # Interior angular-acceleration points
    model.link(
        f"angular.w_B[:{num_nodes - 2}, :]", "angular_rate_central.w_prev"
    )
    model.link("angular.w_B[2:, :]", "angular_rate_central.w_next")

    # Final angular-acceleration point
    model.link(
        f"angular.w_B[{num_nodes - 2}, :]",
        "angular_rate_backward.w_prev[0, :]",
    )
    model.link(
        f"angular.w_B[{num_nodes - 1}, :]",
        "angular_rate_backward.w_last[0, :]",
    )

    # Angular velocity to torque
    model.link("angular.w_B", "torque.w_B")

    # Angular acceleration to torque
    model.link("angular_rate_forward.wdot[0, :]", "torque.wdot_B[0, :]")
    model.link(
        "angular_rate_central.wdot", f"torque.wdot_B[1:{num_nodes - 1}, :]"
    )
    model.link(
        "angular_rate_backward.wdot[0, :]",
        f"torque.wdot_B[{num_nodes - 1}, :]",
    )

    # Orbit and orientation to sideslip
    model.link("orbit.q", "sideslip.q")
    model.link("attitude.O_BI", "sideslip.O_BI")

    # Orbit initial guesses
    for i in range(6):
        model.set_meta("value", f"orbit.q[:, {i}]", reference_states[:, i])
        model.set_meta("value", f"orbit.qdot[:, {i}]", reference_rates[:, i])

    # Roll initial guess
    model.set_meta("value", "attitude.gamma[:]", gamma_reference)

    # Attitude initial guesses
    for i in range(9):
        model.set_meta("value", f"attitude.O_RI[:, {i}]", O_RI_flat[:, i])
        model.set_meta("value", f"attitude.O_BR[:, {i}]", O_BR_flat[:, i])
        model.set_meta("value", f"attitude.O_BI[:, {i}]", O_BI_flat[:, i])

    # Orientation-rate initial guesses
    for i in range(9):
        row = i // 3
        column = i % 3

        model.set_meta(
            "value",
            f"rate_forward.Odot[:, {i}]",
            np.array([reference_Odot_BI[0, row, column]]),
        )
        model.set_meta(
            "value",
            f"rate_central.Odot[:, {i}]",
            reference_Odot_BI[1:-1, row, column],
        )
        model.set_meta(
            "value",
            f"rate_backward.Odot[:, {i}]",
            np.array([reference_Odot_BI[-1, row, column]]),
        )

    # Angular velocity initial guesses
    for i in range(3):
        model.set_meta("value", f"angular.w_B[:, {i}]", reference_w_B[:, i])

    # Angular acceleration initial guesses
    for i in range(3):
        model.set_meta(
            "value",
            f"angular_rate_forward.wdot[:, {i}]",
            np.array([reference_wdot_B[0, i]]),
        )
        model.set_meta(
            "value",
            f"angular_rate_central.wdot[:, {i}]",
            reference_wdot_B[1:-1, i],
        )
        model.set_meta(
            "value",
            f"angular_rate_backward.wdot[:, {i}]",
            np.array([reference_wdot_B[-1, i]]),
        )

    # Torque initial guesses
    for i in range(3):
        model.set_meta("value", f"torque.T_tot[:, {i}]", reference_T_tot[:, i])

    # Sideslip initial guesses
    for i in range(3):
        model.set_meta("value", f"sideslip.v_B[:, {i}]", reference_v_B[:, i])

    # Build model
    print()
    print("Building coupled AMIGO orbit-attitude model...")

    model.build_module()
    print("Build successful.")

    model.initialize()
    print("Initialization successful.")

    # Solve model
    x = model.create_vector()
    solve_model(model, x)

    # Extract AMIGO orbit
    amigo_states = np.zeros((num_nodes, 6))

    for i in range(6):
        amigo_states[:, i] = x[f"orbit.q[:, {i}]"]

    # Extract AMIGO attitude matrices
    O_RI_amigo = extract_matrix(x, "attitude.O_RI", num_nodes)
    O_BR_amigo = extract_matrix(x, "attitude.O_BR", num_nodes)
    O_BI_amigo = extract_matrix(x, "attitude.O_BI", num_nodes)

    # Extract AMIGO orientation rates
    Odot_BI_amigo = np.zeros((num_nodes, 3, 3))

    for i in range(9):
        row = i // 3
        column = i % 3

        Odot_BI_amigo[0, row, column] = x[f"rate_forward.Odot[:, {i}]"][0]
        Odot_BI_amigo[1:-1, row, column] = x[f"rate_central.Odot[:, {i}]"]
        Odot_BI_amigo[-1, row, column] = x[f"rate_backward.Odot[:, {i}]"][0]

    # Extract AMIGO angular velocity
    w_B_amigo = np.zeros((num_nodes, 3))

    for i in range(3):
        w_B_amigo[:, i] = x[f"angular.w_B[:, {i}]"]

    # Extract AMIGO angular acceleration
    wdot_B_amigo = np.zeros((num_nodes, 3))

    for i in range(3):
        wdot_B_amigo[0, i] = x[f"angular_rate_forward.wdot[:, {i}]"][0]
        wdot_B_amigo[1:-1, i] = x[f"angular_rate_central.wdot[:, {i}]"]
        wdot_B_amigo[-1, i] = x[f"angular_rate_backward.wdot[:, {i}]"][0]

    # Extract AMIGO torque
    T_tot_amigo = np.zeros((num_nodes, 3))

    for i in range(3):
        T_tot_amigo[:, i] = x[f"torque.T_tot[:, {i}]"]

    # Extract AMIGO body-frame velocity
    v_B_amigo = np.zeros((num_nodes, 3))

    for i in range(3):
        v_B_amigo[:, i] = x[f"sideslip.v_B[:, {i}]"]

    # Build reference using same AMIGO orbit
    O_RI_same_orbit = attitude_from_orbit(amigo_states)
    O_BR_same_orbit = attitude_roll(gamma_reference)
    O_BI_same_orbit = combine_rotation_matrices(
        O_BR_same_orbit, O_RI_same_orbit
    )
    Odot_BI_same_orbit = rotation_matrix_rates(O_BI_same_orbit, dt)
    w_B_same_orbit = angular_velocity(O_BI_same_orbit, Odot_BI_same_orbit)
    wdot_B_same_orbit = angular_acceleration(w_B_same_orbit, dt)
    T_tot_same_orbit = attitude_torque(w_B_same_orbit, wdot_B_same_orbit)
    v_B_same_orbit = body_frame_velocity(amigo_states, O_BI_same_orbit)

    # Attitude-only differences
    O_RI_attitude_difference = np.max(np.abs(O_RI_amigo - O_RI_same_orbit))
    O_BR_attitude_difference = np.max(np.abs(O_BR_amigo - O_BR_same_orbit))
    O_BI_attitude_difference = np.max(np.abs(O_BI_amigo - O_BI_same_orbit))
    Odot_attitude_difference = np.max(
        np.abs(Odot_BI_amigo - Odot_BI_same_orbit)
    )
    w_B_attitude_difference = np.max(np.abs(w_B_amigo - w_B_same_orbit))
    wdot_B_attitude_difference = np.max(
        np.abs(wdot_B_amigo - wdot_B_same_orbit)
    )
    T_tot_attitude_difference = np.max(np.abs(T_tot_amigo - T_tot_same_orbit))
    v_B_attitude_difference = np.max(np.abs(v_B_amigo - v_B_same_orbit))

    # Full RK4 versus AMIGO differences
    O_RI_total_difference = np.max(np.abs(O_RI_amigo - O_RI_reference))
    O_BI_total_difference = np.max(np.abs(O_BI_amigo - O_BI_reference))
    Odot_total_difference = np.max(np.abs(Odot_BI_amigo - reference_Odot_BI))
    w_B_total_difference = np.max(np.abs(w_B_amigo - reference_w_B))
    wdot_B_total_difference = np.max(np.abs(wdot_B_amigo - reference_wdot_B))
    T_tot_total_difference = np.max(np.abs(T_tot_amigo - reference_T_tot))
    v_B_total_difference = np.max(np.abs(v_B_amigo - reference_v_B))

    # Print attitude-only validation
    print()
    print("=" * 70)
    print("ATTITUDE-ONLY VALIDATION")
    print("=" * 70)

    print()
    print("Maximum O_RI difference using same orbit:")
    print(O_RI_attitude_difference)

    print()
    print("Maximum O_BR difference using same orbit:")
    print(O_BR_attitude_difference)

    print()
    print("Maximum O_BI difference using same orbit:")
    print(O_BI_attitude_difference)

    print()
    print("Maximum Odot_BI difference using same orbit:")
    print(Odot_attitude_difference)

    print()
    print("Maximum w_B difference using same orbit:")
    print(w_B_attitude_difference)

    print()
    print("Maximum wdot_B difference using same orbit:")
    print(wdot_B_attitude_difference)

    print()
    print("Maximum T_tot difference using same orbit:")
    print(T_tot_attitude_difference)

    print()
    print("Maximum v_B difference using same orbit:")
    print(v_B_attitude_difference)

    # Print full coupled comparison
    print()
    print("=" * 70)
    print("FULL COUPLED RK4 VS AMIGO DIFFERENCE")
    print("=" * 70)

    print()
    print("Maximum O_RI difference:")
    print(O_RI_total_difference)

    print()
    print("Maximum O_BI difference:")
    print(O_BI_total_difference)

    print()
    print("Maximum Odot_BI difference:")
    print(Odot_total_difference)

    print()
    print("Maximum w_B difference:")
    print(w_B_total_difference)

    print()
    print("Maximum wdot_B difference:")
    print(wdot_B_total_difference)

    print()
    print("Maximum T_tot difference:")
    print(T_tot_total_difference)

    print()
    print("Maximum v_B difference:")
    print(v_B_total_difference)


if __name__ == "__main__":
    main()
