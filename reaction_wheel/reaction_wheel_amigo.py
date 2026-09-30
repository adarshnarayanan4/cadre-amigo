import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path
from reaction_wheel.reaction_wheel_reference import (
    J_RW,
    reaction_wheel_dynamics,
    reaction_wheel_power,
    reaction_wheel_torque,
    propagate_reaction_wheel,
    propagate_reaction_wheel_trapezoid,
)


V = 4.0
a = 4.9e-4
b = 4.5e2
I0 = 0.017


class ReactionWheelTorque(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("T_tot", shape=(3,))
        self.add_input("T_RW", shape=(3,))

        self.add_constraint("res", shape=(3,))

    def compute(self):
        T_tot = self.inputs["T_tot"]
        T_RW = self.inputs["T_RW"]

        res = 3 * [None]

        for i in range(3):
            res[i] = T_RW[i] - T_tot[i]

        self.constraints["res"] = res


class ReactionWheelDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("J_RW", value=J_RW)

        self.add_input("w_B", shape=(3,))
        self.add_input("T_RW", shape=(3,))

        self.add_input("w_RW", shape=(3,))
        self.add_input("w_RW_dot", shape=(3,))

        self.add_constraint("res", shape=(3,))

    def compute(self):
        J_RW = self.constants["J_RW"]

        w_B = self.inputs["w_B"]
        T_RW = self.inputs["T_RW"]

        w_RW = self.inputs["w_RW"]
        w_RW_dot = self.inputs["w_RW_dot"]

        wx = 3 * [None]

        wx[0] = (
            -T_RW[0] / J_RW
            + w_B[2] * w_RW[1]
            - w_B[1] * w_RW[2]
        )

        wx[1] = (
            -T_RW[1] / J_RW
            - w_B[2] * w_RW[0]
            + w_B[0] * w_RW[2]
        )

        wx[2] = (
            -T_RW[2] / J_RW
            + w_B[1] * w_RW[0]
            - w_B[0] * w_RW[1]
        )

        res = 3 * [None]

        for i in range(3):
            res[i] = wx[i] - w_RW_dot[i]

        self.constraints["res"] = res


class TrapezoidRule(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)

        self.add_input("w0", shape=(3,))
        self.add_input("w1", shape=(3,))

        self.add_input("wdot0", shape=(3,))
        self.add_input("wdot1", shape=(3,))

        self.add_constraint("res", shape=(3,))

    def compute(self):
        dt = self.constants["dt"]

        w0 = self.inputs["w0"]
        w1 = self.inputs["w1"]

        wdot0 = self.inputs["wdot0"]
        wdot1 = self.inputs["wdot1"]

        res = 3 * [None]

        for i in range(3):
            res[i] = (
                w1[i]
                - w0[i]
                - 0.5 * dt * (wdot0[i] + wdot1[i])
            )

        self.constraints["res"] = res


class InitialConditions(am.Component):
    def __init__(self, w_RW0):
        super().__init__()

        self.add_constant("w0", value=float(w_RW0[0]))
        self.add_constant("w1", value=float(w_RW0[1]))
        self.add_constant("w2", value=float(w_RW0[2]))

        self.add_input("w_RW", shape=(3,))

        self.add_constraint("res", shape=(3,))
        self.add_objective("obj")

    def compute(self):
        w_RW = self.inputs["w_RW"]

        w0 = self.constants["w0"]
        w1 = self.constants["w1"]
        w2 = self.constants["w2"]

        self.constraints["res"] = [
            w_RW[0] - w0,
            w_RW[1] - w1,
            w_RW[2] - w2,
        ]

        self.objective["obj"] = (
            1.0e-12
            * (
                w_RW[0] * w_RW[0]
                + w_RW[1] * w_RW[1]
                + w_RW[2] * w_RW[2]
            )
        )


class ReactionWheelPower(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("V", value=V)
        self.add_constant("a", value=a)
        self.add_constant("b", value=b)
        self.add_constant("I0", value=I0)

        self.add_input("w_RW", shape=(3,))
        self.add_input("T_RW", shape=(3,))
        self.add_input("P_RW", shape=(3,))

        self.add_constraint("res", shape=(3,))

    def compute(self):
        V = self.constants["V"]
        a = self.constants["a"]
        b = self.constants["b"]
        I0 = self.constants["I0"]

        w_RW = self.inputs["w_RW"]
        T_RW = self.inputs["T_RW"]
        P_RW = self.inputs["P_RW"]

        res = 3 * [None]

        for i in range(3):
            power = (
                V
                * (a * w_RW[i] + b * T_RW[i]) ** 2
                + V * I0
            )

            res[i] = P_RW[i] - power

        self.constraints["res"] = res


def extract_vector(x, variable_name, n, width):
    values = np.zeros((n, width))

    for i in range(width):
        values[:, i] = x[f"{variable_name}[:, {i}]"]

    return values


def main():
    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    T_tot = data["0:T_tot"]
    T_RW_source = data["0:T_RW"]

    w_B = data["0:w_B"]

    w_RW_source = data["0:w_RW"]
    P_RW_source = data["0:P_RW"]

    w_RW0 = np.zeros(3)

    n = w_B.shape[1]
    num_intervals = n - 1

    dt = 43200.0 / num_intervals

    print()
    print("=" * 70)
    print("CADRE REACTION WHEEL AMIGO")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Number of intervals:")
    print(num_intervals)

    print()
    print("Time step [s]:")
    print(dt)

    # Reaction-wheel torque reference
    T_RW_reference = reaction_wheel_torque(
        T_tot,
    )

    # Original CADRE-style RK4
    w_RW_rk4 = propagate_reaction_wheel(
        w_RW0,
        w_B,
        T_RW_reference,
        dt,
    )

    P_RW_rk4 = reaction_wheel_power(
        w_RW_rk4,
        T_RW_reference,
    )

    # Python trapezoid reference
    w_RW_trap, w_RW_dot_trap = (
        propagate_reaction_wheel_trapezoid(
            w_RW0,
            w_B,
            T_RW_reference,
            dt,
        )
    )

    P_RW_trap = reaction_wheel_power(
        w_RW_trap,
        T_RW_reference,
    )

    print()
    print("=" * 70)
    print("PYTHON PRECHECK")
    print("=" * 70)

    # Dynamics residual
    max_dynamics_residual = 0.0

    for k in range(n):
        rate = reaction_wheel_dynamics(
            w_RW_trap[:, k],
            w_B[:, k],
            T_RW_reference[:, k],
        )

        max_dynamics_residual = max(
            max_dynamics_residual,
            np.max(
                np.abs(
                    rate - w_RW_dot_trap[:, k]
                )
            ),
        )

    # Trapezoid residual
    max_trapezoid_residual = 0.0

    for k in range(num_intervals):
        residual = (
            w_RW_trap[:, k + 1]
            - w_RW_trap[:, k]
            - 0.5
            * dt
            * (
                w_RW_dot_trap[:, k]
                + w_RW_dot_trap[:, k + 1]
            )
        )

        max_trapezoid_residual = max(
            max_trapezoid_residual,
            np.max(np.abs(residual)),
        )

    print()
    print("Python dynamics residual:")
    print(max_dynamics_residual)

    print()
    print("Python trapezoid residual:")
    print(max_trapezoid_residual)

    # Components
    torque = ReactionWheelTorque()
    dynamics = ReactionWheelDynamics()
    trapezoid = TrapezoidRule(dt)
    initial = InitialConditions(w_RW0)
    power = ReactionWheelPower()

    # Model
    model = am.Model("cadre_reaction_wheel")

    model.add_component(
        "torque",
        n,
        torque,
    )

    model.add_component(
        "dynamics",
        n,
        dynamics,
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

    model.add_component(
        "power",
        n,
        power,
    )

    # Torque links
    model.link(
        "torque.T_RW",
        "dynamics.T_RW",
    )

    model.link(
        "torque.T_RW",
        "power.T_RW",
    )

    # Wheel-speed links
    model.link(
        "dynamics.w_RW",
        "power.w_RW",
    )

    # Trapezoid links
    model.link(
        "dynamics.w_RW[:-1]",
        "trapezoid.w0",
    )

    model.link(
        "dynamics.w_RW[1:]",
        "trapezoid.w1",
    )

    model.link(
        "dynamics.w_RW_dot[:-1]",
        "trapezoid.wdot0",
    )

    model.link(
        "dynamics.w_RW_dot[1:]",
        "trapezoid.wdot1",
    )

    # Initial condition
    model.link(
        "dynamics.w_RW[0]",
        "initial.w_RW[0]",
    )

    # Fix total attitude torque
    for i in range(3):
        model.set_meta(
            "value",
            f"torque.T_tot[:, {i}]",
            T_tot[i, :],
        )

        model.set_meta(
            "lower",
            f"torque.T_tot[:, {i}]",
            T_tot[i, :],
        )

        model.set_meta(
            "upper",
            f"torque.T_tot[:, {i}]",
            T_tot[i, :],
        )

    # Fix spacecraft body angular velocity
    for i in range(3):
        model.set_meta(
            "value",
            f"dynamics.w_B[:, {i}]",
            w_B[i, :],
        )

        model.set_meta(
            "lower",
            f"dynamics.w_B[:, {i}]",
            w_B[i, :],
        )

        model.set_meta(
            "upper",
            f"dynamics.w_B[:, {i}]",
            w_B[i, :],
        )

    # Initial torque guess
    for i in range(3):
        model.set_meta(
            "value",
            f"torque.T_RW[:, {i}]",
            T_RW_reference[i, :],
        )

    # Initial wheel-speed guess
    for i in range(3):
        model.set_meta(
            "value",
            f"dynamics.w_RW[:, {i}]",
            w_RW_trap[i, :],
        )

        model.set_meta(
            "value",
            f"dynamics.w_RW_dot[:, {i}]",
            w_RW_dot_trap[i, :],
        )

    # Initial power guess
    for i in range(3):
        model.set_meta(
            "value",
            f"power.P_RW[:, {i}]",
            P_RW_trap[i, :],
        )

    print()
    print("Building AMIGO reaction-wheel model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

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
    print("Solving AMIGO reaction-wheel model...")

    opt.optimize(opt_options)

    print()
    print("AMIGO solve complete.")

    # Extract AMIGO results
    T_RW_amigo = extract_vector(
        x,
        "torque.T_RW",
        n,
        3,
    ).T

    w_RW_amigo = extract_vector(
        x,
        "dynamics.w_RW",
        n,
        3,
    ).T

    w_RW_dot_amigo = extract_vector(
        x,
        "dynamics.w_RW_dot",
        n,
        3,
    ).T

    P_RW_amigo = extract_vector(
        x,
        "power.P_RW",
        n,
        3,
    ).T

    # Differences
    torque_difference = np.max(
        np.abs(
            T_RW_amigo - T_RW_reference
        )
    )

    wheel_speed_difference = np.max(
        np.abs(
            w_RW_amigo - w_RW_trap
        )
    )

    wheel_rate_difference = np.max(
        np.abs(
            w_RW_dot_amigo - w_RW_dot_trap
        )
    )

    power_difference = np.max(
        np.abs(
            P_RW_amigo - P_RW_trap
        )
    )

    # AMIGO dynamics residual
    max_amigo_dynamics_residual = 0.0

    for k in range(n):
        rate = reaction_wheel_dynamics(
            w_RW_amigo[:, k],
            w_B[:, k],
            T_RW_amigo[:, k],
        )

        max_amigo_dynamics_residual = max(
            max_amigo_dynamics_residual,
            np.max(
                np.abs(
                    rate - w_RW_dot_amigo[:, k]
                )
            ),
        )

    # AMIGO trapezoid residual
    max_amigo_trapezoid_residual = 0.0

    for k in range(num_intervals):
        residual = (
            w_RW_amigo[:, k + 1]
            - w_RW_amigo[:, k]
            - 0.5
            * dt
            * (
                w_RW_dot_amigo[:, k]
                + w_RW_dot_amigo[:, k + 1]
            )
        )

        max_amigo_trapezoid_residual = max(
            max_amigo_trapezoid_residual,
            np.max(np.abs(residual)),
        )

    print()
    print("=" * 70)
    print("REACTION WHEEL AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("AMIGO vs Python torque difference [N*m]:")
    print(torque_difference)

    print()
    print("AMIGO vs Python trapezoid wheel-speed difference [1/s]:")
    print(wheel_speed_difference)

    print()
    print("AMIGO vs Python wheel-rate difference [1/s^2]:")
    print(wheel_rate_difference)

    print()
    print("AMIGO vs Python reaction-wheel power difference [W]:")
    print(power_difference)

    print()
    print("AMIGO dynamics residual:")
    print(max_amigo_dynamics_residual)

    print()
    print("AMIGO trapezoid residual:")
    print(max_amigo_trapezoid_residual)

    print()
    print("=" * 70)
    print("AMIGO VS ORIGINAL CADRE RK4")
    print("=" * 70)

    print()
    print("Maximum wheel-speed difference [1/s]:")
    print(
        np.max(
            np.abs(
                w_RW_amigo - w_RW_source
            )
        )
    )

    print()
    print("Maximum power difference [W]:")
    print(
        np.max(
            np.abs(
                P_RW_amigo - P_RW_source
            )
        )
    )

    print()
    print("Python RK4 vs CADRE wheel-speed difference [1/s]:")
    print(
        np.max(
            np.abs(
                w_RW_rk4 - w_RW_source
            )
        )
    )

    print()
    print("Python RK4 vs CADRE power difference [W]:")
    print(
        np.max(
            np.abs(
                P_RW_rk4 - P_RW_source
            )
        )
    )

    print()
    print("AMIGO wheel-speed min / max [1/s]:")
    print(
        np.min(w_RW_amigo),
        np.max(w_RW_amigo),
    )

    print()
    print("AMIGO power min / max [W]:")
    print(
        np.min(P_RW_amigo),
        np.max(P_RW_amigo),
    )


if __name__ == "__main__":
    main()