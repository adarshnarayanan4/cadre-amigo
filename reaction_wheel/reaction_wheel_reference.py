import pickle

import numpy as np

from cadre_paths import cadre_path


J_RW = 2.8e-5

V = 4.0
a = 4.9e-4
b = 4.5e2
I0 = 0.017


def reaction_wheel_torque(T_tot):
    return T_tot.copy()


def reaction_wheel_dynamics(state, w_B, T_RW):
    w_Bx = np.zeros((3, 3))

    w_Bx[0, :] = [0.0, -w_B[2], w_B[1]]
    w_Bx[1, :] = [w_B[2], 0.0, -w_B[0]]
    w_Bx[2, :] = [-w_B[1], w_B[0], 0.0]

    return -T_RW / J_RW - w_Bx @ state


def rk4_step(state, w_B, T_RW, h):
    a1 = reaction_wheel_dynamics(
        state,
        w_B,
        T_RW,
    )

    b1 = reaction_wheel_dynamics(
        state + h / 2.0 * a1,
        w_B,
        T_RW,
    )

    c1 = reaction_wheel_dynamics(
        state + h / 2.0 * b1,
        w_B,
        T_RW,
    )

    d1 = reaction_wheel_dynamics(
        state + h * c1,
        w_B,
        T_RW,
    )

    return state + h / 6.0 * (
        a1 + 2.0 * b1 + 2.0 * c1 + d1
    )


def propagate_reaction_wheel(w_RW0, w_B, T_RW, h):
    n = w_B.shape[1]

    w_RW = np.zeros((3, n))
    w_RW[:, 0] = w_RW0

    for k in range(n - 1):
        w_RW[:, k + 1] = rk4_step(
            w_RW[:, k],
            w_B[:, k],
            T_RW[:, k],
            h,
        )

    return w_RW


def propagate_reaction_wheel_trapezoid(w_RW0, w_B, T_RW, h):
    n = w_B.shape[1]

    w_RW = np.zeros((3, n))
    w_RW_dot = np.zeros((3, n))

    w_RW[:, 0] = w_RW0

    for k in range(n - 1):
        state0 = w_RW[:, k]

        f0 = reaction_wheel_dynamics(
            state0,
            w_B[:, k],
            T_RW[:, k],
        )

        state1 = state0 + h * f0

        for _ in range(100):
            f1 = reaction_wheel_dynamics(
                state1,
                w_B[:, k + 1],
                T_RW[:, k + 1],
            )

            state1_new = state0 + 0.5 * h * (
                f0 + f1
            )

            if np.max(np.abs(state1_new - state1)) < 1e-14:
                state1 = state1_new
                break

            state1 = state1_new

        w_RW[:, k + 1] = state1

    for k in range(n):
        w_RW_dot[:, k] = reaction_wheel_dynamics(
            w_RW[:, k],
            w_B[:, k],
            T_RW[:, k],
        )

    return w_RW, w_RW_dot


def reaction_wheel_power(w_RW, T_RW):
    return V * (a * w_RW + b * T_RW) ** 2 + V * I0


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
    h = 43200.0 / (n - 1)

    # Reaction-wheel torque
    T_RW = reaction_wheel_torque(T_tot)

    # Original CADRE-style RK4
    w_RW = propagate_reaction_wheel(
        w_RW0,
        w_B,
        T_RW,
        h,
    )

    P_RW = reaction_wheel_power(
        w_RW,
        T_RW,
    )

    # Trapezoidal reference for AMIGO
    w_RW_trap, w_RW_dot_trap = propagate_reaction_wheel_trapezoid(
        w_RW0,
        w_B,
        T_RW,
        h,
    )

    P_RW_trap = reaction_wheel_power(
        w_RW_trap,
        T_RW,
    )

    # Trapezoid checks
    max_dynamics_residual = 0.0
    max_trapezoid_residual = 0.0

    for k in range(n):
        rate = reaction_wheel_dynamics(
            w_RW_trap[:, k],
            w_B[:, k],
            T_RW[:, k],
        )

        max_dynamics_residual = max(
            max_dynamics_residual,
            np.max(
                np.abs(
                    rate - w_RW_dot_trap[:, k]
                )
            ),
        )

    for k in range(n - 1):
        residual = (
            w_RW_trap[:, k + 1]
            - w_RW_trap[:, k]
            - 0.5
            * h
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
    print("=" * 70)
    print("CADRE REACTION WHEEL REFERENCE")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(h)

    print()
    print("=" * 70)
    print("SOURCE VALIDATION")
    print("=" * 70)

    print()
    print("Maximum torque difference [N*m]:")
    print(np.max(np.abs(T_RW - T_RW_source)))

    print()
    print("Maximum wheel-speed difference [1/s]:")
    print(np.max(np.abs(w_RW - w_RW_source)))

    print()
    print("Mean wheel-speed difference [1/s]:")
    print(np.mean(np.abs(w_RW - w_RW_source)))

    print()
    print("Maximum power difference [W]:")
    print(np.max(np.abs(P_RW - P_RW_source)))

    print()
    print("Mean power difference [W]:")
    print(np.mean(np.abs(P_RW - P_RW_source)))

    print()
    print("=" * 70)
    print("TRAPEZOID PRECHECK")
    print("=" * 70)

    print()
    print("Python dynamics residual:")
    print(max_dynamics_residual)

    print()
    print("Python trapezoid residual:")
    print(max_trapezoid_residual)

    print()
    print("RK4 vs trapezoid maximum wheel-speed difference [1/s]:")
    print(np.max(np.abs(w_RW - w_RW_trap)))

    print()
    print("RK4 vs trapezoid mean wheel-speed difference [1/s]:")
    print(np.mean(np.abs(w_RW - w_RW_trap)))

    print()
    print("RK4 vs trapezoid maximum power difference [W]:")
    print(np.max(np.abs(P_RW - P_RW_trap)))

    print()
    print("Source wheel-speed min / max [1/s]:")
    print(np.min(w_RW_source), np.max(w_RW_source))

    print()
    print("Trapezoid wheel-speed min / max [1/s]:")
    print(np.min(w_RW_trap), np.max(w_RW_trap))

    print()
    print("Source power min / max [W]:")
    print(np.min(P_RW_source), np.max(P_RW_source))

    print()
    print("Trapezoid power min / max [W]:")
    print(np.min(P_RW_trap), np.max(P_RW_trap))


if __name__ == "__main__":
    main()