import pickle

import numpy as np

from cadre_paths import cadre_path
from power.mbi_modern import (
    paramuni,
    build_1d_matrix,
)


def build_bspline_matrix(n, m):
    k = 4

    parameter_values = paramuni(
        k,
        m,
        n,
    )

    B = build_1d_matrix(
        parameter_values,
        k,
        m,
    )

    return B.toarray()


def bspline_parameters(
    CP_P_comm,
    CP_gamma,
    CP_Isetpt,
    n,
):
    m = len(CP_P_comm)

    B = build_bspline_matrix(
        n,
        m,
    )

    P_comm = B @ CP_P_comm
    gamma = B @ CP_gamma

    Isetpt = np.zeros((12, n))

    for i in range(12):
        Isetpt[i, :] = B @ CP_Isetpt[i, :]

    return P_comm, gamma, Isetpt


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(data_path, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1",
        )

    CP_P_comm = data["0:CP_P_comm"]
    CP_gamma = data["0:CP_gamma"]
    CP_Isetpt = data["0:CP_Isetpt"]

    P_comm_source = data["0:P_comm"]
    gamma_source = data["0:gamma"]
    Isetpt_source = data["0:Isetpt"]

    n = len(P_comm_source)
    m = len(CP_P_comm)

    P_comm, gamma, Isetpt = bspline_parameters(
        CP_P_comm,
        CP_gamma,
        CP_Isetpt,
        n,
    )

    print()
    print("=" * 70)
    print("CADRE BSPLINE PARAMETERS REFERENCE")
    print("=" * 70)

    print()
    print("Number of time nodes:")
    print(n)

    print()
    print("Number of control points:")
    print(m)

    print()
    print("B-spline matrix shape:")
    print((n, m))

    print()
    print("=" * 70)
    print("COMMUNICATION POWER")
    print("=" * 70)

    print()
    print("Maximum P_comm difference [W]:")
    print(
        np.max(
            np.abs(
                P_comm - P_comm_source
            )
        )
    )

    print()
    print("Mean P_comm difference [W]:")
    print(
        np.mean(
            np.abs(
                P_comm - P_comm_source
            )
        )
    )

    print()
    print("=" * 70)
    print("ROLL ANGLE")
    print("=" * 70)

    print()
    print("Maximum gamma difference [rad]:")
    print(
        np.max(
            np.abs(
                gamma - gamma_source
            )
        )
    )

    print()
    print("Mean gamma difference [rad]:")
    print(
        np.mean(
            np.abs(
                gamma - gamma_source
            )
        )
    )

    print()
    print("=" * 70)
    print("SOLAR PANEL CURRENT")
    print("=" * 70)

    print()
    print("Maximum Isetpt difference [A]:")
    print(
        np.max(
            np.abs(
                Isetpt - Isetpt_source
            )
        )
    )

    print()
    print("Mean Isetpt difference [A]:")
    print(
        np.mean(
            np.abs(
                Isetpt - Isetpt_source
            )
        )
    )

    print()
    print("Source P_comm min / max [W]:")
    print(
        np.min(P_comm_source),
        np.max(P_comm_source),
    )

    print()
    print("Reference P_comm min / max [W]:")
    print(
        np.min(P_comm),
        np.max(P_comm),
    )

    print()
    print("Source gamma min / max [rad]:")
    print(
        np.min(gamma_source),
        np.max(gamma_source),
    )

    print()
    print("Reference gamma min / max [rad]:")
    print(
        np.min(gamma),
        np.max(gamma),
    )


if __name__ == "__main__":
    main()