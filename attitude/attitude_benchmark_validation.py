import pickle

import numpy as np

from cadre_paths import cadre_path
from attitude.attitude_reference import (
    attitude_from_orbit,
    attitude_roll,
    combine_rotation_matrices,
    rotation_matrix_rates,
    angular_velocity,
    angular_acceleration,
    attitude_torque,
)


def get_case_value(data, *names):
    """Load a variable from CADRE case 0."""

    for name in names:
        case_key = f"0:{name}"

        if case_key in data:
            return np.asarray(
                data[case_key],
                dtype=float,
            )

        if name in data:
            return np.asarray(
                data[name],
                dtype=float,
            )

    raise KeyError(
        f"Could not find any of these variables: {names}"
    )


def matrix_history(array):
    """
    Convert CADRE matrix history from
    (3, 3, n) to (n, 3, 3).
    """

    array = np.asarray(
        array,
        dtype=float,
    )

    return np.moveaxis(
        array,
        -1,
        0,
    )


def vector_history(array):
    """
    Convert CADRE vector history from
    (3, n) to (n, 3).
    """

    array = np.asarray(
        array,
        dtype=float,
    )

    return array.T


def report_difference(
    name,
    reference,
    source,
):
    difference = np.abs(
        reference - source
    )

    print()
    print(
        f"Maximum {name} difference:"
    )
    print(
        np.max(difference)
    )

    print(
        f"Mean {name} difference:"
    )
    print(
        np.mean(difference)
    )


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(
        data_path,
        "rb",
    ) as file:
        data = pickle.load(
            file,
            encoding="latin1",
        )

    # --------------------------------------------------------
    # CADRE benchmark inputs
    # --------------------------------------------------------

    orbit_source = get_case_value(
        data,
        "r_e2b_I",
    ).T

    gamma_source = get_case_value(
        data,
        "gamma",
        "Gamma",
    ).reshape(-1)

    n = orbit_source.shape[0]

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    print()
    print("=" * 70)
    print(
        "CADRE ATTITUDE BENCHMARK VALIDATION"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Gamma minimum [rad]:")
    print(
        np.min(gamma_source)
    )

    print()
    print("Gamma maximum [rad]:")
    print(
        np.max(gamma_source)
    )

    # --------------------------------------------------------
    # Our recreated attitude calculations
    # --------------------------------------------------------

    O_RI_reference = attitude_from_orbit(
        orbit_source
    )

    O_BR_reference = attitude_roll(
        gamma_source
    )

    O_BI_reference = (
        combine_rotation_matrices(
            O_BR_reference,
            O_RI_reference,
        )
    )

    Odot_BI_reference = (
        rotation_matrix_rates(
            O_BI_reference,
            dt,
        )
    )

    w_B_reference = angular_velocity(
        O_BI_reference,
        Odot_BI_reference,
    )

    wdot_B_reference = (
        angular_acceleration(
            w_B_reference,
            dt,
        )
    )

    T_tot_reference = attitude_torque(
        w_B_reference,
        wdot_B_reference,
    )

    # --------------------------------------------------------
    # Original CADRE outputs
    # --------------------------------------------------------

    O_RI_source = matrix_history(
        get_case_value(
            data,
            "O_RI",
        )
    )

    O_BR_source = matrix_history(
        get_case_value(
            data,
            "O_BR",
        )
    )

    O_BI_source = matrix_history(
        get_case_value(
            data,
            "O_BI",
        )
    )

    Odot_BI_source = matrix_history(
        get_case_value(
            data,
            "Odot_BI",
        )
    )

    w_B_source = vector_history(
        get_case_value(
            data,
            "w_B",
        )
    )

    wdot_B_source = vector_history(
        get_case_value(
            data,
            "wdot_B",
        )
    )

    T_tot_source = vector_history(
        get_case_value(
            data,
            "T_tot",
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PYTHON ATTITUDE VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "O_RI",
        O_RI_reference,
        O_RI_source,
    )

    report_difference(
        "O_BR",
        O_BR_reference,
        O_BR_source,
    )

    report_difference(
        "O_BI",
        O_BI_reference,
        O_BI_source,
    )

    report_difference(
        "Odot_BI",
        Odot_BI_reference,
        Odot_BI_source,
    )

    report_difference(
        "angular velocity",
        w_B_reference,
        w_B_source,
    )

    report_difference(
        "angular acceleration",
        wdot_B_reference,
        wdot_B_source,
    )

    report_difference(
        "torque",
        T_tot_reference,
        T_tot_source,
    )

    # --------------------------------------------------------
    # Rotation-matrix sanity check
    # --------------------------------------------------------

    maximum_orthogonality_error = 0.0

    for i in range(n):
        error = np.max(
            np.abs(
                O_BI_reference[i]
                @ O_BI_reference[i].T
                - np.eye(3)
            )
        )

        maximum_orthogonality_error = max(
            maximum_orthogonality_error,
            error,
        )

    print()
    print("=" * 70)
    print(
        "ATTITUDE SANITY CHECKS"
    )
    print("=" * 70)

    print()
    print(
        "Maximum rotation-matrix "
        "orthogonality error:"
    )
    print(
        maximum_orthogonality_error
    )

    print()
    print(
        "Maximum angular velocity magnitude [1/s]:"
    )
    print(
        np.max(
            np.linalg.norm(
                w_B_reference,
                axis=1,
            )
        )
    )

    print()
    print(
        "Maximum angular acceleration magnitude [1/s^2]:"
    )
    print(
        np.max(
            np.linalg.norm(
                wdot_B_reference,
                axis=1,
            )
        )
    )

    print()
    print(
        "Maximum torque magnitude [N*m]:"
    )
    print(
        np.max(
            np.linalg.norm(
                T_tot_reference,
                axis=1,
            )
        )
    )


if __name__ == "__main__":
    main()