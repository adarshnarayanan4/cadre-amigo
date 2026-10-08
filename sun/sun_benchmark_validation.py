import pickle

import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_benchmark_validation import (
    get_case_value,
    matrix_history,
    vector_history,
)

from sun.sun_reference import (
    sun_position_eci,
    sun_line_of_sight,
    sun_position_body,
    sun_position_spherical,
)


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
    # --------------------------------------------------------
    # Load original CADRE benchmark
    # --------------------------------------------------------

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
    # Original CADRE inputs
    # --------------------------------------------------------

    orbit_source = get_case_value(
        data,
        "r_e2b_I",
    ).T

    O_BI_source = matrix_history(
        get_case_value(
            data,
            "O_BI",
        )
    )

    LD = float(
        get_case_value(
            data,
            "LD",
        ).reshape(-1)[0]
    )

    n = orbit_source.shape[0]

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    times = np.linspace(
        t1,
        t2,
        n,
    )

    print()
    print("=" * 70)
    print(
        "CADRE SUN BENCHMARK VALIDATION"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Launch date parameter LD:")
    print(LD)

    # --------------------------------------------------------
    # Our recreated Sun calculations
    # --------------------------------------------------------

    r_e2s_I_reference = (
        sun_position_eci(
            times,
            LD=LD,
        )
    )

    LOS_reference = (
        sun_line_of_sight(
            orbit_source,
            r_e2s_I_reference,
        )
    )

    r_e2s_B_reference = (
        sun_position_body(
            O_BI_source,
            r_e2s_I_reference,
        )
    )

    (
        azimuth_reference,
        elevation_reference,
    ) = sun_position_spherical(
        r_e2s_B_reference
    )

    # --------------------------------------------------------
    # Original CADRE outputs
    # --------------------------------------------------------

    r_e2s_I_source = vector_history(
        get_case_value(
            data,
            "r_e2s_I",
        )
    )

    LOS_source = get_case_value(
        data,
        "LOS",
    ).reshape(-1)

    r_e2s_B_source = vector_history(
        get_case_value(
            data,
            "r_e2s_B",
        )
    )

    azimuth_source = get_case_value(
        data,
        "azimuth",
    ).reshape(-1)

    elevation_source = get_case_value(
        data,
        "elevation",
    ).reshape(-1)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PYTHON SUN VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "r_e2s_I",
        r_e2s_I_reference,
        r_e2s_I_source,
    )

    report_difference(
        "LOS",
        LOS_reference,
        LOS_source,
    )

    report_difference(
        "r_e2s_B",
        r_e2s_B_reference,
        r_e2s_B_source,
    )

    report_difference(
        "azimuth",
        azimuth_reference,
        azimuth_source,
    )

    report_difference(
        "elevation",
        elevation_reference,
        elevation_source,
    )

    # --------------------------------------------------------
    # LOS region counts
    # --------------------------------------------------------

    full_sunlight = np.sum(
        LOS_reference == 1.0
    )

    full_eclipse = np.sum(
        LOS_reference == 0.0
    )

    transition = (
        n
        - full_sunlight
        - full_eclipse
    )

    print()
    print("=" * 70)
    print(
        "SUN SANITY CHECKS"
    )
    print("=" * 70)

    print()
    print("Full sunlight nodes:")
    print(full_sunlight)

    print()
    print("Full eclipse nodes:")
    print(full_eclipse)

    print()
    print("Transition nodes:")
    print(transition)

    sun_norm = np.linalg.norm(
        r_e2s_I_reference,
        axis=1,
    )

    print()
    print(
        "Minimum Sun-vector magnitude:"
    )
    print(
        np.min(sun_norm)
    )

    print()
    print(
        "Maximum Sun-vector magnitude:"
    )
    print(
        np.max(sun_norm)
    )

    print()
    print("Azimuth range [rad]:")
    print(
        np.min(azimuth_reference),
        np.max(azimuth_reference),
    )

    print()
    print("Elevation range [rad]:")
    print(
        np.min(elevation_reference),
        np.max(elevation_reference),
    )


if __name__ == "__main__":
    main()