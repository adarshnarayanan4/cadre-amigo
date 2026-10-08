import pickle

import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_benchmark_validation import (
    get_case_value,
)

from thermal.thermal_reference import (
    propagate_temperature,
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
    # Original CADRE thermal inputs
    # --------------------------------------------------------

    exposed_area = get_case_value(
        data,
        "exposedArea",
    )

    LOS = get_case_value(
        data,
        "LOS",
    ).reshape(-1)

    P_comm = get_case_value(
        data,
        "P_comm",
    ).reshape(-1)

    # cellInstd is a shared design variable in CADRE.
    cellInstd = get_case_value(
        data,
        "cellInstd",
    )

    # CADRE ThermalTemperature defaults to 273 K
    # for all five thermal states.
    T0 = 273.0 * np.ones(5)

    n = LOS.size

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    print()
    print("=" * 70)
    print(
        "CADRE THERMAL BENCHMARK VALIDATION"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Exposed-area shape:")
    print(
        exposed_area.shape
    )

    print()
    print("cellInstd shape:")
    print(
        cellInstd.shape
    )

    print()
    print(
        "P_comm minimum / maximum [W]:"
    )
    print(
        np.min(P_comm),
        np.max(P_comm),
    )

    print()
    print(
        "LOS minimum / maximum:"
    )
    print(
        np.min(LOS),
        np.max(LOS),
    )

    # --------------------------------------------------------
    # Recreate original CADRE RK4 thermal trajectory
    # --------------------------------------------------------

    temperature_reference = (
        propagate_temperature(
            T0,
            exposed_area,
            LOS,
            P_comm,
            cellInstd,
            dt,
        )
    )

    # --------------------------------------------------------
    # Original CADRE output
    # --------------------------------------------------------

    temperature_source = (
        get_case_value(
            data,
            "temperature",
        )
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PYTHON RK4 THERMAL VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "temperature [K]",
        temperature_reference,
        temperature_source,
    )

    final_difference = np.abs(
        temperature_reference[:, -1]
        - temperature_source[:, -1]
    )

    print()
    print(
        "Maximum final temperature difference [K]:"
    )
    print(
        np.max(final_difference)
    )

    print()
    print(
        "Original CADRE final temperatures [K]:"
    )
    print(
        temperature_source[:, -1]
    )

    print()
    print(
        "Python RK4 final temperatures [K]:"
    )
    print(
        temperature_reference[:, -1]
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "THERMAL SANITY CHECKS"
    )
    print("=" * 70)

    print()
    print(
        "Original CADRE minimum temperature [K]:"
    )
    print(
        np.min(temperature_source)
    )

    print()
    print(
        "Original CADRE maximum temperature [K]:"
    )
    print(
        np.max(temperature_source)
    )

    print()
    print(
        "Python minimum temperature [K]:"
    )
    print(
        np.min(temperature_reference)
    )

    print()
    print(
        "Python maximum temperature [K]:"
    )
    print(
        np.max(temperature_reference)
    )

    print()
    print(
        "Minimum temperature by state [K]:"
    )
    print(
        np.min(
            temperature_reference,
            axis=1,
        )
    )

    print()
    print(
        "Maximum temperature by state [K]:"
    )
    print(
        np.max(
            temperature_reference,
            axis=1,
        )
    )


if __name__ == "__main__":
    main()