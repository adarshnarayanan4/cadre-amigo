import pickle

import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_benchmark_validation import (
    get_case_value,
)

from battery.battery_reference import (
    battery_current,
    propagate_soc,
)


def ks_function(values, rho=50.0):
    """
    CADRE Kreisselmeier-Steinhauser aggregation.
    """

    maximum = np.max(values)

    exponents = np.exp(
        rho * (values - maximum)
    )

    return (
        maximum
        + np.log(np.sum(exponents)) / rho
    )


def battery_constraints(
    SOC,
    I_bat,
):
    rho = 50.0

    Imin = -10.0
    Imax = 5.0

    SOC0 = 0.2
    SOC1 = 1.0

    ConCh = ks_function(
        I_bat - Imax,
        rho,
    )

    ConDs = ks_function(
        Imin - I_bat,
        rho,
    )

    ConS0 = ks_function(
        SOC0 - SOC,
        rho,
    )

    ConS1 = ks_function(
        SOC - SOC1,
        rho,
    )

    return (
        ConCh,
        ConDs,
        ConS0,
        ConS1,
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
    # Load CADRE benchmark
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

    initial_soc = float(
        get_case_value(
            data,
            "iSOC",
        ).reshape(-1)[0]
    )

    P_bat = get_case_value(
        data,
        "P_bat",
    ).reshape(-1)

    temperature = get_case_value(
        data,
        "temperature",
    )

    n = len(P_bat)

    dt = 43200.0 / (n - 1)

    print()
    print("=" * 70)
    print(
        "CADRE BATTERY BENCHMARK VALIDATION"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Initial SOC:")
    print(initial_soc)

    print()
    print(
        "Battery power min / max [W]:"
    )
    print(
        np.min(P_bat),
        np.max(P_bat),
    )

    print()
    print(
        "Body temperature min / max [K]:"
    )
    print(
        np.min(temperature[4]),
        np.max(temperature[4]),
    )

    # --------------------------------------------------------
    # Python RK4 reproduction
    # --------------------------------------------------------

    SOC_reference = propagate_soc(
        initial_soc,
        P_bat,
        temperature,
        dt,
    )

    I_reference = battery_current(
        P_bat,
        SOC_reference,
        temperature[4],
    )

    (
        ConCh_reference,
        ConDs_reference,
        ConS0_reference,
        ConS1_reference,
    ) = battery_constraints(
        SOC_reference,
        I_reference,
    )

    # --------------------------------------------------------
    # Original CADRE outputs
    # --------------------------------------------------------

    SOC_source = get_case_value(
        data,
        "SOC",
    ).reshape(-1)

    I_source = get_case_value(
        data,
        "I_bat",
    ).reshape(-1)

    # --------------------------------------------------------
    # State and current validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "PYTHON RK4 BATTERY VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "SOC",
        SOC_reference,
        SOC_source,
    )

    report_difference(
        "battery current [A]",
        I_reference,
        I_source,
    )

    print()
    print(
        "Original CADRE final SOC:"
    )
    print(
        SOC_source[-1]
    )

    print()
    print(
        "Python RK4 final SOC:"
    )
    print(
        SOC_reference[-1]
    )

    print()
    print(
        "Final SOC difference:"
    )
    print(
        abs(
            SOC_reference[-1]
            - SOC_source[-1]
        )
    )

    # --------------------------------------------------------
    # Battery constraints
    # --------------------------------------------------------

    ConCh_source = float(
        get_case_value(
            data,
            "ConCh",
        ).reshape(-1)[0]
    )

    ConDs_source = float(
        get_case_value(
            data,
            "ConDs",
        ).reshape(-1)[0]
    )

    ConS0_source = float(
        get_case_value(
            data,
            "ConS0",
        ).reshape(-1)[0]
    )

    ConS1_source = float(
        get_case_value(
            data,
            "ConS1",
        ).reshape(-1)[0]
    )

    print()
    print("=" * 70)
    print(
        "BATTERY CONSTRAINT VALIDATION"
    )
    print("=" * 70)

    print()
    print("ConCh:")
    print(
        "CADRE =",
        ConCh_source,
    )
    print(
        "Python =",
        ConCh_reference,
    )
    print(
        "Difference =",
        abs(
            ConCh_reference
            - ConCh_source
        ),
    )

    print()
    print("ConDs:")
    print(
        "CADRE =",
        ConDs_source,
    )
    print(
        "Python =",
        ConDs_reference,
    )
    print(
        "Difference =",
        abs(
            ConDs_reference
            - ConDs_source
        ),
    )

    print()
    print("ConS0:")
    print(
        "CADRE =",
        ConS0_source,
    )
    print(
        "Python =",
        ConS0_reference,
    )
    print(
        "Difference =",
        abs(
            ConS0_reference
            - ConS0_source
        ),
    )

    print()
    print("ConS1:")
    print(
        "CADRE =",
        ConS1_source,
    )
    print(
        "Python =",
        ConS1_reference,
    )
    print(
        "Difference =",
        abs(
            ConS1_reference
            - ConS1_source
        ),
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BATTERY SANITY CHECKS"
    )
    print("=" * 70)

    print()
    print("SOC minimum / maximum:")
    print(
        np.min(SOC_reference),
        np.max(SOC_reference),
    )

    print()
    print(
        "Battery current minimum / maximum [A]:"
    )
    print(
        np.min(I_reference),
        np.max(I_reference),
    )


if __name__ == "__main__":
    main()