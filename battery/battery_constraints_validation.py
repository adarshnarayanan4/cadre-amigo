import pickle
from pathlib import Path

import numpy as np

from battery.battery_reference import (
    battery_current,
    propagate_soc,
    propagate_soc_trapezoid,
)


def ks_function(g, rho=50.0):
    g = np.asarray(g)

    g_max = np.max(g)
    g_diff = g - g_max
    exponents = np.exp(rho * g_diff)
    summation = np.sum(exponents)

    return g_max + 1.0 / rho * np.log(summation)


def battery_constraints(I_bat, SOC):
    rho = 50.0

    Imin = -10.0
    Imax = 5.0

    SOC0 = 0.2
    SOC1 = 1.0

    ConCh = ks_function(I_bat - Imax, rho)
    ConDs = ks_function(Imin - I_bat, rho)
    ConS0 = ks_function(SOC0 - SOC, rho)
    ConS1 = ks_function(SOC - SOC1, rho)

    return ConCh, ConDs, ConS0, ConS1


def print_constraints(name, values):
    ConCh, ConDs, ConS0, ConS1 = values

    print()
    print(name)

    print("ConCh:")
    print(ConCh)

    print("ConDs:")
    print(ConDs)

    print("ConS0:")
    print(ConS0)

    print("ConS1:")
    print(ConS1)


def main():
    root = Path(__file__).resolve().parents[1]
    cadre = root.parent / "CADRE"

    data_path = cadre / "src/CADRE/test/data1346.pkl"

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    initial_soc = data["0:iSOC"][0]
    P_bat = data["0:P_bat"]
    temperature = data["0:temperature"]

    SOC_source = data["0:SOC"][0, :]
    I_source = data["0:I_bat"]

    n = len(P_bat)
    dt = 43200.0 / (n - 1)

    # Recreated CADRE RK4 trajectory
    SOC_rk4 = propagate_soc(
        initial_soc,
        P_bat,
        temperature,
        dt,
    )

    I_rk4 = battery_current(
        P_bat,
        SOC_rk4,
        temperature[4, :],
    )

    # Trapezoidal trajectory used by AMIGO
    SOC_trap = propagate_soc_trapezoid(
        initial_soc,
        P_bat,
        temperature,
        dt,
    )

    I_trap = battery_current(
        P_bat,
        SOC_trap,
        temperature[4, :],
    )

    constraints_source = battery_constraints(
        I_source,
        SOC_source,
    )

    constraints_rk4 = battery_constraints(
        I_rk4,
        SOC_rk4,
    )

    constraints_trap = battery_constraints(
        I_trap,
        SOC_trap,
    )

    print()
    print("=" * 70)
    print("CADRE BATTERY CONSTRAINT VALIDATION")
    print("=" * 70)

    print_constraints(
        "Original CADRE trajectory:",
        constraints_source,
    )

    print_constraints(
        "Recreated Python RK4 trajectory:",
        constraints_rk4,
    )

    print_constraints(
        "Python trapezoid / AMIGO trajectory:",
        constraints_trap,
    )

    print()
    print("=" * 70)
    print("POINTWISE EXTREMES")
    print("=" * 70)

    print()
    print("CADRE minimum current [A]:")
    print(np.min(I_source))

    print()
    print("Trapezoid minimum current [A]:")
    print(np.min(I_trap))

    print()
    print("CADRE minimum SOC:")
    print(np.min(SOC_source))

    print()
    print("Trapezoid minimum SOC:")
    print(np.min(SOC_trap))

    print()
    print("=" * 70)
    print("SAVED CADRE CONSTRAINT VALUES")
    print("=" * 70)

    for name in ["ConCh", "ConDs", "ConS0", "ConS1"]:
        key = f"0:{name}"

        if key in data:
            print()
            print(key)
            print(data[key])
        else:
            print()
            print(key, "not found in pickle")


if __name__ == "__main__":
    main()