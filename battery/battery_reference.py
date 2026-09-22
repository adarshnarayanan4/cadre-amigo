import pickle

import numpy as np

from cadre_paths import cadre_path


sigma = 1e-10
eta = 0.99
Cp = 2900.0 * 0.001 * 3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1 / 1.1**5)


def battery_voltage(SOC, temperature):
    voc = 3.0 + np.expm1(SOC) / (np.e - 1.0)
    return IR * voc * (2.0 - np.exp(alpha * (temperature - T0) / T0))


def battery_current(P_bat, SOC, temperature):
    return P_bat / battery_voltage(SOC, temperature)


def soc_rate(SOC, P_bat, temperature):
    current = battery_current(P_bat, SOC, temperature)
    return -sigma / 24.0 * SOC + eta / Cp * current


def rk4_step(SOC, P_bat, temperature, h):
    a = soc_rate(SOC, P_bat, temperature)
    b = soc_rate(SOC + h / 2.0 * a, P_bat, temperature)
    c = soc_rate(SOC + h / 2.0 * b, P_bat, temperature)
    d = soc_rate(SOC + h * c, P_bat, temperature)
    return SOC + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)


def propagate_soc(iSOC, P_bat, temperature, h):
    n = len(P_bat)
    SOC = np.zeros(n)
    SOC[0] = iSOC

    for k in range(n - 1):
        SOC[k + 1] = rk4_step(SOC[k], P_bat[k], temperature[4, k], h)

    return SOC


def propagate_soc_trapezoid(iSOC, P_bat, temperature, h):
    n = len(P_bat)
    SOC = np.zeros(n)
    SOC[0] = iSOC

    for k in range(n - 1):
        f0 = soc_rate(SOC[k], P_bat[k], temperature[4, k])
        SOC_next = SOC[k] + h * f0

        for _ in range(100):
            f1 = soc_rate(SOC_next, P_bat[k + 1], temperature[4, k + 1])
            new_SOC = SOC[k] + 0.5 * h * (f0 + f1)

            if abs(new_SOC - SOC_next) < 1e-14:
                SOC_next = new_SOC
                break

            SOC_next = new_SOC

        SOC[k + 1] = SOC_next

    return SOC


def main():
    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    iSOC = data["0:iSOC"][0]
    P_bat = data["0:P_bat"]
    temperature = data["0:temperature"]

    n = len(P_bat)
    h = (43200.0 - 0.0) / (n - 1)

    SOC = propagate_soc(iSOC, P_bat, temperature, h)
    SOC_trap = propagate_soc_trapezoid(iSOC, P_bat, temperature, h)
    I_bat = battery_current(P_bat, SOC, temperature[4, :])

    print()
    print("=" * 70)
    print("CADRE BATTERY REFERENCE")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(h)

    print()
    print("Initial SOC:")
    print(SOC[0])

    print()
    print("Final SOC:")
    print(SOC[-1])

    print()
    print("Minimum SOC:")
    print(np.min(SOC))

    print()
    print("Maximum SOC:")
    print(np.max(SOC))

    print()
    print("Minimum battery current [A]:")
    print(np.min(I_bat))

    print()
    print("Maximum battery current [A]:")
    print(np.max(I_bat))

    print()
    print("RK4 vs trapezoid maximum SOC difference:")
    print(np.max(np.abs(SOC - SOC_trap)))

    print()
    print("Trapezoid final SOC:")
    print(SOC_trap[-1])

    print()
    print("Trapezoid minimum SOC:")
    print(np.min(SOC_trap))

    print()
    print("Trapezoid maximum SOC:")
    print(np.max(SOC_trap))


if __name__ == "__main__":
    main()
