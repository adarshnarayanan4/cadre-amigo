import pickle

import numpy as np

from cadre_paths import cadre_path
from battery.battery_reference import battery_current, propagate_soc


def main():
    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    iSOC = data["0:iSOC"][0]
    P_bat = data["0:P_bat"]
    temperature = data["0:temperature"]

    SOC_source = data["0:SOC"][0, :]
    I_bat_source = data["0:I_bat"]

    n = len(P_bat)
    h = 43200.0 / (n - 1)

    SOC_test = propagate_soc(iSOC, P_bat, temperature, h)

    # Validate BatteryPower separately using CADRE's exact SOC
    I_bat_same_soc = battery_current(P_bat, SOC_source, temperature[4, :])

    # Full reproduced battery chain
    I_bat_test = battery_current(P_bat, SOC_test, temperature[4, :])

    print()
    print("=" * 70)
    print("CADRE BATTERY SOURCE VALIDATION")
    print("=" * 70)

    print()
    print("Time step [s]:")
    print(h)

    print()
    print("Source final SOC:")
    print(SOC_source[-1])

    print()
    print("Reference final SOC:")
    print(SOC_test[-1])

    print()
    print("Maximum SOC difference:")
    print(np.max(np.abs(SOC_test - SOC_source)))

    print()
    print("Mean SOC difference:")
    print(np.mean(np.abs(SOC_test - SOC_source)))

    print()
    print("BatteryPower validation using same SOC:")

    print()
    print("Maximum current difference [A]:")
    print(np.max(np.abs(I_bat_same_soc - I_bat_source)))

    print()
    print("Mean current difference [A]:")
    print(np.mean(np.abs(I_bat_same_soc - I_bat_source)))

    print()
    print("Full reference current validation:")

    print()
    print("Maximum current difference [A]:")
    print(np.max(np.abs(I_bat_test - I_bat_source)))

    print()
    print("Mean current difference [A]:")
    print(np.mean(np.abs(I_bat_test - I_bat_source)))

    print()
    print("Source SOC min / max:")
    print(np.min(SOC_source), np.max(SOC_source))

    print()
    print("Reference SOC min / max:")
    print(np.min(SOC_test), np.max(SOC_test))

    print()
    print("Source current min / max [A]:")
    print(np.min(I_bat_source), np.max(I_bat_source))

    print()
    print("Reference current min / max [A]:")
    print(np.min(I_bat_test), np.max(I_bat_test))


if __name__ == "__main__":
    main()
