import pickle
import numpy as np


def solar_power(V_sol, Isetpt):
    return np.sum(V_sol * Isetpt, axis=0)


def main():
    with open("../../../CADRE/src/CADRE/test/data1346.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    V_sol = data["0:V_sol"]
    Isetpt = data["0:Isetpt"]
    P_sol_source = data["0:P_sol"]

    P_sol_test = solar_power(V_sol, Isetpt)

    max_difference = np.max(np.abs(P_sol_test - P_sol_source))
    mean_difference = np.mean(np.abs(P_sol_test - P_sol_source))

    print()
    print("=" * 70)
    print("CADRE POWER_SOLARPOWER VALIDATION")
    print("=" * 70)

    print()
    print("V_sol shape:")
    print(V_sol.shape)

    print()
    print("Isetpt shape:")
    print(Isetpt.shape)

    print()
    print("P_sol shape:")
    print(P_sol_source.shape)

    print()
    print("Source minimum solar power [W]:")
    print(np.min(P_sol_source))

    print()
    print("Calculated minimum solar power [W]:")
    print(np.min(P_sol_test))

    print()
    print("Source maximum solar power [W]:")
    print(np.max(P_sol_source))

    print()
    print("Calculated maximum solar power [W]:")
    print(np.max(P_sol_test))

    print()
    print("Maximum solar-power difference [W]:")
    print(max_difference)

    print()
    print("Mean solar-power difference [W]:")
    print(mean_difference)


if __name__ == "__main__":
    main()