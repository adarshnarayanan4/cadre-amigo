import pickle
import numpy as np

from cadre_paths import cadre_path


def power_total(P_sol, P_comm, P_RW):
    return P_sol - 5.0 * P_comm - np.sum(P_RW, axis=0) - 2.0


def main():
    with open(cadre_path("test/data1346.pkl", "../../../CADRE"), "rb") as f:
        data = pickle.load(f, encoding="latin1")

    P_sol = data["0:P_sol"]
    P_comm = data["0:P_comm"]
    P_RW = data["0:P_RW"]
    P_bat_source = data["0:P_bat"]

    P_bat_test = power_total(P_sol, P_comm, P_RW)

    max_difference = np.max(np.abs(P_bat_test - P_bat_source))
    mean_difference = np.mean(np.abs(P_bat_test - P_bat_source))

    print()
    print("=" * 70)
    print("CADRE POWER_TOTAL VALIDATION")
    print("=" * 70)

    print()
    print("P_sol shape:")
    print(P_sol.shape)

    print()
    print("P_comm shape:")
    print(P_comm.shape)

    print()
    print("P_RW shape:")
    print(P_RW.shape)

    print()
    print("P_bat shape:")
    print(P_bat_source.shape)

    print()
    print("Source minimum battery power [W]:")
    print(np.min(P_bat_source))

    print()
    print("Calculated minimum battery power [W]:")
    print(np.min(P_bat_test))

    print()
    print("Source maximum battery power [W]:")
    print(np.max(P_bat_source))

    print()
    print("Calculated maximum battery power [W]:")
    print(np.max(P_bat_test))

    print()
    print("Maximum battery-power difference [W]:")
    print(max_difference)

    print()
    print("Mean battery-power difference [W]:")
    print(mean_difference)


if __name__ == "__main__":
    main()
