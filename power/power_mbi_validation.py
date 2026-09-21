import pickle
import numpy as np

from power.mbi_modern import ModernMBI


def load_power_data():
    dat = np.loadtxt("../../../CADRE/src/CADRE/data/Power/curve.dat")

    nT = int(dat[0])
    nA = int(dat[1])
    nI = int(dat[2])

    index = 3

    T = dat[index:index + nT]
    index += nT

    A = dat[index:index + nA]
    index += nA

    I = dat[index:index + nI]
    index += nI

    V = dat[index:].reshape((nT, nA, nI), order="F")

    return T, A, I, V


def power_cell_voltage(LOS, temperature, exposedArea, Isetpt, mbi):
    num_nodes = LOS.size
    V_sol = np.zeros((12, num_nodes))

    for p in range(12):
        temp_index = 4 if p < 4 else p % 4

        print("Evaluating panel", p + 1, "of 12")

        for c in range(7):
            effective_area = LOS * exposedArea[c, p, :]
            points = np.column_stack((temperature[temp_index, :], effective_area, Isetpt[p, :]))
            V_sol[p, :] += mbi.evaluate(points)

    return V_sol


def solar_power(V_sol, Isetpt):
    return np.sum(V_sol * Isetpt, axis=0)


def main():
    print()
    print("=" * 70)
    print("CADRE MODERN MBI POWER VALIDATION")
    print("=" * 70)

    with open("../../../CADRE/src/CADRE/test/data1346.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    LOS = data["0:LOS"]
    exposedArea = data["0:exposedArea"]
    temperature = data["0:temperature"]
    Isetpt = data["0:Isetpt"]
    V_sol_source = data["0:V_sol"]
    P_sol_source = data["0:P_sol"]

    T, A, I, V = load_power_data()

    print()
    print("Building modern MBI...")

    mbi = ModernMBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

    print()
    print("Evaluating CADRE power inputs...")

    V_sol_test = power_cell_voltage(LOS, temperature, exposedArea, Isetpt, mbi)
    P_sol_test = solar_power(V_sol_test, Isetpt)

    voltage_difference = np.max(np.abs(V_sol_test - V_sol_source))
    voltage_mean_difference = np.mean(np.abs(V_sol_test - V_sol_source))
    power_difference = np.max(np.abs(P_sol_test - P_sol_source))
    power_mean_difference = np.mean(np.abs(P_sol_test - P_sol_source))

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print()
    print("Source minimum panel voltage [V]:")
    print(np.min(V_sol_source))

    print()
    print("Modern MBI minimum panel voltage [V]:")
    print(np.min(V_sol_test))

    print()
    print("Source maximum panel voltage [V]:")
    print(np.max(V_sol_source))

    print()
    print("Modern MBI maximum panel voltage [V]:")
    print(np.max(V_sol_test))

    print()
    print("Maximum panel-voltage difference [V]:")
    print(voltage_difference)

    print()
    print("Mean panel-voltage difference [V]:")
    print(voltage_mean_difference)

    print()
    print("Source minimum solar power [W]:")
    print(np.min(P_sol_source))

    print()
    print("Modern MBI minimum solar power [W]:")
    print(np.min(P_sol_test))

    print()
    print("Source maximum solar power [W]:")
    print(np.max(P_sol_source))

    print()
    print("Modern MBI maximum solar power [W]:")
    print(np.max(P_sol_test))

    print()
    print("Maximum solar-power difference [W]:")
    print(power_difference)

    print()
    print("Mean solar-power difference [W]:")
    print(power_mean_difference)


if __name__ == "__main__":
    main()