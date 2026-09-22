import pickle
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cadre_paths import cadre_path


def load_power_data():
    path = cadre_path("data/Power/curve.dat", "../../../CADRE")
    dat = np.loadtxt(path)

    nT = int(dat[0])
    nA = int(dat[1])
    nI = int(dat[2])

    index = 3
    T = dat[index : index + nT]
    index += nT
    A = dat[index : index + nA]
    index += nA
    current_grid = dat[index : index + nI]
    index += nI
    V = dat[index:].reshape((nT, nA, nI), order="F")

    return T, A, current_grid, V


def build_power_interpolator(T, A, I, V):  # noqa: E741
    return RegularGridInterpolator(
        (T, A, I),
        V,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )


def power_cell_voltage(LOS, temperature, exposedArea, Isetpt, interpolator):
    num_nodes = LOS.size
    V_sol = np.zeros((12, num_nodes))

    for p in range(12):
        temp_index = 4 if p < 4 else p % 4

        for c in range(7):
            effective_area = LOS * exposedArea[c, p, :]
            points = np.column_stack(
                (temperature[temp_index, :], effective_area, Isetpt[p, :])
            )
            V_sol[p, :] += interpolator(points)

    return V_sol


def solar_power(V_sol, Isetpt):
    return np.sum(V_sol * Isetpt, axis=0)


def main():
    with open(cadre_path("test/data1346.pkl", "../../../CADRE"), "rb") as f:
        data = pickle.load(f, encoding="latin1")

    LOS = data["0:LOS"]
    exposedArea = data["0:exposedArea"]
    temperature = data["0:temperature"]
    Isetpt = data["0:Isetpt"]
    V_sol_source = data["0:V_sol"]
    P_sol_source = data["0:P_sol"]

    print()
    print("=" * 70)
    print("CADRE POWER SOURCE VALIDATION")
    print("=" * 70)

    print()
    print("LOS shape:")
    print(LOS.shape)

    print()
    print("Exposed-area shape:")
    print(exposedArea.shape)

    print()
    print("Temperature shape:")
    print(temperature.shape)

    print()
    print("Isetpt shape:")
    print(Isetpt.shape)

    print()
    print("Source V_sol shape:")
    print(V_sol_source.shape)

    print()
    print("Source P_sol shape:")
    print(P_sol_source.shape)

    T_grid, A_grid, I_grid, V_data = load_power_data()
    interpolator = build_power_interpolator(T_grid, A_grid, I_grid, V_data)

    V_sol_test = power_cell_voltage(
        LOS, temperature, exposedArea, Isetpt, interpolator
    )
    P_sol_test = solar_power(V_sol_test, Isetpt)

    voltage_difference = np.max(np.abs(V_sol_test - V_sol_source))
    power_difference = np.max(np.abs(P_sol_test - P_sol_source))

    voltage_mean_difference = np.mean(np.abs(V_sol_test - V_sol_source))
    power_mean_difference = np.mean(np.abs(P_sol_test - P_sol_source))

    print()
    print("Source minimum panel voltage [V]:")
    print(np.min(V_sol_source))

    print()
    print("Test minimum panel voltage [V]:")
    print(np.min(V_sol_test))

    print()
    print("Source maximum panel voltage [V]:")
    print(np.max(V_sol_source))

    print()
    print("Test maximum panel voltage [V]:")
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
    print("Test minimum solar power [W]:")
    print(np.min(P_sol_test))

    print()
    print("Source maximum solar power [W]:")
    print(np.max(P_sol_source))

    print()
    print("Test maximum solar power [W]:")
    print(np.max(P_sol_test))

    print()
    print("Maximum solar-power difference [W]:")
    print(power_difference)

    print()
    print("Mean solar-power difference [W]:")
    print(power_mean_difference)


if __name__ == "__main__":
    main()
