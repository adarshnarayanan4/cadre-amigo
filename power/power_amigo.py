import amigo as am
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


def total_power(P_sol, P_comm, P_RW):
    return P_sol - 5.0 * P_comm - np.sum(P_RW, axis=0) - 2.0


class PowerSolar(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("V_sol", shape=(12,))
        self.add_input("Isetpt", shape=(12,))
        self.add_input("P_sol")
        self.add_constraint("res")

    def compute(self):
        V_sol = self.inputs["V_sol"]
        Isetpt = self.inputs["Isetpt"]
        P_sol = self.inputs["P_sol"]

        power = 0.0

        for p in range(12):
            power += V_sol[p] * Isetpt[p]

        self.constraints["res"] = P_sol - power


class PowerTotal(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("P_sol")
        self.add_input("P_comm")
        self.add_input("P_RW", shape=(3,))
        self.add_input("P_bat")
        self.add_constraint("res")

    def compute(self):
        P_sol = self.inputs["P_sol"]
        P_comm = self.inputs["P_comm"]
        P_RW = self.inputs["P_RW"]
        P_bat = self.inputs["P_bat"]

        self.constraints["res"] = P_bat - (P_sol - 5.0 * P_comm - P_RW[0] - P_RW[1] - P_RW[2] - 2.0)


def extract_scalar(x, variable_name):
    return np.array(x[f"{variable_name}[:]"])


def main():
    print()
    print("=" * 70)
    print("CADRE POWER AMIGO")
    print("=" * 70)

    # Load original CADRE benchmark
    with open("../../../CADRE/src/CADRE/test/data1346.pkl", "rb") as f:
        data = pickle.load(f, encoding="latin1")

    LOS = data["0:LOS"]
    exposedArea = data["0:exposedArea"]
    temperature = data["0:temperature"]
    Isetpt = data["0:Isetpt"]
    V_sol_source = data["0:V_sol"]
    P_sol_source = data["0:P_sol"]
    P_comm = data["0:P_comm"]
    P_RW = data["0:P_RW"]
    P_bat_source = data["0:P_bat"]

    num_nodes = LOS.size

    print()
    print("Number of nodes:")
    print(num_nodes)

    # Build modern MBI
    T, A, I, V = load_power_data()

    print()
    print("Building modern MBI...")

    mbi = ModernMBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

    print()
    print("Calculating panel voltage with modern MBI...")

    V_sol_mbi = power_cell_voltage(LOS, temperature, exposedArea, Isetpt, mbi)

    # Python reference using modern MBI voltage
    P_sol_mbi = solar_power(V_sol_mbi, Isetpt)
    P_bat_mbi = total_power(P_sol_mbi, P_comm, P_RW)

    print()
    print("=" * 70)
    print("PYTHON PRECHECK")
    print("=" * 70)

    print()
    print("Modern MBI vs CADRE maximum voltage difference [V]:")
    print(np.max(np.abs(V_sol_mbi - V_sol_source)))

    print()
    print("Modern MBI vs CADRE mean voltage difference [V]:")
    print(np.mean(np.abs(V_sol_mbi - V_sol_source)))

    print()
    print("Modern MBI solar power vs CADRE maximum difference [W]:")
    print(np.max(np.abs(P_sol_mbi - P_sol_source)))

    print()
    print("Modern MBI battery power vs CADRE maximum difference [W]:")
    print(np.max(np.abs(P_bat_mbi - P_bat_source)))

    # AMIGO components
    solar = PowerSolar()
    total = PowerTotal()

    # AMIGO model
    model = am.Model("cadre_power")

    model.add_component("solar", num_nodes, solar)
    model.add_component("total", num_nodes, total)

    # Link solar power into total power
    model.link("solar.P_sol[:]", "total.P_sol[:]")

    # Fix V_sol
    for p in range(12):
        model.set_meta("value", f"solar.V_sol[:, {p}]", V_sol_mbi[p, :])
        model.set_meta("lower", f"solar.V_sol[:, {p}]", V_sol_mbi[p, :])
        model.set_meta("upper", f"solar.V_sol[:, {p}]", V_sol_mbi[p, :])

    # Fix Isetpt
    for p in range(12):
        model.set_meta("value", f"solar.Isetpt[:, {p}]", Isetpt[p, :])
        model.set_meta("lower", f"solar.Isetpt[:, {p}]", Isetpt[p, :])
        model.set_meta("upper", f"solar.Isetpt[:, {p}]", Isetpt[p, :])

    # Initial solar-power guess
    model.set_meta("value", "solar.P_sol[:]", P_sol_mbi)

    # Fix communication power
    model.set_meta("value", "total.P_comm[:]", P_comm)
    model.set_meta("lower", "total.P_comm[:]", P_comm)
    model.set_meta("upper", "total.P_comm[:]", P_comm)

    # Fix reaction-wheel power
    for i in range(3):
        model.set_meta("value", f"total.P_RW[:, {i}]", P_RW[i, :])
        model.set_meta("lower", f"total.P_RW[:, {i}]", P_RW[i, :])
        model.set_meta("upper", f"total.P_RW[:, {i}]", P_RW[i, :])

    # Initial battery-power guess
    model.set_meta("value", "total.P_bat[:]", P_bat_mbi)

    print()
    print("Building AMIGO power model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

    x = model.create_vector()
    opt = am.Optimizer(model, x)

    opt_options = {"max_iterations": 100, "convergence_tolerance": 1e-10, "initial_barrier_param": 0.1, "barrier_strategy": "heuristic", "max_line_search_iterations": 4}

    print()
    print("Solving AMIGO power model...")

    opt.optimize(opt_options)

    print()
    print("AMIGO solve complete.")

    # Extract AMIGO results
    P_sol_amigo = extract_scalar(x, "solar.P_sol")
    P_bat_amigo = extract_scalar(x, "total.P_bat")

    # Validation
    amigo_python_solar_difference = np.max(np.abs(P_sol_amigo - P_sol_mbi))
    amigo_python_battery_difference = np.max(np.abs(P_bat_amigo - P_bat_mbi))
    amigo_cadre_solar_difference = np.max(np.abs(P_sol_amigo - P_sol_source))
    amigo_cadre_battery_difference = np.max(np.abs(P_bat_amigo - P_bat_source))

    # Residual checks
    solar_residual = np.max(np.abs(P_sol_amigo - np.sum(V_sol_mbi * Isetpt, axis=0)))
    battery_residual = np.max(np.abs(P_bat_amigo - (P_sol_amigo - 5.0 * P_comm - np.sum(P_RW, axis=0) - 2.0)))

    print()
    print("=" * 70)
    print("POWER AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("AMIGO vs Python solar-power difference [W]:")
    print(amigo_python_solar_difference)

    print()
    print("AMIGO vs Python battery-power difference [W]:")
    print(amigo_python_battery_difference)

    print()
    print("AMIGO solar-power residual:")
    print(solar_residual)

    print()
    print("AMIGO battery-power residual:")
    print(battery_residual)

    print()
    print("AMIGO vs original CADRE solar-power difference [W]:")
    print(amigo_cadre_solar_difference)

    print()
    print("AMIGO vs original CADRE battery-power difference [W]:")
    print(amigo_cadre_battery_difference)

    print()
    print("CADRE minimum solar power [W]:")
    print(np.min(P_sol_source))

    print()
    print("AMIGO minimum solar power [W]:")
    print(np.min(P_sol_amigo))

    print()
    print("CADRE maximum solar power [W]:")
    print(np.max(P_sol_source))

    print()
    print("AMIGO maximum solar power [W]:")
    print(np.max(P_sol_amigo))

    print()
    print("CADRE minimum battery power [W]:")
    print(np.min(P_bat_source))

    print()
    print("AMIGO minimum battery power [W]:")
    print(np.min(P_bat_amigo))

    print()
    print("CADRE maximum battery power [W]:")
    print(np.max(P_bat_source))

    print()
    print("AMIGO maximum battery power [W]:")
    print(np.max(P_bat_amigo))


if __name__ == "__main__":
    main()