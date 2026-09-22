import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path
from battery.battery_reference import (
    battery_current,
    propagate_soc,
    propagate_soc_trapezoid,
    soc_rate,
)


sigma = 1e-10
eta = 0.99
Cp = 2900.0 * 0.001 * 3600.0
IR = 0.9
T0 = 293.0
alpha = np.log(1 / 1.1**5)


class BatteryDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant("sigma", value=sigma)
        self.add_constant("eta", value=eta)
        self.add_constant("Cp", value=Cp)
        self.add_constant("IR", value=IR)
        self.add_constant("T0", value=T0)
        self.add_constant("alpha", value=alpha)
        self.add_constant("e_minus_1", value=np.e - 1.0)

        # Fixed inputs from the other CADRE subsystems
        self.add_data("P_bat")
        self.add_data("temperature")

        # State and state rate
        self.add_input("SOC")
        self.add_input("SOCdot")

        # Battery dynamics residual
        self.add_constraint("res")

    def compute(self):
        sigma = self.constants["sigma"]
        eta = self.constants["eta"]
        Cp = self.constants["Cp"]
        IR = self.constants["IR"]
        T0 = self.constants["T0"]
        alpha = self.constants["alpha"]
        e_minus_1 = self.constants["e_minus_1"]

        P = self.data["P_bat"]
        T = self.data["temperature"]

        SOC = self.inputs["SOC"]
        SOCdot = self.inputs["SOCdot"]

        voc = 3.0 + (am.exp(SOC) - 1.0) / e_minus_1
        V = IR * voc * (2.0 - am.exp(alpha * (T - T0) / T0))
        current = P / V

        rate = -sigma / 24.0 * SOC + eta / Cp * current

        self.constraints["res"] = rate - SOCdot


class TrapezoidRule(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant("dt", value=dt)

        self.add_input("SOC1")
        self.add_input("SOC2")

        self.add_input("SOCdot1")
        self.add_input("SOCdot2")

        self.add_constraint("res")

    def compute(self):
        dt = self.constants["dt"]

        SOC1 = self.inputs["SOC1"]
        SOC2 = self.inputs["SOC2"]

        SOCdot1 = self.inputs["SOCdot1"]
        SOCdot2 = self.inputs["SOCdot2"]

        self.constraints["res"] = SOC2 - SOC1 - 0.5 * dt * (SOCdot1 + SOCdot2)


class InitialSOC(am.Component):
    def __init__(self, initial_soc):
        super().__init__()

        self.add_constant("initial_soc", value=float(initial_soc))

        self.add_input("SOC")

        self.add_constraint("res")

        # Small objective so this can be treated as a feasibility solve
        self.add_objective("obj")

    def compute(self):
        SOC = self.inputs["SOC"]
        initial_soc = self.constants["initial_soc"]

        self.constraints["res"] = SOC - initial_soc
        self.objective["obj"] = 1.0e-12 * SOC * SOC


def calculate_rates(SOC, P_bat, temperature):
    n = len(SOC)
    rates = np.zeros(n)

    for k in range(n):
        rates[k] = soc_rate(
            SOC[k],
            P_bat[k],
            temperature[4, k],
        )

    return rates


def build_model(
    initial_soc,
    P_bat,
    temperature,
    dt,
    SOC_guess,
    SOCdot_guess,
):
    n = len(P_bat)
    num_intervals = n - 1

    battery = BatteryDynamics()
    trap = TrapezoidRule(dt)
    ic = InitialSOC(initial_soc)

    model = am.Model("cadre_battery")

    model.add_component("battery", n, battery)
    model.add_component("trap", num_intervals, trap)
    model.add_component("ic", 1, ic)

    # Connect battery states to trapezoid intervals
    model.link(
        f"battery.SOC[:{num_intervals}]",
        "trap.SOC1",
    )

    model.link(
        "battery.SOC[1:]",
        "trap.SOC2",
    )

    model.link(
        f"battery.SOCdot[:{num_intervals}]",
        "trap.SOCdot1",
    )

    model.link(
        "battery.SOCdot[1:]",
        "trap.SOCdot2",
    )

    # Initial condition
    model.link(
        "battery.SOC[0]",
        "ic.SOC[0]",
    )

    # P_bat and body temperature are fixed data
    model.set_data(
        "battery.P_bat",
        P_bat,
    )

    model.set_data(
        "battery.temperature",
        temperature[4, :],
    )

    # Start AMIGO from the Python trapezoidal solution
    model.set_meta(
        "value",
        "battery.SOC",
        SOC_guess,
    )

    model.set_meta(
        "value",
        "battery.SOCdot",
        SOCdot_guess,
    )

    print()
    print("Building AMIGO battery model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

    return model


def solve_model(model):
    x = model.create_vector()

    opt = am.Optimizer(model, x)

    options = {
        "max_iterations": 200,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print("Solving AMIGO battery model...")

    opt.optimize(options)

    print()
    print("AMIGO solve complete.")

    return x


def main():
    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    initial_soc = data["0:iSOC"][0]

    P_bat = data["0:P_bat"]
    temperature = data["0:temperature"]

    SOC_source = data["0:SOC"][0, :]
    I_source = data["0:I_bat"]

    n = len(P_bat)

    dt = 43200.0 / (n - 1)

    # Original CADRE-style RK4 reference
    SOC_rk4 = propagate_soc(
        initial_soc,
        P_bat,
        temperature,
        dt,
    )

    # Python trapezoidal reference
    SOC_trap = propagate_soc_trapezoid(
        initial_soc,
        P_bat,
        temperature,
        dt,
    )

    SOCdot_trap = calculate_rates(
        SOC_trap,
        P_bat,
        temperature,
    )

    print()
    print("=" * 70)
    print("CADRE BATTERY AMIGO")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Number of intervals:")
    print(n - 1)

    print()
    print("Time step [s]:")
    print(dt)

    model = build_model(
        initial_soc,
        P_bat,
        temperature,
        dt,
        SOC_trap,
        SOCdot_trap,
    )

    x = solve_model(model)

    SOC_amigo = np.array(x["battery.SOC"])
    SOCdot_amigo = np.array(x["battery.SOCdot"])

    I_amigo = battery_current(
        P_bat,
        SOC_amigo,
        temperature[4, :],
    )

    I_trap = battery_current(
        P_bat,
        SOC_trap,
        temperature[4, :],
    )

    # Check AMIGO battery dynamics residual
    expected_rates = calculate_rates(
        SOC_amigo,
        P_bat,
        temperature,
    )

    dynamics_residual = expected_rates - SOCdot_amigo

    # Check AMIGO trapezoid residual
    trapezoid_residual = (
        SOC_amigo[1:]
        - SOC_amigo[:-1]
        - 0.5 * dt * (SOCdot_amigo[:-1] + SOCdot_amigo[1:])
    )

    print()
    print("=" * 70)
    print("BATTERY AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("Python trapezoid final SOC:")
    print(SOC_trap[-1])

    print()
    print("AMIGO final SOC:")
    print(SOC_amigo[-1])

    print()
    print("Maximum AMIGO vs Python trapezoid SOC difference:")
    print(np.max(np.abs(SOC_amigo - SOC_trap)))

    print()
    print("Mean AMIGO vs Python trapezoid SOC difference:")
    print(np.mean(np.abs(SOC_amigo - SOC_trap)))

    print()
    print("Maximum AMIGO battery dynamics residual:")
    print(np.max(np.abs(dynamics_residual)))

    print()
    print("Maximum AMIGO trapezoid residual:")
    print(np.max(np.abs(trapezoid_residual)))

    print()
    print("AMIGO SOC min / max:")
    print(np.min(SOC_amigo), np.max(SOC_amigo))

    print()
    print("AMIGO current min / max [A]:")
    print(np.min(I_amigo), np.max(I_amigo))

    print()
    print("=" * 70)
    print("AMIGO VS ORIGINAL CADRE RK4")
    print("=" * 70)

    print()
    print("Maximum SOC difference:")
    print(np.max(np.abs(SOC_amigo - SOC_source)))

    print()
    print("Mean SOC difference:")
    print(np.mean(np.abs(SOC_amigo - SOC_source)))

    print()
    print("Final SOC difference:")
    print(abs(SOC_amigo[-1] - SOC_source[-1]))

    print()
    print("Maximum current difference [A]:")
    print(np.max(np.abs(I_amigo - I_source)))

    print()
    print("Mean current difference [A]:")
    print(np.mean(np.abs(I_amigo - I_source)))

    print()
    print("=" * 70)
    print("RK4 REPRODUCTION CHECK")
    print("=" * 70)

    print()
    print("Maximum Python RK4 vs CADRE SOC difference:")
    print(np.max(np.abs(SOC_rk4 - SOC_source)))

    print()
    print("Maximum Python trapezoid vs CADRE SOC difference:")
    print(np.max(np.abs(SOC_trap - SOC_source)))

    print()
    print("Maximum AMIGO vs Python trapezoid current difference [A]:")
    print(np.max(np.abs(I_amigo - I_trap)))


if __name__ == "__main__":
    main()
