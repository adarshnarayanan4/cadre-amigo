"""Replay one complete CADRE case through the integrated AMIGO model."""

import pickle

import amigo as am
import numpy as np

from battery.battery_amigo import TrapezoidRule as BatteryTrapezoidRule
from battery.battery_reference import battery_current, propagate_soc_trapezoid
from cadre_paths import cadre_path
from communication.communication_amigo import (
    CommunicationBitRate,
    DataDynamics,
    TrapezoidRule as DataTrapezoidRule,
    propagate_data_trapezoid,
)
from communication.communication_data_validation import communication_bit_rate
from integrated.reporting.console import print_heading, print_values
from integrated.model.components import DataInitialCondition
from integrated.model.components import ReactionWheelInitialCondition
from integrated.reference.case import calculate_battery_rates
from integrated.reference.case import extract_vector
from parameters.parameters_amigo import (
    ParameterControlPoints,
    ParameterSplineKernel,
    ParameterValues,
    build_sparse_basis,
)
from parameters.parameters_reference import bspline_parameters
from power.power_amigo import PowerSolar, PowerTotal, solar_power, total_power
from reaction_wheel.reaction_wheel_amigo import (
    ReactionWheelDynamics,
    ReactionWheelPower,
    ReactionWheelTorque,
    TrapezoidRule as ReactionWheelTrapezoidRule,
)
from reaction_wheel.reaction_wheel_reference import (
    propagate_reaction_wheel_trapezoid,
    reaction_wheel_power,
    reaction_wheel_torque,
)
from thermal.thermal_amigo import (
    InitialConditions as ThermalInitialConditions,
    ThermalDynamics,
    TrapezoidRule as ThermalTrapezoidRule,
    propagate_temperature_trapezoid,
)


class BatteryDynamicsLinked(am.Component):
    """
    Same CADRE battery dynamics as battery_amigo.py,
    but P_bat and temperature are AMIGO inputs rather
    than fixed data so other subsystems can link to them.
    """

    def __init__(self):
        super().__init__()

        self.add_constant("sigma", value=1.0e-10)
        self.add_constant("eta", value=0.99)
        self.add_constant("Cp", value=2900.0 * 0.001 * 3600.0)
        self.add_constant("IR", value=0.9)
        self.add_constant("T0", value=293.0)
        self.add_constant("alpha", value=np.log(1.0 / 1.1**5))
        self.add_constant("e_minus_1", value=np.e - 1.0)

        self.add_input("P_bat")
        self.add_input("temperature")
        self.add_input("SOC")
        self.add_input("SOCdot")

        self.add_constraint("res")

    def compute(self):
        sigma = self.constants["sigma"]
        eta = self.constants["eta"]
        Cp = self.constants["Cp"]
        IR = self.constants["IR"]
        T0 = self.constants["T0"]
        alpha = self.constants["alpha"]
        e_minus_1 = self.constants["e_minus_1"]

        P_bat = self.inputs["P_bat"]
        temperature = self.inputs["temperature"]

        SOC = self.inputs["SOC"]
        SOCdot = self.inputs["SOCdot"]

        voc = 3.0 + (am.exp(SOC) - 1.0) / e_minus_1

        voltage = IR * voc * (2.0 - am.exp(alpha * (temperature - T0) / T0))

        current = P_bat / voltage

        rate = -sigma / 24.0 * SOC + eta / Cp * current
        self.constraints["res"] = rate - SOCdot


class BatteryInitialCondition(am.Component):
    def __init__(self, initial_soc):
        super().__init__()

        self.add_constant("initial_soc", value=float(initial_soc))

        self.add_input("SOC")

        self.add_constraint("res")

    def compute(self):
        SOC = self.inputs["SOC"]

        self.constraints["res"] = SOC - self.constants["initial_soc"]


def max_difference(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b)))


def main():

    # Load original CADRE case

    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    P_comm_source = data["0:P_comm"]

    n = len(P_comm_source)
    num_intervals = n - 1

    dt = 43200.0 / num_intervals

    print_heading("CADRE SINGLE-CASE INTEGRATED AMIGO")

    print_values(
        [
            ("Time nodes", f"{n:,}"),
            ("Time intervals", f"{num_intervals:,}"),
            ("Time step", f"{dt:.4f} s"),
        ]
    )

    # CADRE saved inputs

    # Parameter control points
    CP_P_comm = data["0:CP_P_comm"]

    CP_gamma = data["0:CP_gamma"]

    CP_Isetpt = data["0:CP_Isetpt"]

    m = len(CP_P_comm)

    # Thermal fixed lookup inputs
    exposed_area = data["0:exposedArea"]

    LOS = data["0:LOS"]

    cellInstd = data["cellInstd"]

    # Attitude outputs used by
    # reaction wheels for this stage
    w_B = data["0:w_B"]

    T_tot = data["0:T_tot"]

    # Power-cell voltage lookup
    # remains fixed for this stage
    V_sol = data["0:V_sol"]

    # Communication geometry/gain
    # remain fixed for this stage
    gain = data["0:gain"]

    GSdist = data["0:GSdist"]

    CommLOS = data["0:CommLOS"]

    initial_soc = float(data["0:iSOC"][0])

    # Sequential Python integrated reference

    print_heading("BUILDING PYTHON INTEGRATED REFERENCE")

    # Parameters

    P_comm_reference, gamma_reference, Isetpt_reference = bspline_parameters(
        CP_P_comm, CP_gamma, CP_Isetpt, n
    )

    # Thermal

    T0 = 273.0 * np.ones(5)

    temperature_reference, temperature_dot_reference = propagate_temperature_trapezoid(
        T0, exposed_area, LOS, P_comm_reference, cellInstd, dt
    )

    # Reaction wheels

    T_RW_reference = reaction_wheel_torque(T_tot)

    w_RW0 = np.zeros(3)

    w_RW_reference, w_RW_dot_reference = propagate_reaction_wheel_trapezoid(
        w_RW0, w_B, T_RW_reference, dt
    )

    P_RW_reference = reaction_wheel_power(w_RW_reference, T_RW_reference)

    # Power

    P_sol_reference = solar_power(V_sol, Isetpt_reference)

    P_bat_reference = total_power(P_sol_reference, P_comm_reference, P_RW_reference)

    # Battery

    SOC_reference = propagate_soc_trapezoid(
        initial_soc, P_bat_reference, temperature_reference, dt
    )

    SOCdot_reference = calculate_battery_rates(
        SOC_reference, P_bat_reference, temperature_reference[4]
    )

    I_bat_reference = battery_current(
        P_bat_reference, SOC_reference, temperature_reference[4]
    )

    # Communication

    Dr_reference = communication_bit_rate(P_comm_reference, gain, GSdist, CommLOS)

    Data_reference, Data_dot_reference = propagate_data_trapezoid(0.0, Dr_reference, dt)

    print_values(
        [
            ("Python reference", "Ready"),
            ("Final SOC", f"{SOC_reference[-1]:.6f}"),
            ("Downloaded data [CADRE units]", f"{Data_reference[-1]:,.6f}"),
        ]
    )

    # Build sparse parameter interpolation

    weights, indices = build_sparse_basis(n, m)

    # AMIGO components

    parameter_control_points = ParameterControlPoints()

    parameter_values = ParameterValues()

    parameter_kernel = ParameterSplineKernel()

    thermal = ThermalDynamics()

    thermal_trapezoid = ThermalTrapezoidRule(dt)
    thermal_trapezoid.name = "ThermalTrapezoidRule"

    thermal_initial = ThermalInitialConditions()

    rw_torque = ReactionWheelTorque()

    rw_dynamics = ReactionWheelDynamics()

    rw_trapezoid = ReactionWheelTrapezoidRule(dt)
    rw_trapezoid.name = "ReactionWheelTrapezoidRule"

    rw_initial = ReactionWheelInitialCondition()

    rw_power = ReactionWheelPower()

    power_solar = PowerSolar()
    power_total = PowerTotal()

    battery = BatteryDynamicsLinked()

    battery_trapezoid = BatteryTrapezoidRule(dt)
    battery_trapezoid.name = "BatteryTrapezoidRule"

    battery_initial = BatteryInitialCondition(initial_soc)

    comm_rate = CommunicationBitRate()

    comm_data = DataDynamics()

    comm_trapezoid = DataTrapezoidRule(dt)
    comm_trapezoid.name = "CommunicationTrapezoidRule"

    comm_initial = DataInitialCondition()

    # AMIGO model

    model = am.Model("cadre_single_case")

    # Parameters
    model.add_component("control_points", m, parameter_control_points)
    model.add_component("values", n, parameter_values)
    model.add_component("parameter_kernel", n, parameter_kernel)

    # Thermal
    model.add_component("thermal", n, thermal)
    model.add_component("thermal_trap", num_intervals, thermal_trapezoid)
    model.add_component("thermal_ic", 1, thermal_initial)

    # Reaction wheels
    model.add_component("rw_torque", n, rw_torque)
    model.add_component("rw_dynamics", n, rw_dynamics)
    model.add_component("rw_trap", num_intervals, rw_trapezoid)
    model.add_component("rw_ic", 1, rw_initial)
    model.add_component("rw_power", n, rw_power)

    # Power
    model.add_component("power_solar", n, power_solar)
    model.add_component("power_total", n, power_total)

    # Battery
    model.add_component("battery", n, battery)
    model.add_component("battery_trap", num_intervals, battery_trapezoid)
    model.add_component("battery_ic", 1, battery_initial)

    # Communication
    model.add_component("comm_rate", n, comm_rate)
    model.add_component("comm_data", n, comm_data)
    model.add_component("comm_trap", num_intervals, comm_trapezoid)
    model.add_component("comm_ic", 1, comm_initial)

    # Parameter links

    print("\n  Creating parameter links...\n", flush=True)

    for i in range(n):
        for j in range(4):
            control_index = indices[i, j]
            model.link(
                ("control_points." f"CP_P_comm[{control_index}]"),
                ("parameter_kernel." f"CP_P_comm[{i}, {j}]"),
            )
            model.link(
                ("control_points." f"CP_gamma[{control_index}]"),
                ("parameter_kernel." f"CP_gamma[{i}, {j}]"),
            )

            for panel in range(12):
                local_index = 4 * panel + j
                model.link(
                    ("control_points." "CP_Isetpt" f"[{control_index}, " f"{panel}]"),
                    ("parameter_kernel." "CP_Isetpt" f"[{i}, " f"{local_index}]"),
                )

    model.link("parameter_kernel.P_comm", "values.P_comm")
    model.link("parameter_kernel.Gamma", "values.Gamma")
    model.link("parameter_kernel.Isetpt", "values.Isetpt")

    model.set_data("parameter_kernel.N", weights)

    # Cross-subsystem links

    # P_comm -> Thermal
    model.link("values.P_comm", "thermal.P_comm")

    # P_comm -> Power
    model.link("values.P_comm", "power_total.P_comm")

    # Isetpt -> solar power
    model.link("values.Isetpt", "power_solar.Isetpt")

    # Reaction-wheel power -> total power
    model.link("rw_power.P_RW", "power_total.P_RW")

    # Solar power -> total power
    model.link("power_solar.P_sol", "power_total.P_sol")

    # Total power -> battery
    model.link("power_total.P_bat", "battery.P_bat")

    # Thermal body temperature -> battery
    model.link("thermal.temperature[:, 4]", "battery.temperature")

    # P_comm -> communication
    model.link("values.P_comm", "comm_rate.P_comm")

    # Bit rate -> data dynamics
    model.link("comm_rate.Dr", "comm_data.Dr")

    # Thermal trajectory links

    model.link("thermal.temperature[:-1]", "thermal_trap.temperature0")
    model.link("thermal.temperature[1:]", "thermal_trap.temperature1")

    model.link("thermal.temperature_dot[:-1]", "thermal_trap.temperature_dot0")
    model.link("thermal.temperature_dot[1:]", "thermal_trap.temperature_dot1")

    model.link("thermal.temperature[0]", "thermal_ic.temperature[0]")

    # Reaction-wheel links

    model.link("rw_torque.T_RW", "rw_dynamics.T_RW")
    model.link("rw_torque.T_RW", "rw_power.T_RW")
    model.link("rw_dynamics.w_RW", "rw_power.w_RW")

    model.link("rw_dynamics.w_RW[:-1]", "rw_trap.w0")
    model.link("rw_dynamics.w_RW[1:]", "rw_trap.w1")

    model.link("rw_dynamics.w_RW_dot[:-1]", "rw_trap.wdot0")
    model.link("rw_dynamics.w_RW_dot[1:]", "rw_trap.wdot1")

    model.link("rw_dynamics.w_RW[0]", "rw_ic.w_RW[0]")

    # Battery trajectory links

    model.link("battery.SOC[:-1]", "battery_trap.SOC1")
    model.link("battery.SOC[1:]", "battery_trap.SOC2")

    model.link("battery.SOCdot[:-1]", "battery_trap.SOCdot1")
    model.link("battery.SOCdot[1:]", "battery_trap.SOCdot2")

    model.link("battery.SOC[0]", "battery_ic.SOC[0]")

    # Communication trajectory links

    model.link("comm_data.Data[:-1]", "comm_trap.Data0")
    model.link("comm_data.Data[1:]", "comm_trap.Data1")

    model.link("comm_data.Data_dot[:-1]", "comm_trap.Data_dot0")
    model.link("comm_data.Data_dot[1:]", "comm_trap.Data_dot1")

    model.link("comm_data.Data[0]", "comm_ic.Data[0]")

    # Fix CADRE control points

    model.set_meta("value", "control_points.CP_P_comm[:]", CP_P_comm)
    model.set_meta("lower", "control_points.CP_P_comm[:]", CP_P_comm)
    model.set_meta("upper", "control_points.CP_P_comm[:]", CP_P_comm)

    model.set_meta("value", "control_points.CP_gamma[:]", CP_gamma)
    model.set_meta("lower", "control_points.CP_gamma[:]", CP_gamma)
    model.set_meta("upper", "control_points.CP_gamma[:]", CP_gamma)

    for panel in range(12):
        values = CP_Isetpt[panel, :]

        variable = "control_points." f"CP_Isetpt[:, {panel}]"
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    # Parameter initial guesses
    model.set_meta("value", "values.P_comm[:]", P_comm_reference)
    model.set_meta("value", "values.Gamma[:]", gamma_reference)

    for panel in range(12):
        model.set_meta(
            "value", ("values." f"Isetpt[:, {panel}]"), Isetpt_reference[panel, :]
        )

    # Fix thermal lookup inputs

    exposed_area_flat = np.zeros((n, 84))

    for k in range(n):
        exposed_area_flat[k] = exposed_area[:, :, k].reshape(84)

    for i in range(84):
        variable = "thermal." f"exposedArea[:, {i}]"

        values = exposed_area_flat[:, i]
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    model.set_meta("value", "thermal.LOS[:]", LOS)
    model.set_meta("lower", "thermal.LOS[:]", LOS)
    model.set_meta("upper", "thermal.LOS[:]", LOS)

    cell_flat = cellInstd.reshape(84)

    for i in range(84):
        values = cell_flat[i] * np.ones(n)

        variable = "thermal." f"cellInstd[:, {i}]"
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    # Thermal initial guesses
    for i in range(5):
        model.set_meta(
            "value", ("thermal." f"temperature[:, {i}]"), temperature_reference[i, :]
        )
        model.set_meta(
            "value",
            ("thermal." f"temperature_dot[:, {i}]"),
            temperature_dot_reference[:, i],
        )

    # Fix attitude inputs to reaction wheels

    for i in range(3):
        model.set_meta("value", ("rw_torque." f"T_tot[:, {i}]"), T_tot[i])
        model.set_meta("lower", ("rw_torque." f"T_tot[:, {i}]"), T_tot[i])
        model.set_meta("upper", ("rw_torque." f"T_tot[:, {i}]"), T_tot[i])

        model.set_meta("value", ("rw_dynamics." f"w_B[:, {i}]"), w_B[i])
        model.set_meta("lower", ("rw_dynamics." f"w_B[:, {i}]"), w_B[i])
        model.set_meta("upper", ("rw_dynamics." f"w_B[:, {i}]"), w_B[i])

        model.set_meta("value", ("rw_torque." f"T_RW[:, {i}]"), T_RW_reference[i])
        model.set_meta("value", ("rw_dynamics." f"w_RW[:, {i}]"), w_RW_reference[i])
        model.set_meta(
            "value", ("rw_dynamics." f"w_RW_dot[:, {i}]"), w_RW_dot_reference[i]
        )

        model.set_meta("value", ("rw_power." f"P_RW[:, {i}]"), P_RW_reference[i])

    # Fix panel voltages

    for panel in range(12):
        variable = "power_solar." f"V_sol[:, {panel}]"

        values = V_sol[panel, :]
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    model.set_meta("value", "power_solar.P_sol[:]", P_sol_reference)
    model.set_meta("value", "power_total.P_bat[:]", P_bat_reference)

    # Battery initial guesses

    model.set_meta("value", "battery.SOC[:]", SOC_reference)
    model.set_meta("value", "battery.SOCdot[:]", SOCdot_reference)

    # Fix communication geometry/gain

    for name, values in [("gain", gain), ("GSdist", GSdist), ("CommLOS", CommLOS)]:
        variable = f"comm_rate.{name}[:]"
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    model.set_meta("value", "comm_rate.Dr[:]", Dr_reference)
    model.set_meta("value", "comm_data.Data[:]", Data_reference)
    model.set_meta("value", "comm_data.Data_dot[:]", Data_dot_reference)

    # Build and solve

    print("\n  Building integrated AMIGO model...\n", flush=True)

    model.build_module()

    print("\n  Build successful.\n", flush=True)

    model.initialize(order_type=am.OrderingType.NATURAL)

    print("\n  Initialization successful.\n", flush=True)

    x = model.create_vector()

    optimizer = am.Optimizer(model, x)

    options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print("\n  Solving integrated AMIGO model...\n", flush=True)

    optimizer.optimize(options)

    print("\n  Integrated AMIGO solve complete.\n", flush=True)

    # Extract results

    P_comm_amigo = np.array(x["values.P_comm[:]"])

    Isetpt_amigo = np.zeros((12, n))

    for panel in range(12):
        Isetpt_amigo[panel] = x[("values." f"Isetpt[:, {panel}]")]

    temperature_amigo = extract_vector(x, "thermal.temperature", n, 5).T

    w_RW_amigo = extract_vector(x, "rw_dynamics.w_RW", n, 3).T

    P_RW_amigo = extract_vector(x, "rw_power.P_RW", n, 3).T

    P_sol_amigo = np.array(x["power_solar.P_sol[:]"])

    P_bat_amigo = np.array(x["power_total.P_bat[:]"])

    SOC_amigo = np.array(x["battery.SOC[:]"])

    Dr_amigo = np.array(x["comm_rate.Dr[:]"])

    Data_amigo = np.array(x["comm_data.Data[:]"])

    # Integrated validation

    print_heading("INTEGRATED AMIGO VS SEQUENTIAL PYTHON")

    print_values(
        [
            (
                "Communication power [W]",
                f"{max_difference(P_comm_amigo, P_comm_reference):.3e}",
            ),
            (
                "Solar current [A]",
                f"{max_difference(Isetpt_amigo, Isetpt_reference):.3e}",
            ),
            (
                "Temperature [K]",
                f"{max_difference(temperature_amigo, temperature_reference):.3e}",
            ),
            (
                "Reaction-wheel speed [1/s]",
                f"{max_difference(w_RW_amigo, w_RW_reference):.3e}",
            ),
            (
                "Reaction-wheel power [W]",
                f"{max_difference(P_RW_amigo, P_RW_reference):.3e}",
            ),
            ("Solar power [W]", f"{max_difference(P_sol_amigo, P_sol_reference):.3e}"),
            (
                "Battery power [W]",
                f"{max_difference(P_bat_amigo, P_bat_reference):.3e}",
            ),
            ("Battery SOC", f"{max_difference(SOC_amigo, SOC_reference):.3e}"),
            (
                "Download rate [CADRE units/s]",
                f"{max_difference(Dr_amigo, Dr_reference):.3e}",
            ),
            (
                "Downloaded data [CADRE units]",
                f"{max_difference(Data_amigo, Data_reference):.3e}",
            ),
        ]
    )

    print_heading("INTEGRATED CASE SUMMARY")

    I_bat_amigo = battery_current(P_bat_amigo, SOC_amigo, temperature_amigo[4])

    print_values(
        [
            ("Final Python SOC", f"{SOC_reference[-1]:.6f}"),
            ("Final AMIGO SOC", f"{SOC_amigo[-1]:.6f}"),
            (
                "Python battery current [A]",
                f"{np.min(I_bat_reference):.6f} to {np.max(I_bat_reference):.6f}",
            ),
            (
                "AMIGO battery current [A]",
                f"{np.min(I_bat_amigo):.6f} to {np.max(I_bat_amigo):.6f}",
            ),
        ]
    )

    print("  Downloaded data [CADRE data units]\n")

    print_values(
        [
            ("Python reference", f"{Data_reference[-1]:,.6f}"),
            ("AMIGO", f"{Data_amigo[-1]:,.6f}"),
            ("Original CADRE", f"{data['0:Data'].reshape(-1)[-1]:,.6f}"),
        ]
    )


if __name__ == "__main__":
    main()
