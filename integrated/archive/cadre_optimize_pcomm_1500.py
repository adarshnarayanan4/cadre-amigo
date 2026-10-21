import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path

from parameters.parameters_reference import (
    bspline_parameters,
)

from parameters.parameters_amigo import (
    ParameterControlPoints,
    ParameterValues,
    ParameterSplineKernel,
    build_sparse_basis,
)

from thermal.thermal_amigo import (
    ThermalDynamics,
    TrapezoidRule as ThermalTrapezoidRule,
    InitialConditions as ThermalInitialConditions,
    propagate_temperature_trapezoid,
)

from reaction_wheel.reaction_wheel_reference import (
    reaction_wheel_torque,
    reaction_wheel_power,
    propagate_reaction_wheel_trapezoid,
)

from reaction_wheel.reaction_wheel_amigo import (
    ReactionWheelTorque,
    ReactionWheelDynamics,
    TrapezoidRule as ReactionWheelTrapezoidRule,
    ReactionWheelPower,
)

from power.power_amigo import (
    PowerSolar,
    PowerTotal,
    solar_power,
    total_power,
)

from battery.battery_reference import (
    battery_current,
    propagate_soc_trapezoid,
    soc_rate,
)

from battery.battery_amigo import (
    TrapezoidRule as BatteryTrapezoidRule,
)

from communication.communication_data_validation import (
    communication_bit_rate,
)

from communication.communication_amigo import (
    CommunicationBitRate,
    DataDynamics,
    TrapezoidRule as DataTrapezoidRule,
    propagate_data_trapezoid,
)

# ================================================================
# Integrated-model components
# ================================================================


class ReactionWheelInitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input(
            "w_RW",
            shape=(3,),
        )

        self.add_constraint(
            "res",
            shape=(3,),
        )

    def compute(self):
        w_RW = self.inputs["w_RW"]

        self.constraints["res"] = [
            w_RW[0],
            w_RW[1],
            w_RW[2],
        ]


class BatteryDynamicsLinked(am.Component):
    """
    CADRE battery dynamics with P_bat and temperature
    supplied by the coupled AMIGO model.
    """

    def __init__(self):
        super().__init__()

        self.add_constant(
            "sigma",
            value=1.0e-10,
        )

        self.add_constant(
            "eta",
            value=0.99,
        )

        self.add_constant(
            "Cp",
            value=2900.0 * 0.001 * 3600.0,
        )

        self.add_constant(
            "IR",
            value=0.9,
        )

        self.add_constant(
            "T0",
            value=293.0,
        )

        self.add_constant(
            "alpha",
            value=np.log(1.0 / 1.1**5),
        )

        self.add_constant(
            "e_minus_1",
            value=np.e - 1.0,
        )

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

        self.add_constant(
            "initial_soc",
            value=float(initial_soc),
        )

        self.add_input("SOC")

        self.add_constraint("res")

    def compute(self):
        SOC = self.inputs["SOC"]

        self.constraints["res"] = SOC - self.constants["initial_soc"]


class DataInitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Data")

        self.add_constraint("res")

    def compute(self):
        self.constraints["res"] = self.inputs["Data"]


class DownloadObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Data")

        self.add_objective("obj")

    def compute(self):
        # AMIGO minimizes.
        # Therefore minimize -Data to maximize downloaded data.
        self.objective["obj"] = -self.inputs["Data"]


# ================================================================
# Helper functions
# ================================================================


def calculate_battery_rates(
    SOC,
    P_bat,
    temperature,
):
    n = len(SOC)

    rates = np.zeros(n)

    for k in range(n):
        rates[k] = soc_rate(
            SOC[k],
            P_bat[k],
            temperature[k],
        )

    return rates


def extract_vector(
    x,
    variable_name,
    n,
    width,
):
    values = np.zeros((n, width))

    for i in range(width):
        values[:, i] = x[f"{variable_name}[:, {i}]"]

    return values


# ================================================================
# Main
# ================================================================


def main():
    # --------------------------------------------------------
    # Load original CADRE benchmark case
    # --------------------------------------------------------

    data_path = cadre_path("test/data1346.pkl")

    with open(
        data_path,
        "rb",
    ) as file:
        data = pickle.load(
            file,
            encoding="latin1",
        )

        # ========================================================
    # Reduced optimization problem
    # ========================================================
    #
    # The final CADRE benchmark uses 1500 time nodes and
    # 300 control points.
    #
    # For the first running optimization, use:
    #
    #   300 time nodes
    #   60 B-spline control points
    #
    # The mission duration remains 12 hours.
    # ========================================================

    full_n = len(data["0:P_comm"])

    n = 300

    sample_indices = np.rint(
        np.linspace(
            0,
            full_n - 1,
            n,
        )
    ).astype(int)

    num_intervals = n - 1

    dt = 43200.0 / num_intervals

    print()
    print("=" * 70)
    print("CADRE REDUCED P_COMM OPTIMIZATION")
    print("=" * 70)

    print()
    print("Original CADRE nodes:")
    print(full_n)

    print()
    print("Optimization nodes:")
    print(n)

    print()
    print("Number of intervals:")
    print(num_intervals)

    print()
    print("Time step [s]:")
    print(dt)

    # ========================================================
    # Reduce the original 300 control points to 60
    # ========================================================

    CP_P_comm_full = np.asarray(data["0:CP_P_comm"])

    CP_gamma_full = np.asarray(data["0:CP_gamma"])

    CP_Isetpt_full = np.asarray(data["0:CP_Isetpt"])

    full_m = len(CP_P_comm_full)

    m = 60

    old_cp_location = np.linspace(
        0.0,
        1.0,
        full_m,
    )

    new_cp_location = np.linspace(
        0.0,
        1.0,
        m,
    )

    CP_P_comm = np.interp(
        new_cp_location,
        old_cp_location,
        CP_P_comm_full,
    )

    CP_gamma = np.interp(
        new_cp_location,
        old_cp_location,
        CP_gamma_full,
    )

    CP_Isetpt = np.zeros((12, m))

    for panel in range(12):
        CP_Isetpt[
            panel,
            :,
        ] = np.interp(
            new_cp_location,
            old_cp_location,
            CP_Isetpt_full[
                panel,
                :,
            ],
        )

    print()
    print("Communication-power control points:")
    print(m)

    # ========================================================
    # Downsample fixed CADRE trajectory inputs
    # ========================================================

    P_comm_source = np.asarray(data["0:P_comm"])[sample_indices]

    exposed_area = np.asarray(data["0:exposedArea"])[
        :,
        :,
        sample_indices,
    ]

    LOS = np.asarray(data["0:LOS"])[sample_indices]

    cellInstd = np.asarray(data["cellInstd"])

    w_B = np.asarray(data["0:w_B"])[
        :,
        sample_indices,
    ]

    T_tot = np.asarray(data["0:T_tot"])[
        :,
        sample_indices,
    ]

    V_sol = np.asarray(data["0:V_sol"])[
        :,
        sample_indices,
    ]

    gain = np.asarray(data["0:gain"])[sample_indices]

    GSdist = np.asarray(data["0:GSdist"])[sample_indices]

    CommLOS = np.asarray(data["0:CommLOS"])[sample_indices]

    initial_soc = float(data["0:iSOC"][0])

    # ========================================================
    # Original CADRE-design baseline
    # ========================================================

    (
        P_comm_baseline,
        gamma_baseline,
        Isetpt_baseline,
    ) = bspline_parameters(
        CP_P_comm,
        CP_gamma,
        CP_Isetpt,
        n,
    )

    T0 = 273.0 * np.ones(5)

    (
        temperature_baseline,
        temperature_dot_baseline,
    ) = propagate_temperature_trapezoid(
        T0,
        exposed_area,
        LOS,
        P_comm_baseline,
        cellInstd,
        dt,
    )

    T_RW_baseline = reaction_wheel_torque(T_tot)

    w_RW0 = np.zeros(3)

    (
        w_RW_baseline,
        w_RW_dot_baseline,
    ) = propagate_reaction_wheel_trapezoid(
        w_RW0,
        w_B,
        T_RW_baseline,
        dt,
    )

    P_RW_baseline = reaction_wheel_power(
        w_RW_baseline,
        T_RW_baseline,
    )

    P_sol_baseline = solar_power(
        V_sol,
        Isetpt_baseline,
    )

    P_bat_baseline = total_power(
        P_sol_baseline,
        P_comm_baseline,
        P_RW_baseline,
    )

    SOC_baseline = propagate_soc_trapezoid(
        initial_soc,
        P_bat_baseline,
        temperature_baseline,
        dt,
    )

    Dr_baseline = communication_bit_rate(
        P_comm_baseline,
        gain,
        GSdist,
        CommLOS,
    )

    (
        Data_baseline,
        Data_dot_baseline,
    ) = propagate_data_trapezoid(
        0.0,
        Dr_baseline,
        dt,
    )

    print()
    print("=" * 70)
    print("BASELINE")
    print("=" * 70)

    print()
    print("Baseline final SOC:")
    print(SOC_baseline[-1])

    print()
    print("Baseline SOC min / max:")
    print(
        np.min(SOC_baseline),
        np.max(SOC_baseline),
    )

    print()
    print("Baseline downloaded data [Gibyte]:")
    print(Data_baseline[-1])

    # ========================================================
    # Interior optimization starting point
    # ========================================================
    #
    # Some original CADRE CP_P_comm values are exactly zero.
    #
    # AMIGO uses an interior-point optimizer, so variables directly
    # on a bound are pushed slightly inside the bound automatically.
    #
    # If we used the original zero-valued CPs while keeping all the
    # state guesses from the original trajectory, that interior push
    # would immediately make the coupled equality constraints
    # inconsistent.
    #
    # Instead, create a slightly interior control-point guess first
    # and regenerate ALL coupled state guesses from that point.
    # ========================================================

    CP_P_comm_guess = np.clip(
        CP_P_comm,
        0.02,
        24.5,
    )

    (
        P_comm_guess,
        gamma_guess,
        Isetpt_guess,
    ) = bspline_parameters(
        CP_P_comm_guess,
        CP_gamma,
        CP_Isetpt,
        n,
    )

    # --------------------------------------------------------
    # Thermal initial guess
    # --------------------------------------------------------

    (
        temperature_guess,
        temperature_dot_guess,
    ) = propagate_temperature_trapezoid(
        T0,
        exposed_area,
        LOS,
        P_comm_guess,
        cellInstd,
        dt,
    )

    # --------------------------------------------------------
    # Reaction wheels
    #
    # These are unchanged because attitude is fixed during this
    # reduced P_comm optimization.
    # --------------------------------------------------------

    T_RW_guess = T_RW_baseline
    w_RW_guess = w_RW_baseline
    w_RW_dot_guess = w_RW_dot_baseline
    P_RW_guess = P_RW_baseline

    # --------------------------------------------------------
    # Power initial guess
    # --------------------------------------------------------

    P_sol_guess = solar_power(
        V_sol,
        Isetpt_guess,
    )

    P_bat_guess = total_power(
        P_sol_guess,
        P_comm_guess,
        P_RW_guess,
    )

    # --------------------------------------------------------
    # Battery initial guess
    # --------------------------------------------------------

    SOC_guess = propagate_soc_trapezoid(
        initial_soc,
        P_bat_guess,
        temperature_guess,
        dt,
    )

    SOCdot_guess = calculate_battery_rates(
        SOC_guess,
        P_bat_guess,
        temperature_guess[4],
    )

    # --------------------------------------------------------
    # Communications initial guess
    # --------------------------------------------------------

    Dr_guess = communication_bit_rate(
        P_comm_guess,
        gain,
        GSdist,
        CommLOS,
    )

    (
        Data_guess,
        Data_dot_guess,
    ) = propagate_data_trapezoid(
        0.0,
        Dr_guess,
        dt,
    )

    print()
    print("=" * 70)
    print("OPTIMIZATION INITIAL GUESS")
    print("=" * 70)

    print()
    print("Initial-guess final SOC:")
    print(SOC_guess[-1])

    print()
    print("Initial-guess SOC min / max:")
    print(
        np.min(SOC_guess),
        np.max(SOC_guess),
    )

    print()
    print("Initial-guess downloaded data [Gibyte]:")
    print(Data_guess[-1])

    # ========================================================
    # Sparse B-spline structure
    # ========================================================

    weights, indices = build_sparse_basis(
        n,
        m,
    )

    # ========================================================
    # AMIGO components
    # ========================================================

    parameter_control_points = ParameterControlPoints()

    parameter_values = ParameterValues()

    parameter_kernel = ParameterSplineKernel()

    thermal = ThermalDynamics()

    thermal_trapezoid = ThermalTrapezoidRule(dt)

    # Important:
    # Different modules all contain classes named TrapezoidRule.
    # Give each one a unique AMIGO/C++ class name.
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

    objective = DownloadObjective()

    # ========================================================
    # AMIGO model
    # ========================================================

    model = am.Model("cadre_optimize_pcomm")

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    model.add_component(
        "control_points",
        m,
        parameter_control_points,
    )

    model.add_component(
        "values",
        n,
        parameter_values,
    )

    model.add_component(
        "parameter_kernel",
        n,
        parameter_kernel,
    )

    # --------------------------------------------------------
    # Thermal
    # --------------------------------------------------------

    model.add_component(
        "thermal",
        n,
        thermal,
    )

    model.add_component(
        "thermal_trap",
        num_intervals,
        thermal_trapezoid,
    )

    model.add_component(
        "thermal_ic",
        1,
        thermal_initial,
    )

    # --------------------------------------------------------
    # Reaction wheels
    # --------------------------------------------------------

    model.add_component(
        "rw_torque",
        n,
        rw_torque,
    )

    model.add_component(
        "rw_dynamics",
        n,
        rw_dynamics,
    )

    model.add_component(
        "rw_trap",
        num_intervals,
        rw_trapezoid,
    )

    model.add_component(
        "rw_ic",
        1,
        rw_initial,
    )

    model.add_component(
        "rw_power",
        n,
        rw_power,
    )

    # --------------------------------------------------------
    # Power
    # --------------------------------------------------------

    model.add_component(
        "power_solar",
        n,
        power_solar,
    )

    model.add_component(
        "power_total",
        n,
        power_total,
    )

    # --------------------------------------------------------
    # Battery
    # --------------------------------------------------------

    model.add_component(
        "battery",
        n,
        battery,
    )

    model.add_component(
        "battery_trap",
        num_intervals,
        battery_trapezoid,
    )

    model.add_component(
        "battery_ic",
        1,
        battery_initial,
    )

    # --------------------------------------------------------
    # Communications
    # --------------------------------------------------------

    model.add_component(
        "comm_rate",
        n,
        comm_rate,
    )

    model.add_component(
        "comm_data",
        n,
        comm_data,
    )

    model.add_component(
        "comm_trap",
        num_intervals,
        comm_trapezoid,
    )

    model.add_component(
        "comm_ic",
        1,
        comm_initial,
    )

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    model.add_component(
        "objective",
        1,
        objective,
    )

    # ========================================================
    # B-spline parameter links
    # ========================================================

    print()
    print("Creating parameter links...")

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

    model.link(
        "parameter_kernel.P_comm",
        "values.P_comm",
    )

    model.link(
        "parameter_kernel.Gamma",
        "values.Gamma",
    )

    model.link(
        "parameter_kernel.Isetpt",
        "values.Isetpt",
    )

    model.set_data(
        "parameter_kernel.N",
        weights,
    )

    # ========================================================
    # Cross-subsystem links
    # ========================================================

    # Communication power affects thermal dynamics.
    model.link(
        "values.P_comm",
        "thermal.P_comm",
    )

    # Communication power affects total spacecraft power.
    model.link(
        "values.P_comm",
        "power_total.P_comm",
    )

    # Solar-panel current settings determine solar power.
    model.link(
        "values.Isetpt",
        "power_solar.Isetpt",
    )

    # Reaction-wheel power contributes to total power usage.
    model.link(
        "rw_power.P_RW",
        "power_total.P_RW",
    )

    # Solar power contributes to battery power.
    model.link(
        "power_solar.P_sol",
        "power_total.P_sol",
    )

    # Net power goes into battery dynamics.
    model.link(
        "power_total.P_bat",
        "battery.P_bat",
    )

    # Body temperature affects the battery.
    model.link(
        "thermal.temperature[:, 4]",
        "battery.temperature",
    )

    # Communication power affects the communication bitrate.
    model.link(
        "values.P_comm",
        "comm_rate.P_comm",
    )

    # Bitrate is the downloaded-data time derivative.
    model.link(
        "comm_rate.Dr",
        "comm_data.Dr",
    )

    # ========================================================
    # Thermal trajectory
    # ========================================================

    model.link(
        "thermal.temperature[:-1]",
        "thermal_trap.temperature0",
    )

    model.link(
        "thermal.temperature[1:]",
        "thermal_trap.temperature1",
    )

    model.link(
        "thermal.temperature_dot[:-1]",
        "thermal_trap.temperature_dot0",
    )

    model.link(
        "thermal.temperature_dot[1:]",
        "thermal_trap.temperature_dot1",
    )

    model.link(
        "thermal.temperature[0]",
        "thermal_ic.temperature[0]",
    )

    # ========================================================
    # Reaction-wheel trajectory
    # ========================================================

    model.link(
        "rw_torque.T_RW",
        "rw_dynamics.T_RW",
    )

    model.link(
        "rw_torque.T_RW",
        "rw_power.T_RW",
    )

    model.link(
        "rw_dynamics.w_RW",
        "rw_power.w_RW",
    )

    model.link(
        "rw_dynamics.w_RW[:-1]",
        "rw_trap.w0",
    )

    model.link(
        "rw_dynamics.w_RW[1:]",
        "rw_trap.w1",
    )

    model.link(
        "rw_dynamics.w_RW_dot[:-1]",
        "rw_trap.wdot0",
    )

    model.link(
        "rw_dynamics.w_RW_dot[1:]",
        "rw_trap.wdot1",
    )

    model.link(
        "rw_dynamics.w_RW[0]",
        "rw_ic.w_RW[0]",
    )

    # ========================================================
    # Battery trajectory
    # ========================================================

    model.link(
        "battery.SOC[:-1]",
        "battery_trap.SOC1",
    )

    model.link(
        "battery.SOC[1:]",
        "battery_trap.SOC2",
    )

    model.link(
        "battery.SOCdot[:-1]",
        "battery_trap.SOCdot1",
    )

    model.link(
        "battery.SOCdot[1:]",
        "battery_trap.SOCdot2",
    )

    model.link(
        "battery.SOC[0]",
        "battery_ic.SOC[0]",
    )

    # ========================================================
    # Downloaded-data trajectory
    # ========================================================

    model.link(
        "comm_data.Data[:-1]",
        "comm_trap.Data0",
    )

    model.link(
        "comm_data.Data[1:]",
        "comm_trap.Data1",
    )

    model.link(
        "comm_data.Data_dot[:-1]",
        "comm_trap.Data_dot0",
    )

    model.link(
        "comm_data.Data_dot[1:]",
        "comm_trap.Data_dot1",
    )

    model.link(
        "comm_data.Data[0]",
        "comm_ic.Data[0]",
    )

    # Final downloaded data is the optimization objective.
    model.link(
        f"comm_data.Data[{n - 1}]",
        "objective.Data[0]",
    )

    # ========================================================
    # Design variables
    # ========================================================

    # --------------------------------------------------------
    # CP_P_comm is FREE.
    #
    # 300 B-spline communication-power control points:
    #
    #       0 W <= CP_P_comm <= 25 W
    # --------------------------------------------------------

    model.set_meta(
        "value",
        "control_points.CP_P_comm[:]",
        CP_P_comm_guess,
    )

    model.set_meta(
        "lower",
        "control_points.CP_P_comm[:]",
        np.zeros_like(CP_P_comm),
    )

    model.set_meta(
        "upper",
        "control_points.CP_P_comm[:]",
        25.0 * np.ones_like(CP_P_comm),
    )

    # --------------------------------------------------------
    # Gamma is FIXED for this first optimization.
    # --------------------------------------------------------

    model.set_meta(
        "value",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )

    model.set_meta(
        "lower",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )

    model.set_meta(
        "upper",
        "control_points.CP_gamma[:]",
        CP_gamma,
    )

    # --------------------------------------------------------
    # Isetpt is FIXED for this first optimization.
    # --------------------------------------------------------

    for panel in range(12):
        values = CP_Isetpt[
            panel,
            :,
        ]

        variable = "control_points." f"CP_Isetpt[:, {panel}]"

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # ========================================================
    # Parameter output initial guesses
    # ========================================================

    model.set_meta(
        "value",
        "values.P_comm[:]",
        P_comm_guess,
    )

    model.set_meta(
        "value",
        "values.Gamma[:]",
        gamma_guess,
    )

    for panel in range(12):
        model.set_meta(
            "value",
            ("values." f"Isetpt[:, {panel}]"),
            Isetpt_guess[
                panel,
                :,
            ],
        )

    # ========================================================
    # Fixed thermal inputs
    # ========================================================

    exposed_area_flat = np.zeros((n, 84))

    for k in range(n):
        exposed_area_flat[k] = exposed_area[
            :,
            :,
            k,
        ].reshape(84)

    for i in range(84):
        variable = "thermal." f"exposedArea[:, {i}]"

        values = exposed_area_flat[
            :,
            i,
        ]

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    model.set_meta(
        "value",
        "thermal.LOS[:]",
        LOS,
    )

    model.set_meta(
        "lower",
        "thermal.LOS[:]",
        LOS,
    )

    model.set_meta(
        "upper",
        "thermal.LOS[:]",
        LOS,
    )

    cell_flat = cellInstd.reshape(84)

    for i in range(84):
        values = cell_flat[i] * np.ones(n)

        variable = "thermal." f"cellInstd[:, {i}]"

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # Thermal initial guesses
    for i in range(5):
        model.set_meta(
            "value",
            ("thermal." f"temperature[:, {i}]"),
            temperature_guess[
                i,
                :,
            ],
        )

        model.set_meta(
            "value",
            ("thermal." f"temperature_dot[:, {i}]"),
            temperature_dot_guess[
                :,
                i,
            ],
        )

    # ========================================================
    # Fixed attitude inputs to reaction wheels
    # ========================================================

    for i in range(3):
        model.set_meta(
            "value",
            ("rw_torque." f"T_tot[:, {i}]"),
            T_tot[i],
        )

        model.set_meta(
            "lower",
            ("rw_torque." f"T_tot[:, {i}]"),
            T_tot[i],
        )

        model.set_meta(
            "upper",
            ("rw_torque." f"T_tot[:, {i}]"),
            T_tot[i],
        )

        model.set_meta(
            "value",
            ("rw_dynamics." f"w_B[:, {i}]"),
            w_B[i],
        )

        model.set_meta(
            "lower",
            ("rw_dynamics." f"w_B[:, {i}]"),
            w_B[i],
        )

        model.set_meta(
            "upper",
            ("rw_dynamics." f"w_B[:, {i}]"),
            w_B[i],
        )

        model.set_meta(
            "value",
            ("rw_torque." f"T_RW[:, {i}]"),
            T_RW_guess[i],
        )

        model.set_meta(
            "value",
            ("rw_dynamics." f"w_RW[:, {i}]"),
            w_RW_guess[i],
        )

        model.set_meta(
            "value",
            ("rw_dynamics." f"w_RW_dot[:, {i}]"),
            w_RW_dot_guess[i],
        )

        model.set_meta(
            "value",
            ("rw_power." f"P_RW[:, {i}]"),
            P_RW_guess[i],
        )

    # ========================================================
    # Fixed solar-cell voltages
    # ========================================================

    for panel in range(12):
        variable = "power_solar." f"V_sol[:, {panel}]"

        values = V_sol[
            panel,
            :,
        ]

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # Power initial guesses
    model.set_meta(
        "value",
        "power_solar.P_sol[:]",
        P_sol_guess,
    )

    model.set_meta(
        "value",
        "power_total.P_bat[:]",
        P_bat_guess,
    )

    # ========================================================
    # Battery initial guesses and constraints
    # ========================================================

    model.set_meta(
        "value",
        "battery.SOC[:]",
        SOC_guess,
    )

    model.set_meta(
        "value",
        "battery.SOCdot[:]",
        SOCdot_guess,
    )

    # The only battery inequality used in this first optimization:
    #
    #              0.2 <= SOC <= 1.0
    #
    model.set_meta(
        "lower",
        "battery.SOC[:]",
        0.2 * np.ones(n),
    )

    model.set_meta(
        "upper",
        "battery.SOC[:]",
        1.0 * np.ones(n),
    )

    # ========================================================
    # Fixed communication geometry and antenna gain
    # ========================================================

    for name, values in [
        (
            "gain",
            gain,
        ),
        (
            "GSdist",
            GSdist,
        ),
        (
            "CommLOS",
            CommLOS,
        ),
    ]:
        variable = f"comm_rate.{name}[:]"

        model.set_meta(
            "value",
            variable,
            values,
        )

        model.set_meta(
            "lower",
            variable,
            values,
        )

        model.set_meta(
            "upper",
            variable,
            values,
        )

    # Communication initial guesses
    model.set_meta(
        "value",
        "comm_rate.Dr[:]",
        Dr_guess,
    )

    model.set_meta(
        "value",
        "comm_data.Data[:]",
        Data_guess,
    )

    model.set_meta(
        "value",
        "comm_data.Data_dot[:]",
        Data_dot_guess,
    )

    # ========================================================
    # Build model
    # ========================================================

    print()
    print("Building reduced CADRE " "optimization model...")

    model.build_module()

    print("Build successful.")

    # NATURAL ordering is currently used because it was stable
    # for the integrated CADRE model on this machine.
    model.initialize(order_type=am.OrderingType.NESTED_DISSECTION)

    print("Initialization successful.")

    # ========================================================
    # Optimize
    # ========================================================

    x = model.create_vector()

    optimizer = am.Optimizer(
        model,
        x,
    )

    options = {
        "max_iterations": 200,
        "convergence_tolerance": 1e-8,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "monotone",
        "max_line_search_iterations": 40,
    }

    print()
    print("=" * 70)
    print("RUNNING P_COMM OPTIMIZATION")
    print("=" * 70)

    opt_data = optimizer.optimize(options)

    print()
    print("Optimization finished.")

    print()
    print("Optimizer converged:")
    print(
        opt_data.get(
            "converged",
            False,
        )
    )

    if "iterations" in opt_data and len(opt_data["iterations"]) > 0:
        last_iteration = opt_data["iterations"][-1]

        print()
        print("Final NLP residual:")
        print(
            last_iteration.get(
                "residual",
                np.nan,
            )
        )

        print()
        print("Final iteration number:")
        print(
            last_iteration.get(
                "iteration",
                -1,
            )
        )

    # ========================================================
    # Extract optimized solution
    # ========================================================

    CP_P_comm_amigo = np.array(x["control_points." "CP_P_comm[:]"])

    P_comm_amigo = np.array(x["values.P_comm[:]"])

    temperature_amigo = extract_vector(
        x,
        "thermal.temperature",
        n,
        5,
    ).T

    P_bat_amigo = np.array(x["power_total.P_bat[:]"])

    SOC_amigo = np.array(x["battery.SOC[:]"])

    Dr_amigo = np.array(x["comm_rate.Dr[:]"])

    Data_amigo = np.array(x["comm_data.Data[:]"])

    # Battery current is only reported here.
    # It is NOT yet constrained in this reduced optimization.
    I_bat_amigo = battery_current(
        P_bat_amigo,
        SOC_amigo,
        temperature_amigo[4],
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("=" * 70)
    print("P_COMM OPTIMIZATION RESULTS")
    print("=" * 70)

    print()
    print("Baseline downloaded data [Gibyte]:")
    print(Data_baseline[-1])

    print()
    print("Initial-guess downloaded data [Gibyte]:")
    print(Data_guess[-1])

    print()
    print("Optimized downloaded data [Gibyte]:")
    print(Data_amigo[-1])

    print()
    print("Improvement over baseline [Gibyte]:")
    print(Data_amigo[-1] - Data_baseline[-1])

    print()
    print("Percent improvement over baseline:")
    print(100.0 * (Data_amigo[-1] - Data_baseline[-1]) / Data_baseline[-1])

    print()
    print("Maximum CP_P_comm change " "from CADRE design [W]:")
    print(np.max(np.abs(CP_P_comm_amigo - CP_P_comm)))

    print()
    print("Optimized CP_P_comm min / max [W]:")
    print(
        np.min(CP_P_comm_amigo),
        np.max(CP_P_comm_amigo),
    )

    print()
    print("Optimized P_comm min / max [W]:")
    print(
        np.min(P_comm_amigo),
        np.max(P_comm_amigo),
    )

    print()
    print("Optimized SOC min / max:")
    print(
        np.min(SOC_amigo),
        np.max(SOC_amigo),
    )

    print()
    print("Initial / final SOC:")
    print(
        SOC_amigo[0],
        SOC_amigo[-1],
    )

    print()
    print("Battery current min / max [A]")
    print("(diagnostic only; not constrained)")
    print(
        np.min(I_bat_amigo),
        np.max(I_bat_amigo),
    )

    print()
    print("Original CADRE final " "downloaded data [Gibyte]:")
    print(data["0:Data"].reshape(-1)[-1])

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
