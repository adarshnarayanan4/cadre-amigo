"""Build and connect the integrated AMIGO optimization model."""

import amigo as am
import numpy as np

from battery.battery_amigo import TrapezoidRule as BatteryTrapezoidRule
from battery.battery_reference import battery_current
from communication.communication_amigo import (
    CommunicationBitRate,
    DataDynamics,
    TrapezoidRule as DataTrapezoidRule,
)
from integrated.model.components import (
    BatteryDynamicsLinked,
    BatteryInitialCondition,
    BatteryPeriodicity,
    DataInitialCondition,
    DownloadObjective,
    ReactionWheelInitialCondition,
)
from integrated.reporting.console import print_heading
from integrated.model.design import (
    LookupResiduals,
    SurfaceAllocation,
    ThermalDesignState,
    add_roll_attitude,
)
from parameters.parameters_amigo import (
    ParameterControlPoints,
    ParameterSplineKernel,
    ParameterValues,
    build_sparse_basis,
)
from power.power_amigo import PowerSolar, PowerTotal
from power.voltage_spline import make_voltage_component
from reaction_wheel.reaction_wheel_amigo import (
    ReactionWheelDynamics,
    ReactionWheelPower,
    ReactionWheelTorque,
    TrapezoidRule as ReactionWheelTrapezoidRule,
)
from thermal.thermal_amigo import (
    InitialConditions as ThermalInitialConditions,
    ThermalDynamics,
    TrapezoidRule as ThermalTrapezoidRule,
)


def stage_model(case, args, include_objective=True):
    """Assemble equations and bounds without compilation or initialization."""
    model = _add_components(case, args, include_objective)
    _link_components(model, case)
    _set_initial_values(model, case, args)
    _register_fixed_inputs(model, case, args)
    return model


def build_model(case, args):
    """Build one case using the same staging path as the joint model."""
    model = stage_model(case, args)
    _initialize_model(model, case, args)
    return model


def _add_components(case, args, include_objective=True):
    """Register the discipline components and optional design lookup residuals."""

    # AMIGO components
    parameter_control_points = ParameterControlPoints()
    parameter_values = ParameterValues()
    parameter_kernel = ParameterSplineKernel()

    thermal = (
        ThermalDesignState()
        if case.design_physics
        else ThermalDynamics(fixed_geometry=case.coupled_voltage)
    )
    thermal_trapezoid = ThermalTrapezoidRule(case.dt)

    # Each subsystem's TrapezoidRule needs a unique AMIGO/C++ class name.
    thermal_trapezoid.name = "ThermalTrapezoidRule"
    thermal_initial = ThermalInitialConditions()

    rw_torque = ReactionWheelTorque()
    rw_dynamics = ReactionWheelDynamics()
    rw_trapezoid = ReactionWheelTrapezoidRule(case.dt)

    rw_trapezoid.name = "ReactionWheelTrapezoidRule"
    rw_initial = ReactionWheelInitialCondition()
    rw_power = ReactionWheelPower()

    power_solar = PowerSolar()
    power_total = PowerTotal()

    battery = BatteryDynamicsLinked()
    battery_trapezoid = BatteryTrapezoidRule(case.dt)

    battery_trapezoid.name = "BatteryTrapezoidRule"
    battery_initial = BatteryInitialCondition()

    comm_rate = CommunicationBitRate()
    comm_data = DataDynamics()
    comm_trapezoid = DataTrapezoidRule(case.dt)

    comm_trapezoid.name = "CommunicationTrapezoidRule"
    comm_initial = DataInitialCondition()

    objective = DownloadObjective()

    # AMIGO model
    model = am.Model(
        "cadre_optimize_design" if case.design_physics else "cadre_optimize_pcomm"
    )

    # Parameters
    model.add_component("control_points", case.m, parameter_control_points)
    model.add_component("values", case.n, parameter_values)
    model.add_component("parameter_kernel", case.n, parameter_kernel)

    # Thermal
    model.add_component("thermal", case.n, thermal)
    model.add_component("thermal_trap", case.num_intervals, thermal_trapezoid)
    model.add_component("thermal_ic", 1, thermal_initial)

    # Reaction wheels
    model.add_component("rw_torque", case.n, rw_torque)
    model.add_component("rw_dynamics", case.n, rw_dynamics)
    model.add_component("rw_trap", case.num_intervals, rw_trapezoid)
    model.add_component("rw_ic", 1, rw_initial)
    model.add_component("rw_power", case.n, rw_power)

    # Power
    model.add_component("power_solar", case.n, power_solar)
    model.add_component("power_total", case.n, power_total)

    # Battery
    model.add_component("battery", case.n, battery)
    model.add_component("battery_trap", case.num_intervals, battery_trapezoid)
    model.add_component("battery_ic", 1, battery_initial)
    model.add_component("battery_periodicity", 1, BatteryPeriodicity())

    # Communications
    model.add_component("comm_rate", case.n, comm_rate)
    model.add_component("comm_data", case.n, comm_data)
    model.add_component("comm_trap", case.num_intervals, comm_trapezoid)
    model.add_component("comm_ic", 1, comm_initial)

    # Objective
    if include_objective:
        model.add_component("objective", 1, objective)

    if case.design_physics:
        model.add_component("surface", 1, SurfaceAllocation())
        model.add_component("design_lookup", case.n, LookupResiduals())

        case.design_external.register(model)

        model.set_meta(
            "value", "surface.cellInstd[0, :]", case.initial_guess["cellInstd"].ravel()
        )

        if args.optimize_allocation:
            model.set_meta("lower", "surface.cellInstd[0, :]", 0.0)
            model.set_meta("upper", "surface.cellInstd[0, :]", 1.0)
        else:
            model.add_fixed("surface.cellInstd[0, :]")

        add_roll_attitude(model, case.lookups, case.start_geometry, case.dt)

        for index in range(5):
            model.set_meta("lower", f"thermal.temperature[:, {index}]", 50.0)
            model.set_meta("upper", f"thermal.temperature[:, {index}]", 400.0)
    elif case.coupled_voltage:
        model.add_component(
            "voltage", case.n * 12, make_voltage_component(case.voltage_spline)
        )
        model.set_data(
            "voltage.coefficients", case.voltage_coefficients.reshape(case.n * 12, 90)
        )

        for panel in range(12):
            temp_index = 4 if panel < 4 else panel % 4
            model.link(
                f"thermal.temperature[:, {temp_index}]",
                f"voltage.temperature[{panel}::12]",
            )
            model.link(f"values.Isetpt[:, {panel}]", f"voltage.current[{panel}::12]")
            model.link(
                f"power_solar.V_sol[:, {panel}]", f"voltage.voltage[{panel}::12]"
            )

        model.set_meta("value", "voltage.tT[:]", case.initial_guess["tT"].ravel())
        model.set_meta("value", "voltage.tI[:]", case.initial_guess["tI"].ravel())

        # Coordinate maps are globally monotone; impose the table domain on
        # physical temperatures and control currents rather than mapped states.
        for index in range(5):
            model.set_meta("lower", f"thermal.temperature[:, {index}]", 50.0)
            model.set_meta("upper", f"thermal.temperature[:, {index}]", 400.0)

    return model


def _link_components(model, case):
    """Connect spline controls and the thermal, wheel, battery, and data trajectories."""
    weights, indices = build_sparse_basis(case.n, case.m)

    # B-spline parameter links
    print_heading("MODEL SETUP")
    print("  Creating parameter links...", flush=True)

    for i in range(case.n):
        for j in range(4):
            control_index = indices[i, j]
            model.link(
                f"control_points.CP_P_comm[{control_index}]",
                f"parameter_kernel.CP_P_comm[{i}, {j}]",
            )
            model.link(
                f"control_points.CP_gamma[{control_index}]",
                f"parameter_kernel.CP_gamma[{i}, {j}]",
            )

            for panel in range(12):
                local_index = 4 * panel + j
                model.link(
                    f"control_points.CP_Isetpt[{control_index}, {panel}]",
                    f"parameter_kernel.CP_Isetpt[{i}, {local_index}]",
                )

    model.link("parameter_kernel.P_comm", "values.P_comm")
    model.link("parameter_kernel.Gamma", "values.Gamma")
    model.link("parameter_kernel.Isetpt", "values.Isetpt")

    model.set_data("parameter_kernel.N", weights)

    # Cross-subsystem links
    # Communication power affects thermal dynamics.
    model.link("values.P_comm", "thermal.P_comm")

    # Communication power affects total spacecraft power.
    model.link("values.P_comm", "power_total.P_comm")

    # Solar-panel current settings determine solar power.
    model.link("values.Isetpt", "power_solar.Isetpt")

    # Reaction-wheel power contributes to total power usage.
    model.link("rw_power.P_RW", "power_total.P_RW")

    # Solar power contributes to battery power.
    model.link("power_solar.P_sol", "power_total.P_sol")

    # Net power goes into battery dynamics.
    model.link("power_total.P_bat", "battery.P_bat")

    # Body temperature affects the battery.
    model.link("thermal.temperature[:, 4]", "battery.temperature")

    # Communication power affects the communication bitrate.
    model.link("values.P_comm", "comm_rate.P_comm")

    # Bitrate is the downloaded-data time derivative.
    model.link("comm_rate.Dr", "comm_data.Dr")

    # Thermal trajectory
    model.link("thermal.temperature[:-1]", "thermal_trap.temperature0")
    model.link("thermal.temperature[1:]", "thermal_trap.temperature1")
    model.link("thermal.temperature_dot[:-1]", "thermal_trap.temperature_dot0")
    model.link("thermal.temperature_dot[1:]", "thermal_trap.temperature_dot1")
    model.link("thermal.temperature[0]", "thermal_ic.temperature[0]")

    # Reaction-wheel trajectory
    model.link("rw_torque.T_RW", "rw_dynamics.T_RW")
    model.link("rw_torque.T_RW", "rw_power.T_RW")
    model.link("rw_dynamics.w_RW", "rw_power.w_RW")
    model.link("rw_dynamics.w_RW[:-1]", "rw_trap.w0")
    model.link("rw_dynamics.w_RW[1:]", "rw_trap.w1")
    model.link("rw_dynamics.w_RW_dot[:-1]", "rw_trap.wdot0")
    model.link("rw_dynamics.w_RW_dot[1:]", "rw_trap.wdot1")
    model.link("rw_dynamics.w_RW[0]", "rw_ic.w_RW[0]")

    # Battery trajectory
    model.link("battery.SOC[:-1]", "battery_trap.SOC1")
    model.link("battery.SOC[1:]", "battery_trap.SOC2")
    model.link("battery.SOCdot[:-1]", "battery_trap.SOCdot1")
    model.link("battery.SOCdot[1:]", "battery_trap.SOCdot2")
    model.link("battery.SOC[0]", "battery_ic.SOC[0]")
    model.link("battery.SOC[0]", "battery_periodicity.SOC_initial[0]")
    model.link(f"battery.SOC[{case.n - 1}]", "battery_periodicity.SOC_final[0]")

    # Downloaded-data trajectory
    model.link("comm_data.Data[:-1]", "comm_trap.Data0")
    model.link("comm_data.Data[1:]", "comm_trap.Data1")
    model.link("comm_data.Data_dot[:-1]", "comm_trap.Data_dot0")
    model.link("comm_data.Data_dot[1:]", "comm_trap.Data_dot1")
    model.link("comm_data.Data[0]", "comm_ic.Data[0]")

    # Final downloaded data is the optimization objective.
    if "objective" in model.comp:
        model.link(f"comm_data.Data[{case.n - 1}]", "objective.Data[0]")


def _set_initial_values(model, case, args):
    """Set consistent trajectory guesses, bounds, and prescribed geometry."""
    _set_control_values(model, case, args)

    _set_thermal_values(model, case)

    _set_reaction_wheel_values(model, case)

    _set_power_values(model, case)

    _set_battery_values(model, case, args)

    _set_communication_values(model, case)


def _set_control_values(model, case, args):
    """Set spline controls, their bounds, and parameter output guesses."""
    # Communication-power control points are bounded between 0 and 25 W.
    model.set_meta(
        "value", "control_points.CP_P_comm[:]", case.initial_guess["CP_P_comm"]
    )
    model.set_meta(
        "lower", "control_points.CP_P_comm[:]", np.zeros_like(case.CP_P_comm)
    )
    model.set_meta(
        "upper", "control_points.CP_P_comm[:]", 25.0 * np.ones_like(case.CP_P_comm)
    )

    # Roll is free only when --optimize-roll is selected.
    model.set_meta(
        "value", "control_points.CP_gamma[:]", case.initial_guess["CP_gamma"]
    )
    model.set_meta(
        "lower",
        "control_points.CP_gamma[:]",
        0.0 if args.optimize_roll else case.initial_guess["CP_gamma"],
    )
    model.set_meta(
        "upper",
        "control_points.CP_gamma[:]",
        np.pi / 2 if args.optimize_roll else case.initial_guess["CP_gamma"],
    )

    # Solar current control points are free in the coupled optimization.
    for panel in range(12):
        values = case.initial_guess["CP_Isetpt"][panel, :]

        variable = f"control_points.CP_Isetpt[:, {panel}]"
        model.set_meta("value", variable, values)
        model.set_meta(
            "lower", variable, np.zeros(case.m) if case.free_solar else values
        )
        model.set_meta(
            "upper", variable, 0.4 * np.ones(case.m) if case.free_solar else values
        )

    # Parameter output initial guesses
    model.set_meta("value", "values.P_comm[:]", case.initial_guess["P_comm"])
    model.set_meta("value", "values.Gamma[:]", case.initial_guess["gamma"])

    for panel in range(12):
        model.set_meta(
            "value",
            f"values.Isetpt[:, {panel}]",
            case.initial_guess["Isetpt"][panel, :],
        )


def _set_thermal_values(model, case):
    """Set prescribed thermal inputs and initial temperature trajectories."""
    # Fixed thermal inputs
    exposed_area_flat = np.zeros((case.n, 84))

    for k in range(case.n):
        exposed_area_flat[k] = case.exposed_area[:, :, k].reshape(84)

    if case.design_physics:
        pass
    elif case.coupled_voltage:
        model.set_data("thermal.exposedArea", exposed_area_flat)
        model.set_data("thermal.LOS", case.LOS)
        model.set_data(
            "thermal.cellInstd",
            np.broadcast_to(case.cellInstd.reshape(84), (case.n, 84)).copy(),
        )
    else:
        for i in range(84):
            variable = f"thermal.exposedArea[:, {i}]"

            values = exposed_area_flat[:, i]
            model.set_meta("value", variable, values)
            model.set_meta("lower", variable, values)
            model.set_meta("upper", variable, values)

        model.set_meta("value", "thermal.LOS[:]", case.LOS)
        model.set_meta("lower", "thermal.LOS[:]", case.LOS)
        model.set_meta("upper", "thermal.LOS[:]", case.LOS)

        cell_flat = case.cellInstd.reshape(84)

        for i in range(84):
            values = cell_flat[i] * np.ones(case.n)

            variable = f"thermal.cellInstd[:, {i}]"
            model.set_meta("value", variable, values)
            model.set_meta("lower", variable, values)
            model.set_meta("upper", variable, values)

    # Thermal initial guesses
    for i in range(5):
        model.set_meta(
            "value",
            f"thermal.temperature[:, {i}]",
            case.initial_guess["temperature"][i, :],
        )
        model.set_meta(
            "value",
            f"thermal.temperature_dot[:, {i}]",
            case.initial_guess["temperature_dot"][:, i],
        )


def _set_reaction_wheel_values(model, case):
    """Set prescribed attitude inputs and reaction-wheel trajectory guesses."""

    # Fixed attitude inputs to reaction wheels
    for i in range(3):
        model.set_meta(
            "value", f"rw_torque.T_tot[:, {i}]", case.initial_guess["T_tot"][i]
        )
        model.set_meta(
            "lower", f"rw_torque.T_tot[:, {i}]", case.initial_guess["T_tot"][i]
        )
        model.set_meta(
            "upper", f"rw_torque.T_tot[:, {i}]", case.initial_guess["T_tot"][i]
        )
        model.set_meta(
            "value", f"rw_dynamics.w_B[:, {i}]", case.initial_guess["w_B"][i]
        )
        model.set_meta(
            "lower", f"rw_dynamics.w_B[:, {i}]", case.initial_guess["w_B"][i]
        )
        model.set_meta(
            "upper", f"rw_dynamics.w_B[:, {i}]", case.initial_guess["w_B"][i]
        )
        model.set_meta(
            "value", f"rw_torque.T_RW[:, {i}]", case.initial_guess["T_RW"][i]
        )
        model.set_meta(
            "value", f"rw_dynamics.w_RW[:, {i}]", case.initial_guess["w_RW"][i]
        )
        model.set_meta(
            "value",
            f"rw_dynamics.w_RW_dot[:, {i}]",
            case.initial_guess["w_RW_dot"][i],
        )
        model.set_meta("value", f"rw_power.P_RW[:, {i}]", case.initial_guess["P_RW"][i])


def _set_power_values(model, case):
    """Set solar voltage bounds and initial solar and battery power."""

    # Solar-cell voltage initial guesses
    for panel in range(12):
        variable = f"power_solar.V_sol[:, {panel}]"

        values = case.initial_guess["V_sol"][panel, :]
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, -np.inf if case.coupled_voltage else values)
        model.set_meta("upper", variable, np.inf if case.coupled_voltage else values)

    # Power initial guesses
    model.set_meta("value", "power_solar.P_sol[:]", case.initial_guess["P_sol"])
    model.set_meta("value", "power_total.P_bat[:]", case.initial_guess["P_bat"])


def _set_battery_values(model, case, args):
    """Set battery trajectory guesses, current limits, and initial SOC."""
    # Battery initial guesses and constraints
    model.set_meta("value", "battery.SOC[:]", case.initial_guess["SOC"])
    model.set_meta("value", "battery.SOCdot[:]", case.initial_guess["SOCdot"])

    # Keep SOC between 0.2 and 1.0 and battery current between -10 and 5 A.
    model.set_meta("lower", "battery.SOC[:]", 0.2 * np.ones(case.n))
    model.set_meta("upper", "battery.SOC[:]", 1.0 * np.ones(case.n))
    model.set_meta(
        "value",
        "battery.I_bat[:]",
        battery_current(
            case.initial_guess["P_bat"],
            case.initial_guess["SOC"],
            case.initial_guess["temperature"][4],
        ),
    )
    model.set_meta("lower", "battery.I_bat[:]", -10.0 * np.ones(case.n))
    model.set_meta("upper", "battery.I_bat[:]", 5.0 * np.ones(case.n))
    model.set_meta("value", "battery_ic.initial_soc[0]", case.initial_soc)

    if args.fixed_initial_soc:
        model.add_fixed("battery_ic.initial_soc[:]")
    else:
        model.set_meta("lower", "battery_ic.initial_soc[0]", 0.2)
        model.set_meta("upper", "battery_ic.initial_soc[0]", 1.0)


def _set_communication_values(model, case):
    """Set prescribed communication geometry and downloaded-data guesses."""

    # Fixed communication geometry and antenna gain
    for name, values in [
        ("gain", case.initial_guess["gain"]),
        ("GSdist", case.GSdist),
        ("CommLOS", case.CommLOS),
    ]:
        variable = f"comm_rate.{name}[:]"
        model.set_meta("value", variable, values)
        model.set_meta("lower", variable, values)
        model.set_meta("upper", variable, values)

    # Communication initial guesses
    model.set_meta("value", "comm_rate.Dr[:]", case.initial_guess["Dr"])
    model.set_meta("value", "comm_data.Data[:]", case.initial_guess["Data"])
    model.set_meta("value", "comm_data.Data_dot[:]", case.initial_guess["Data_dot"])


def _initialize_model(model, case, args):
    """Compile and initialize, keeping prescribed rows out of the bound barrier."""

    # Build model
    print("\n  Building CADRE optimization model...\n", flush=True)

    model.build_module()

    print("\n  Build successful. Initializing the model...\n", flush=True)

    model.initialize(order_type=am.OrderingType.NATURAL)

    print("\n  Model initialization successful.\n", flush=True)


def _register_fixed_inputs(model, case, args):
    """Remove prescribed inputs from the barrier before submodel assembly."""

    # The solver orders the complete KKT matrix rather than only variables.
    # Equal bounds alone do not mark a variable FIXED in this AMIGO version.
    # Register prescribed inputs and remove their zero-width barrier bounds.
    if case.design_physics:
        for variable, width in (
            ("rw_torque.T_tot", 3),
            ("rw_dynamics.w_B", 3),
            ("comm_rate.gain", 1),
        ):
            for column in range(width):
                expression = (
                    f"{variable}[:]" if width == 1 else f"{variable}[:, {column}]"
                )
                model.set_meta("lower", expression, -np.inf)
                model.set_meta("upper", expression, np.inf)

    for variable, width in (
        ("control_points.CP_gamma", 1),
        ("control_points.CP_Isetpt", 12),
        ("thermal.exposedArea", 84),
        ("thermal.LOS", 1),
        ("thermal.cellInstd", 84),
        ("rw_torque.T_tot", 3),
        ("rw_dynamics.w_B", 3),
        ("power_solar.V_sol", 12),
        ("comm_rate.gain", 1),
        ("comm_rate.GSdist", 1),
        ("comm_rate.CommLOS", 1),
    ):
        if case.design_physics and variable in (
            "rw_torque.T_tot",
            "rw_dynamics.w_B",
            "comm_rate.gain",
        ):
            continue

        if variable == "control_points.CP_gamma" and args.optimize_roll:
            continue

        if case.coupled_voltage and variable in (
            "thermal.exposedArea",
            "thermal.LOS",
            "thermal.cellInstd",
        ):
            continue

        if (variable == "control_points.CP_Isetpt" and case.free_solar) or (
            variable == "power_solar.V_sol" and case.coupled_voltage
        ):
            continue

        # AMIGO's fixed-index staging expects one-dimensional slices.
        for column in range(width):
            expression = f"{variable}[:]" if width == 1 else f"{variable}[:, {column}]"
            model.add_fixed(expression)
            # FIXED rows still appear in AMIGO's primal index array. Remove
            # finite equal bounds so the barrier never divides by a zero gap.
            model.set_meta("lower", expression, -np.inf)
            model.set_meta("upper", expression, np.inf)
