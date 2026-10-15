"""Reference trajectories and initialization for an integrated CADRE case."""

from dataclasses import dataclass
import pickle

import numpy as np

from battery.battery_reference import battery_current, propagate_soc_trapezoid, soc_rate
from cadre_paths import cadre_path
from communication.communication_amigo import propagate_data_trapezoid
from communication.communication_data_validation import communication_bit_rate
from integrated.cadre_feasibility import feasibility_metrics, repair_baseline
from integrated.cadre_reporting import print_feasibility, print_heading, print_values
from integrated.design_amigo import DesignPhysics
from integrated.design_reference import DesignLookups
from parameters.parameters_reference import bspline_parameters, build_bspline_matrix
from power.power_amigo import solar_power, total_power
from power.voltage_spline import VoltageSpline
from reaction_wheel.reaction_wheel_reference import (
    propagate_reaction_wheel_trapezoid,
    reaction_wheel_power,
    reaction_wheel_torque,
)
from thermal.thermal_amigo import propagate_temperature_trapezoid


@dataclass
class CadreCase:
    """Benchmark inputs, repaired comparison, and a consistent initial trajectory.

    Array names retain the CADRE discipline notation used elsewhere in the repo.
    Node-major arrays are created only when wiring AMIGO components.
    """

    design_physics: bool
    coupled_voltage: bool
    free_solar: bool
    initial_guess: dict[str, np.ndarray | None]
    baseline: dict[str, np.ndarray]
    original: dict[str, np.ndarray]
    data: dict
    n: int
    m: int
    num_intervals: int
    dt: float
    window_duration: float
    exposed_area: np.ndarray
    LOS: np.ndarray
    cellInstd: np.ndarray
    V_sol: np.ndarray
    gain: np.ndarray
    GSdist: np.ndarray
    CommLOS: np.ndarray
    initial_soc: float
    B: np.ndarray
    CP_P_comm: np.ndarray

    voltage_spline: VoltageSpline | None
    lookups: DesignLookups | None
    design_external: DesignPhysics | None

    start_geometry: tuple | None

    T0: np.ndarray
    w_RW0: np.ndarray

    voltage_coefficients: np.ndarray | None

    original_metrics: dict
    original_data: float

    baseline_metrics: dict


def calculate_battery_rates(SOC, P_bat, temperature):
    n = len(SOC)

    rates = np.zeros(n)

    for k in range(n):
        rates[k] = soc_rate(SOC[k], P_bat[k], temperature[k])

    return rates


def extract_vector(x, variable_name, n, width):
    values = np.zeros((n, width))

    for i in range(width):
        values[:, i] = x[f"{variable_name}[:, {i}]"]

    return values


def prepare_case(args):
    """Load the selected case, repair its baseline, and replay starting controls."""
    design_physics = (
        args.design_physics or args.optimize_roll or args.optimize_allocation
    )
    coupled_voltage = not args.legacy_fixed_voltage
    free_solar = coupled_voltage and not args.fixed_solar_currents

    start_geometry = tT_guess = tI_guess = None

    data_path = cadre_path("test/data1346.pkl")

    with open(data_path, "rb") as file:
        data = pickle.load(file, encoding="latin1")

    # Preserve the original time resolution and use a contiguous prefix.
    full_n = len(data["0:P_comm"])
    n = args.nodes
    m = args.control_points

    if not 4 <= m <= n <= full_n:
        raise ValueError("Require 4 <= control-points <= nodes <= benchmark nodes")

    if not np.any(np.asarray(data["0:CommLOS"])[:n] > 0):
        raise ValueError(
            "Selected window contains no ground-station contact; increase --nodes"
        )

    num_intervals = n - 1

    # Keep original CADRE time resolution.
    dt = 43200.0 / (full_n - 1)

    window_duration = dt * num_intervals

    print_heading("CADRE P_COMM OPTIMIZATION")

    print_values(
        [
            ("Original CADRE nodes", f"{full_n:,}"),
            ("Optimization nodes", f"{n:,}"),
            ("Time intervals", f"{num_intervals:,}"),
            ("Time step", f"{dt:.4f} s"),
            ("Optimization window", f"{window_duration / 3600.0:.2f} hours"),
        ]
    )

    # Fixed CADRE trajectory inputs for the selected window

    P_comm_target = np.asarray(data["0:P_comm"])[:n]

    gamma_target = np.asarray(data["0:gamma"])[:n]

    Isetpt_target = np.asarray(data["0:Isetpt"])[:, :n]

    exposed_area = np.asarray(data["0:exposedArea"])[:, :, :n]

    LOS = np.asarray(data["0:LOS"])[:n]

    cellInstd = np.asarray(data["cellInstd"])

    w_B = np.asarray(data["0:w_B"])[:, :n]

    T_tot = np.asarray(data["0:T_tot"])[:, :n]

    V_sol = np.asarray(data["0:V_sol"])[:, :n]

    gain = np.asarray(data["0:gain"])[:n]

    GSdist = np.asarray(data["0:GSdist"])[:n]

    CommLOS = np.asarray(data["0:CommLOS"])[:n]

    initial_soc = float(data["0:iSOC"][0])

    # Fit B-spline control points to this trajectory window

    B = build_bspline_matrix(n, m)

    if n == full_n and m == len(data["0:CP_P_comm"]):
        # Retain the benchmark design exactly on its native grid.
        CP_P_comm = np.asarray(data["0:CP_P_comm"]).copy()
        CP_gamma = np.asarray(data["0:CP_gamma"]).copy()
        CP_Isetpt = np.asarray(data["0:CP_Isetpt"]).copy()
    else:
        CP_P_comm = np.linalg.lstsq(B, P_comm_target, rcond=None)[0]

        CP_gamma = np.linalg.lstsq(B, gamma_target, rcond=None)[0]

        CP_Isetpt = np.linalg.lstsq(B, Isetpt_target.T, rcond=None)[0].T

    print_values(
        [
            ("Communication-power controls", f"{m:,}"),
            (
                "Fitted power maximum error",
                f"{np.max(np.abs(B @ CP_P_comm - P_comm_target)):.3e} W",
            ),
        ]
    )

    # Benchmark design represented on the selected spline grid

    P_comm_baseline, gamma_baseline, Isetpt_baseline = bspline_parameters(
        CP_P_comm, CP_gamma, CP_Isetpt, n
    )

    voltage_spline = VoltageSpline() if coupled_voltage else None

    lookups = DesignLookups(data, n, voltage_spline) if design_physics else None
    design_external = DesignPhysics(lookups) if design_physics else None

    if design_physics:
        lookups.dt = dt
        baseline_geometry = lookups.geometry(gamma_baseline)
        exposed_area, gain, w_B, T_tot = baseline_geometry[:4]

    CP_gamma_start = (
        np.clip(CP_gamma, 0.02, np.pi / 2 - 0.02)
        if args.optimize_roll
        else CP_gamma.copy()
    )
    gamma_start = B @ CP_gamma_start
    cellInstd_start = (
        np.clip(cellInstd, 0.02, 0.98) if args.optimize_allocation else cellInstd.copy()
    )

    if design_physics:
        start_geometry = lookups.geometry(gamma_start)
        area_start, gain_start, w_B_start, T_tot_start = start_geometry[:4]
    else:
        area_start, gain_start, w_B_start, T_tot_start = exposed_area, gain, w_B, T_tot

    T0 = 273.0 * np.ones(5)

    temperature_baseline, temperature_dot_baseline = propagate_temperature_trapezoid(
        T0, exposed_area, LOS, P_comm_baseline, cellInstd, dt
    )

    T_RW_baseline = reaction_wheel_torque(T_tot)

    w_RW0 = np.zeros(3)

    w_RW_baseline, w_RW_dot_baseline = propagate_reaction_wheel_trapezoid(
        w_RW0, w_B, T_RW_baseline, dt
    )

    P_RW_baseline = reaction_wheel_power(w_RW_baseline, T_RW_baseline)

    voltage_coefficients = (
        voltage_spline.panel_coefficients(exposed_area, LOS)
        if coupled_voltage
        else None
    )

    if coupled_voltage:
        V_sol = voltage_spline.reference_panels(
            temperature_baseline, Isetpt_baseline, exposed_area, LOS
        )

    CP_Isetpt_start = (
        np.clip(CP_Isetpt, 0.005, 0.395) if free_solar else CP_Isetpt.copy()
    )
    Isetpt_start = CP_Isetpt_start @ B.T

    P_sol_baseline = solar_power(V_sol, Isetpt_baseline)

    P_bat_baseline = total_power(P_sol_baseline, P_comm_baseline, P_RW_baseline)

    SOC_baseline = propagate_soc_trapezoid(
        initial_soc, P_bat_baseline, temperature_baseline, dt
    )

    Dr_baseline = communication_bit_rate(P_comm_baseline, gain, GSdist, CommLOS)

    Data_baseline, Data_dot_baseline = propagate_data_trapezoid(0.0, Dr_baseline, dt)

    def replay_controls(
        cp, currents=Isetpt_baseline, roll=gamma_baseline, allocation=cellInstd
    ):
        if design_physics:
            area, antenna_gain, angular, torque = lookups.geometry(roll)[:4]
            wheel_torque = reaction_wheel_torque(torque)
            wheels, _ = propagate_reaction_wheel_trapezoid(
                w_RW0, angular, wheel_torque, dt
            )
            wheel_power = reaction_wheel_power(wheels, wheel_torque)
        else:
            area, antenna_gain, wheel_power = exposed_area, gain, P_RW_baseline

        power = B @ cp
        temperature, temperature_dot = propagate_temperature_trapezoid(
            T0, area, LOS, power, allocation, dt
        )
        solar_voltage = (
            voltage_spline.reference_panels(temperature, currents, area, LOS)
            if coupled_voltage
            else V_sol
        )
        solar = solar_power(solar_voltage, currents)
        battery_power = total_power(solar, power, wheel_power)
        soc = propagate_soc_trapezoid(initial_soc, battery_power, temperature, dt)
        rate = communication_bit_rate(power, antenna_gain, GSdist, CommLOS)
        downloaded, downloaded_dot = propagate_data_trapezoid(0.0, rate, dt)

        return {
            "P_comm": power,
            "temperature": temperature,
            "temperature_dot": temperature_dot,
            "P_sol": solar,
            "V_sol": solar_voltage,
            "P_RW": wheel_power,
            "P_bat": battery_power,
            "SOC": soc,
            "Dr": rate,
            "Data": downloaded,
            "Data_dot": downloaded_dot,
            "I_bat": battery_current(battery_power, soc, temperature[4]),
        }

    original_state = {
        "SOC": SOC_baseline,
        "temperature": temperature_baseline,
        "I_bat": battery_current(P_bat_baseline, SOC_baseline, temperature_baseline[4]),
    }

    original_metrics = feasibility_metrics(original_state, initial_soc)
    original_data = float(Data_baseline[-1])
    original_P_comm = P_comm_baseline.copy()
    original_SOC = SOC_baseline.copy()
    original_Data = Data_baseline.copy()

    print_feasibility("ORIGINAL BENCHMARK FEASIBILITY", original_metrics)

    CP_P_comm_baseline, baseline_state, baseline_metrics = repair_baseline(
        B,
        CP_P_comm,
        initial_soc,
        dt,
        lambda state: total_power(state["P_sol"], np.zeros(n), state["P_RW"]),
        replay_controls,
    )

    if args.warm_start:
        # These model guesses are replaced by the complete checkpoint below.
        CP_P_comm_start = CP_P_comm_baseline.copy()
    else:
        CP_P_comm_start, _, _ = repair_baseline(
            B,
            CP_P_comm,
            initial_soc,
            dt,
            lambda state: total_power(state["P_sol"], np.zeros(n), state["P_RW"]),
            lambda cp: replay_controls(cp, Isetpt_start, gamma_start, cellInstd_start),
            interior=True,
        )

    P_sol_baseline = baseline_state["P_sol"]
    V_sol_baseline = baseline_state["V_sol"]
    P_comm_baseline = baseline_state["P_comm"]
    temperature_baseline = baseline_state["temperature"]
    temperature_dot_baseline = baseline_state["temperature_dot"]
    P_bat_baseline = baseline_state["P_bat"]
    SOC_baseline = baseline_state["SOC"]
    Dr_baseline = baseline_state["Dr"]
    Data_baseline = baseline_state["Data"]
    Data_dot_baseline = baseline_state["Data_dot"]

    print_heading("FEASIBLE BASELINE")

    print_values(
        [
            ("Final SOC", f"{SOC_baseline[-1]:.6f}"),
            ("SOC range", f"{np.min(SOC_baseline):.6f} to {np.max(SOC_baseline):.6f}"),
            ("Downloaded data [CADRE units]", f"{Data_baseline[-1]:,.6f}"),
        ]
    )

    # Interior optimization starting point

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

    CP_P_comm_guess = CP_P_comm_start.copy()

    P_comm_guess, gamma_guess, Isetpt_guess = bspline_parameters(
        CP_P_comm_guess, CP_gamma_start, CP_Isetpt_start, n
    )

    # Thermal initial guess

    temperature_guess, temperature_dot_guess = propagate_temperature_trapezoid(
        T0, area_start, LOS, P_comm_guess, cellInstd_start, dt
    )

    # Reaction wheels
    #
    # Use the prescribed or roll-dependent starting attitude.

    T_RW_guess = reaction_wheel_torque(T_tot_start)
    w_RW_guess, w_RW_dot_guess = propagate_reaction_wheel_trapezoid(
        w_RW0, w_B_start, T_RW_guess, dt
    )
    P_RW_guess = reaction_wheel_power(w_RW_guess, T_RW_guess)

    # Power initial guess

    if design_physics:
        V_sol_guess = voltage_spline.reference_panels(
            temperature_guess, Isetpt_guess, area_start, LOS
        )
    elif coupled_voltage:
        V_sol_guess, tT_guess, tI_guess = voltage_spline.evaluate_panels(
            temperature_guess, Isetpt_guess, voltage_coefficients
        )
    else:
        V_sol_guess = V_sol

    P_sol_guess = solar_power(V_sol_guess, Isetpt_guess)

    P_bat_guess = total_power(P_sol_guess, P_comm_guess, P_RW_guess)

    # Battery initial guess

    SOC_guess = propagate_soc_trapezoid(initial_soc, P_bat_guess, temperature_guess, dt)

    SOCdot_guess = calculate_battery_rates(SOC_guess, P_bat_guess, temperature_guess[4])

    # Communications initial guess

    Dr_guess = communication_bit_rate(P_comm_guess, gain_start, GSdist, CommLOS)

    Data_guess, Data_dot_guess = propagate_data_trapezoid(0.0, Dr_guess, dt)

    print_heading("OPTIMIZATION INITIAL GUESS")

    print_values(
        [
            ("Final SOC", f"{SOC_guess[-1]:.6f}"),
            ("SOC range", f"{np.min(SOC_guess):.6f} to {np.max(SOC_guess):.6f}"),
            ("Downloaded data [CADRE units]", f"{Data_guess[-1]:,.6f}"),
        ]
    )

    initial_guess = {
        "CP_P_comm": CP_P_comm_guess,
        "P_comm": P_comm_guess,
        "gamma": gamma_guess,
        "Isetpt": Isetpt_guess,
        "temperature": temperature_guess,
        "temperature_dot": temperature_dot_guess,
        "T_RW": T_RW_guess,
        "w_RW": w_RW_guess,
        "w_RW_dot": w_RW_dot_guess,
        "P_RW": P_RW_guess,
        "V_sol": V_sol_guess,
        "tI": tI_guess,
        "tT": tT_guess,
        "P_sol": P_sol_guess,
        "P_bat": P_bat_guess,
        "SOC": SOC_guess,
        "SOCdot": SOCdot_guess,
        "Dr": Dr_guess,
        "Data": Data_guess,
        "Data_dot": Data_dot_guess,
        "CP_gamma": CP_gamma_start,
        "cellInstd": cellInstd_start,
        "w_B": w_B_start,
        "gain": gain_start,
        "T_tot": T_tot_start,
        "CP_Isetpt": CP_Isetpt_start,
    }
    baseline_state.update(
        {
            "gamma": gamma_baseline,
            "Isetpt": Isetpt_baseline,
            "P_RW": P_RW_baseline,
            "SOC": SOC_baseline,
            "CP_P_comm": CP_P_comm_baseline,
            "V_sol": V_sol_baseline,
            "P_comm": P_comm_baseline,
            "temperature": temperature_baseline,
            "Data": Data_baseline,
        }
    )

    original = {"P_comm": original_P_comm, "SOC": original_SOC, "Data": original_Data}

    return CadreCase(
        design_physics=design_physics,
        coupled_voltage=coupled_voltage,
        free_solar=free_solar,
        data=data,
        n=n,
        m=m,
        num_intervals=num_intervals,
        dt=dt,
        window_duration=window_duration,
        exposed_area=exposed_area,
        LOS=LOS,
        cellInstd=cellInstd,
        V_sol=V_sol,
        gain=gain,
        GSdist=GSdist,
        CommLOS=CommLOS,
        initial_soc=initial_soc,
        B=B,
        CP_P_comm=CP_P_comm,
        voltage_spline=voltage_spline,
        lookups=lookups,
        design_external=design_external,
        start_geometry=start_geometry,
        T0=T0,
        w_RW0=w_RW0,
        voltage_coefficients=voltage_coefficients,
        original_metrics=original_metrics,
        original_data=original_data,
        baseline_metrics=baseline_metrics,
        initial_guess=initial_guess,
        baseline=baseline_state,
        original=original,
    )
