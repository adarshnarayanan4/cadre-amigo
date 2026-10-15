"""Independent trajectory validation and saved optimization results."""

import json

import numpy as np

from battery.battery_reference import battery_current, propagate_soc_trapezoid
from communication.communication_amigo import propagate_data_trapezoid
from communication.communication_data_validation import communication_bit_rate
from integrated.cadre_feasibility import feasibility_metrics
from integrated.cadre_reference import extract_vector
from integrated.cadre_reporting import print_results
from power.power_amigo import solar_power, total_power
from reaction_wheel.reaction_wheel_reference import (
    propagate_reaction_wheel_trapezoid,
    reaction_wheel_power,
    reaction_wheel_torque,
)
from thermal.thermal_amigo import propagate_temperature_trapezoid


def validate_and_save_results(case, args, result):
    """Replay optimized controls, check feasibility, and save diagnostics.

    A failed check still saves its diagnostics, then raises instead of reporting
    an unvalidated run as a successful optimization.
    """
    design_physics = case.design_physics
    coupled_voltage = case.coupled_voltage
    free_solar = case.free_solar

    data = case.data
    n = case.n
    m = case.m
    dt = case.dt
    window_duration = case.window_duration

    exposed_area = case.exposed_area
    LOS = case.LOS
    cellInstd = case.cellInstd
    V_sol = case.V_sol

    gain = case.gain
    GSdist = case.GSdist
    CommLOS = case.CommLOS

    initial_soc = case.initial_soc
    B = case.B
    CP_P_comm = case.CP_P_comm

    gamma_baseline = case.baseline["gamma"]
    Isetpt_baseline = case.baseline["Isetpt"]

    voltage_spline = case.voltage_spline
    lookups = case.lookups

    T0 = case.T0
    w_RW0 = case.w_RW0

    P_RW_baseline = case.baseline["P_RW"]
    SOC_baseline = case.baseline["SOC"]

    original_metrics = case.original_metrics
    original_data = case.original_data

    original_P_comm = case.original["P_comm"]
    original_SOC = case.original["SOC"]
    original_Data = case.original["Data"]

    baseline_state = case.baseline
    baseline_metrics = case.baseline_metrics

    CP_P_comm_baseline = case.baseline["CP_P_comm"]
    V_sol_baseline = case.baseline["V_sol"]
    P_comm_baseline = case.baseline["P_comm"]
    temperature_baseline = case.baseline["temperature"]
    Data_baseline = case.baseline["Data"]

    Data_guess = case.initial_guess["Data"]

    x, opt_data = result.point, result.history

    CP_P_comm_amigo = np.array(x["control_points." "CP_P_comm[:]"])

    P_comm_amigo = np.array(x["values.P_comm[:]"])

    temperature_amigo = extract_vector(x, "thermal.temperature", n, 5).T

    P_bat_amigo = np.array(x["power_total.P_bat[:]"])

    SOC_amigo = np.array(x["battery.SOC[:]"])

    Dr_amigo = np.array(x["comm_rate.Dr[:]"])

    Data_amigo = np.array(x["comm_data.Data[:]"])

    # Compare the constrained current variable with the independent formula.
    I_bat_amigo = battery_current(P_bat_amigo, SOC_amigo, temperature_amigo[4])

    I_bat_variable = np.array(x["battery.I_bat[:]"])

    optimized_initial_soc = float(x["battery_ic.initial_soc[0]"])

    CP_Isetpt_amigo = extract_vector(x, "control_points.CP_Isetpt", m, 12).T

    Isetpt_amigo = extract_vector(x, "values.Isetpt", n, 12).T

    V_sol_amigo = extract_vector(x, "power_solar.V_sol", n, 12).T
    P_sol_amigo = np.array(x["power_solar.P_sol[:]"])

    CP_gamma_amigo = np.asarray(x["control_points.CP_gamma[:]"]).copy()
    gamma_amigo = np.asarray(x["values.Gamma[:]"]).copy()

    cell_amigo = (
        np.asarray(x["surface.cellInstd[0, :]"]).reshape(7, 12).copy()
        if design_physics
        else cellInstd.copy()
    )
    # Independently replay the optimized controls through every free dependency.
    P_replay = B @ CP_P_comm_amigo
    gamma_replay = B @ CP_gamma_amigo

    if design_physics:
        area_replay, gain_replay, angular_replay, torque_replay = lookups.geometry(
            gamma_replay, reference=True
        )[:4]
        wheel_torque_replay = reaction_wheel_torque(torque_replay)
        wheel_replay, _ = propagate_reaction_wheel_trapezoid(
            w_RW0, angular_replay, wheel_torque_replay, dt
        )
        wheel_power_replay = reaction_wheel_power(wheel_replay, wheel_torque_replay)
    else:
        area_replay, gain_replay, wheel_power_replay = exposed_area, gain, P_RW_baseline

    temperature_replay, _ = propagate_temperature_trapezoid(
        T0, area_replay, LOS, P_replay, cell_amigo, dt
    )

    Isetpt_replay = CP_Isetpt_amigo @ B.T
    V_sol_replay = (
        voltage_spline.reference_panels(
            temperature_replay, Isetpt_replay, area_replay, LOS
        )
        if coupled_voltage
        else V_sol
    )

    P_sol_replay = solar_power(V_sol_replay, Isetpt_replay)
    P_bat_replay = total_power(P_sol_replay, P_replay, wheel_power_replay)

    SOC_replay = propagate_soc_trapezoid(
        optimized_initial_soc, P_bat_replay, temperature_replay, dt
    )

    Dr_replay = communication_bit_rate(P_replay, gain_replay, GSdist, CommLOS)
    Data_replay, _ = propagate_data_trapezoid(0.0, Dr_replay, dt)

    I_bat_replay = battery_current(P_bat_replay, SOC_replay, temperature_replay[4])

    optimized_metrics = feasibility_metrics(
        {"SOC": SOC_replay, "I_bat": I_bat_replay}, optimized_initial_soc
    )

    replay_errors = {
        "solar_current_A": float(np.max(np.abs(Isetpt_replay - Isetpt_amigo))),
        "solar_voltage_V": float(np.max(np.abs(V_sol_replay - V_sol_amigo))),
        "solar_power_W": float(np.max(np.abs(P_sol_replay - P_sol_amigo))),
        "current_A": float(np.max(np.abs(I_bat_replay - I_bat_variable))),
        "initial_SOC": float(abs(SOC_amigo[0] - optimized_initial_soc)),
        "P_comm_W": float(np.max(np.abs(P_replay - P_comm_amigo))),
        "temperature_K": float(np.max(np.abs(temperature_replay - temperature_amigo))),
        "P_bat_W": float(np.max(np.abs(P_bat_replay - P_bat_amigo))),
        "SOC": float(np.max(np.abs(SOC_replay - SOC_amigo))),
        "Dr": float(np.max(np.abs(Dr_replay - Dr_amigo))),
        "Data": float(np.max(np.abs(Data_replay - Data_amigo))),
    }

    replay_errors["roll_angle_rad"] = float(np.max(np.abs(gamma_replay - gamma_amigo)))

    if design_physics:
        replay_errors["antenna_gain"] = float(
            np.max(np.abs(gain_replay - np.asarray(x["comm_rate.gain[:]"])))
        )
        replay_errors["angular_velocity"] = float(
            np.max(
                np.abs(angular_replay - extract_vector(x, "rw_dynamics.w_B", n, 3).T)
            )
        )
        replay_errors["attitude_torque"] = float(
            np.max(np.abs(torque_replay - extract_vector(x, "rw_torque.T_tot", n, 3).T))
        )
        replay_errors["reaction_wheel_power_W"] = float(
            np.max(
                np.abs(wheel_power_replay - extract_vector(x, "rw_power.P_RW", n, 3).T)
            )
        )

    mapped_domain_violation = 0.0

    if coupled_voltage and not design_physics:
        for variable in ("voltage.tT", "voltage.tI"):
            mapped = np.asarray(x[variable + "[:]"])
            mapped_domain_violation = max(
                mapped_domain_violation,
                float(-np.min(mapped)),
                float(np.max(mapped) - 1.0),
            )

    last = opt_data["iterations"][-1]
    bound_violation = max(
        0.0,
        float(-np.min(CP_P_comm_amigo)),
        float(np.max(CP_P_comm_amigo) - 25.0),
        optimized_metrics["inequality_violation"],
        mapped_domain_violation,
        float(-np.min(CP_Isetpt_amigo)) if free_solar else 0.0,
        float(np.max(CP_Isetpt_amigo) - 0.4) if free_solar else 0.0,
        float(-np.min(CP_gamma_amigo)) if args.optimize_roll else 0.0,
        float(np.max(CP_gamma_amigo) - np.pi / 2) if args.optimize_roll else 0.0,
        float(-np.min(cell_amigo)),
        float(np.max(cell_amigo) - 1.0),
        float(50.0 - np.min(temperature_replay)) if design_physics else 0.0,
        float(np.max(temperature_replay) - 400.0) if design_physics else 0.0,
    )

    validated = bool(
        opt_data.get("converged", False)
        and np.isfinite(last["nlp_error"])
        and last["nlp_error"] < 1e-6
        and last["inf_pr"] < 1e-6
        and bound_violation < 1e-7
        and abs(optimized_metrics["periodicity_error"]) < 1e-7
        and abs(baseline_metrics["periodicity_error"]) < 1e-7
        and baseline_metrics["inequality_violation"] < 1e-7
        and all(np.isfinite(v) and v < 1e-5 for v in replay_errors.values())
        and Data_replay[-1] > Data_baseline[-1]
    )

    summary = {
        "validated": validated,
        "converged": bool(opt_data.get("converged", False)),
        "acceptable": bool(opt_data.get("acceptable", False)),
        "nodes": n,
        "control_points": m,
        "duration_hours": window_duration / 3600.0,
        "dt_s": dt,
        "contact_nodes": int(np.count_nonzero(CommLOS)),
        "baseline_data": float(Data_baseline[-1]),
        "baseline_type": "Feasible repair minimizing L1 change from benchmark controls; separate interior starting guess",
        "baseline_feasibility": baseline_metrics,
        "original_benchmark_data": original_data,
        "original_benchmark_feasibility": original_metrics,
        "periodicity_error": optimized_metrics["periodicity_error"],
        "current_min_A": optimized_metrics["current_min_A"],
        "current_max_A": optimized_metrics["current_max_A"],
        "initial_soc_optimized": not args.fixed_initial_soc,
        "solar_currents_optimized": free_solar,
        "solar_voltage_coupled": coupled_voltage,
        "roll_optimized": args.optimize_roll,
        "allocation_optimized": args.optimize_allocation,
        "design_physics_restored": design_physics,
        "allocation_min": float(np.min(cell_amigo)),
        "allocation_max": float(np.max(cell_amigo)),
        "roll_min_rad": float(np.min(gamma_replay)),
        "roll_max_rad": float(np.max(gamma_replay)),
        "design_variable_count": m * (13 if free_solar else 1)
        + int(not args.fixed_initial_soc)
        + m * int(args.optimize_roll)
        + 84 * int(args.optimize_allocation),
        "baseline_SOC_initial": initial_soc,
        "SOC_initial": optimized_initial_soc,
        "solver_time_s": float(last["time"]),
        "linear_solver": args.linear_solver,
        "objective_evaluator": "ConsistentEvaluator",
        "warm_start": str(args.warm_start) if args.warm_start else None,
        "solver_time_scope": (
            "warm-start segment" if args.warm_start else "cold solver phase"
        ),
        "initial_barrier_param": result.initial_barrier_param,
        "optimized_data": float(Data_replay[-1]),
        "improvement_percent": (
            float(100 * (Data_replay[-1] / Data_baseline[-1] - 1))
            if Data_baseline[-1] > 0
            else None
        ),
        "data_units": "CADRE benchmark convention (legacy label: Gibyte)",
        "SOC_min": float(np.min(SOC_replay)),
        "SOC_max": float(np.max(SOC_replay)),
        "SOC_final": float(SOC_replay[-1]),
        "iterations": int(last["iteration"]),
        "nlp_error": float(last["nlp_error"]),
        "primal_infeasibility": float(last["inf_pr"]),
        "mapped_voltage_domain_violation": mapped_domain_violation,
        "bound_violation": bound_violation,
        "reference_max_errors": replay_errors,
        "scope": "P_comm, optional initial SOC and 12 solar-current profiles; voltage responds to temperature and currents unless --legacy-fixed-voltage; optional roll and cell/radiator allocation; orbit, fin and antenna angles fixed; battery limits and SOC periodicity enforced",
    }

    args.output.mkdir(parents=True, exist_ok=True)

    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    (args.output / "solver_history.json").write_text(
        json.dumps(opt_data["iterations"], indent=2) + "\n"
    )

    np.savez_compressed(
        args.output / "trajectories.npz",
        time_s=np.arange(n) * dt,
        CP_P_comm=CP_P_comm_amigo,
        P_comm=P_comm_amigo,
        CP_gamma=CP_gamma_amigo,
        gamma=gamma_amigo,
        cellInstd=cell_amigo,
        baseline_gamma=gamma_baseline,
        baseline_cellInstd=cellInstd,
        exposed_area=area_replay,
        gain=gain_replay,
        P_RW=wheel_power_replay,
        CP_Isetpt=CP_Isetpt_amigo,
        Isetpt=Isetpt_amigo,
        V_sol=V_sol_amigo,
        P_sol=P_sol_amigo,
        baseline_Isetpt=Isetpt_baseline,
        baseline_V_sol=V_sol_baseline,
        baseline_P_sol=baseline_state["P_sol"],
        temperature=temperature_amigo,
        SOC=SOC_amigo,
        P_bat=P_bat_amigo,
        Dr=Dr_amigo,
        Data=Data_amigo,
        I_bat=I_bat_amigo,
        CommLOS=CommLOS,
        baseline_P_comm=P_comm_baseline,
        baseline_SOC=SOC_baseline,
        baseline_Data=Data_baseline,
        baseline_temperature=temperature_baseline,
        baseline_I_bat=baseline_state["I_bat"],
        baseline_CP_P_comm=CP_P_comm_baseline,
        original_P_comm=original_P_comm,
        original_SOC=original_SOC,
        original_Data=original_Data,
        replay_SOC=SOC_replay,
        replay_Data=Data_replay,
        replay_I_bat=I_bat_replay,
    )

    diagnostics = {
        "initial_guess_data": Data_guess[-1],
        "optimized_data": Data_amigo[-1],
        "data_gain": Data_amigo[-1] - Data_baseline[-1],
        "improvement_percent": (
            100.0 * (Data_amigo[-1] / Data_baseline[-1] - 1.0)
            if Data_baseline[-1] > 0
            else float("nan")
        ),
        "control_change": np.max(np.abs(CP_P_comm_amigo - CP_P_comm)),
        "control_range": (np.min(CP_P_comm_amigo), np.max(CP_P_comm_amigo)),
        "power_range": (np.min(P_comm_amigo), np.max(P_comm_amigo)),
        "endpoint_soc": (SOC_amigo[0], SOC_amigo[-1]),
        "original_data": data["0:Data"].reshape(-1)[n - 1],
        "battery_metrics": optimized_metrics,
    }

    print_results(summary, diagnostics, args.output.resolve())

    if not validated:
        raise RuntimeError(
            "Optimization failed convergence or independent validation; inspect saved diagnostics"
        )

    return summary
