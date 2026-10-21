"""Independent joint-design replay through the coupled CADRE disciplines."""

import numpy as np

from battery.battery_reference import battery_current, propagate_soc_trapezoid
from communication.communication_amigo import propagate_data_trapezoid
from communication.communication_data_validation import communication_bit_rate
from integrated.reference.feasibility import feasibility_metrics
from power.power_amigo import solar_power, total_power
from reaction_wheel.reaction_wheel_reference import (
    propagate_reaction_wheel_trapezoid,
    reaction_wheel_power,
    reaction_wheel_torque,
)
from thermal.thermal_amigo import propagate_temperature_trapezoid


def replay_joint(joint, state):
    model = joint.model
    vector = state.get_vector()
    vector.copy_device_to_host()
    point = vector.get_array()
    allocation = point[model.get_indices("shared_surface.cellInstd[0, :]")].reshape(
        7, 12
    )
    fin = float(point[model.get_indices("shared_angles.finAngle[:]")][0])
    antenna = float(point[model.get_indices("shared_angles.antAngle[:]")][0])
    reports = []
    total = 0.0

    for case in joint.cases:
        prefix = f"case{case.case_index}."

        def values(name):
            return point[model.get_indices(prefix + name)]

        power = case.B @ values("control_points.CP_P_comm[:]")
        roll = case.B @ values("control_points.CP_gamma[:]")
        currents = (case.B @ values("control_points.CP_Isetpt[:, :]")).T
        previous_angles = case.lookups.fin_angle, case.lookups.antenna_angle

        try:
            case.lookups.set_angles(fin, antenna)
            area, gain, angular, torque = case.lookups.geometry(roll, reference=True)[
                :4
            ]
        finally:
            case.lookups.set_angles(*previous_angles)

        wheel_torque = reaction_wheel_torque(torque)
        wheel, _ = propagate_reaction_wheel_trapezoid(
            case.w_RW0, angular, wheel_torque, case.dt
        )
        wheel_power = reaction_wheel_power(wheel, wheel_torque)
        temperature, _ = propagate_temperature_trapezoid(
            case.T0, area, case.LOS, power, allocation, case.dt
        )
        voltage = case.voltage_spline.reference_panels(
            temperature, currents, area, case.LOS
        )
        solar = solar_power(voltage, currents)
        net_power = total_power(solar, power, wheel_power)
        initial_soc = float(values("battery_ic.initial_soc[:]")[0])
        soc = propagate_soc_trapezoid(initial_soc, net_power, temperature, case.dt)
        current = battery_current(net_power, soc, temperature[4])
        rate = communication_bit_rate(power, gain, case.GSdist, case.CommLOS)
        downloaded, _ = propagate_data_trapezoid(0, rate, case.dt)
        pairs = {
            "roll_rad": (roll, values("values.Gamma[:]")),
            "communication_power_W": (power, values("values.P_comm[:]")),
            "solar_current_A": (currents, values("values.Isetpt[:, :]").T),
            "temperature_K": (temperature, values("thermal.temperature[:, :]").T),
            "voltage_V": (voltage, values("power_solar.V_sol[:, :]").T),
            "solar_power_W": (solar, values("power_solar.P_sol[:]")),
            "battery_power_W": (net_power, values("power_total.P_bat[:]")),
            "SOC": (soc, values("battery.SOC[:]")),
            "current_A": (current, values("battery.I_bat[:]")),
            "gain": (gain, values("comm_rate.gain[:]")),
            "download_rate": (rate, values("comm_rate.Dr[:]")),
            "downloaded_data": (downloaded, values("comm_data.Data[:]")),
        }
        errors = {
            name: float(np.max(np.abs(actual - expected)))
            for name, (actual, expected) in pairs.items()
        }

        if not all(np.isfinite(error) for error in errors.values()):
            raise FloatingPointError(f"Nonfinite replay for case {case.case_index}")

        feasibility = feasibility_metrics({"SOC": soc, "I_bat": current}, initial_soc)
        reports.append(
            {
                "case_index": case.case_index,
                "downloaded_data": float(downloaded[-1]),
                "reference_max_errors": errors,
                "battery_feasibility": feasibility,
            }
        )
        total += float(downloaded[-1])

    return {
        "total_downloaded_data": total,
        "fin_angle_rad": fin,
        "antenna_angle_rad": antenna,
        "cases": reports,
    }
