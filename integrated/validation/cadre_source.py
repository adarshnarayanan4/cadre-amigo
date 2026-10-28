"""Compare reference equations with numerical methods read from OpenMDAO/CADRE.

This checks forward calculations, not OpenMDAO's driver or total derivatives.
Lookup and AMIGO derivative checks remain separate commands.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from attitude import attitude_reference as attitude
from battery.battery_reference import battery_current, soc_rate
from cadre_paths import source_provenance
from cadre_source import source_class
from communication import communication_reference as communication
from communication.communication_data_validation import communication_bit_rate
from integrated.reference.benchmark import load_benchmark, select_case
from orbit.orbit_reference import orbit_dynamics
from power.power_amigo import solar_power, total_power
from reaction_wheel.reaction_wheel_reference import (
    reaction_wheel_dynamics,
    reaction_wheel_power,
    reaction_wheel_torque,
)
from sun import sun_reference as sun
from thermal.thermal_reference import thermal_dynamics


def check_case(data, case_index):
    selected = select_case(data, case_index)
    selected["Gamma"] = selected["gamma"]
    n = len(selected["P_comm"])
    dt = 43200 / (n - 1)
    errors = {}

    def compare(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        np.testing.assert_allclose(
            actual, expected, rtol=1e-10, atol=1e-10, err_msg=name
        )
        errors[name] = float(np.max(np.abs(actual - expected)))

    def compute(module, name, expected, *arguments):
        component = source_class(module, name)(*arguments)
        outputs = {key: np.zeros_like(value) for key, value in expected.items()}
        component.compute(selected, outputs)

        for key, value in expected.items():
            compare(f"{name}.{key}", outputs[key], value)

    orientation = selected["O_BI"].transpose(2, 0, 1)
    orientation_rate = selected["Odot_BI"].transpose(2, 0, 1)
    states = selected["r_e2b_I"].T
    compute(
        "attitude",
        "Attitude_Attitude",
        {"O_RI": attitude.attitude_from_orbit(states).transpose(1, 2, 0)},
        n,
    )
    compute(
        "attitude",
        "Attitude_Roll",
        {"O_BR": attitude.attitude_roll(selected["gamma"]).transpose(1, 2, 0)},
        n,
    )
    compute(
        "attitude",
        "Attitude_RotationMtx",
        {
            "O_BI": attitude.combine_rotation_matrices(
                selected["O_BR"].transpose(2, 0, 1), selected["O_RI"].transpose(2, 0, 1)
            ).transpose(1, 2, 0)
        },
        n,
    )
    compute(
        "attitude",
        "Attitude_RotationMtxRates",
        {"Odot_BI": attitude.rotation_matrix_rates(orientation, dt).transpose(1, 2, 0)},
        n,
        dt,
    )
    compute(
        "attitude",
        "Attitude_Angular",
        {"w_B": attitude.angular_velocity(orientation, orientation_rate).T},
        n,
    )
    compute(
        "attitude",
        "Attitude_AngularRates",
        {"wdot_B": attitude.angular_acceleration(selected["w_B"].T, dt).T},
        n,
        dt,
    )
    compute(
        "attitude",
        "Attitude_Torque",
        {"T_tot": attitude.attitude_torque(selected["w_B"].T, selected["wdot_B"].T).T},
        n,
    )

    compute(
        "sun",
        "Sun_PositionECI",
        {"r_e2s_I": sun.sun_position_eci(selected["t"], float(selected["LD"][0])).T},
        n,
    )
    compute(
        "sun",
        "Sun_LOS",
        {"LOS": sun.sun_line_of_sight(states, selected["r_e2s_I"].T)},
        n,
    )
    compute(
        "sun",
        "Sun_PositionBody",
        {"r_e2s_B": sun.sun_position_body(orientation, selected["r_e2s_I"].T).T},
        n,
    )
    azimuth, elevation = sun.sun_position_spherical(selected["r_e2s_B"].T)
    compute(
        "sun", "Sun_PositionSpherical", {"azimuth": azimuth, "elevation": elevation}, n
    )

    compute(
        "comm",
        "Comm_AntRotation",
        {
            "q_A": communication.antenna_rotation(
                float(np.asarray(selected["antAngle"]).item()), n
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_AntRotationMtx",
        {"O_AB": communication.quaternion_rotation_matrix(selected["q_A"])},
        n,
    )
    compute(
        "comm", "Comm_EarthsSpin", {"q_E": communication.earth_spin(selected["t"])}, n
    )
    compute(
        "comm",
        "Comm_EarthsSpinMtx",
        {"O_IE": communication.quaternion_rotation_matrix(selected["q_E"])},
        n,
    )
    compute(
        "comm",
        "Comm_GSposEarth",
        {
            "r_e2g_E": communication.ground_station_position_earth(
                selected["lon"], selected["lat"], selected["alt"], n
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_GSposECI",
        {
            "r_e2g_I": communication.rotate_vectors(
                selected["O_IE"], selected["r_e2g_E"]
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_VectorECI",
        {"r_b2g_I": communication.vector_eci(selected["r_e2g_I"], selected["r_e2b_I"])},
        n,
    )
    compute(
        "comm",
        "Comm_VectorBody",
        {
            "r_b2g_B": communication.rotate_vectors(
                selected["O_BI"], selected["r_b2g_I"]
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_VectorAnt",
        {
            "r_b2g_A": communication.rotate_vectors(
                selected["O_AB"], selected["r_b2g_B"]
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_Distance",
        {"GSdist": communication.ground_station_distance(selected["r_b2g_A"])},
        n,
    )
    compute(
        "comm",
        "Comm_LOS",
        {
            "CommLOS": communication.communication_los(
                selected["r_b2g_I"], selected["r_e2g_I"]
            )
        },
        n,
    )
    compute(
        "comm",
        "Comm_BitRate",
        {
            "Dr": communication_bit_rate(
                selected["P_comm"],
                selected["gain"],
                selected["GSdist"],
                selected["CommLOS"],
            )
        },
        n,
    )

    compute(
        "power",
        "Power_SolarPower",
        {"P_sol": solar_power(selected["V_sol"], selected["Isetpt"])},
        n,
    )
    compute(
        "power",
        "Power_Total",
        {"P_bat": total_power(selected["P_sol"], selected["P_comm"], selected["P_RW"])},
        n,
    )
    compute(
        "battery",
        "BatteryPower",
        {
            "I_bat": battery_current(
                selected["P_bat"], selected["SOC"][0], selected["temperature"][4]
            )
        },
        n,
    )
    compute(
        "reactionwheel",
        "ReactionWheel_Torque",
        {"T_RW": reaction_wheel_torque(selected["T_tot"])},
        n,
    )
    compute(
        "reactionwheel",
        "ReactionWheel_Power",
        {"P_RW": reaction_wheel_power(selected["w_RW"], selected["T_RW"])},
        n,
    )

    dynamics = source_class("orbit", "Orbit_Dynamics")(n, dt)
    thermal = source_class("thermal_temperature", "ThermalTemperature")(n, dt)
    battery = source_class("battery", "BatterySOC")(n, dt)
    wheel = source_class("reactionwheel", "ReactionWheel_Dynamics")(n, dt)
    wheel.jy = np.zeros((3, 3))  # Normally allocated by OpenMDAO setup.

    for node in np.linspace(0, n - 1, 17, dtype=int):
        compare(
            f"orbit_rate_{node}",
            dynamics.f_dot(None, states[node]),
            orbit_dynamics(states[node]),
        )
        external = np.r_[
            selected["exposedArea"][:, :, node].ravel(),
            selected["LOS"][node],
            selected["P_comm"][node],
            selected["cellInstd"].ravel(),
        ]
        compare(
            f"temperature_rate_{node}",
            thermal.f_dot(external, selected["temperature"][:, node]),
            thermal_dynamics(
                selected["temperature"][:, node],
                selected["exposedArea"][:, :, node],
                selected["LOS"][node],
                selected["P_comm"][node],
                selected["cellInstd"],
            ),
        )
        external = np.r_[selected["P_bat"][node], selected["temperature"][:, node]]
        compare(
            f"soc_rate_{node}",
            battery.f_dot(external, selected["SOC"][:, node]),
            soc_rate(
                selected["SOC"][0, node],
                selected["P_bat"][node],
                selected["temperature"][4, node],
            ),
        )
        external = np.r_[selected["w_B"][:, node], selected["T_RW"][:, node]]
        compare(
            f"wheel_rate_{node}",
            wheel.f_dot(external, selected["w_RW"][:, node]),
            reaction_wheel_dynamics(
                selected["w_RW"][:, node],
                selected["w_B"][:, node],
                selected["T_RW"][:, node],
            ),
        )

    return {"nodes": n, "checks": len(errors), "max_absolute_errors": errors}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("integrated/results/validation/cadre_source_validation.json"),
    )
    args = parser.parse_args(argv)
    provenance = source_provenance()
    data = load_benchmark()
    cases = {}

    for index in range(6):
        cases[str(index)] = check_case(data, index)
        print(f"OpenMDAO/CADRE forward equations passed for case {index}", flush=True)

    report = {
        "cadre_source": provenance,
        "scope": "Actual source forward methods; no OpenMDAO framework, total derivatives, or optimization",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
