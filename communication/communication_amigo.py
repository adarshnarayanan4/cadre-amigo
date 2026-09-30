import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path
from communication.communication_data_validation import (
    ALPHA,
    communication_bit_rate,
)
from communication.communication_gain_validation import (
    ModernMBI2D,
    fixangles,
    load_gain_data,
)


class CommunicationBitRate(am.Component):
    def __init__(self):
        super().__init__()

        self.add_constant(
            "alpha",
            value=ALPHA,
        )

        self.add_input("P_comm")
        self.add_input("gain")
        self.add_input("GSdist")
        self.add_input("CommLOS")
        self.add_input("Dr")

        self.add_constraint("res")

    def compute(self):
        alpha = self.constants["alpha"]

        P_comm = self.inputs["P_comm"]
        gain = self.inputs["gain"]
        GSdist = self.inputs["GSdist"]
        CommLOS = self.inputs["CommLOS"]
        Dr = self.inputs["Dr"]

        distance = GSdist * 1.0e3

        rate = (
            alpha
            * P_comm
            * gain
            * CommLOS
            / distance**2
        )

        self.constraints["res"] = (
            Dr - rate
        )


class DataDynamics(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Dr")
        self.add_input("Data")
        self.add_input("Data_dot")

        self.add_constraint("res")

    def compute(self):
        Dr = self.inputs["Dr"]
        Data_dot = self.inputs["Data_dot"]

        self.constraints["res"] = (
            Data_dot - Dr
        )


class TrapezoidRule(am.Component):
    def __init__(self, dt):
        super().__init__()

        self.add_constant(
            "dt",
            value=dt,
        )

        self.add_input("Data0")
        self.add_input("Data1")

        self.add_input("Data_dot0")
        self.add_input("Data_dot1")

        self.add_constraint("res")

    def compute(self):
        dt = self.constants["dt"]

        Data0 = self.inputs["Data0"]
        Data1 = self.inputs["Data1"]

        Data_dot0 = self.inputs["Data_dot0"]
        Data_dot1 = self.inputs["Data_dot1"]

        self.constraints["res"] = (
            Data1
            - Data0
            - 0.5
            * dt
            * (
                Data_dot0
                + Data_dot1
            )
        )


class InitialCondition(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("Data")

        self.add_constraint("res")
        self.add_objective("obj")

    def compute(self):
        Data = self.inputs["Data"]

        self.constraints["res"] = Data

        self.objective["obj"] = (
            1.0e-12 * Data * Data
        )


def propagate_data_trapezoid(
    Data0,
    Dr,
    dt,
):
    n = len(Dr)

    Data = np.zeros(n)
    Data_dot = np.zeros(n)

    Data[0] = Data0

    for k in range(n):
        Data_dot[k] = Dr[k]

    for k in range(n - 1):
        Data[k + 1] = (
            Data[k]
            + 0.5
            * dt
            * (
                Data_dot[k]
                + Data_dot[k + 1]
            )
        )

    return Data, Data_dot


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(data_path, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1",
        )

    P_comm = data["0:P_comm"]
    GSdist = data["0:GSdist"]
    CommLOS = data["0:CommLOS"]

    azimuthGS = data["0:azimuthGS"]
    elevationGS = data["0:elevationGS"]

    Dr_source = data["0:Dr"]
    Data_source = data["0:Data"].reshape(-1)

    n = len(P_comm)
    num_intervals = n - 1

    dt = 43200.0 / num_intervals

    print()
    print("=" * 70)
    print("CADRE COMMUNICATION AMIGO")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    # Build communication gain MBI
    azimuth, elevation = fixangles(
        azimuthGS,
        elevationGS,
    )

    (
        azimuth_grid,
        elevation_grid,
        gain_data,
    ) = load_gain_data()

    print()
    print("Building communication gain MBI...")

    mbi = ModernMBI2D(
        gain_data,
        [
            azimuth_grid,
            elevation_grid,
        ],
        [
            15,
            15,
        ],
        [
            4,
            4,
        ],
    )

    points = np.column_stack(
        (
            azimuth,
            elevation,
        )
    )

    gain = mbi.evaluate(
        points
    )

    # Python bitrate
    Dr_reference = communication_bit_rate(
        P_comm,
        gain,
        GSdist,
        CommLOS,
    )

    # Python trapezoid reference
    Data_trap, Data_dot_trap = (
        propagate_data_trapezoid(
            0.0,
            Dr_reference,
            dt,
        )
    )

    print()
    print("=" * 70)
    print("PYTHON PRECHECK")
    print("=" * 70)

    print()
    print("Maximum bitrate difference from CADRE [Gibyte/s]:")
    print(
        np.max(
            np.abs(
                Dr_reference - Dr_source
            )
        )
    )

    print()
    print("CADRE RK4 vs trapezoid maximum Data difference [Gibyte]:")
    print(
        np.max(
            np.abs(
                Data_trap - Data_source
            )
        )
    )

    max_trapezoid_residual = 0.0

    for k in range(num_intervals):
        residual = (
            Data_trap[k + 1]
            - Data_trap[k]
            - 0.5
            * dt
            * (
                Data_dot_trap[k]
                + Data_dot_trap[k + 1]
            )
        )

        max_trapezoid_residual = max(
            max_trapezoid_residual,
            abs(residual),
        )

    print()
    print("Python trapezoid residual:")
    print(
        max_trapezoid_residual
    )

    # Components
    bitrate = CommunicationBitRate()
    dynamics = DataDynamics()
    trapezoid = TrapezoidRule(dt)
    initial = InitialCondition()

    # Model
    model = am.Model(
        "cadre_communication"
    )

    model.add_component(
        "bitrate",
        n,
        bitrate,
    )

    model.add_component(
        "dynamics",
        n,
        dynamics,
    )

    model.add_component(
        "trapezoid",
        num_intervals,
        trapezoid,
    )

    model.add_component(
        "initial",
        1,
        initial,
    )

    # Link bitrate to data derivative
    model.link(
        "bitrate.Dr",
        "dynamics.Dr",
    )

    # Trapezoid links
    model.link(
        "dynamics.Data[:-1]",
        "trapezoid.Data0",
    )

    model.link(
        "dynamics.Data[1:]",
        "trapezoid.Data1",
    )

    model.link(
        "dynamics.Data_dot[:-1]",
        "trapezoid.Data_dot0",
    )

    model.link(
        "dynamics.Data_dot[1:]",
        "trapezoid.Data_dot1",
    )

    # Initial condition
    model.link(
        "dynamics.Data[0]",
        "initial.Data[0]",
    )

    # Fix P_comm
    model.set_meta(
        "value",
        "bitrate.P_comm[:]",
        P_comm,
    )

    model.set_meta(
        "lower",
        "bitrate.P_comm[:]",
        P_comm,
    )

    model.set_meta(
        "upper",
        "bitrate.P_comm[:]",
        P_comm,
    )

    # Fix gain
    model.set_meta(
        "value",
        "bitrate.gain[:]",
        gain,
    )

    model.set_meta(
        "lower",
        "bitrate.gain[:]",
        gain,
    )

    model.set_meta(
        "upper",
        "bitrate.gain[:]",
        gain,
    )

    # Fix distance
    model.set_meta(
        "value",
        "bitrate.GSdist[:]",
        GSdist,
    )

    model.set_meta(
        "lower",
        "bitrate.GSdist[:]",
        GSdist,
    )

    model.set_meta(
        "upper",
        "bitrate.GSdist[:]",
        GSdist,
    )

    # Fix communication LOS
    model.set_meta(
        "value",
        "bitrate.CommLOS[:]",
        CommLOS,
    )

    model.set_meta(
        "lower",
        "bitrate.CommLOS[:]",
        CommLOS,
    )

    model.set_meta(
        "upper",
        "bitrate.CommLOS[:]",
        CommLOS,
    )

    # Initial bitrate guess
    model.set_meta(
        "value",
        "bitrate.Dr[:]",
        Dr_reference,
    )

    # Initial downloaded-data guess
    model.set_meta(
        "value",
        "dynamics.Data[:]",
        Data_trap,
    )

    model.set_meta(
        "value",
        "dynamics.Data_dot[:]",
        Data_dot_trap,
    )

    print()
    print("Building AMIGO communication model...")

    model.build_module()

    print("Build successful.")

    model.initialize()

    print("Initialization successful.")

    x = model.create_vector()

    opt = am.Optimizer(
        model,
        x,
    )

    opt_options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print("Solving AMIGO communication model...")

    opt.optimize(
        opt_options
    )

    print()
    print("AMIGO solve complete.")

    # Extract results
    Dr_amigo = np.array(
        x["bitrate.Dr[:]"]
    )

    Data_amigo = np.array(
        x["dynamics.Data[:]"]
    )

    Data_dot_amigo = np.array(
        x["dynamics.Data_dot[:]"]
    )

    print()
    print("=" * 70)
    print("COMMUNICATION AMIGO VALIDATION")
    print("=" * 70)

    print()
    print("AMIGO vs Python bitrate difference [Gibyte/s]:")
    print(
        np.max(
            np.abs(
                Dr_amigo - Dr_reference
            )
        )
    )

    print()
    print("AMIGO vs Python trapezoid Data difference [Gibyte]:")
    print(
        np.max(
            np.abs(
                Data_amigo - Data_trap
            )
        )
    )

    print()
    print("AMIGO vs Python Data-dot difference [Gibyte/s]:")
    print(
        np.max(
            np.abs(
                Data_dot_amigo - Data_dot_trap
            )
        )
    )

    max_amigo_trapezoid_residual = 0.0

    for k in range(num_intervals):
        residual = (
            Data_amigo[k + 1]
            - Data_amigo[k]
            - 0.5
            * dt
            * (
                Data_dot_amigo[k]
                + Data_dot_amigo[k + 1]
            )
        )

        max_amigo_trapezoid_residual = max(
            max_amigo_trapezoid_residual,
            abs(residual),
        )

    print()
    print("AMIGO trapezoid residual:")
    print(
        max_amigo_trapezoid_residual
    )

    print()
    print("=" * 70)
    print("AMIGO VS ORIGINAL CADRE")
    print("=" * 70)

    print()
    print("Maximum Data difference [Gibyte]:")
    print(
        np.max(
            np.abs(
                Data_amigo - Data_source
            )
        )
    )

    print()
    print("Final CADRE downloaded data [Gibyte]:")
    print(
        Data_source[-1]
    )

    print()
    print("Final AMIGO downloaded data [Gibyte]:")
    print(
        Data_amigo[-1]
    )


if __name__ == "__main__":
    main()