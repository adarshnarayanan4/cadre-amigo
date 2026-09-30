import pickle

import numpy as np

from cadre_paths import cadre_path
from communication.communication_gain_validation import (
    ModernMBI2D,
    fixangles,
    load_gain_data,
)


PI = np.pi
C = 299792458.0
GR = 10.0 ** (12.9 / 10.0)
LL = 10.0 ** (-2.0 / 10.0)
FREQ = 437.0e6
K = 1.3806503e-23
SNR = 10.0 ** (5.0 / 10.0)
T = 500.0

ALPHA = (
    C**2
    * GR
    * LL
    / 16.0
    / PI**2
    / FREQ**2
    / K
    / SNR
    / T
    / 1.0e6
)


def communication_bit_rate(
    P_comm,
    gain,
    GSdist,
    CommLOS,
):
    n = len(P_comm)

    Dr = np.zeros(n)

    for i in range(n):
        if abs(GSdist[i]) > 1.0e-10:
            distance = GSdist[i] * 1.0e3
        else:
            distance = 1.0e-10

        Dr[i] = (
            ALPHA
            * P_comm[i]
            * gain[i]
            * CommLOS[i]
            / distance**2
        )

    return Dr


def propagate_data_rk4(
    Data0,
    Dr,
    dt,
):
    n = len(Dr)

    Data = np.zeros(n)
    Data[0] = Data0

    for k in range(n - 1):
        a = Dr[k]
        b = Dr[k]
        c = Dr[k]
        d = Dr[k]

        Data[k + 1] = (
            Data[k]
            + dt
            / 6.0
            * (
                a
                + 2.0 * b
                + 2.0 * c
                + d
            )
        )

    return Data


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
    gain_source = data["0:gain"]
    GSdist = data["0:GSdist"]
    CommLOS = data["0:CommLOS"]

    azimuthGS = data["0:azimuthGS"]
    elevationGS = data["0:elevationGS"]

    Dr_source = data["0:Dr"]
    Data_source = data["0:Data"]

    n = len(P_comm)
    dt = 43200.0 / (n - 1)

    # Source-exact bitrate check
    Dr_reference = communication_bit_rate(
        P_comm,
        gain_source,
        GSdist,
        CommLOS,
    )

    # Source-exact downloaded-data check
    Data_reference = propagate_data_rk4(
        0.0,
        Dr_reference,
        dt,
    )

    # Modern gain MBI
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
    print("=" * 70)
    print("CADRE COMMUNICATION DATA VALIDATION")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

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

    gain_mbi = mbi.evaluate(
        points
    )

    # End-to-end using recreated MBI
    Dr_mbi = communication_bit_rate(
        P_comm,
        gain_mbi,
        GSdist,
        CommLOS,
    )

    Data_mbi = propagate_data_rk4(
        0.0,
        Dr_mbi,
        dt,
    )

    print()
    print("=" * 70)
    print("SOURCE FORMULA VALIDATION")
    print("=" * 70)

    print()
    print("Maximum bitrate difference [Gibyte/s]:")
    print(
        np.max(
            np.abs(
                Dr_reference - Dr_source
            )
        )
    )

    print()
    print("Mean bitrate difference [Gibyte/s]:")
    print(
        np.mean(
            np.abs(
                Dr_reference - Dr_source
            )
        )
    )

    print()
    print("Maximum downloaded-data difference [Gibyte]:")
    print(
        np.max(
            np.abs(
                Data_reference - Data_source.reshape(-1)
            )
        )
    )

    print()
    print("Final source downloaded data [Gibyte]:")
    print(
        Data_source.reshape(-1)[-1]
    )

    print()
    print("Final reference downloaded data [Gibyte]:")
    print(
        Data_reference[-1]
    )

    print()
    print("=" * 70)
    print("MODERN MBI END-TO-END")
    print("=" * 70)

    print()
    print("Maximum gain difference:")
    print(
        np.max(
            np.abs(
                gain_mbi - gain_source
            )
        )
    )

    print()
    print("Maximum bitrate difference [Gibyte/s]:")
    print(
        np.max(
            np.abs(
                Dr_mbi - Dr_source
            )
        )
    )

    print()
    print("Mean bitrate difference [Gibyte/s]:")
    print(
        np.mean(
            np.abs(
                Dr_mbi - Dr_source
            )
        )
    )

    print()
    print("Maximum downloaded-data difference [Gibyte]:")
    print(
        np.max(
            np.abs(
                Data_mbi - Data_source.reshape(-1)
            )
        )
    )

    print()
    print("Final modern-MBI downloaded data [Gibyte]:")
    print(
        Data_mbi[-1]
    )

    print()
    print("Source bitrate min / max [Gibyte/s]:")
    print(
        np.min(Dr_source),
        np.max(Dr_source),
    )

    print()
    print("Modern MBI bitrate min / max [Gibyte/s]:")
    print(
        np.min(Dr_mbi),
        np.max(Dr_mbi),
    )


if __name__ == "__main__":
    main()