import pickle

import numpy as np

from cadre_paths import cadre_path
from communication.mbi_2d import ModernMBI2D


def cadre_modulo(a, p):
    return a - int(a / p) * p


def fixangles(azimuth0, elevation0):
    n = len(azimuth0)

    azimuth = np.zeros(n)
    elevation = np.zeros(n)

    for i in range(n):
        azimuth[i] = cadre_modulo(
            azimuth0[i],
            2.0 * np.pi,
        )

        elevation[i] = cadre_modulo(
            elevation0[i],
            2.0 * np.pi,
        )

        if elevation[i] > np.pi:
            elevation[i] = (
                2.0 * np.pi
                - elevation[i]
            )

            azimuth[i] = (
                np.pi
                + azimuth[i]
            )

            azimuth[i] = cadre_modulo(
                azimuth[i],
                2.0 * np.pi,
            )

    return azimuth, elevation


def load_gain_data():
    path = cadre_path(
        "data/Comm/Gain.txt"
    )

    raw_gain = np.genfromtxt(
        path
    )

    gain_data = (
        10.0 ** (raw_gain / 10.0)
    ).reshape(
        (361, 361),
        order="F",
    )

    azimuth_grid = np.linspace(
        0.0,
        2.0 * np.pi,
        361,
    )

    elevation_grid = np.linspace(
        0.0,
        2.0 * np.pi,
        361,
    )

    return (
        azimuth_grid,
        elevation_grid,
        gain_data,
    )


def main():
    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(data_path, "rb") as f:
        data = pickle.load(
            f,
            encoding="latin1",
        )

    azimuth_source = data[
        "0:azimuthGS"
    ]

    elevation_source = data[
        "0:elevationGS"
    ]

    gain_source = data[
        "0:gain"
    ]

    azimuth, elevation = fixangles(
        azimuth_source,
        elevation_source,
    )

    (
        azimuth_grid,
        elevation_grid,
        gain_data,
    ) = load_gain_data()

    print()
    print("=" * 70)
    print("CADRE COMMUNICATION GAIN MBI VALIDATION")
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(len(azimuth))

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

    print()
    print("Evaluating antenna gain...")

    gain_test = mbi.evaluate(
        points
    )

    difference = np.abs(
        gain_test - gain_source
    )

    print()
    print("=" * 70)
    print("COMMUNICATION GAIN RESULTS")
    print("=" * 70)

    print()
    print("Source gain minimum:")
    print(np.min(gain_source))

    print()
    print("Modern MBI gain minimum:")
    print(np.min(gain_test))

    print()
    print("Source gain maximum:")
    print(np.max(gain_source))

    print()
    print("Modern MBI gain maximum:")
    print(np.max(gain_test))

    print()
    print("Maximum gain difference:")
    print(np.max(difference))

    print()
    print("Mean gain difference:")
    print(np.mean(difference))

    print()
    print("Maximum source gain index:")
    print(np.argmax(gain_source))

    print()
    print("Maximum test gain index:")
    print(np.argmax(gain_test))


if __name__ == "__main__":
    main()