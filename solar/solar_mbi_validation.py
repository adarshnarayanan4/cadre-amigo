import pickle

import numpy as np

from cadre_paths import cadre_path
from power.mbi_modern import ModernMBI
from solar.solar_reference import (
    fixangles,
    load_solar_data,
)


def build_solar_mbi():
    angle, azimuth, elevation, data = load_solar_data()

    na = len(angle)
    nz = len(azimuth)
    ne = len(elevation)

    data_mbi = data.reshape(
        (na, nz, ne, 84),
        order="F",
    )

    mbi = ModernMBI(
        data_mbi,
        [angle, azimuth, elevation],
        [4, 10, 8],
        [4, 4, 4],
    )

    return mbi


def solar_exposed_area_mbi(fin_angle, azimuth, elevation, mbi):
    azimuth, elevation = fixangles(
        azimuth.copy(),
        elevation.copy(),
    )

    num_nodes = len(azimuth)

    points = np.zeros((num_nodes, 3))

    points[:, 0] = fin_angle
    points[:, 1] = azimuth
    points[:, 2] = elevation

    values = mbi.evaluate(points)

    exposed_area = values.T.reshape(
        (7, 12, num_nodes),
        order="F",
    )

    return exposed_area


def main():
    print()
    print("=" * 70)
    print("CADRE SOLAR MBI VALIDATION")
    print("=" * 70)

    # Load original CADRE benchmark
    with open(cadre_path("test/data1346.pkl"), "rb") as f:
        data = pickle.load(f, encoding="latin1")

    fin_angle = float(
        np.asarray(data["finAngle"]).reshape(-1)[0]
    )

    azimuth = np.asarray(data["0:azimuth"])
    elevation = np.asarray(data["0:elevation"])
    exposed_area_source = np.asarray(data["0:exposedArea"])

    num_nodes = len(azimuth)

    print()
    print("Number of nodes:")
    print(num_nodes)

    print()
    print("Fin angle [rad]:")
    print(fin_angle)

    print()
    print("Building solar MBI...")

    mbi = build_solar_mbi()

    print()
    print("Evaluating solar exposed area...")

    exposed_area_mbi = solar_exposed_area_mbi(
        fin_angle,
        azimuth,
        elevation,
        mbi,
    )

    difference = exposed_area_mbi - exposed_area_source

    print()
    print("=" * 70)
    print("SOLAR MBI RESULTS")
    print("=" * 70)

    print()
    print("Source exposed-area shape:")
    print(exposed_area_source.shape)

    print()
    print("Modern MBI exposed-area shape:")
    print(exposed_area_mbi.shape)

    print()
    print("Source minimum exposed area:")
    print(np.min(exposed_area_source))

    print()
    print("Modern MBI minimum exposed area:")
    print(np.min(exposed_area_mbi))

    print()
    print("Source maximum exposed area:")
    print(np.max(exposed_area_source))

    print()
    print("Modern MBI maximum exposed area:")
    print(np.max(exposed_area_mbi))

    print()
    print("Maximum exposed-area difference [m^2]:")
    print(np.max(np.abs(difference)))

    print()
    print("Mean exposed-area difference [m^2]:")
    print(np.mean(np.abs(difference)))

    print()
    print("Maximum relative difference:")
    denominator = np.maximum(
        np.abs(exposed_area_source),
        1.0e-12,
    )
    print(np.max(np.abs(difference) / denominator))


if __name__ == "__main__":
    main()