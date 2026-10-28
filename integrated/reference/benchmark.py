"""Read the original six-case benchmark without running a model or optimizer."""

import pickle

import numpy as np

from cadre_paths import cadre_path, source_provenance


def load_benchmark():
    source_provenance()
    with open(cadre_path("test/data1346.pkl"), "rb") as source:
        return pickle.load(source, encoding="latin1")


def select_case(data, case_index=0):
    """Combine shared inputs with one case's fields, without changing the source."""
    prefix = f"{case_index}:"
    selected = {
        key[len(prefix) :]: value
        for key, value in data.items()
        if key.startswith(prefix)
    }

    if not selected:
        raise ValueError(f"Benchmark case {case_index} does not exist")

    shared = {key: value for key, value in data.items() if ":" not in key}
    return shared | selected


def resample_benchmark(data, nodes):
    """Sample fixed benchmark geometry over all 12 hours on a smaller grid.

    Spline controls and shared designs retain their source values. This is a
    coarse discretization for optimization checks, not a new reference solution.
    """
    from attitude.attitude_reference import attitude_from_orbit

    original_nodes = len(data["0:P_comm"])
    if not 4 <= nodes <= original_nodes:
        raise ValueError("Require 4 <= nodes <= source time nodes")
    if nodes == original_nodes:
        return data

    original_time = np.linspace(0, 43200, original_nodes)
    time = np.linspace(0, 43200, nodes)
    sampled = {}

    for name, values in data.items():
        values = np.asarray(values)
        if values.ndim and values.shape[-1] == original_nodes:
            rows = values.reshape(-1, original_nodes)
            sampled[name] = np.array(
                [np.interp(time, original_time, row) for row in rows]
            ).reshape(values.shape[:-1] + (nodes,))
        else:
            sampled[name] = values.copy()

    sampled["t"] = time
    sampled["source_time_nodes"] = original_nodes
    for index in range(6):
        # Interpolating rotation-matrix entries would lose orthogonality.
        states = sampled[f"{index}:r_e2b_I"].T
        sampled[f"{index}:O_RI"] = attitude_from_orbit(states).transpose(1, 2, 0)

    return sampled
