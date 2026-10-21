"""Read the original six-case benchmark without running a model or optimizer."""

import pickle

from cadre_paths import cadre_path


def load_benchmark():
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
