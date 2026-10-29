"""Small NumPy-backed AMIGO vector/matrix adapters for solver checks."""

from functools import partial
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix


def saved_bound_response(saved):
    """Rebuild a captured reduced inverse without compiling a CADRE model."""
    from integrated.solvers.joint import JointKKTSolver, scaled_solve

    fields = ("scaling", "owner", "shared", "shared_scale", "shared_bound_positions")
    inverse = SimpleNamespace(**{name: saved[name] for name in fields})
    inverse.shared_factor = (saved["shared_factor"], bool(saved["shared_lower"]))
    inverse.shared_lookup = {int(index): i for i, index in enumerate(inverse.shared)}
    inverse.step_bounds = SimpleNamespace(indices=saved["bound_indices"])
    inverse.cases = []
    for i in range(int(saved["case_count"])):
        fields = (
            "controls",
            "bound_core",
            "bound_response",
            "cross",
            "scale",
            "control_bound_positions",
            "core_bound_positions",
        )
        case = SimpleNamespace(**{name: saved[f"case_{i}_{name}"] for name in fields})
        case.private_count = len(case.controls)
        factor = (
            saved[f"case_{i}_private_factor"],
            bool(saved[f"case_{i}_private_lower"]),
        )
        case.solve_private = partial(scaled_solve, factor, case.scale)
        case.factor = SimpleNamespace(fixed_diagonal=saved[f"case_{i}_fixed_diagonal"])
        for name, indices in (
            ("control_lookup", case.controls),
            ("bound_core_lookup", case.bound_core),
            ("fixed_lookup", saved[f"case_{i}_fixed_indices"]),
        ):
            setattr(case, name, {int(index): j for j, index in enumerate(indices)})
        inverse.cases.append(case)
    return partial(JointKKTSolver.bound_inverse_response, inverse)


class Vector:
    def __init__(self, a):
        self.array = np.asarray(a, dtype=float).copy()

    def get_array(self):
        return self.array

    def copy_device_to_host(self):
        pass

    def copy_host_to_device(self):
        pass


class Matrix:
    def __init__(self, a):
        self.array = csr_matrix(a)

    def get_nonzero_structure(self):
        a = self.array

        return *a.shape, a.nnz, a.indptr, a.indices

    def get_data(self):
        return self.array.data

    def copy_data_device_to_host(self):
        pass
