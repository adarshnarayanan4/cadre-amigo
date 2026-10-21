"""Check shared-angle residuals and analytic Hessians without optimization."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from communication.communication_reference import (
    antenna_rotation,
    quaternion_rotation_matrix,
)
from integrated.model.shared_design import SharedDesignPhysics
from integrated.reference.benchmark import load_benchmark
from integrated.reference.shared_angles import SharedAngleLookups, antenna_matrices


def check_case(data, case_index):
    lookups = SharedAngleLookups(data, 7, case_index=case_index)
    lookups.dt = 43200 / 1499
    physics = SharedDesignPhysics(lookups)
    base = physics.base
    rng = np.random.default_rng(846 + case_index)
    x = np.zeros(physics.nvars + physics.ncon)
    x[base.g] = rng.uniform(0.3, 1.2, physics.n)
    x[base.t] = rng.uniform(270, 300, (physics.n, 5))
    x[base.i] = rng.uniform(0.04, 0.22, (physics.n, 12))
    x[base.cell] = rng.uniform(0.2, 0.8, (7, 12))
    x[physics.fin], x[physics.antenna] = 1.02, 0.17
    x[physics.nvars :] = rng.normal(size=physics.ncon)
    con, grad, jac = (
        np.zeros(physics.ncon),
        np.zeros(physics.nvars),
        np.zeros(len(physics.cols)),
    )

    def evaluate(point):
        physics.evaluate(point, con, grad, jac)
        return con.copy(), csr_matrix(
            (jac.copy(), physics.cols, physics.rowp),
            shape=(physics.ncon, physics.nvars),
        )

    class Identity:
        def get_indices(self, expressions):
            return np.arange(physics.nvars if expressions == "all" else base.nvars)

    physics.model = base.model = Identity()
    physics.inputs = "all"
    base.inputs = "base"
    direction = rng.normal(size=physics.nvars)
    direction[base.g] *= 0.1
    direction[base.i] *= 0.01
    direction[base.cell] *= 0.1
    direction[-2:] *= 0.1
    h = 1e-5
    value, matrix = evaluate(x)
    xp, xm = x.copy(), x.copy()
    xp[: physics.nvars] += h * direction
    xm[: physics.nvars] -= h * direction
    plus, jp = evaluate(xp)
    minus, jm = evaluate(xm)
    fd = (plus - minus) / (2 * h)
    jac_error = np.max(np.abs(matrix @ direction - fd)) / max(1, np.max(np.abs(fd)))
    evaluate(x)
    hessian = physics.hessian_correction((physics.nvars, physics.nvars))
    fd_h = ((jp - jm).T @ x[physics.nvars :]) / (2 * h)
    h_error = np.max(np.abs(hessian @ direction - fd_h)) / max(1, np.max(np.abs(fd_h)))
    angle_errors = []

    for index in [physics.fin, physics.antenna]:
        xp, xm = x.copy(), x.copy()
        xp[index] += h
        xm[index] -= h
        plus, _ = evaluate(xp)
        minus, _ = evaluate(xm)
        angle_errors.append(
            np.max(
                np.abs((plus - minus) / (2 * h) - matrix[:, index].toarray().ravel())
            )
        )

    evaluate(x)
    contracted = lookups.geometry(x[base.g])
    independent = lookups.geometry(x[base.g], reference=True)
    replay = max(
        np.max(np.abs(contracted[0] - independent[0])),
        np.max(np.abs(contracted[1] - independent[1])),
    )
    asymmetry = np.max(np.abs((hessian - hessian.T).data), initial=0)
    assert jac_error < 1e-6, jac_error
    assert h_error < 1e-5, h_error
    assert max(angle_errors) < 1e-6, angle_errors
    assert replay < 1e-8, replay
    assert asymmetry < 1e-9, asymmetry
    return {
        "jacobian_relative_error": float(jac_error),
        "hessian_relative_error": float(h_error),
        "angle_column_max_error": float(max(angle_errors)),
        "independent_lookup_max_error": float(replay),
        "hessian_asymmetry": float(asymmetry),
    }


def main():
    for angle in [-np.pi / 4, 0, 0.23, np.pi / 4]:
        reference = quaternion_rotation_matrix(antenna_rotation(angle, 1))[:, :, 0]
        np.testing.assert_allclose(antenna_matrices(angle)[0], reference, atol=1e-14)

    data = load_benchmark()
    nodes = np.linspace(0, 1499, 7, dtype=int)
    data = {
        k: (
            v[..., nodes]
            if isinstance(v, np.ndarray) and v.ndim and v.shape[-1] == 1500
            else v
        )
        for k, v in data.items()
    }
    report = {}

    for index in range(6):
        report[str(index)] = check_case(data, index)
        print(
            f"Shared-angle derivatives passed for case {index}: {report[str(index)]}",
            flush=True,
        )

    path = Path("integrated/results/validation/shared_angle_validation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
