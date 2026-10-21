"""Compile a small joint model and check assembly; never optimize or factor KKT."""

import argparse
import json
from pathlib import Path

import amigo as am
import numpy as np
from scipy.sparse import csr_matrix

from integrated.model.joint import (
    assemble_joint_model,
    build_directory,
    prepare_joint_cases,
)
from integrated.solvers.native_ldl import temporal_kkt_ordering


def check_joint(joint):
    model = joint.model
    design = joint.design_indices()
    assert len(design) == len(np.unique(design)) == joint.design_variable_count
    shared = model.get_indices("shared_surface.cellInstd[0, :]")
    fin = int(model.get_indices("shared_angles.finAngle[:]")[0])
    antenna = int(model.get_indices("shared_angles.antAngle[:]")[0])
    lower = model.get_lower().get_vector().get_array()
    upper = model.get_upper().get_vector().get_array()

    for name, lo, hi in [
        ("shared_surface.cellInstd[0, :]", 0, 1),
        ("shared_angles.finAngle[:]", 0, np.pi / 2),
        ("shared_angles.antAngle[:]", -np.pi / 4, np.pi / 4),
    ]:
        indices = model.get_indices(name)
        np.testing.assert_array_equal(lower[indices], lo)
        np.testing.assert_array_equal(upper[indices], hi)

    for case in joint.cases:
        prefix = f"case{case.case_index}."
        np.testing.assert_array_equal(
            shared, model.get_indices(prefix + "surface.cellInstd[0, :]")
        )
        assert fin == model.get_indices(prefix + "angles.finAngle[:]")[0]
        assert antenna == model.get_indices(prefix + "angles.antAngle[:]")[0]

        for name, lo, hi in [
            ("control_points.CP_P_comm[:]", 0, 25),
            ("control_points.CP_Isetpt[:, :]", 0, 0.4),
            ("control_points.CP_gamma[:]", 0, np.pi / 2),
            ("battery_ic.initial_soc[:]", 0.2, 1),
            ("battery.SOC[:]", 0.2, 1),
            ("battery.I_bat[:]", -10, 5),
        ]:
            indices = model.get_indices(prefix + name)
            np.testing.assert_array_equal(lower[indices], lo)
            np.testing.assert_array_equal(upper[indices], hi)

    order = temporal_kkt_ordering(model, joint.cases[0].n)
    np.testing.assert_array_equal(np.sort(order), np.arange(len(lower)))

    state = model.get_initial_point()
    vector = state.get_vector()
    values = vector.get_array()
    original = values.copy()
    replay = joint.replay(state)
    replay_error = max(
        error
        for case in replay["cases"]
        for error in case["reference_max_errors"].values()
    )
    assert replay_error < 1e-7, replay_error
    gradient = model.create_vector()
    duals = model.problem.get_constraint_indices()
    final_data, periodicity = [], []

    for index, case in enumerate(joint.cases):
        prefix = f"case{case.case_index}."
        ids = np.atleast_1d(model.get_indices(prefix + f"comm_data.Data[{case.n-1}]"))
        final_data.extend(ids)
        values[ids] = index + 1
        np.testing.assert_array_equal(
            ids, np.atleast_1d(model.get_indices(f"objective.Data[0, {index}]"))
        )
        values[model.get_indices(prefix + "battery.SOC[0]")] = 0.4
        values[model.get_indices(prefix + f"battery.SOC[{case.n-1}]")] = 0.6
        periodicity.extend(model.get_indices(prefix + "battery_periodicity.res[:]"))

    values[duals] = 0
    model.problem.update(vector)
    objective = model.problem.lagrangian(1.0, vector)
    expected = -len(joint.cases) * (len(joint.cases) + 1) / 2
    assert abs(objective - expected) < 1e-12, (objective, expected)
    model.eval_gradient(state, gradient)
    np.testing.assert_allclose(
        gradient.get_vector().get_array()[final_data], -1, atol=1e-12
    )
    assert len(set(periodicity)) == len(joint.cases)
    np.testing.assert_allclose(
        gradient.get_vector().get_array()[periodicity], 0.2, atol=1e-12
    )

    values[:] = original
    values[shared] = 0.5
    values[fin], values[antenna] = 1.02, 0.17

    for case in joint.cases:
        prefix = f"case{case.case_index}."
        values[model.get_indices(prefix + "values.Gamma[:]")] = np.linspace(
            0.4, 1.0, case.n
        )

        for k in range(5):
            values[model.get_indices(prefix + f"thermal.temperature[:, {k}]")] = 280 + k

        for panel in range(12):
            values[model.get_indices(prefix + f"values.Isetpt[:, {panel}]")] = 0.12

    rng = np.random.default_rng(974)
    values[duals] = rng.normal(size=len(duals)) * 0.1
    point = values.copy()
    joint.set_point_provider(lambda: vector.get_array())
    model.problem.update(vector)
    matrix = model.create_matrix()
    model.eval_hessian(state, matrix)
    matrix.copy_data_device_to_host()
    n, m, _, rowp, cols = matrix.get_nonzero_structure()
    native = csr_matrix(
        (np.array(matrix.get_data(), copy=True), cols, rowp), shape=(n, m)
    )
    hessian = native + joint.hessian_correction((n, m))
    direction = rng.normal(size=len(values))

    for expression, indices in model._staged_fixed_vars:
        fixed = model.get_indices(expression)
        direction[fixed if indices is None else fixed[indices]] = 0

    step = 1e-5
    probes = []

    for sign in [1, -1]:
        values[:] = point + sign * step * direction
        model.problem.update(vector)
        model.eval_gradient(state, gradient)
        probes.append(gradient.get_vector().get_array().copy())

    values[:] = point
    model.problem.update(vector)
    finite = (probes[0] - probes[1]) / (2 * step)
    error = np.max(np.abs(hessian @ direction - finite)) / max(
        1, np.max(np.abs(finite))
    )
    asymmetry = np.max(np.abs((hessian - hessian.T).data), initial=0)
    assert np.all(np.isfinite(hessian.data))
    assert error < 2e-6, error
    assert asymmetry < 1e-8, asymmetry
    return {
        "cases": [case.case_index for case in joint.cases],
        "design_variables": joint.design_variable_count,
        "shared_design_variables": 86,
        "shared_aliases_verified": True,
        "design_and_battery_bounds_verified": True,
        "scoped_temporal_permutation_verified": True,
        "summed_objective_error": float(abs(objective - expected)),
        "separate_soc_periodicity_verified": True,
        "independent_initial_replay_max_error": float(replay_error),
        "assembled_hessian_direction_relative_error": float(error),
        "assembled_hessian_asymmetry": float(asymmetry),
        "optimization_run": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir", type=Path, default=Path("/tmp/cadre-joint-validation/build")
    )
    parser.add_argument(
        "--reuse-build",
        action="store_true",
        help="Reuse a build from the identical source revision and six-case configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("integrated/results/validation/joint_model_validation.json"),
    )
    args = parser.parse_args(argv)
    cases, options = prepare_joint_cases(list(range(6)), nodes=8, controls=4)
    joint = assemble_joint_model(cases, options)

    if args.reuse_build:
        with build_directory(args.build_dir):
            joint.model._built = True
            joint.model.initialize(order_type=am.OrderingType.NATURAL)
    else:
        joint.initialize(args.build_dir)

    report = check_joint(joint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
