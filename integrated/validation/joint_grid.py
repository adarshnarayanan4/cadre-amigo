"""Check full-window sampling and joint command defaults without optimizing."""

import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from attitude.attitude_reference import attitude_from_orbit
from integrated.cadre_joint import parse_args
from cadre_paths import source_provenance
from integrated.optimization.joint import load_initial_design, refit_control_profile
from integrated.reference.benchmark import load_benchmark, resample_benchmark
from integrated.validation.helpers import Vector


def check_control_transfer():
    """Transfer a known bounded design and reject incompatible source metadata."""
    indices = {
        "shared_surface.cellInstd[0, :]": np.arange(84),
        "shared_angles.finAngle[:]": np.array([84]),
        "shared_angles.antAngle[:]": np.array([85]),
        "case0.control_points.CP_P_comm[:]": np.arange(86, 94),
        "case0.control_points.CP_gamma[:]": np.arange(94, 102),
        "case0.control_points.CP_Isetpt[:, :]": np.arange(102, 198).reshape(8, 12),
        "case0.battery_ic.initial_soc[:]": np.array([198]),
    }
    lower, upper = np.zeros(199), np.ones(199)
    lower[85], upper[85] = -np.pi / 4, np.pi / 4
    upper[86:94], upper[94:102], upper[102:198] = 25, np.pi / 2, 0.4
    model = SimpleNamespace(
        get_indices=indices.__getitem__,
        get_lower=lambda: SimpleNamespace(get_vector=lambda: Vector(lower)),
        get_upper=lambda: SimpleNamespace(get_vector=lambda: Vector(upper)),
    )
    joint = SimpleNamespace(
        model=model,
        cases=[SimpleNamespace(case_index=0, m=8, window_duration=43200)],
        design_indices=lambda: np.arange(199),
    )
    point = Vector(np.zeros(199))
    summary = {
        "cadre_source": source_provenance(),
        "duration_hours": 12,
        "fin_angle_rad": 0.5,
        "antenna_angle_rad": -0.5,
    }
    with TemporaryDirectory(prefix="cadre-control-transfer-") as directory:
        directory = Path(directory)
        metadata = directory / "summary.json"
        metadata.write_text(json.dumps(summary))
        np.savez(
            directory / "controls.npz",
            surface_allocation=np.full(84, 0.9),
            case0_CP_P_comm=np.full(4, 2.5),
            case0_CP_gamma=np.full(4, 0.6),
            case0_CP_Isetpt=np.full((4, 12), 0.2),
            case0_SOC=[0.7],
        )
        load_initial_design(joint, point, directory, refit=True)
        np.testing.assert_allclose(point.get_array()[86:94], 2.5, atol=1e-8)
        np.testing.assert_allclose(point.get_array()[94:102], 0.6, atol=1e-8)
        np.testing.assert_allclose(point.get_array()[102:198], 0.2, atol=1e-8)
        assert point.get_array()[198] == 0.7
        for refit, duration in [(False, 12), (True, 6)]:
            summary["duration_hours"] = duration
            metadata.write_text(json.dumps(summary))
            try:
                load_initial_design(joint, point, directory, refit=refit)
            except ValueError:
                pass
            else:
                raise AssertionError("Incompatible control import was accepted")
    for invalid in [
        np.full(4, np.nan),
        np.full(4, -1),
        np.full(4, 26),
        np.zeros((4, 2)),
    ]:
        try:
            refit_control_profile(invalid, 8, 25)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid source controls were fitted")
    return True


def main():
    original = load_benchmark()
    sampled = resample_benchmark(original, 150)
    np.testing.assert_array_equal(sampled["t"], np.linspace(0, 43200, 150))
    assert len(original["t"]) == 1500
    assert resample_benchmark(original, 1500) is original
    contacts = {}
    for index in range(6):
        prefix = f"{index}:"
        for name in ["CP_P_comm", "CP_Isetpt", "CP_gamma"]:
            np.testing.assert_array_equal(
                sampled[prefix + name], original[prefix + name]
            )
        states = sampled[prefix + "r_e2b_I"].T
        np.testing.assert_allclose(
            sampled[prefix + "O_RI"], attitude_from_orbit(states).transpose(1, 2, 0)
        )
        contacts[str(index)] = int(np.count_nonzero(sampled[prefix + "CommLOS"] > 0))
        assert contacts[str(index)] > 0
    assert parse_args(["--optimize"]).time_grid == "full"
    assert (
        parse_args(
            ["--optimize", "--cases", "0", "--linear-solver", "sparse"]
        ).linear_solver
        == "sparse"
    )
    assert (
        parse_args(["--build", "--nodes", "8", "--control-points", "4"]).time_grid
        == "prefix"
    )
    for invalid in [
        ["--spectral-curvature"],
        ["--optimize", "--box-qp"],
        ["--optimize", "--spectral-curvature", "--linear-solver", "native"],
        ["--optimize", "--refit-controls"],
        ["--case-workers", "0"],
        ["--primal-barrier"],
        ["--optimize", "--primal-barrier", "--full-space-trials"],
        ["--optimize", "--primal-barrier", "--linear-solver", "native"],
        ["--repair-start"],
        ["--optimize", "--repair-start", "--warm-start", "checkpoint.npz"],
        ["--warm-start", "checkpoint.npz"],
        ["--cases", "0", "0"],
        ["--optimize", "--nodes", "4", "--control-points", "5"],
        ["--optimize", "--initial-barrier", "nan"],
        ["--optimize", "--initial-barrier", "inf"],
        ["--optimize", "--objective-scale", "nan"],
        ["--optimize", "--restore-periodicity", "--full-space-trials"],
        ["--optimize", "--linear-solver", "sparse"],
        ["--initial-design", "results"],
        ["--optimize", "--initial-design", "results", "--warm-start", "checkpoint.npz"],
    ]:
        with contextlib.redirect_stderr(io.StringIO()):
            try:
                parse_args(invalid)
            except SystemExit as error:
                assert error.code == 2
            else:
                raise AssertionError(f"Invalid command was accepted: {invalid}")
    report = {
        "active_box_step_option": parse_args(
            ["--optimize", "--primal-barrier", "--box-qp", "--spectral-curvature"]
        ).box_qp,
        "bounded_control_transfer_and_window_checks": check_control_transfer(),
        "control_refitting_repairs_start": parse_args(
            ["--optimize", "--initial-design", "results", "--refit-controls"]
        ).repair_start,
        "primal_barrier_repairs_cold_start": parse_args(
            ["--optimize", "--primal-barrier"]
        ).repair_start,
        "primal_barrier_restores_periodicity": parse_args(
            ["--optimize", "--primal-barrier"]
        ).restore_periodicity,
        "parallel_case_configuration": parse_args(
            ["--optimize", "--case-workers", "2"]
        ).case_workers
        == 2,
        "cold_battery_repair_option": parse_args(
            ["--optimize", "--repair-start", "--full-space-trials"]
        ).repair_start,
        "source_data_unchanged": True,
        "controls_unchanged": True,
        "duration_hours": 12,
        "coarse_nodes": 150,
        "contact_nodes": contacts,
        "optimization_defaults_to_full_window": True,
        "build_only_prefix_compatibility": True,
        "invalid_commands_rejected": True,
    }
    path = Path("integrated/results/validation/joint_grid_validation.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
