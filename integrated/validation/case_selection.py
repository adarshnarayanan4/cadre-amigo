"""Check case isolation and output naming without model compilation or solving."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from integrated.optimization.options import parse_args
from integrated.reference.benchmark import load_benchmark, select_case
from integrated.reporting.plots import find_reference


def main():
    data = load_benchmark()

    for index in range(6):
        selected = select_case(data, index)
        np.testing.assert_array_equal(selected["P_comm"], data[f"{index}:P_comm"])
        np.testing.assert_array_equal(selected["O_RI"], data[f"{index}:O_RI"])
        np.testing.assert_array_equal(selected["cellInstd"], data["cellInstd"])
        args = parse_args(
            ["--case-index", str(index), "--optimize-roll", "--optimize-allocation"]
        )
        expected = "pcomm_roll_allocation" + (f"_case_{index}" if index else "")
        assert args.output.name == expected

    assert parse_args([]).output.name == "pcomm_solar"
    explicit = parse_args(["--case-index", "2", "--output", "/tmp/cadre-case-output"])
    assert explicit.output == Path("/tmp/cadre-case-output")

    try:
        select_case(data, 6)
    except ValueError:
        pass
    else:
        raise AssertionError("A missing case was accepted")

    with TemporaryDirectory(prefix="cadre-reference-") as directory:
        root = Path(directory)
        reference = root / "pcomm_design_reference_case_1"
        reference.mkdir()
        metadata = {
            "case_index": 1,
            "validated": True,
            "nodes": 1500,
            "control_points": 300,
            "dt_s": 43200 / 1499,
            "design_physics_restored": True,
            "solar_voltage_coupled": True,
            "solar_currents_optimized": True,
            "initial_soc_optimized": True,
        }
        (reference / "summary.json").write_text(json.dumps(metadata))
        np.savez(reference / "trajectories.npz", time_s=np.arange(3))
        current = dict(metadata, roll_optimized=True)
        assert find_reference(root / "trial", current) is not None

        metadata["case_index"] = 0
        (reference / "summary.json").write_text(json.dumps(metadata))
        assert find_reference(root / "trial", current) is None

    print(
        "Case selection, output naming, and comparison isolation passed for all six cases."
    )


if __name__ == "__main__":
    main()
