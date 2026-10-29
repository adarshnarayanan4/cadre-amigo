"""Check warm-start compatibility, multiplier retention, and barrier overrides."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from integrated.reporting.console import print_validation
from integrated.optimization.checkpoint import load_checkpoint
from integrated.optimization.options import parse_args
from integrated.validation.helpers import Vector


def main():
    signature = {
        "nodes": 1500,
        "control_points": 300,
        "fixed_initial_soc": False,
        "fixed_solar_currents": False,
        "legacy_fixed_voltage": False,
        "design_physics": True,
        "optimize_roll": False,
        "optimize_allocation": False,
    }

    solution = np.array([0.3, 0.4, 1.2, -2.0])
    lower_duals, upper_duals = np.array([1.0, 2.0]), np.array([3.0, 4.0])

    with TemporaryDirectory(prefix="cadre-checkpoint-") as directory:
        checkpoint = Path(directory) / "checkpoint.npz"
        np.savez(
            checkpoint,
            signature=json.dumps(signature),
            solution=solution,
            zl=lower_duals,
            zu=upper_duals,
            barrier_param=1e-7,
            nlp_error=1e-6,
            complementarity=1e-8,
            linear_solver_diagnostics=json.dumps({"curvature_shift": 0.1}),
        )

        args = parse_args(["--design-physics", "--warm-start", str(checkpoint)])
        optimizer = SimpleNamespace(x_init=Vector(np.zeros(4)))

        same, duals = load_checkpoint(optimizer, args, signature)

        assert same and args.initial_barrier == 1e-7

        np.testing.assert_array_equal(optimizer.x_init.array, solution)
        np.testing.assert_array_equal(duals[0], lower_duals)
        np.testing.assert_array_equal(duals[1], upper_duals)

        # An old checkpoint without a case field belongs to case 0 only.
        try:
            load_checkpoint(optimizer, args, dict(signature, case_index=1))
        except ValueError:
            pass
        else:
            raise AssertionError("A case-0 checkpoint was accepted for case 1")

        args.initial_barrier = 2e-5
        load_checkpoint(optimizer, args, signature)

        assert args.initial_barrier == 2e-5

        released = dict(signature, optimize_roll=True, optimize_allocation=True)
        args.release_design_controls = True

        same, duals = load_checkpoint(optimizer, args, released)

        assert not same and duals is None

        rejected = 0

        for invalid in (
            dict(released, nodes=600),
            dict(released, fixed_solar_currents=True),
        ):
            try:
                load_checkpoint(optimizer, args, invalid)
            except ValueError:
                rejected += 1

        args.release_design_controls = False

        try:
            load_checkpoint(optimizer, args, released)
        except ValueError:
            rejected += 1

        try:
            load_checkpoint(
                SimpleNamespace(x_init=Vector(np.zeros(3))), args, signature
            )
        except ValueError:
            rejected += 1

        assert rejected == 4

        joint_signature = {
            "formulation": "joint-cadre-v1",
            "cases": [0, 1, 2, 3, 4, 5],
            "nodes": 150,
            "controls": 30,
            "time_grid": "full",
            "model_source_sha256": "tested-model",
            "cadre_source": {"revision": "tested-reference"},
        }
        np.savez(
            checkpoint,
            signature=json.dumps(joint_signature),
            solution=solution,
            zl=lower_duals,
            zu=upper_duals,
            barrier_param=1e-7,
        )
        same, duals = load_checkpoint(optimizer, args, joint_signature)
        assert same and duals is not None
        joint_rejected = 0
        for invalid in (
            dict(joint_signature, cases=[1, 0, 2, 3, 4, 5]),
            dict(joint_signature, time_grid="prefix"),
            dict(joint_signature, model_source_sha256="changed-model"),
            dict(joint_signature, cadre_source={"revision": "other-reference"}),
        ):
            try:
                load_checkpoint(optimizer, args, invalid)
            except ValueError:
                joint_rejected += 1
        assert joint_rejected == 4
        np.savez(
            checkpoint,
            signature=json.dumps(joint_signature),
            solution=np.full(4, np.nan),
        )
        try:
            load_checkpoint(optimizer, args, joint_signature)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite checkpoint was accepted")

    result = {
        "compatible_restore": True,
        "joint_restore_verified": True,
        "joint_signature_mismatches_rejected": joint_rejected,
        "nonfinite_checkpoint_rejected": True,
        "cross_case_restore_rejected": True,
        "explicit_barrier_preserved": True,
        "design_release_supported": True,
        "incompatible_checkpoints_rejected": rejected,
    }

    print_validation("CHECKPOINT VALIDATION", result)

    output = Path("integrated/results/validation/checkpoint_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
