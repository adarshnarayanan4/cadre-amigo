"""Check warm-start compatibility, multiplier retention, and barrier overrides."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from integrated.cadre_checkpoint import load_checkpoint
from integrated.cadre_options import parse_args
from integrated.validation_helpers import Vector


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
        )

        args = parse_args(["--design-physics", "--warm-start", str(checkpoint)])
        optimizer = SimpleNamespace(x_init=Vector(np.zeros(4)))

        same, duals = load_checkpoint(optimizer, args, signature)

        assert same and args.initial_barrier == 1e-7

        np.testing.assert_array_equal(optimizer.x_init.array, solution)
        np.testing.assert_array_equal(duals[0], lower_duals)
        np.testing.assert_array_equal(duals[1], upper_duals)

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

    result = {
        "compatible_restore": True,
        "explicit_barrier_preserved": True,
        "design_release_supported": True,
        "incompatible_checkpoints_rejected": rejected,
    }

    print(result)

    output = Path("integrated/results/checkpoint_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
