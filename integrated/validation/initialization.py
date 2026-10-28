"""Check eclipse current seeds and the joint interior-bound projection."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from integrated.optimization.checkpoint import project_into_bounds
from integrated.reference.initialization import maximum_power_controls
from parameters.parameters_reference import build_bspline_matrix
from power.voltage_spline import VoltageSpline


def check_reference_fallback():
    from integrated.model.joint import prepare_joint_cases

    reference = SimpleNamespace(lookups=object())
    with (
        patch("integrated.model.joint.load_benchmark", return_value={}),
        patch("integrated.model.joint.SharedDesignPhysics"),
        patch(
            "integrated.model.joint.prepare_case",
            side_effect=[RuntimeError("Current seed repair failed"), reference],
        ) as prepare,
    ):
        cases, _ = prepare_joint_cases([1], nodes=16, controls=16, repair=True)
    assert cases == [reference]
    assert prepare.call_args_list[0].kwargs["maximum_power_start"]
    assert not prepare.call_args_list[1].kwargs["maximum_power_start"]


def main():
    check_reference_fallback()
    nodes = controls = 16
    basis = build_bspline_matrix(nodes, controls)
    los = np.ones(nodes)
    los[:3] = los[-3:] = 0
    area = np.full((7, 12, nodes), 0.01)
    temperature = np.full((5, nodes), 273.0)
    voltage = VoltageSpline()
    cp = maximum_power_controls(basis, temperature, area, los, voltage)
    currents = cp @ basis.T
    assert np.all(cp >= 0) and np.all(cp <= 0.4)
    np.testing.assert_allclose(currents[:, los == 0], 0, atol=1e-10)
    power = currents * voltage.reference_panels(temperature, currents, area, los)
    assert np.sum(power[:, los > 0]) > 0
    values = project_into_bounds(
        np.array([-0.1, 1e-5, 0.5]), np.zeros(3), np.full(3, 0.4), 1e-6
    )
    assert 0 < values[0] < 1e-5
    assert values[1] == 1e-5
    assert 0.399 < values[2] < 0.4
    result = {
        "bounded_solar_controls": True,
        "zero_eclipse_seed_current": True,
        "positive_daylight_power": True,
        "small_interior_currents_preserved": True,
        "failed_power_seed_uses_reference_currents": True,
    }
    path = Path("integrated/results/validation/initialization_validation.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
