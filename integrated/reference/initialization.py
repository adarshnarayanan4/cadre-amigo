"""Construct solar-current starting controls for the selected geometry."""

import numpy as np
from scipy.optimize import lsq_linear


def maximum_power_controls(basis, temperature, area, los, voltage):
    """Fit bounded controls to a panel-wise maximum-power current scan.

    A control whose spline support crosses a dark panel sample is left at zero.
    This sacrifices some edge-of-daylight power to avoid negative panel power
    during eclipse. The optimizer is free to change every control afterward.
    """
    nodes, controls = basis.shape
    coefficients = voltage.panel_coefficients(area, los)
    peak_power = np.zeros((12, nodes))
    target = np.zeros_like(peak_power)
    for current in np.geomspace(1e-5, 0.4, 80):
        currents = np.full((12, nodes), current)
        panel_voltage = voltage.evaluate_panels(temperature, currents, coefficients)[0]
        power = current * panel_voltage
        improved = power > peak_power
        peak_power[improved] = power[improved]
        target[improved] = current
    result = np.zeros((12, controls))
    for panel in range(12):
        dark = peak_power[panel] < 1e-5
        eligible = np.all(basis[dark] < 1e-10, axis=0)
        if np.any(eligible):
            result[panel, eligible] = lsq_linear(
                basis[:, eligible], target[panel], bounds=(0, 0.4)
            ).x
    return result
