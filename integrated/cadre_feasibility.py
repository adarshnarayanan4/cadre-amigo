"""Repair a benchmark power schedule before the constrained AMIGO solve."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from battery.battery_reference import Cp, battery_voltage, eta, sigma
from integrated.cadre_reporting import print_repair_header, print_repair_iteration


def feasibility_metrics(state, initial_soc):
    soc, current = state["SOC"], state["I_bat"]

    return {
        "SOC_min": float(np.min(soc)),
        "SOC_max": float(np.max(soc)),
        "current_min_A": float(np.min(current)),
        "current_max_A": float(np.max(current)),
        "periodicity_error": float(soc[-1] - initial_soc),
        "inequality_violation": max(
            0.0,
            float(0.2 - np.min(soc)),
            float(np.max(soc) - 1.0),
            float(-10.0 - np.min(current)),
            float(np.max(current) - 5.0),
        ),
    }


def repair_baseline(
    basis, target_cp, initial_soc, dt, fixed_power, replay, *, interior=False
):
    """Find a nearby feasible schedule by successive frozen-voltage LPs.

    Minimize the L1 distance from the benchmark controls, not downloaded data.
    Each LP freezes battery voltage at the independently propagated trajectory.
    Replaying and updating the voltage makes the SOC periodicity condition
    consistent with the nonlinear model. Optional interior margins prepare
    AMIGO's starting guess separately from the comparison baseline.
    """
    n, m = basis.shape
    cp_lower, cp_upper = (0.02, 24.5) if interior else (0.0, 25.0)
    current_lower, current_upper = (-9.85, 4.9) if interior else (-9.9999, 4.9999)
    soc_lower, soc_upper = (0.215, 0.985) if interior else (0.20001, 0.99999)

    cp = np.clip(target_cp, cp_lower, cp_upper)

    identity = eye(m, format="csr")
    distance_rows = vstack(
        [hstack([identity, -identity]), hstack([-identity, -identity])], format="csr"
    )
    distance_rhs = np.concatenate([target_cp, -target_cp])

    print_repair_header(interior)

    for iteration in range(30):
        state = replay(cp)
        power_offset = fixed_power(state) if callable(fixed_power) else fixed_power
        voltage = battery_voltage(state["SOC"], state["temperature"][4])

        current_matrix = -5.0 * basis / voltage[:, None]
        current_offset = power_offset / voltage

        rate_matrix = eta / Cp * current_matrix
        rate_offset = eta / Cp * current_offset - sigma / 24.0 * state["SOC"]

        soc_matrix = np.zeros((n, m))
        soc_matrix[1:] = np.cumsum(
            0.5 * dt * (rate_matrix[:-1] + rate_matrix[1:]), axis=0
        )

        soc_offset = np.full(n, initial_soc)
        soc_offset[1:] += np.cumsum(0.5 * dt * (rate_offset[:-1] + rate_offset[1:]))

        inequalities = csr_matrix(
            np.vstack([current_matrix, -current_matrix, soc_matrix, -soc_matrix])
        )
        rhs = np.concatenate(
            [
                current_upper - current_offset,
                current_offset - current_lower,
                soc_upper - soc_offset,
                soc_offset - soc_lower,
            ]
        )
        inequalities = hstack([inequalities, csr_matrix((4 * n, m))], format="csr")

        result = linprog(
            np.r_[np.zeros(m), np.ones(m)],
            A_ub=vstack([inequalities, distance_rows], format="csr"),
            b_ub=np.r_[rhs, distance_rhs],
            A_eq=csr_matrix(np.r_[soc_matrix[-1], np.zeros(m)][None, :]),
            b_eq=[initial_soc - soc_offset[-1]],
            bounds=[(cp_lower, cp_upper)] * m + [(0, None)] * m,
            method="highs",
        )

        if not result.success:
            raise RuntimeError(f"Feasible-baseline repair failed: {result.message}")

        cp = result.x[:m]
        state = replay(cp)

        metrics = feasibility_metrics(state, initial_soc)

        print_repair_iteration(iteration + 1, metrics)

        if (
            abs(metrics["periodicity_error"]) < 1e-9
            and metrics["inequality_violation"] < 1e-9
        ):
            return cp, state, metrics

    raise RuntimeError("Feasible-baseline repair did not converge to SOC periodicity")
