"""Check CADRE cases and write a run plan without compilation or optimization."""

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

from cadre_paths import cadre_path, source_provenance
from integrated.reference.benchmark import load_benchmark, select_case
from integrated.reporting.console import print_heading, print_values
from parameters.parameters_reference import build_bspline_matrix


def check_case(data, case_index, nodes, controls, check_lookups=False, voltage=None):
    """Check source fields and independent lookup replay; never repair a baseline."""
    selected = select_case(data, case_index)
    full_n = len(selected["P_comm"])
    native_controls = len(selected["CP_P_comm"])

    if not 4 <= controls <= nodes <= full_n:
        raise ValueError("Require 4 <= control-points <= nodes <= benchmark nodes")

    shapes = {
        "P_comm": (full_n,),
        "gamma": (full_n,),
        "Isetpt": (12, full_n),
        "LOS": (full_n,),
        "exposedArea": (7, 12, full_n),
        "O_RI": (3, 3, full_n),
        "O_AB": (3, 3, full_n),
        "r_e2s_I": (3, full_n),
        "r_b2g_I": (3, full_n),
        "w_B": (3, full_n),
        "T_tot": (3, full_n),
        "V_sol": (12, full_n),
        "gain": (full_n,),
        "GSdist": (full_n,),
        "CommLOS": (full_n,),
        "temperature": (5, full_n),
        "cellInstd": (7, 12),
        "CP_P_comm": (native_controls,),
        "CP_gamma": (native_controls,),
        "CP_Isetpt": (12, native_controls),
        "iSOC": (1,),
        "LD": (1,),
    }

    for name, expected in shapes.items():
        values = np.asarray(selected[name])

        if values.shape != expected or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Case {case_index}: invalid {name} shape or nonfinite values"
            )

    contact_nodes = int(np.count_nonzero(selected["CommLOS"][:nodes] > 0))

    if not contact_nodes:
        raise ValueError(f"Case {case_index}: selected window has no ground contact")

    basis = build_bspline_matrix(full_n, native_controls)
    reconstruction = {
        "communication_power_W": (basis @ selected["CP_P_comm"], selected["P_comm"]),
        "roll_rad": (basis @ selected["CP_gamma"], selected["gamma"]),
        "solar_current_A": (selected["CP_Isetpt"] @ basis.T, selected["Isetpt"]),
    }
    errors = {
        name: float(np.max(np.abs(actual - expected)))
        for name, (actual, expected) in reconstruction.items()
    }

    if max(errors.values()) > 1e-9:
        raise ValueError(
            f"Case {case_index}: benchmark spline reconstruction failed: {errors}"
        )

    report = {
        "case_index": case_index,
        "benchmark_LD_days": float(np.asarray(selected["LD"]).item()),
        "source_nodes": full_n,
        "nodes": nodes,
        "control_points": controls,
        "duration_hours": float(12 * (nodes - 1) / (full_n - 1)),
        "contact_nodes": contact_nodes,
        "single_case_design_variables_with_roll_and_allocation": 14 * controls + 1 + 84,
        "spline_reconstruction_max_errors": errors,
        "lookup_replay_checked": check_lookups,
    }

    if check_lookups:
        from integrated.reference.design import DesignLookups

        lookups = DesignLookups(data, nodes, voltage, case_index=case_index)
        lookups.dt = 43200 / (full_n - 1)
        gamma = selected["gamma"][:nodes]
        contracted = lookups.geometry(gamma)
        independent = lookups.geometry(gamma, reference=True)
        area_error = float(np.max(np.abs(contracted[0] - independent[0])))
        gain_error = float(np.max(np.abs(contracted[1] - independent[1])))

        if (
            not np.isfinite(area_error + gain_error)
            or area_error > 1e-10
            or gain_error > 1e-8
        ):
            raise ValueError(f"Case {case_index}: independent lookup replay failed")

        report["lookup_replay_max_errors"] = {
            "exposed_area_m2": area_error,
            "antenna_gain": gain_error,
        }

    return report


def environment():
    packages = {}

    for name in ("amigo", "numpy", "scipy", "matplotlib"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "build_tools": {name: shutil.which(name) for name in ("cmake", "c++")},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", type=int, nargs="+", choices=range(6), default=list(range(6))
    )
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--control-points", type=int, default=300)
    parser.add_argument(
        "--check-lookups",
        action="store_true",
        help="Also compare analytic lookup contractions with independent tensor replay",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("integrated/results/preparation.json")
    )
    args = parser.parse_args(argv)
    cases = list(dict.fromkeys(args.cases))
    data = load_benchmark()
    voltage = None

    if args.check_lookups:
        from power.voltage_spline import VoltageSpline

        voltage = VoltageSpline()

    reports = []

    for case_index in cases:
        print_heading(f"PREPARING CADRE CASE {case_index}")
        report = check_case(
            data,
            case_index,
            args.nodes,
            args.control_points,
            args.check_lookups,
            voltage,
        )
        reports.append(report)
        print_values(
            [
                ("Time nodes", report["nodes"]),
                ("Ground-contact nodes", report["contact_nodes"]),
                ("Spline reconstruction", "Passed"),
                (
                    "Independent lookup replay",
                    "Passed" if args.check_lookups else "Not requested",
                ),
            ]
        )

    report = {
        "scope": "Input and reference checks only; no baseline repair, compilation, KKT factorization, or optimization",
        "cadre_source": source_provenance(),
        "source_sha256": hashlib.sha256(
            cadre_path("test/data1346.pkl").read_bytes()
        ).hexdigest(),
        "environment": environment(),
        "cases": reports,
        "joint_model_implemented": True,
        "joint_optimizer_implemented": True,
        "joint_optimization_validated": False,
        "planned_joint_design_variables": len(cases) * (14 * args.control_points + 1)
        + 86,
        "remaining_model_work": [
            "Coarse-grid convergence, including bounded SOC periodicity restoration",
            "Server-scale compilation and numerical profiling",
            "Full-resolution joint solver profiling",
            "Full-resolution optimization convergence and independent validation",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Preparation report saved to {args.output.resolve()}", flush=True)
    return report


if __name__ == "__main__":
    main()
