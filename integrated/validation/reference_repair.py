"""Repair saved communication controls on a new grid using the reference model."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from cadre_paths import source_provenance
from integrated.model.joint import prepare_joint_cases
from integrated.reference.feasibility import repair_baseline
from integrated.reference.joint_replay import replay_joint
from integrated.validation.helpers import Vector
from power.power_amigo import total_power


def repair_reference(source, output, nodes):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output directory to preserve the source")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")

    summary = json.loads((source / "summary.json").read_text())
    provenance = source_provenance()
    for key in ("repository", "reference_revision", "benchmark_sha256"):
        if summary["cadre_source"][key] != provenance[key]:
            raise ValueError(f"Source CADRE provenance does not match: {key}")
    if not np.isclose(summary["duration_hours"], 12, rtol=0, atol=1e-9):
        raise ValueError("The source must cover the full 12-hour window")
    if not (
        summary.get("feasibility_validated")
        or summary.get("reference_feasibility_validated")
    ):
        raise ValueError("The source must have passed its feasibility checks")
    controls = summary["controls_per_profile"]
    if not 4 <= controls <= nodes <= 1500:
        raise ValueError("Require 4 <= control points <= nodes <= 1500")
    case_ids = [case["case_index"] for case in summary["cases"]]
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("Source must contain distinct cases")

    with np.load(source / "controls.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}

    def checked(name, shape, lower, upper):
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid shape or values for {name}")
        if np.any(value < lower) or np.any(value > upper):
            raise ValueError(f"{name} exceeds its design bounds")
        return value

    allocation = checked("surface_allocation", (7, 12), 0, 1)
    fin, antenna = summary["fin_angle_rad"], summary["antenna_angle_rad"]
    if not (
        np.isfinite(fin)
        and 0 <= fin <= np.pi / 2
        and np.isfinite(antenna)
        and -np.pi / 4 <= antenna <= np.pi / 4
    ):
        raise ValueError("Shared angles exceed their design bounds")
    inputs = {
        "shared_surface.cellInstd[0, :]": allocation.ravel(),
        "shared_angles.finAngle[:]": np.array([fin]),
        "shared_angles.antAngle[:]": np.array([antenna]),
    }
    for i in case_ids:
        prefix = f"case{i}"
        inputs[prefix + ".control_points.CP_P_comm[:]"] = checked(
            prefix + "_CP_P_comm", (controls,), 0, 25
        )
        inputs[prefix + ".control_points.CP_gamma[:]"] = checked(
            prefix + "_CP_gamma", (controls,), 0, np.pi / 2
        )
        inputs[prefix + ".control_points.CP_Isetpt[:, :]"] = checked(
            prefix + "_CP_Isetpt", (controls, 12), 0, 0.4
        )
        soc = np.asarray(arrays[prefix + "_SOC"])
        if (
            soc.ndim != 1
            or not len(soc)
            or not np.isfinite(soc[0])
            or not 0.2 <= soc[0] <= 1
        ):
            raise ValueError(f"Invalid initial SOC for {prefix}")
        inputs[prefix + ".battery_ic.initial_soc[:]"] = np.array([soc[0]])

    indices, offset = {}, 0
    for name, value in inputs.items():
        indices[name] = np.arange(offset, offset + value.size).reshape(value.shape)
        offset += value.size
    point = Vector(np.concatenate([value.ravel() for value in inputs.values()]))
    state = SimpleNamespace(get_vector=lambda: point)
    model = SimpleNamespace(get_indices=indices.__getitem__)
    cases, _ = prepare_joint_cases(
        case_ids, nodes=nodes, controls=controls, full_duration=True
    )
    joint = SimpleNamespace(model=model, cases=cases)
    repaired = {"surface_allocation": allocation}
    reports, failures = [], []
    output.mkdir(parents=True, exist_ok=True)

    for case in cases:
        prefix = f"case{case.case_index}"
        name = prefix + ".control_points.CP_P_comm[:]"
        ids, starting = indices[name], inputs[name].copy()
        initial_soc = float(inputs[prefix + ".battery_ic.initial_soc[:]"][0])

        def replay(cp):
            point.get_array()[ids] = cp
            data = replay_joint(joint, state, collect=True, cases=[case])
            return {
                "SOC": data[prefix + ".battery.SOC"],
                "I_bat": data[prefix + ".battery.I_bat"],
                "temperature": data[prefix + ".thermal.temperature"].T,
                "P_sol": data[prefix + ".power_solar.P_sol"],
                "P_RW": data[prefix + ".rw_power.P_RW"].T,
                "P_comm": data[prefix + ".values.P_comm"],
                "Data": data[prefix + ".comm_data.Data"],
            }

        print(f"\nRepairing case {case.case_index} at {nodes:,} nodes...", flush=True)
        started = perf_counter()
        try:
            cp, trajectory, metrics = repair_baseline(
                case.B,
                starting,
                initial_soc,
                case.dt,
                lambda data: total_power(data["P_sol"], np.zeros(case.n), data["P_RW"]),
                replay,
            )
            if (
                not np.all(np.isfinite(cp))
                or not all(
                    np.all(np.isfinite(values)) for values in trajectory.values()
                )
                or not all(np.isfinite(value) for value in metrics.values())
                or abs(metrics["periodicity_error"]) >= 1e-9
                or metrics["inequality_violation"] >= 1e-9
                or np.any(cp < 0)
                or np.any(cp > 25)
            ):
                raise ValueError(
                    "Repaired trajectory did not pass battery or power bounds"
                )
            repaired[prefix + "_CP_P_comm"] = cp
            for field, values in {
                "time_s": np.arange(case.n) * case.dt,
                "P_comm": trajectory["P_comm"],
                "SOC": trajectory["SOC"],
                "I_bat": trajectory["I_bat"],
                "Data": trajectory["Data"],
            }.items():
                repaired[prefix + "_" + field] = values
            for field in ("CP_gamma", "CP_Isetpt"):
                repaired[prefix + "_" + field] = arrays[prefix + "_" + field]
            reports.append(
                {
                    "case_index": case.case_index,
                    "downloaded_data": float(trajectory["Data"][-1]),
                    "battery_feasibility": metrics,
                    "maximum_power_control_change_W": float(np.max(abs(cp - starting))),
                    "repair_seconds": perf_counter() - started,
                }
            )
        except (RuntimeError, ValueError, FloatingPointError) as error:
            failures.append({"case_index": case.case_index, "reason": str(error)})
            print(f"Case {case.case_index} failed: {error}", flush=True)

    passed = not failures and len(reports) == len(cases)
    result = {
        "source": str(source),
        "nodes_per_case": nodes,
        "controls_per_profile": controls,
        "duration_hours": 12.0,
        "cadre_source": provenance,
        "reference_feasibility_validated": passed,
        "feasibility_validated": False,
        "validated_initialization": False,
        "native_model_checked": False,
        "optimization_run": False,
        "power_repair_uses_linear_programs": True,
        "converged": False,
        "validated": False,
        "method": "Reference battery repair using linear programs; initial SOC and other design controls are held fixed.",
        "objective_scale": summary["objective_scale"],
        "fin_angle_rad": fin,
        "antenna_angle_rad": antenna,
        "cases": reports,
        "failures": failures,
    }
    if passed:
        result["total_downloaded_data"] = sum(
            item["downloaded_data"] for item in reports
        )
        np.savez_compressed(output / "controls.npz", **repaired)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    if not passed:
        raise RuntimeError(
            f"Reference repair failed for {len(failures)} case(s); see {output / 'summary.json'}"
        )
    print(
        f"\nReference repair passed; no native model check or optimization was run.\nResults: {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--nodes", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair_reference(args.source, args.output, args.nodes)


if __name__ == "__main__":
    main()
