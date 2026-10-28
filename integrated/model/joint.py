"""Assemble case-local trajectories around one shared 86-variable design."""

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import amigo as am
import numpy as np
from scipy.sparse import csr_matrix

from integrated.model.builder import stage_model
from integrated.model.design import SurfaceAllocation
from integrated.model.shared_design import DesignAngles, SharedDesignPhysics
from integrated.optimization.options import parse_args
from integrated.reference.case import prepare_case
from integrated.reference.benchmark import load_benchmark, resample_benchmark
from integrated.reference.shared_angles import SharedAngleLookups


class TotalDownloadedData(am.Component):
    def __init__(self, count, scale=1.0):
        super().__init__()
        self.count = count
        self.name = f"TotalDownloadedData{count}"
        self.add_constant("data_scale", value=float(scale))
        self.add_input("Data", shape=(count,))
        self.add_objective("obj")

    def compute(self):
        self.objective["obj"] = -self.constants["data_scale"] * sum(
            self.inputs["Data"][i] for i in range(self.count)
        )


@dataclass
class JointCadreModel:
    model: object
    cases: list
    physics: list
    objective_scale: float = 1.0

    @property
    def design_variable_count(self):
        return sum(14 * case.m + 1 for case in self.cases) + 86

    def design_indices(self):
        names = []

        for case in self.cases:
            prefix = f"case{case.case_index}."
            names += [
                prefix + name
                for name in [
                    "control_points.CP_P_comm[:]",
                    "control_points.CP_Isetpt[:, :]",
                    "control_points.CP_gamma[:]",
                    "battery_ic.initial_soc[:]",
                ]
            ]

        names += [
            "shared_surface.cellInstd[0, :]",
            "shared_angles.finAngle[:]",
            "shared_angles.antAngle[:]",
        ]
        return self.model.get_indices(names)

    def hessian_correction(self, shape):
        """Accumulate every case's curvature at the aliased shared indices."""
        result = csr_matrix(shape)

        for physics in self.physics:
            result = result + physics.hessian_correction(shape)

        return result

    def set_point_provider(self, provider):
        for physics in self.physics:
            physics.point_provider = provider

    def replay(self, state):
        from integrated.reference.joint_replay import replay_joint

        return replay_joint(self, state)

    def initialize(self, directory):
        """Compile in an isolated directory and initialize, without optimization."""

        with build_directory(directory):
            self.model.build_module()
            self.model.initialize(order_type=am.OrderingType.NATURAL)


@contextmanager
def build_directory(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    sys.path.insert(0, str(directory))
    os.chdir(directory)

    try:
        yield
    finally:
        os.chdir(previous)
        sys.path.remove(str(directory))


def prepare_joint_cases(
    indices, nodes=1500, controls=300, *, full_duration=False, repair=False
):
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Choose distinct benchmark cases")

    cases, options = [], []
    data = load_benchmark()
    if full_duration:
        data = resample_benchmark(data, nodes)

    for index in indices:
        args = parse_args(
            [
                "--case-index",
                str(index),
                "--nodes",
                str(nodes),
                "--control-points",
                str(controls),
                "--optimize-roll",
                "--optimize-allocation",
            ]
        )
        try:
            case = prepare_case(
                args,
                repair_initialization=repair,
                lookups_type=SharedAngleLookups,
                data_override=data,
                bounded_controls=True,
                current_margin=1e-5,
                maximum_power_start=repair,
                allow_unrepaired_baseline=True,
            )
        except RuntimeError as error:
            if not repair or "repair" not in str(error).lower():
                raise
            print(
                f"Case {index}: power-only seed repair unavailable; using an unrepaired reference. {error}",
                flush=True,
            )
            case = prepare_case(
                args,
                repair_initialization=False,
                lookups_type=SharedAngleLookups,
                data_override=data,
                bounded_controls=True,
                current_margin=1e-5,
                maximum_power_start=False,
                allow_unrepaired_baseline=True,
            )
        case.design_external = SharedDesignPhysics(case.lookups)
        cases.append(case)
        options.append(args)

    return cases, options


def assemble_joint_model(cases, options, objective_scale=1.0):
    if not cases or len(cases) != len(options):
        raise ValueError("Supply one option set for each case")

    if len({case.case_index for case in cases}) != len(cases):
        raise ValueError("Joint case identifiers must be unique")

    for case, args in zip(cases, options):
        if not (
            case.design_physics
            and case.free_solar
            and args.optimize_roll
            and args.optimize_allocation
            and not args.fixed_initial_soc
            and isinstance(case.design_external, SharedDesignPhysics)
        ):
            raise ValueError(
                "The joint formulation requires all local controls and the shared-angle physics"
            )

    model = am.Model("cadre_joint")
    model.add_component("shared_surface", 1, SurfaceAllocation())
    model.add_component("shared_angles", 1, DesignAngles())
    model.add_component(
        "objective", 1, TotalDownloadedData(len(cases), objective_scale)
    )
    shared = cases[0]

    for name, value, lower, upper in [
        (
            "shared_surface.cellInstd[0, :]",
            shared.initial_guess["cellInstd"].ravel(),
            0,
            1,
        ),
        ("shared_angles.finAngle[:]", shared.lookups.fin_angle, 0, np.pi / 2),
        (
            "shared_angles.antAngle[:]",
            shared.lookups.antenna_angle,
            -np.pi / 4,
            np.pi / 4,
        ),
    ]:
        model.set_meta("value", name, value)
        model.set_meta("lower", name, lower)
        model.set_meta("upper", name, upper)

    physics = []

    for position, (case, args) in enumerate(zip(cases, options)):
        if not np.array_equal(
            case.initial_guess["cellInstd"], shared.initial_guess["cellInstd"]
        ):
            raise ValueError("Cases must start with the same shared surface allocation")

        if (case.lookups.fin_angle, case.lookups.antenna_angle) != (
            shared.lookups.fin_angle,
            shared.lookups.antenna_angle,
        ):
            raise ValueError("Cases must start with the same shared angles")

        prefix = f"case{case.case_index}"
        submodel = stage_model(case, args, include_objective=False)
        model.add_model(prefix, submodel)

        for source, target in [
            ("shared_surface.cellInstd[0, :]", "surface.cellInstd[0, :]"),
            ("shared_angles.finAngle[:]", "angles.finAngle[:]"),
            ("shared_angles.antAngle[:]", "angles.antAngle[:]"),
        ]:
            model.link(source, prefix + "." + target)

        model.link(
            f"{prefix}.comm_data.Data[{case.n-1}]", f"objective.Data[0, {position}]"
        )
        case.design_external.bind(model, prefix + ".")
        physics.append(case.design_external)

    return JointCadreModel(model, cases, physics, objective_scale)
