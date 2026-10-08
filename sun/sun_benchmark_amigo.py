import pickle

import amigo as am
import numpy as np

from cadre_paths import cadre_path

from attitude.attitude_benchmark_validation import (
    get_case_value,
    matrix_history,
    vector_history,
)

from sun.sun_reference import (
    sun_position_eci,
    sun_line_of_sight,
    sun_position_body,
    sun_position_spherical,
)

from sun.sun_amigo import (
    SunPositionBody,
    SunPositionSpherical,
    SunLOSVisible,
    SunLOSEclipse,
    SunLOSTransition,
    extract_vector,
)


class SunPositionECIBenchmark(am.Component):
    def __init__(self, LD):
        super().__init__()

        self.add_constant(
            "d2r",
            value=np.pi / 180.0,
        )

        self.add_constant(
            "LD",
            value=LD,
        )

        self.add_input("t")

        self.add_input(
            "r_e2s_I",
            shape=(3,),
        )

        self.add_constraint(
            "res",
            shape=(3,),
        )

    def compute(self):
        d2r = self.constants["d2r"]
        LD = self.constants["LD"]

        t = self.inputs["t"]
        r = self.inputs["r_e2s_I"]

        T = (
            LD
            + t / 3600.0 / 24.0
        )

        L = (
            d2r * 280.460
            + d2r * 0.9856474 * T
        )

        g = (
            d2r * 357.528
            + d2r * 0.9856003 * T
        )

        Lambda = (
            L
            + d2r
            * 1.914666
            * am.sin(g)
            + d2r
            * 0.01999464
            * am.sin(2.0 * g)
        )

        eps = (
            d2r * 23.439
            - d2r
            * 3.56e-7
            * T
        )

        x = am.cos(Lambda)

        y = (
            am.sin(Lambda)
            * am.cos(eps)
        )

        z = (
            am.sin(Lambda)
            * am.sin(eps)
        )

        self.constraints["res"] = [
            r[0] - x,
            r[1] - y,
            r[2] - z,
        ]


class DummyObjective(am.Component):
    def __init__(self):
        super().__init__()

        self.add_input("value")
        self.add_objective("obj")

    def compute(self):
        value = self.inputs["value"]

        self.objective["obj"] = (
            1.0e-12
            * value
            * value
        )


def report_difference(
    name,
    amigo_value,
    reference_value,
):
    difference = np.abs(
        amigo_value
        - reference_value
    )

    print()
    print(
        f"Maximum {name} difference:"
    )
    print(
        np.max(difference)
    )

    print(
        f"Mean {name} difference:"
    )
    print(
        np.mean(difference)
    )


def classify_los(
    states,
    r_e2s_I,
):
    r1 = 6378.137 * 0.85
    r2 = 6378.137

    visible = []
    eclipse = []
    transition = []

    for i in range(
        states.shape[0]
    ):
        r_b = states[i, 0:3]
        r_s = r_e2s_I[i]

        dot = np.dot(
            r_b,
            r_s,
        )

        dist = np.linalg.norm(
            np.cross(
                r_b,
                r_s,
            )
        )

        if dot >= 0.0:
            visible.append(i)

        elif dist <= r1:
            eclipse.append(i)

        elif dist >= r2:
            visible.append(i)

        else:
            transition.append(i)

    return (
        np.asarray(
            visible,
            dtype=int,
        ),
        np.asarray(
            eclipse,
            dtype=int,
        ),
        np.asarray(
            transition,
            dtype=int,
        ),
    )


def main():
    # --------------------------------------------------------
    # Load CADRE benchmark
    # --------------------------------------------------------

    data_path = cadre_path(
        "test/data1346.pkl"
    )

    with open(
        data_path,
        "rb",
    ) as file:
        data = pickle.load(
            file,
            encoding="latin1",
        )

    orbit_source = get_case_value(
        data,
        "r_e2b_I",
    ).T

    O_BI_source = matrix_history(
        get_case_value(
            data,
            "O_BI",
        )
    )

    LD = float(
        get_case_value(
            data,
            "LD",
        ).reshape(-1)[0]
    )

    n = orbit_source.shape[0]

    t1 = 0.0
    t2 = 43200.0

    dt = (
        t2 - t1
    ) / (n - 1)

    times = np.linspace(
        t1,
        t2,
        n,
    )

    print()
    print("=" * 70)
    print(
        "CADRE SUN AMIGO BENCHMARK"
    )
    print("=" * 70)

    print()
    print("Number of nodes:")
    print(n)

    print()
    print("Time step [s]:")
    print(dt)

    print()
    print("Launch date parameter LD:")
    print(LD)

    # --------------------------------------------------------
    # Python reference
    # --------------------------------------------------------

    r_e2s_I_reference = (
        sun_position_eci(
            times,
            LD=LD,
        )
    )

    LOS_reference = (
        sun_line_of_sight(
            orbit_source,
            r_e2s_I_reference,
        )
    )

    r_e2s_B_reference = (
        sun_position_body(
            O_BI_source,
            r_e2s_I_reference,
        )
    )

    (
        azimuth_reference,
        elevation_reference,
    ) = sun_position_spherical(
        r_e2s_B_reference
    )

    O_BI_flat = (
        O_BI_source.reshape(
            n,
            9,
        )
    )

    (
        visible_indices,
        eclipse_indices,
        transition_indices,
    ) = classify_los(
        orbit_source,
        r_e2s_I_reference,
    )

    print()
    print("Full sunlight nodes:")
    print(
        len(visible_indices)
    )

    print()
    print("Full eclipse nodes:")
    print(
        len(eclipse_indices)
    )

    print()
    print("Transition nodes:")
    print(
        len(transition_indices)
    )

    # --------------------------------------------------------
    # Components
    # --------------------------------------------------------

    sun_eci = (
        SunPositionECIBenchmark(
            LD
        )
    )

    sun_body = SunPositionBody()

    sun_spherical = (
        SunPositionSpherical()
    )

    objective = DummyObjective()

    model = am.Model(
        "cadre_sun_benchmark"
    )

    model.add_component(
        "sun_eci",
        n,
        sun_eci,
    )

    model.add_component(
        "sun_body",
        n,
        sun_body,
    )

    model.add_component(
        "sun_spherical",
        n,
        sun_spherical,
    )

    model.add_component(
        "objective",
        1,
        objective,
    )

    # Add LOS components only when
    # that branch actually exists.
    if len(visible_indices) > 0:
        model.add_component(
            "los_visible",
            len(visible_indices),
            SunLOSVisible(),
        )

    if len(eclipse_indices) > 0:
        model.add_component(
            "los_eclipse",
            len(eclipse_indices),
            SunLOSEclipse(),
        )

    if len(transition_indices) > 0:
        model.add_component(
            "los_transition",
            len(transition_indices),
            SunLOSTransition(),
        )

    # --------------------------------------------------------
    # Main Sun links
    # --------------------------------------------------------

    model.link(
        "sun_eci.r_e2s_I",
        "sun_body.r_e2s_I",
    )

    model.link(
        "sun_body.r_e2s_B",
        "sun_spherical.r_e2s_B",
    )

    model.link(
        "sun_eci.r_e2s_I[0, 0]",
        "objective.value[0]",
    )

    # --------------------------------------------------------
    # Fix benchmark time
    # --------------------------------------------------------

    model.set_meta(
        "value",
        "sun_eci.t[:]",
        times,
    )

    model.set_meta(
        "lower",
        "sun_eci.t[:]",
        times,
    )

    model.set_meta(
        "upper",
        "sun_eci.t[:]",
        times,
    )

    # --------------------------------------------------------
    # Fix benchmark attitude
    # --------------------------------------------------------

    for i in range(9):
        variable = (
            f"sun_body.O_BI[:, {i}]"
        )

        model.set_meta(
            "value",
            variable,
            O_BI_flat[:, i],
        )

        model.set_meta(
            "lower",
            variable,
            O_BI_flat[:, i],
        )

        model.set_meta(
            "upper",
            variable,
            O_BI_flat[:, i],
        )

    # --------------------------------------------------------
    # Initial Sun guesses
    # --------------------------------------------------------

    for i in range(3):
        model.set_meta(
            "value",
            f"sun_eci.r_e2s_I[:, {i}]",
            r_e2s_I_reference[:, i],
        )

        model.set_meta(
            "value",
            f"sun_body.r_e2s_B[:, {i}]",
            r_e2s_B_reference[:, i],
        )

    model.set_meta(
        "value",
        "sun_spherical.azimuth[:]",
        azimuth_reference,
    )

    model.set_meta(
        "value",
        "sun_spherical.elevation[:]",
        elevation_reference,
    )

    # --------------------------------------------------------
    # LOS branches
    # --------------------------------------------------------

    if len(visible_indices) > 0:
        model.set_meta(
            "value",
            "los_visible.LOS[:]",
            LOS_reference[
                visible_indices
            ],
        )

    if len(eclipse_indices) > 0:
        model.set_meta(
            "value",
            "los_eclipse.LOS[:]",
            LOS_reference[
                eclipse_indices
            ],
        )

    if len(transition_indices) > 0:
        model.set_meta(
            "value",
            "los_transition.LOS[:]",
            LOS_reference[
                transition_indices
            ],
        )

        transition_r_b = (
            orbit_source[
                transition_indices,
                0:3,
            ]
        )

        transition_r_s = (
            r_e2s_I_reference[
                transition_indices
            ]
        )

        for i in range(3):
            rb_variable = (
                f"los_transition.r_b[:, {i}]"
            )

            rs_variable = (
                f"los_transition.r_s[:, {i}]"
            )

            model.set_meta(
                "value",
                rb_variable,
                transition_r_b[:, i],
            )

            model.set_meta(
                "lower",
                rb_variable,
                transition_r_b[:, i],
            )

            model.set_meta(
                "upper",
                rb_variable,
                transition_r_b[:, i],
            )

            model.set_meta(
                "value",
                rs_variable,
                transition_r_s[:, i],
            )

            model.set_meta(
                "lower",
                rs_variable,
                transition_r_s[:, i],
            )

            model.set_meta(
                "upper",
                rs_variable,
                transition_r_s[:, i],
            )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    print()
    print(
        "Building AMIGO Sun benchmark..."
    )

    model.build_module()

    print("Build successful.")

    model.initialize()

    print(
        "Initialization successful."
    )

    x = model.create_vector()

    optimizer = am.Optimizer(
        model,
        x,
    )

    options = {
        "max_iterations": 100,
        "convergence_tolerance": 1e-10,
        "initial_barrier_param": 0.1,
        "barrier_strategy": "heuristic",
        "max_line_search_iterations": 4,
    }

    print()
    print(
        "Solving AMIGO Sun benchmark..."
    )

    optimizer.optimize(
        options
    )

    print()
    print(
        "AMIGO solve complete."
    )

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    r_e2s_I_amigo = extract_vector(
        x,
        "sun_eci.r_e2s_I",
        n,
        3,
    )

    r_e2s_B_amigo = extract_vector(
        x,
        "sun_body.r_e2s_B",
        n,
        3,
    )

    azimuth_amigo = np.asarray(
        x[
            "sun_spherical.azimuth[:]"
        ]
    )

    elevation_amigo = np.asarray(
        x[
            "sun_spherical.elevation[:]"
        ]
    )

    LOS_amigo = np.zeros(n)

    if len(visible_indices) > 0:
        LOS_amigo[
            visible_indices
        ] = x[
            "los_visible.LOS[:]"
        ]

    if len(eclipse_indices) > 0:
        LOS_amigo[
            eclipse_indices
        ] = x[
            "los_eclipse.LOS[:]"
        ]

    if len(transition_indices) > 0:
        LOS_amigo[
            transition_indices
        ] = x[
            "los_transition.LOS[:]"
        ]

    # --------------------------------------------------------
    # AMIGO vs Python
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AMIGO VS PYTHON SUN"
    )
    print("=" * 70)

    report_difference(
        "r_e2s_I",
        r_e2s_I_amigo,
        r_e2s_I_reference,
    )

    report_difference(
        "LOS",
        LOS_amigo,
        LOS_reference,
    )

    report_difference(
        "r_e2s_B",
        r_e2s_B_amigo,
        r_e2s_B_reference,
    )

    report_difference(
        "azimuth",
        azimuth_amigo,
        azimuth_reference,
    )

    report_difference(
        "elevation",
        elevation_amigo,
        elevation_reference,
    )

    # --------------------------------------------------------
    # AMIGO vs original CADRE
    # --------------------------------------------------------

    r_e2s_I_source = vector_history(
        get_case_value(
            data,
            "r_e2s_I",
        )
    )

    LOS_source = get_case_value(
        data,
        "LOS",
    ).reshape(-1)

    r_e2s_B_source = vector_history(
        get_case_value(
            data,
            "r_e2s_B",
        )
    )

    azimuth_source = get_case_value(
        data,
        "azimuth",
    ).reshape(-1)

    elevation_source = get_case_value(
        data,
        "elevation",
    ).reshape(-1)

    print()
    print("=" * 70)
    print(
        "AMIGO VS ORIGINAL CADRE"
    )
    print("=" * 70)

    report_difference(
        "r_e2s_I",
        r_e2s_I_amigo,
        r_e2s_I_source,
    )

    report_difference(
        "LOS",
        LOS_amigo,
        LOS_source,
    )

    report_difference(
        "r_e2s_B",
        r_e2s_B_amigo,
        r_e2s_B_source,
    )

    report_difference(
        "azimuth",
        azimuth_amigo,
        azimuth_source,
    )

    report_difference(
        "elevation",
        elevation_amigo,
        elevation_source,
    )


if __name__ == "__main__":
    main()