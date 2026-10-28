# CADRE-AMIGO

A port of the [OpenMDAO/CADRE](https://github.com/OpenMDAO/CADRE) CubeSat model
to AMIGO, with subsystem comparisons and coupled spacecraft optimization.

## Setup

Keep the reference checkout beside this repository:

```sh
git clone https://github.com/OpenMDAO/CADRE.git ../CADRE-OpenMDAO
git -C ../CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303
```

Set `CADRE_ROOT` if it is elsewhere. Use AMIGO's Pixi environment, with CMake
and a C++ compiler available. Change the manifest path below if needed.
The [source notes](integrated/docs/cadre_source.md) explain the pinned revision
and the move from the earlier OpenMDAO-Plugins checkout.

## Run

From this repository's root:

```sh
# Optimize one case: communication power, solar currents, and initial SOC.
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_optimize_pcomm

# Replot the saved prescribed-geometry result without optimizing.
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.plot_pcomm_results

# Generate subsystem figures from the reference calculations.
pixi run --manifest-path ../amigo/pixi.toml python -m visualizations
```

Single-case options include `--case-index 0` through `5`, `--design-physics`,
`--optimize-roll`, and `--optimize-allocation`. Use `--help` for all options.
The default uses 1,500 nodes and 300 spline controls over 12 hours. Reducing
single-case nodes shortens the window; the joint runner instead supports a
reduced grid spanning all 12 hours.

The [joint optimization guide](integrated/docs/joint_optimization.md) covers
shared geometry, solver options, checkpoints, and mesh checks. For a Linux
machine, see [server setup](integrated/docs/server_preparation.md).

## Results

[Run the Single Case 600 node optimization](integrated/docs/single_case_600.md).

The all-variable single case has converged at **600 nodes and 120 controls**
over 12 hours, with 1,767 design variables. Downloaded data is 4,605.910122
CADRE units and NLP error is 1.000e-9. The fresh run took 126 iterations and
39 minutes 27 seconds of solver time. Native equations, independent replay,
battery bounds, and SOC periodicity pass.

A separate validated 1,500-node case with prescribed roll and allocation
has 3,901 design variables and downloads 5,264.536 units. These are different
optimization problems; their data values are not a matched comparison.

The six-case optimization and the 1,500-node all-variable optimization have
not converged. The coarse single-case design violates battery limits when
replayed on the fine grid; a repaired fine-grid start is saved separately.
Our formulation uses trapezoidal dynamics and battery bounds at each node;
upstream CADRE uses RK4 and KS-aggregated constraints.

See the [saved results](integrated/results/README.md) for validation status and
the [meeting notes](integrated/docs/meeting_result.md) for figures and timings.

## Code

| Location | Contents |
| --- | --- |
| `orbit/`, `attitude/`, `sun/`, `solar/` | Motion, orientation, sunlight, and exposure |
| `thermal/`, `power/`, `battery/`, `reaction_wheel/`, `communication/` | Subsystem equations and reference checks |
| `parameters/` | Spline controls |
| `integrated/model/` | Coupled components, bounds, and shared designs |
| `integrated/reference/` | Benchmark inputs, lookup models, repair, and replay |
| `integrated/optimization/`, `integrated/solvers/` | Optimizer setup, checkpoints, and solver adapters |
| `integrated/reporting/` | Terminal output, saved results, and plots |
| `integrated/validation/` | Numerical checks |
| `visualizations/` | [Subsystem figure gallery](visualizations/README.md) and plotting code |

The entry points are `integrated.cadre_optimize_pcomm` for one case and
`integrated.cadre_joint` for shared-design optimization. The coupled fixed-input
benchmark is `integrated.cadre_benchmark_amigo`.

## Checks and formatting

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.prepare_cadre --check-lookups
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.block_solver
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.evaluator
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.checkpoint
```

Joint checks are listed in the [joint guide](integrated/docs/joint_optimization.md#reports-and-numerical-checks).
Saved reports are in `integrated/results/validation/`. Source comparisons and
integration-error checks are kept separate.

Format with `python -m black .`. Use blank lines between calculations, preserve
CADRE variable names, and explain units or numerical choices in comments.
