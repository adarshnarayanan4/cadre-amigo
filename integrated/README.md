# Integrated CADRE optimization

This package couples thermal, reaction-wheel, solar-power, battery, and
communication dynamics for one CADRE benchmark case. Run commands from the
repository root with the AMIGO environment:

```sh
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.cadre_optimize_pcomm
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.plot_pcomm_results
```

Set `CADRE_ROOT` if the external CADRE checkout is not at `../CADRE`. Compilation
requires the AMIGO environment's CMake and compiler on `PATH`.

## Code layout

| File | Responsibility |
| --- | --- |
| `cadre_optimize_pcomm.py` | CLI entry point and workflow |
| `cadre_options.py` | Arguments, defaults, and option validation |
| `cadre_reference.py` | Benchmark inputs, baseline repair, and initial trajectories |
| `cadre_feasibility.py` | Battery feasibility metrics and baseline-repair LPs |
| `cadre_amigo.py` | Integrated AMIGO components |
| `cadre_model.py` | Component registration, links, bounds, and initialization |
| `cadre_optimizer.py` | Solver selection and optimization execution |
| `cadre_checkpoint.py` | Compatible warm starts and atomic checkpoint saves |
| `cadre_results.py` | Independent replay, validation, and result files |
| `cadre_reporting.py` | Spaced terminal sections, aligned metrics, and repair progress tables |
| `design_reference.py` | Roll-dependent exposure, antenna gain, and voltage references |
| `design_amigo.py` | Roll components and external lookup derivatives |
| `solvers/` | Sparse LU, native LDL, and the CADRE evaluator adapter |
| `plot_pcomm_results.py` | Saved-result figures and matched comparisons |
| `cadre_benchmark_amigo.py` | Integrated fixed-input benchmark replay |
| `*_validation.py` | Numerical checks, matching the subsystem naming convention |
| `docs/` | Meeting result, extension status, and historical results |
| `archive/` | Original SOC-only formulation, retained for reproducibility |

`power/voltage_spline.py` contains the temperature/current voltage model;
`power/power_voltage_spline_validation.py` checks it. Equation variables such as
`SOC`, `P_comm`, `CP_Isetpt`, and `cellInstd` retain the CADRE names used in the
existing subsystem files. Python functions use snake_case and classes use
PascalCase. The former `cadre_single_case` command remains a compatibility entry
point for `cadre_benchmark_amigo`.

## Formulation

The default is a complete 12-hour case with 1,500 nodes and 300 cubic B-spline
controls per profile: communication power, 12 solar-current profiles, and initial
SOC give **3,901 design variables**. Roll and cell/radiator allocation are prescribed.
Fin angle, antenna mounting, orbit, and the other five seasonal cases are fixed or
absent. Trajectory constraints use trapezoidal integration.

Bounds are 0–25 W for communication controls, 0–0.4 A for solar-current controls,
0.2–1.0 for SOC, and −10–5 A for battery current. Final SOC equals optimized initial
SOC. Coupled voltage uses the original CADRE table; temperatures stay in its
50–400 K domain. The comparison baseline is repaired for current limits and SOC
periodicity by minimizing changes to its original controls, separately from the
interior starting guess.

Useful options (`--help` lists all):

- `--design-physics`: restore roll-dependent lookup/attitude physics while retaining
  fixed roll/allocation; provides the matched reference for design optimization.
- `--optimize-roll --optimize-allocation`: add 300 roll controls and 84 continuous
  surface allocations, for **4,285 design variables**. Allocation interpolates
  cell/radiator thermal properties; temperature-dependent voltage carries its effect
  into power. Full-size convergence is still unfinished; see the status note below.
- `--fixed-solar-currents`: retain prescribed currents with coupled voltage.
- `--fixed-initial-soc`: retain the benchmark initial SOC.
- `--legacy-fixed-voltage`: reproduce the earlier fixed-voltage formulation.
- `--nodes`, `--control-points`, `--output`: select a grid and result directory.
  Short windows retain the original timestep and must include ground contact.

The default SuperLU backend retains AMIGO's interior-point algorithm. The native
LDL backend (`--linear-solver native`) adds full inertia checks with explicit
temporal ordering. Both include analytic external lookup Hessians. The native
backend drops stored zeros and expands its symbolic pattern if a zero becomes
nonzero. `--reduce-all-controls` selects an experimental SuperLU reduction and
cannot be combined with native LDL.

The evaluator adapter excludes constraint rows from objective directional
products and refreshes external caches before Hessian evaluation. These local
adapters leave the external AMIGO checkout unchanged.

## Results and restarts

Results stay under `integrated/results/`; use `--output` for a separate run.
Each completed attempt writes `summary.json`, `solver_history.json`, and
`trajectories.npz`. Plotting adds `optimization.png`, `solar_currents.png`, and,
for restored design physics, `design_controls.png`. `--output` on the plotting
command saves figures separately from the source result.

Validation requires solver convergence, KKT and primal residuals below `1e-6`,
bound violations and SOC periodicity below `1e-7`, replay errors below `1e-5`,
a feasible comparison baseline, and improved data. Failures save diagnostics
and raise an error. Downloaded data uses the legacy CADRE numerical convention,
labeled **CADRE data units** rather than asserting a bytes conversion.

`checkpoint.npz` saves the latest point; `best_checkpoint.npz` saves the lowest
recorded KKT error. Restart the same formulation using `--warm-start PATH`.
Same-problem warm starts restore primal, constraint, and bound multipliers and
the saved barrier. `--warm-start-push` controls the strict-bound projection.
`--release-design-controls` permits a restored-physics reference to release fixed
roll/allocation; other model settings must match. `--initial-barrier` overrides
the saved barrier. Filter/inertia history and iteration counters restart, so
report these runtimes as warm-start segments.

## Validation

```sh
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.block_solver_validation
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.native_ldl_validation
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.evaluator_validation
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.checkpoint_validation
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.design_physics_validation
pixi run --manifest-path ~/git/amigo/pixi.toml python -m power.power_voltage_spline_validation
```

The last two checks reuse compiled `cadre_optimize_design` and
`cadre_optimize_pcomm` modules. Run the corresponding optimization model build
first if those modules are absent.

The full-size fixed-design reference is validated: **58 iterations, 776.73 solver
seconds, 4.0783% more data**, KKT `1.00e-9`. The cleanup was also checked with an
identical generated C++ model and a successful full-size checkpoint continuation.
Saved meeting results were preserved.

- [Meeting result and figures](docs/meeting_result.md)
- [Roll/allocation extension status and restart command](docs/roll_allocation_status.md)
- [Earlier optimization stages](docs/history.md)
- [Saved result index](results/README.md)
