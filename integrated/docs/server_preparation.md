# Preparing the full CADRE problem

The integrated command now accepts `--case-index 0` through `--case-index 5`.
Each selects its own orbit, Sun geometry, ground contact, attitude, controls,
and initial SOC from the original benchmark. Case 0 remains the default.
Cases 1–5 get separate default output directories, such as
`integrated/results/pcomm_roll_allocation_case_1/`.

`cadre_optimize_pcomm` builds one case per run. The separate `cadre_joint` command
assembles the joint model, including both free shared angles. No optimization
was run for this model-development work.

## Check the inputs without optimizing

From the repository root, in an environment with NumPy, SciPy, Matplotlib,
and AMIGO:

```sh
python -m integrated.prepare_cadre --check-lookups
```

This checks all six cases and writes `integrated/results/preparation.json`.
It checks array shapes and finite values, reconstructs the original spline
schedules, and compares exposure/gain contractions with independent tensor
replay. It does not repair a baseline, compile a model, factor a KKT matrix,
or call an optimizer.

For selected cases or a separate report destination:

```sh
python -m integrated.prepare_cadre --cases 0 3 5 --check-lookups --output /tmp/cadre-preparation.json
```

The report includes the source-data checksum, package versions, compiler/CMake
locations, case identifiers, and measured reference differences. The lookup
checks establish agreement between two implementations of the same equations;
they do not establish optimization convergence or validate new fin/antenna derivatives.

## Server setup

Use the same CADRE-AMIGO, CADRE, and AMIGO revisions on the server. Install AMIGO
using its repository's environment/build instructions. Generated Python extension
modules must be compiled on that machine; the Mac `.so` files are platform-specific.

Set the CADRE checkout location and run preparation through the AMIGO environment:

```sh
export CADRE_ROOT=/path/to/CADRE
pixi run --manifest-path /path/to/amigo/pixi.toml python -m integrated.prepare_cadre --check-lookups --output /scratch/cadre/preparation.json
```

Check that CMake and a C++ compiler appear in the report's `build_tools` entries.
Preparation can check the reference data even when these tools are missing;
model compilation will need both on `PATH`.

Keep the preparation report with the eventual run. `trajectories.npz` files
preserve completed results; selected checkpoints preserve unfinished runs.
Checkpoints are ignored by Git, so transfer the intended restart explicitly:
`integrated/results/pcomm_roll_allocation_native/restart_evaluator.npz` is the
preserved case-0 restart. Old checkpoints without a case identifier are treated
as case 0, and a checkpoint from another case is rejected even when shapes match.

Use a new output directory for each attempt. Model generation currently writes
build files in the working directory, so concurrent case runs need separate
checkouts/build directories. Independent runs do not enforce shared designs
across the six cases.

The slow native regularized factorization remains unresolved. Profile it on the
server before committing to a long optimization. Server thread counts and job
scheduler settings can be selected once the hardware and scheduler are known.

## Joint model

To check a small model locally:

```sh
python -m integrated.cadre_joint --nodes 8 --control-points 4 --build --output /tmp/cadre-joint-check
python -m integrated.validation.shared_angles
python -m integrated.validation.joint_model
```

The assembly check compiles a small six-case model, checks shared aliases,
the summed objective, separate SOC periodicity, an independent replay, and the
complete assembled Hessian. It does not factor a KKT matrix or optimize.
Short contact-free windows are permitted for this structural check; they are
not meaningful optimization cases.

On the server, the original-resolution build command is:

```sh
python -m integrated.cadre_joint --build --output /scratch/cadre/joint-model
```

Without `--build`, the command stages the equations and writes `model.json`.
With `--build`, it also compiles and initializes in `output/build/`. Both modes
leave optimization disabled. Reference initial guesses are replayed without
LP baseline repairs, so they are not guaranteed feasible.

The model API is `prepare_joint_cases`, `assemble_joint_model`, and
`JointCadreModel.initialize`. Native AMIGO Hessians need the external correction
from `JointCadreModel.hessian_correction`, with `set_point_provider` attached to
the current point. `JointCadreModel.replay` independently propagates every case
at the current shared design. The existing single-case restart driver should
not be used for a joint vector; joint optimizer/checkpoint/reporting integration
is remaining run infrastructure.

## Design variables

The original formulation in `CADRE/CADRE_mdp.py` uses:

| Design variables | Scope | Bounds |
| --- | --- | --- |
| Communication-power spline weights | Each case | 0–25 W |
| Twelve solar-current spline profiles | Each case | 0–0.4 A |
| Roll spline weights | Each case | 0–π/2 rad |
| Initial SOC | Each case | 0.2–1.0 |
| 84 cell/radiator allocations | Shared across all cases | 0–1 |
| Fin angle | Shared across all cases | 0–π/2 rad |
| Antenna angle | Shared across all cases | −π/4–π/4 rad |

At 300 controls per profile, each case has 4,201 local design variables.
Six cases plus 86 shared variables give **25,292 design variables**. This is
the design count, excluding trajectory states and constraint multipliers.
The joint objective maximizes the sum of final downloaded data, with separate
SOC periodicity for each case. The original uses aggregated battery inequalities;
the current implementation uses limits at each node. That formulation difference
must be recorded when comparing results.

The shared-angle implementation includes first derivatives, angle-angle and
angle/local-state mixed derivatives, and multiplier-weighted Hessians. Fin
affects exposure, thermal dynamics, and voltage; antenna angle affects gain.
The compiled assembly has passed a reduced six-case check with 428 design
variables, and angle derivative checks cover all six benchmark geometries.
Full-resolution compilation, solver profiling, convergence, and validation of
an optimized joint result remain for the server.

## Checks available now

```sh
python -m integrated.validation.case_selection
python -m integrated.validation.checkpoint
python -m integrated.validation.design_physics --case-index 0
```

The last command accepts any index from 0 to 5. These checks do not optimize;
the derivative check initializes a small in-memory model without compilation.
Case isolation, legacy checkpoint compatibility, cross-case rejection, and
analytic lookup Jacobians/Hessians have been checked for this preparation step.
