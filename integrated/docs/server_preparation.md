# Server setup

Use the same CADRE-AMIGO, AMIGO, and reference-source revisions on each machine.
The reference checkout belongs beside `cadre-amigo`, outside its Git repository.
From the project root:

```sh
git clone https://github.com/OpenMDAO/CADRE.git ../CADRE-OpenMDAO
git -C ../CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303
```

The code finds `../CADRE-OpenMDAO` automatically. Set `CADRE_ROOT` if the checkout
is elsewhere. The old OpenMDAO-Plugins layout is rejected. Source hashes are
pinned in `cadre_source.json`; see the [source audit](cadre_source.md).

Install AMIGO using its own build instructions. Run through its Pixi environment,
with CMake and a C++ compiler on `PATH`:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.prepare_cadre \
  --check-lookups --output /scratch/cadre/preparation.json
```

This checks all six cases, spline reconstruction, and independent lookup replay.
It records source identity, package versions, and available build tools. It does
not compile or optimize. Mac extension modules cannot be reused on Linux; models
must be compiled on the server.

## Build and run

The single-case command is `integrated.cadre_optimize_pcomm`. Select a case with
`--case-index 0` through `--case-index 5`. Separate single-case runs have independent
designs. The joint command shares allocation, fin angle, and antenna angle across
all selected cases:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_joint \
  --build --output /scratch/cadre/joint-model
```

The default full model has six cases, 1,500 nodes, 300 control points per profile,
and 25,292 design variables. Trajectory states and multipliers are additional.
`--build` compiles and initializes without optimizing. The
[joint optimization guide](joint_optimization.md) covers optimization commands,
the validated all-variable single case over 12 hours, unfinished six-case tests,
and checkpoints.

Use a new output directory for each attempt. The joint runner compiles into its
output directory's `build/` folder. Single-case commands can generate build files
in the working directory, so concurrent single-case runs need separate working
directories. Generated builds are disposable and ignored by Git.

## Transfer results and checkpoints

Keep `summary.json`, `solver_history.json`, and `trajectories.npz` together.
Transfer checkpoints explicitly; Git ignores them except for the included
300-node preparation checkpoint. The retained unfinished runs
are listed in the [result index](../results/README.md).

Checkpoint settings must match the model. Single-case checkpoints cannot be
used for joint runs. Joint checkpoints also check case order, grid, objective
scale, source fingerprint, and vector shape. A source edit may require a new
run even when an old checkpoint has the same number of variables.

A validated six-case starting point is saved in `integrated/results/joint_start_600`.
Its checkpoint is ignored by Git and must be transferred explicitly. Use
`--reset-multipliers` for its first optimization run, as shown in the joint guide.
`--repair-start` generates a fresh feasible starting point on a cold run.
An all-variable single-case start at 1,500 nodes and 300 controls is saved in
`integrated/results/joint_case0_start_1500`. Feasibility and checkpoint loading
pass; this folder contains a feasible start, not an optimization result.
Run it with:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_joint \
  --optimize --cases 0 --nodes 1500 --control-points 300 --time-grid full \
  --objective-scale 1 --full-space-trials --reset-multipliers \
  --warm-start integrated/results/joint_case0_start_1500/checkpoint.npz \
  --output /scratch/cadre/case0-fine
```

Its tracked `controls.npz` is also available for `--initial-design` when the
checkpoint cannot be transferred.
To change the control count for a finer grid, combine `--initial-design DIR`
with `--refit-controls`. The importer checks the reference and time window, fits
bounded controls, recomputes trajectories, and repairs the battery before
optimization. A checkpoint restart requires the original grid instead.
The experimental `--primal-barrier` mode includes that cold-start repair and
keeps trial trajectories and SOC periodicity consistent with the controls.
The projected mode prepared the coarse six-case point; the subsequent
full-space continuation established convergence at 300 nodes. Start with one case worker;
benchmark `--case-workers N` on the server before assuming it is faster. Set
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` when
factoring cases concurrently.

Full-resolution joint convergence is still outstanding. Test convergence before
committing to a long server run; more CPU power alone will not fix a stalled
nonlinear solve. Thread counts and scheduler commands depend on the server.
