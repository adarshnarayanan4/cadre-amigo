# Single Case 600 node optimization

This is a fresh all-variable solve of case 0 over 12 hours, with 600 time nodes,
120 spline controls per profile, and 1,767 design variables. Communication power,
12 solar-current profiles, roll, initial SOC, surface allocation, fin angle, and
antenna angle are free.

The recorded run converged in **126 iterations and 2,366.77 solver seconds
(39 minutes 27 seconds)**. Preparation, initial repairs, compilation, and replay
are additional. Iteration counts and runtimes can differ on another machine.
Downloaded data is 4,605.910122 CADRE units, NLP error is 1.000e-9, and all
independent validation checks passed.

## First-time setup

Requires Git and Pixi. Keep the three repositories beside one another. From a
parent directory such as `~/git`, clone repositories you do not already have:

```sh
git clone https://github.com/adarshnarayanan4/cadre-amigo.git
git clone https://github.com/smdogroup/amigo.git
git clone https://github.com/OpenMDAO/CADRE.git CADRE-OpenMDAO
```

For a new AMIGO checkout, use the tested revision and its installation tasks:

```sh
cd amigo
git checkout af83ffb2024816ea5fe93f969e40ef819c15acad
pixi install
pixi run install-amigo
cd ../cadre-amigo
git -C ../CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303
```

If the repositories and AMIGO environment already exist, update `cadre-amigo`
and check the reference revision before running. Set `CADRE_ROOT` only if the
reference checkout is elsewhere. Do not reuse a Mac build on Linux; this command
compiles a new model for the local machine.

## Run

From the `cadre-amigo` directory:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_joint \
  --optimize --cases 0 \
  --nodes 600 --control-points 120 --time-grid full \
  --objective-scale 1 --initial-barrier 0.1 \
  --initial-bound-push 1e-4 --full-space-trials \
  --max-iterations 300 \
  --output /tmp/cadre-case0-full
```

Choose a new output directory if that path contains a result you want to keep.
The command starts fresh and does not need a checkpoint. Let initialization
repairs and compilation finish before the iteration table starts. The limit is
300 iterations; the recorded run needed 126. The table objective includes
barrier terms, so use the final replayed data total for comparisons.

## Check the result

Read `/tmp/cadre-case0-full/summary.json`. Successful completion has
`validated: true`, `converged: true`, and every `validation_checks` entry true.
The folder also contains trajectories, convergence history, and plots.

The [published summary](../results/single_case_600/summary.json),
[trajectory plot](../results/single_case_600/case_0.png), and
[convergence plot](../results/single_case_600/convergence.png) preserve the
measured run. `reproduction.json` records its command and dependency revisions.
Local checkpoints and compiled builds are excluded from Git.

This is a single-case, reduced-grid result. Full six-case convergence and
fine-grid optimality remain unfinished. Our formulation uses trapezoidal
dynamics and nodewise battery bounds; upstream CADRE uses RK4 and KS aggregation.
