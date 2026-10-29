# Six-case optimization: 5,132 design variables

This result optimizes six 12-hour cases together. It uses 300 time nodes and
60 control points per profile. The cases share the spacecraft's surface
allocation, fin angle, and antenna angle.

The saved result converged with an NLP error of **8.73e-9**. Its native
model equations, bounds, battery limits, periodicity, and reference replay
checks passed. Total downloaded data was **17,138.66 CADRE data units**.
The final warm-start segment took **19 iterations and 8 minutes 30 seconds**
of solver time. That time does not include the preceding work, setup,
compilation, or validation.

The [convergence plot](../results/joint_300/convergence.png),
[case plots](../results/joint_300/), and
[summary](../results/joint_300/summary.json) are saved with the result.
The 600-node replay of these controls failed battery checks, so this result
does not establish convergence at a finer grid. The model uses trapezoidal
integration and nodewise battery bounds; upstream CADRE uses RK4 and KS
aggregation.

## Setup

You need Git, Pixi, CMake, and a C++ compiler. From a directory such as `~/git`,
clone the project, AMIGO, and CADRE:

```sh
git clone https://github.com/adarshnarayanan4/cadre-amigo.git
git clone https://github.com/smdogroup/amigo.git
git clone https://github.com/OpenMDAO/CADRE.git CADRE-OpenMDAO

git -C amigo checkout af83ffb2024816ea5fe93f969e40ef819c15acad
git -C CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303

cd amigo
pixi install
pixi run install-amigo
cd ../cadre-amigo
```

Keep AMIGO and CADRE beside `cadre-amigo`. If CADRE is elsewhere, set
`CADRE_ROOT` to the CADRE checkout. Do not copy CADRE into this repository.

## Repeat the final segment

From `cadre-amigo`, run:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_joint \
  --optimize --cases 0 1 2 3 4 5 --full-space-trials \
  --nodes 300 --control-points 60 --time-grid full --objective-scale 1 \
  --warm-start integrated/results/joint_300/checkpoint_central.npz \
  --max-iterations 40 --output /tmp/cadre-six-case-300
```

This resumes the preparation checkpoint included in the repository; it is not
a fresh run. The model is compiled on the machine running the command.
Iteration count and runtime can differ.

Results will be in `/tmp/cadre-six-case-300`. Check `summary.json`: a successful
run has `converged` and `validated` both set to `true`. Plots and solver history
are saved alongside it. Use another output directory to keep multiple runs.
