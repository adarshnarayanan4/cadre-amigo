# Single Case 600 node optimization

This run optimizes one 12-hour case using 600 time points and 120 controls per
profile. It changes communication power, solar currents, roll, starting battery
charge, surface allocation, fin angle, and antenna angle.

It converged in **126 iterations and 39 minutes 27 seconds** of solver time.
Setup and model compilation take extra time. All checks passed, including
battery limits and a separate calculation of the final result. Your runtime
and iteration count may differ.

## Set up once

You need Git and Pixi. Keep the project, AMIGO, and CADRE folders next to each
other. From a folder such as `~/git`, download any you do not already have:

```sh
git clone https://github.com/adarshnarayanan4/cadre-amigo.git
git clone https://github.com/smdogroup/amigo.git
git clone https://github.com/OpenMDAO/CADRE.git CADRE-OpenMDAO
```

For a new installation, use the AMIGO and CADRE versions used for this run:

```sh
cd amigo
git checkout af83ffb2024816ea5fe93f969e40ef819c15acad
pixi install
pixi run install-amigo
cd ../cadre-amigo
git -C ../CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303
```

If everything is already installed, update `cadre-amigo` with `git pull` and
check that CADRE is on the version above. See [setup notes](server_preparation.md)
if your folders are in different locations.

## Run the optimization

From the `cadre-amigo` folder:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.cadre_joint \
  --optimize --cases 0 \
  --nodes 600 --control-points 120 --time-grid full \
  --objective-scale 1 --initial-barrier 0.1 \
  --initial-bound-push 1e-4 --full-space-trials \
  --max-iterations 300 \
  --output /tmp/cadre-case0-full
```

This starts a new run. It does not need a saved checkpoint. Wait for the starting
point repairs and model compilation before the iteration table appears.
The run can take up to 300 iterations; the saved run took 126.
Use a different output folder if you want to keep an earlier result.

## Find the results

Open `/tmp/cadre-case0-full/summary.json`. A successful run has `validated: true`,
`converged: true`, and all entries under `validation_checks` set to `true`.
Plots and the iteration history are in the same folder. Copy the folder out of
`/tmp` if you want to keep it permanently.

The saved run downloaded **4,605.910122 CADRE data units**, with a final solver
error of **1.000e-9**. You can view its [summary](../results/single_case_600/summary.json),
[trajectory plot](../results/single_case_600/case_0.png), and
[convergence plot](../results/single_case_600/convergence.png).

This is one case on a smaller grid. The six-case model has converged on a 300-node grid, but the full-resolution
optimization is still unfinished. Our time integration and battery constraints also differ from
upstream CADRE; see the [model notes](cadre_source.md).
