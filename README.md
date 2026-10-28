# CADRE-AMIGO

This project uses AMIGO to optimize the [CADRE CubeSat model](https://github.com/OpenMDAO/CADRE).
It includes the spacecraft systems, optimization code, checks, and plots.

## Run the 600-node single case

Start with the [setup and run instructions](integrated/docs/single_case_600.md).
They include the software versions and the exact command used for the saved run.

This case covers 12 hours with 600 time points and 1,767 design variables.
It optimizes communication power, solar currents, roll, starting battery charge,
surface allocation, fin angle, and antenna angle.

The run converged in **126 iterations and 39 minutes 27 seconds** of solver time.
Setup and compilation take extra time. Downloaded data was 4,605.910122 CADRE
units, with a final solver error of 1.000e-9. Battery limits and separate checks
of the result all passed.

See the [saved results](integrated/results/README.md) and
[meeting notes](integrated/docs/meeting_result.md) for plots and measurements.

## What is still unfinished?

The full six-case optimization and the 1,500-node run with all design variables
free have not converged. A coarse-grid solution can violate battery limits on
a finer grid, so the 600-node result does not establish a fine-grid solution.

Our model uses trapezoidal time integration and checks battery bounds at each
time point. Upstream CADRE uses RK4 and KS aggregation. The
[source notes](integrated/docs/cadre_source.md) explain these differences.

## Other commands

Run these from the `cadre-amigo` folder after setup:

```sh
# Plot the saved single-case result.
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.plot_pcomm_results

# Make the spacecraft-system plots without optimizing.
pixi run --manifest-path ../amigo/pixi.toml python -m visualizations

# Check the CADRE inputs and lookup tables.
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.prepare_cadre --check-lookups
```

## Where to find the code

- The spacecraft systems are in `orbit/`, `attitude/`, `sun/`, `solar/`,
  `thermal/`, `power/`, `battery/`, `reaction_wheel/`, and `communication/`.
- `parameters/` contains the spline controls.
- `integrated/` contains the combined model, optimization, solver adapters,
  result reporting, and numerical checks.
- `visualizations/` contains the [figure gallery](visualizations/README.md).

The [joint optimization guide](integrated/docs/joint_optimization.md) explains
solver options, checkpoints, and checks. The [server guide](integrated/docs/server_preparation.md)
covers running on another machine. Format Python code with `python -m black .`.
