# Saved optimization results

Each optimization directory contains its summary, iteration history, trajectories,
and figures. Numerical regression reports are grouped in `validation/`.
Optimization directory names remain stable so checkpoint restart paths stay valid.

| Directory | Meaning |
| --- | --- |
| `pcomm_design_reference/` | Validated full 12-hour, 3,901-variable restored-physics reference; meeting result |
| `pcomm_solar/` | Validated full 12-hour solar-current optimization |
| `pcomm_coupled_voltage/` | Matched fixed-current comparison |
| `pcomm_initial_soc/` | Earlier fixed-voltage, free-initial-SOC case |
| `pcomm_constrained/` | Earlier constrained, fixed-initial-SOC case |
| `pcomm/` | Historical SOC-only result without current/periodicity constraints |
| `design_smoke/` | Validated shorter roll/allocation demonstration |
| `pcomm_roll_allocation*/` | Unfinished full-size solver trials and restart checkpoints |
| `validation/` | Solver, checkpoint, evaluator, and derivative checks |

Use each `summary.json` to determine validation status; trial objectives are not
completed optimization results.

Every saved case with trajectories can be replotted without running AMIGO:

```sh
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.plot_pcomm_results integrated/results/pcomm_solar
pixi run --manifest-path ~/git/amigo/pixi.toml python -m integrated.plot_pcomm_results --all-results
```

`--all-results` refreshes each case in its existing directory. Add `--output`
to collect figures in a separate directory, with one subfolder per saved case.

| Figure | Contents |
| --- | --- |
| `optimization.png` | Communication power, SOC, current, solar power when modeled, and cumulative data; visible bounds and baseline comparisons |
| `solar_currents.png` | Twelve panel currents with one shared scale and legend |
| `design_controls.png` | Roll schedule and the labeled 7 × 12 thermal-allocation matrix |
| `convergence.png` | KKT error, primal violation, dual stationarity, and complementarity, when history is available |

Headers identify validated results and unvalidated trials. Footers show data
totals and SOC periodicity where available. Trial curves are labeled as trial
iterates. These figures use the same formatting as the
[subsystem guide](../../visualizations/README.md).

Start with [`pcomm_design_reference/`](pcomm_design_reference/) for the full
validated fixed-design case. Its `optimization.png` shows the trajectories and
`convergence.png` shows the solver residuals. The `design_smoke/` directory is a
shorter demonstration of free roll and allocation.
