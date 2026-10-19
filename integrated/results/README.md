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

Start with [`pcomm_design_reference/`](pcomm_design_reference/) for the full
validated fixed-design case. Its `optimization.png` shows the trajectories and
`convergence.png` shows the solver residuals. The `design_smoke/` directory is a
shorter demonstration of free roll and allocation.
