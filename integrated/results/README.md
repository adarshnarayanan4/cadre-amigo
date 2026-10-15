# Saved optimization results

Existing directory names and checkpoints are preserved so restart paths and
recorded provenance remain valid.

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
| `*_validation.json` | Numerical validation measurements |

Use each `summary.json` to determine validation status; trial objectives are not
completed optimization results. The original meeting summaries and trajectories
were retained during source cleanup.
