# Saved results

| Folder | Status |
| --- | --- |
| `single_case_600/` | Fresh converged all-variable case 0; 600 nodes, 120 controls, 12 hours; 126 iterations, 2,366.77 solver seconds |
| `joint_case0_full_600/` | Converged all-variable case 0; 600 nodes, 120 controls, 12 hours |
| `joint_case0_prefix_600/` | Converged all-variable case 0 over 4.795 hours; sparse reference solve agrees |
| `openmdao_default_1500/` | Validated default single case at 1,500 nodes; prescribed geometry |
| `pcomm_design_reference/` | Validated 1,500-node case with restored design physics; prescribed roll/allocation |
| `joint_300/` | Converged and validated six-case result at 300 nodes; 19 iterations in the final warm-start segment |
| `joint_300/candidate_600/` | Feasible six-case checkpoint at 600 nodes; **not converged** |
| `joint_300/candidate_600_latest/` | Later feasible six-case checkpoint at 600 nodes; **not converged** |
| `joint_300/reference_start_1500/` | Six-case reference battery repair at 1,500 nodes; **native check and optimization not run** |
| `joint_start_600/` | Feasible six-case start; **not optimized** |
| `joint_case0_start_1500/` | Feasible single-case start at 1,500 nodes / 300 controls; **not optimized** |
| `validation/` | Numerical check reports |

The [single-case instructions](../docs/single_case_600.md) reproduce the fresh
`single_case_600` run.

Read `summary.json` before using a result. `validated` includes solver
convergence; `feasibility_validated` alone does not. Keep summaries, histories,
trajectories, and figures together. Checkpoints are ignored by Git except for
the included `joint_300/checkpoint_central.npz`.

The full-duration 600-node result's 39.6074% gain uses a separately audited
feasible baseline on the same grid. Its 220.51-second timing covers the final
restart only; `continuation.json` preserves earlier segments. Fine-grid replay
fails battery bounds despite a 0.3659% change in data. `mesh_replay.json` records
that check; the repaired fine start is not an optimization result.

The six-case result downloads 17,138.658661 units and passes native equations,
replay, bounds, and periodicity. NLP error is 8.726e-9, with no bound violation.
The final full-space segment took 19 iterations and 509.75 seconds; this is not
a cold-run timing. `continuation.json` records its preceding preparation and
`reproduction.json` records the restart command. The central checkpoint is
included in the repository. Its reference is infeasible, so no
percentage gain is reported. The earlier candidate is preserved in the external
archive under `joint_300_candidate/`.

Holding its controls fixed and replaying at 600 nodes gives 15,803.085184 units
(−7.7928%). All six cases fail SOC bounds or periodicity. This reference-only
check is saved in `mesh_replay_600.json` and its arrays; no fine-grid optimization
or native fine-grid check was run. The [grid figure](joint_300/mesh_replay_600.png)
shows the data change and per-case battery errors.

The derived [600-node start](joint_300/refined_start_600/summary.json) refits to
120 controls and passes native equations, bounds, replay, battery checks, and
checkpoint loading. It has 10,172 design variables and 15,762.256739 data units.
It is a feasible initialization, not an optimized result.

The separate [interior start](joint_300/interior_start_600/summary.json) keeps
SOC between 0.215 and 0.985 across all six cases and power controls at least
0.001 W inside their bounds. It passes the same checks. Data totals
15,349.629783 units; native equation error is 4.485e-12. Its six repairs took
114.14 seconds, excluding setup and verification. Both starts are
unoptimized; the original minimal repair retains the larger data total.

The [latest 600-node candidate](joint_300/candidate_600_latest/candidate_overview.png)
downloads 25,385.917042 data units and passes native equation, bound, replay,
and battery checks. Its NLP error is 475.447, so it is **not converged**. The
reference replay is also used to project trial trajectories; the compiled
native equation check is a separate cross-check. Its final warm-start step
took 336.35 solver seconds, not a complete cold-run time. Its checkpoint is
local and ignored by Git. The fine-grid checks below use the earlier candidate,
not this one.

The [600-node candidate](joint_300/candidate_600/candidate_overview.png) has
24,878.542032 data units, compared with 15,762.256739 for the feasible repaired
start on the same grid, a 57.8362% increase. Its native equation error is
4.581e-12; bounds, battery limits, and periodicity pass. Its NLP error is
2.153e3, so it is not a converged optimization result. The reference replay is
also used to project trial trajectories, so the separate compiled native
equation check is the stronger cross-check. Its `summary.json` records the
checks and warm-start segment; the checkpoint is ignored by Git. The saved
controls reconstruct the same candidate and pass native and battery checks
without the checkpoint; the [joint guide](../docs/joint_optimization.md) has
the replay command.

Holding its 120 controls fixed at 1,500 nodes changes data by +0.5743%, but
all six cases fail SOC periodicity. Two also violate SOC bounds and two exceed
the charging-current limit. The largest errors are 0.000267 SOC, 0.021827 A,
and 0.000509 SOC periodicity. The [grid figure](joint_300/candidate_600/mesh_replay_1500.png)
shows the per-case checks. This is a reference replay, with no native 1,500-node
model check or fine-grid optimization.

To regenerate its figure:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.reporting.joint_plots \
  integrated/results/joint_300/candidate_600 \
  --checkpoint-baseline integrated/results/joint_300/refined_start_600
```

The [1,500-node reference repair](joint_300/reference_start_1500/reference_start_overview.png)
adjusts communication-power controls while keeping the starting SOC, solar
currents, roll, and shared design fixed. All six cases pass the reference
battery and periodicity checks; saved controls reproduce the same result.
Data totals 25,013.758615 units, and the largest power-control change is
0.022820 W. This uses 120 controls per profile. It has not been checked in the
compiled 1,500-node model or optimized; the full target uses 300 controls.

To reproduce the reference repair, use an empty output directory:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.reference_repair \
  integrated/results/joint_300/candidate_600 --nodes 1500 \
  --output /tmp/cadre-reference-repair-1500
```

Replot a single-case result:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.plot_pcomm_results \
  integrated/results/pcomm_design_reference
```

Earlier experiments and superseded notes are preserved outside this repository
in `../cadre-results-archive/20261028-173811/`, relative to the repository root.
Its manifest records verified file checksums. Copy a folder back to restore it,
or pass its absolute path to the plot command. Historical source differences
are described in the [source notes](../docs/cadre_source.md).

See [the joint guide](../docs/joint_optimization.md) for run and restart commands.
