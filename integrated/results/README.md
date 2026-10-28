# Saved results

| Folder | Status |
| --- | --- |
| `single_case_600/` | Fresh converged all-variable case 0; 600 nodes, 120 controls, 12 hours; 126 iterations, 2,366.77 solver seconds |
| `joint_case0_full_600/` | Converged all-variable case 0; 600 nodes, 120 controls, 12 hours |
| `joint_case0_prefix_600/` | Converged all-variable case 0 over 4.795 hours; sparse reference solve agrees |
| `openmdao_default_1500/` | Validated default single case at 1,500 nodes; prescribed geometry |
| `pcomm_design_reference/` | Validated 1,500-node case with restored design physics; prescribed roll/allocation |
| `joint_300/` | Six-case candidate at 300 nodes; physically feasible, **not converged** |
| `joint_start_600/` | Feasible six-case start; **not optimized** |
| `joint_case0_start_1500/` | Feasible single-case start at 1,500 nodes / 300 controls; **not optimized** |
| `validation/` | Numerical check reports |

The [single-case instructions](../docs/single_case_600.md) reproduce the fresh
`single_case_600` run.

Read `summary.json` before using a result. `validated` includes solver
convergence; `feasibility_validated` alone does not. Keep summaries, histories,
trajectories, and figures together. Checkpoints are local files ignored by Git.

The full-duration 600-node result's 39.6074% gain uses a separately audited
feasible baseline on the same grid. Its 220.51-second timing covers the final
restart only; `continuation.json` preserves earlier segments. Fine-grid replay
fails battery bounds despite a 0.3659% change in data. `mesh_replay.json` records
that check; the repaired fine start is not an optimization result.

The six-case candidate downloads 17,099.976926 units and passes native equations,
replay, bounds, and periodicity. NLP error is 69.02 and complementarity is 0.01.
Its reference is infeasible, so no percentage gain is reported.

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

See [meeting notes](../docs/meeting_result.md) for measured results and
[the joint guide](../docs/joint_optimization.md) for run and restart commands.
