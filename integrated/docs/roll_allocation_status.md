# Roll/allocation optimization status

Full fixed-design reference: **validated**, in `../results/pcomm_design_reference/`.
58 iterations, 776.73 solver seconds, KKT error 1.00e-9, data 5264.536115.
The shorter 600-node roll/allocation case remains validated in `../results/design_smoke/`.

The full 12-hour roll/allocation optimization is unfinished and currently stopped.
Last executed log: `../results/pcomm_roll_allocation_native/run_consistent_evaluator.log`.
The next sparse-pattern restart was not executed: automatic approval review hit
a usage limit (review failure, not a safety determination). The user agreed an
earlier validated case can be used for the meeting.
Its summary must say `validated: true` before reporting a completed result.

Next restart command from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 pixi run --manifest-path ~/git/amigo/pixi.toml python -u -m integrated.cadre_optimize_pcomm --optimize-roll --optimize-allocation --linear-solver native --warm-start integrated/results/pcomm_roll_allocation_native/restart_evaluator.npz --warm-start-push 1e-11 --output integrated/results/pcomm_roll_allocation_native
```

Current backend: full native LDL with analytic external lookup Hessians, explicit
temporal ordering, full inertia checks, nominal-unit primal regularization, and
checked/refined full KKT solves. METIS crashes in this environment on the full
matrix; explicit ordering avoids it. Native LDL regression checks include external
Hessian sparsity, permutation, repeated factorization, and barrier-preserving
regularization. The alternative SuperLU/control-reduction paths remain experimental
for this full roll case; they did not produce a validated full-size result.

`solvers/evaluator.py` works around two issues in the installed AMIGO evaluator:
objective directional derivatives must exclude constraint-gradient rows (otherwise
they include the spurious multiplier-step term dlambda^T c(x)), and Hessian probes
must first refresh external caches at the current point. Objective scaling is also
preserved. `evaluator_validation.py` reproduces the derivative bug and verifies the
fix with finite differences and an external-cache regression. The external AMIGO
checkout is unchanged.

Checkpoints contain primal and constraint multipliers plus bound multipliers and
barrier parameter. Same-problem warm starts restore them, then apply a small strict
bound push. Filter history, inertia history, and iteration counter restart; this is
not an exact optimizer resume. Report solver time/iterations as warm-start segments,
not a cold benchmark. `restart_evaluator.npz` is an immutable copy of the best
previous full-design point (primal residual 2.10e-4, raw dual residual 4.30e-4).
`checkpoint.npz` tracks the latest point; `best_checkpoint.npz` tracks lowest KKT
error in the current segment. Preserve those before launching another segment.

Passed checks: full-orbit lookup derivatives and original tensor replay, native
assembled Hessian including fixed allocation, block solves against dense references,
positive-curvature preservation, chunked elimination, and exact trajectory solves.
An additional full-checkpoint Hessian finite-difference diagnostic is recorded in
`../results/pcomm_roll_allocation_native/point_diagnostic.log`.

Earlier unsuccessful paths and logs remain in their respective result folders.
They are not validated full-size roll/allocation results.

Remaining for the extension: restart and converge, independently replay it, generate matched
fixed-design comparison plots, record measured convergence/time with warm-start
provenance, and update this note with the completed status.

Meeting result: use `../results/pcomm_design_reference/summary.json` and
`optimization.png`; see `meeting_result.md`. No new solve is needed to present it.
The native backend now drops stored numerical zeros and safely expands its symbolic
pattern if previously zero Hessian entries become nonzero. Dense-reference
regressions passed, but this performance change has not yet run on the full case.
The existing ordering already uses four-control spline aliases to capture local
support; no additional spline-ordering change was made.

After cleanup, run the maintained CLI above; the temporary debug launcher
uses old internal import paths and is no longer needed. Model construction is in
`cadre_model.py`, checkpoint continuation in `cadre_checkpoint.py`, and solver
adapters in `solvers/`. The 1,500-node fixed-design workflow was revalidated after
refactoring: identical generated C++, three warm-start iterations, KKT 1.00e-9.
Its check output is separate from the original reference result.
