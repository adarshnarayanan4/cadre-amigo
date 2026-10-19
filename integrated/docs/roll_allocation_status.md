# Roll/allocation optimization status

Full fixed-design reference: **validated**, in `../results/pcomm_design_reference/`.
58 iterations, 776.73 solver seconds, KKT error 1.00e-9, data 5264.536115.
The shorter 600-node roll/allocation case remains validated in `../results/design_smoke/`.

The full 12-hour roll/allocation optimization is unfinished and currently stopped.
Latest completed trial: `../results/pcomm_roll_allocation_superlu_resume/run.log`.
The sparse-pattern update has passed regression checks but has not completed a
full-size solve. A completed result requires solver convergence and independent
validation, recorded as `validated: true` in its summary.

Next restart command from repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=4 pixi run --manifest-path ~/git/amigo/pixi.toml python -u -m integrated.cadre_optimize_pcomm --optimize-roll --optimize-allocation --linear-solver native --max-iterations 20 --warm-start integrated/results/pcomm_roll_allocation_native/restart_evaluator.npz --warm-start-push 1e-11 --output integrated/results/pcomm_roll_allocation_next
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
preserved. `validation/evaluator.py` reproduces the derivative bug and verifies the
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
The native backend drops stored numerical zeros and safely expands its symbolic
pattern if previously zero Hessian entries become nonzero. Dense-reference
regressions passed. A full-size restart in `../results/pcomm_roll_allocation_resume/`
reduced the symbolic pattern from 12,624,480 to 2,212,328 stored entries. It was
stopped during numerical factorization before accepting a new iterate. A process
sample confirmed work inside native LDL; fewer stored entries alone did not
establish a speedup. The original restart remains unchanged.
Native factorization now prints a running marker, elapsed seconds, and inertia
counts for every attempt. This distinguishes repeated inertia corrections from
time spent in one factorization. `--max-iterations` limits a restart segment;
the default remains 200.

A SuperLU comparison restarted from the same point for 20 iterations. It took
602.14 solver seconds and failed convergence and independent validation. Primal
infeasibility rose from 2.096e-4 to 9.982; the final dual residual was 4.616e-2.
The independently replayed SOC periodicity error was -0.03248. This trial does
not replace the preserved native restart or the validated fixed-design result.
Its summary and convergence plot remain local in
`integrated/results/pcomm_roll_allocation_superlu_resume/`; unsuccessful trial
outputs are not published as working results. Warm-start plots omit iteration 0 because AMIGO logs it
before the checkpoint callback restores the point; the raw history retains it.

A subsequent native timing trial isolated the slow phase. The first numerical
factorization took 3.42 seconds, with inertia (133257 positive, 126029 negative)
instead of the required (133285 positive, 126001 negative). The second
factorization, requested by inertia correction, was stopped after roughly 132
seconds without finishing. No new iterate was accepted. The diagnostic remains
local at `integrated/results/pcomm_roll_allocation_native_timed/factorization_diagnostic.json`.
The next investigation is why regularization makes the numerical factorization
so much slower, rather than another long uninstrumented restart. Any scaling or
ordering change must retain correct inertia and checked full KKT solve residuals.
The existing ordering already uses four-control spline aliases to capture local
support; no additional spline-ordering change was made.

Model construction is in `model/builder.py`, checkpoint handling in
`optimization/checkpoint.py`, and linear-solver adapters in `solvers/`.
The full fixed-design case remains the validated reference for this extension.
