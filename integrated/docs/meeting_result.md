# Results for the research meeting

## All-variable single case

The most recent validated result frees communication power, all twelve
solar-current profiles, roll, initial SOC, surface allocation, fin angle, and
antenna angle. It covers a full 12-hour case at 600 nodes and 120 spline controls
per profile, with 1,767 design variables.

Use the [trajectory figure](../results/single_case_600/case_0.png),
[convergence figure](../results/single_case_600/convergence.png), and
[shared design](../results/single_case_600/shared_design.png).
[Reproduction instructions](single_case_600.md) start a fresh run.

| Measurement | Result |
| --- | --- |
| Repaired feasible baseline | 3,299.079088 CADRE data units |
| Optimized downloaded data | 4,605.910122 CADRE data units |
| Improvement over that baseline | 39.6120% |
| Final NLP error | 1.000e-9 |
| Native primal residual | 9.966e-10 |
| Independent replay, battery limits, periodicity, bounds | Passed |
| Fresh-run iterations | 126 |
| Solver time | 2,366.77 seconds (39 minutes 27 seconds) |

This solve started without a checkpoint. Timing excludes initial repairs,
preparation, compilation, and independent replay. The gain compares two designs
on the same 600-node formulation; it is not a gain over the original six-case
OpenMDAO optimization. This implementation uses trapezoidal dynamics and
nodewise battery bounds; upstream CADRE uses RK4 and KS aggregation.

The earlier `joint_case0_full_600` result remains separate: data 4,605.758184,
NLP error 2.866e-9, and 17 iterations / 220.51 seconds in its final restart.
Its `continuation.json` retains the preceding segments and baseline audit.
The grid checks below used that earlier solution, not the new fresh run.

## Check at 1,500 nodes

Replaying the 600-node design at 1,500 nodes with the same 120 controls changes
downloaded data by 0.3659%, but exposes a charging-current violation: −10.0861 A
against the −10 A limit. SOC also slightly exceeds its bounds. The coarse result
must therefore be presented with its grid resolution.

The controls were then refitted to 300 points per profile, and communication
power and initial SOC were repaired on the fine grid. The resulting
`joint_case0_start_1500` point passes native equations, independent replay,
battery limits, periodicity, and variable bounds:

- 1,500 nodes and a 4,287-variable model.
- Downloaded data: 4,603.227195 CADRE units.
- Native equality error: 1.596e-12; bound violation: zero.
- SOC range: 0.201221–1.000000; battery current: −10.000000–0.865857 A.
- SOC periodicity error: 1.077e-14.
- Its checkpoint loads with the matching optimizer configuration.

This is a verified feasible starting point, not a fine-grid optimization result.
The [grid comparison](../results/joint_case0_start_1500/grid_check_case_0.png)
shows the original charging-current violation and the repaired trajectory.

## Earlier 1,500-node case with prescribed geometry

Use the [trajectory figure](../results/pcomm_design_reference/optimization.png) and
[solar-current figure](../results/pcomm_design_reference/solar_currents.png).
The measured result is saved in `../results/pcomm_design_reference/summary.json`.

- One complete 12-hour case: 1,500 time nodes, 300 spline control points.
- 3,901 design variables: communication power, 12 solar-current profiles,
  and initial battery SOC. Roll and cell/radiator allocation are prescribed.
- 58 iterations, 776.73 seconds (12 minutes 57 seconds) of solver time.
  This excludes baseline preparation, compilation, initialization, and replay.
- Downloaded data: 5,264.536 CADRE units versus a feasible repaired benchmark
  of 5,058.245 units, an increase of 4.0783%.
- KKT error 1.00e-9; primal infeasibility 2.15e-11; no bound violation.
- SOC stays within 0.2–1.0; battery current stays within the −10–5 A limits;
  initial/final SOC agree within 1.78e-15.
- Independent replay using the original tensor lookup splines agrees within
  2.15e-11 in the largest reported replay error. `validated` is true.

This is a reduced study relative to the original CADRE problem: one seasonal
case, prescribed roll/allocation, and fixed fin and antenna angles. Do not present
its runtime as a measured speedup over the original six-case study's reported
100 hours. The CADRE data units retain the legacy code's numerical convention.

## Remaining optimization work

The six-case model shares one spacecraft allocation and the two angles across
all cases. A 300-node / 60-control candidate now passes native equations,
independent replay, battery bounds, and periodicity for all six cases. It has
5,132 design variables and downloads 17,099.976926 data units in total.
The native equality residual is 4.183e-12 and bound violation is zero.

Its 40-step continuation did **not** converge: NLP error is 69.02 and
complementarity is 0.01. The
[six-case convergence figure](../results/joint_300/convergence.png) and
[case-0 trajectory](../results/joint_300/case_0.png) identify it as a feasible,
unconverged candidate. The stored reference is infeasible, so it is not used for
a percentage gain. Six-case continuation is still running. A separate 30-step 1,500-node
all-variable trial failed convergence and physical validation; it does not
replace the validated single-case results.

The next checks are six-case convergence, the all-variable 1,500-node / 300-control
run, and grid refinement. The full six-case model has 25,292 design variables.
The earlier 1,500-node fixed-geometry result and the newer all-variable 600-node
result are separate results; neither establishes completion of that model.
