# Validated result for the research meeting

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

Suggested description: “We validated an AMIGO optimization of communication
power, solar-current schedules, and initial SOC for one full 12-hour CADRE case.
It converged in 58 iterations, with 12.95 minutes of solver time, and improved
downloaded data by 4.08% over a feasible repaired benchmark.”

This is a reduced study relative to the original CADRE problem: one seasonal
case, prescribed roll/allocation, and fixed fin and antenna angles. Do not present
its runtime as a measured speedup over the original six-case study's reported
100 hours. The CADRE data units retain the legacy code's numerical convention.

Roll/allocation extension: implemented and derivative-tested, with a validated
shorter 600-node demonstration in `../results/design_smoke/`. The full 1,500-node,
4,285-design-variable extension still needs convergence, independent replay,
and final comparison plots. Its trial points are not completed results.
