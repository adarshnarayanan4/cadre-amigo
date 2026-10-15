# Earlier optimization stages

## Validated solar-current stage

| Case | Free design variables | Downloaded data | Gain vs feasible benchmark |
| --- | ---: | ---: | ---: |
| Feasible repaired benchmark | — | 5057.986516 | — |
| Coupled voltage, fixed solar currents, free initial SOC | 301 | 5256.737633 | 3.9295% |
| Coupled voltage, optimized solar currents and initial SOC | 3901 | 5264.456877 | 4.0821% |

Optimizing the currents adds **0.1468%** downloaded data over the matched
fixed-current optimum. Both runs use temperature-dependent voltage, the same
1500-node grid, free initial SOC, battery limits, and SOC periodicity.
Integrated solar energy rises from 37.5140 to 40.0626 Wh (**6.7937%**).
The larger energy gain does not translate directly into downloaded data because
communication opportunities and battery constraints also limit the objective.

The solar-current solve took **73 iterations and 1002.61 seconds (16.71 minutes)**
of solver time. It stopped under AMIGO's default acceptable-convergence test,
with KKT error **1.534e-7**, rather than reaching the requested primary 1e-8
target. Primal infeasibility was 6.253e-13. The fixed-current comparison also
stopped under acceptable convergence: 42 iterations, 73.75 seconds, KKT error
1.570e-8. These measured times exclude baseline preparation, compilation,
initialization, and replay; the runs partly overlapped and are not an isolated
runtime benchmark or a speedup measurement against the original full CADRE study.

The optimizer chose initial/final SOC **1.000000**, with periodicity error
1.554e-15. SOC spans 0.2–1.0, battery current spans −10–0.680033 A,
and all checked bounds pass. Panel-current controls span approximately
0–0.4 A; the actual spline trajectories reach 0.337414 A. Independent replay
matches voltage within 6.777e-13 V, solar power within 1.807e-13 W,
battery current within 6.040e-14 A, and downloaded data within 9.095e-13 units.
Native gradient and Hessian-product checks have relative errors below 3e-9.

This validates the current port's voltage-table spline and trajectory model.
The inherited fitted voltage table can give negative net panel power for some
fixed benchmark currents in darkness; it is not clamped in this stage. The
optimized trajectory has positive total solar power throughout. Separate
validation against the original CADRE implementation remains necessary before
claiming an exact reproduction of that study.

See [the meeting figure](../results/pcomm_solar/optimization.png),
[the panel-current figure](../results/pcomm_solar/solar_currents.png),
[validated results](../results/pcomm_solar/summary.json),
[the matched comparison](../results/pcomm_solar/comparison.json), and
[derivative validation](../results/voltage_derivative_validation.json).

## Previous validated run: optimized initial SOC

The optimizer chose initial SOC **0.763584**, compared with the
benchmark value 0.815707. Final SOC is
0.763584, with independently verified periodicity error
5.814e-13. This adds one genuine design variable for a
total of 301.

Downloaded data is **5256.382558** benchmark units: **0.8581%**
above the prior fixed-initial-SOC optimum (5211.663029), and
**3.9316%** above the feasible repaired benchmark
(5057.539582). Both optimized runs enforce the same current and SOC
limits and endpoint periodicity; the new run also chooses the starting SOC.

The run converged in 30 iterations with KKT error
8.574e-09 and primal infeasibility 2.454e-09.
Solver time was 158.58 seconds, excluding baseline preparation,
compilation, initialization, and independent validation. Reference replay
matched SOC within 3.116e-11 and current within
2.454e-09 A. All bounds passed to the stated
numerical tolerance. SOC ranges from 0.2 to 1.0; current ranges from −10 to
0.697871 A. The lower and upper SOC bounds and discharge-current
limit are active.

The historical initial-SOC plot compares the feasible benchmark, the saved fixed-initial-SOC
optimum, and the new optimum when all trajectories share the same grid.
See [the new meeting figure](../results/pcomm_initial_soc/optimization.png),
[validated results](../results/pcomm_initial_soc/summary.json), and
[the numerical comparison](../results/pcomm_initial_soc/comparison.json).

The `--fixed-initial-soc` regression run also passed validation and reproduced
the saved fixed-SOC optimum with zero difference in downloaded data.

That historical run optimized only communication power and initial SOC, with fixed
solar voltage and currents. Its results are not a matched comparison for the new
temperature-dependent voltage formulation.

## Previous constrained run: fixed initial SOC

The strengthened run converged in 31 iterations with KKT error
5.246e-09, primal infeasibility 2.260e-09, and
151.88 seconds of solver time (excluding baseline repair,
compilation, initialization, and independent validation).

Downloaded data increased from **5057.539582** for the feasible
benchmark repair to **5211.663029**, an improvement of
**3.0474%**. The original unrepaired benchmark is retained
in the results for context, with its current-limit and periodicity violations
reported separately.

Battery current ranged from -10.000000 to 0.687902 A.
SOC ranged from 0.261166 to 1.000000, started at
0.8157066534, and ended at 0.8157066534. Independent replay
verified SOC periodicity to 1.329e-12, SOC trajectories
within 5.335e-11, and battery current within
2.260e-09 A. All bounds passed.

The discharge-current limit and upper SOC bound are active; the lower SOC
bound and charge-current limit are inactive. This remains a single-case,
communication-power-only optimization with fixed solar voltage, attitude,
geometry, and solar-current settings.

See [the constrained meeting figure](../results/pcomm_constrained/optimization.png)
and [the validated summary](../results/pcomm_constrained/summary.json).

## Previous run: SOC bounds only

The October 14, 2026 run converged in 38 iterations with KKT error
8.67e-9 and primal infeasibility 7.76e-10. Downloaded data increased from
5066.372936 to 9049.344054 benchmark units, an improvement of **78.6158%**.
SOC ranged from 0.2000 to 1.0000 and ended at 0.3235, compared with baseline
final SOC of 0.8143. No terminal SOC equality is imposed.

Independent Python replay matched temperature within 6.51e-10 K, SOC within
2.54e-8, and downloaded data within 1.82e-12 benchmark units. All control and
SOC bounds passed. Battery current ranged from -29.964 to 0.697 A and remains
unconstrained; this result establishes a working communication-power/SOC
optimization, with the fixed-input assumptions described above.

See [the meeting figure](../results/pcomm/optimization.png) and
[the machine-readable summary](../results/pcomm/summary.json).
