# CADRE reference source

The reference is [OpenMDAO/CADRE](https://github.com/OpenMDAO/CADRE), revision
`05f008489e34a123637b213419fd980162ccd303`. The repository and checksums are
pinned in [`cadre_source.json`](../../cadre_source.json).

Earlier work used OpenMDAO-Plugins/CADRE revision
`a09cecb6e9977b6c50a8a62c59799fcacfa033c4`. That is a different checkout and
must not be described as the requested source. The old checkout has been left
untouched so historical results can still be traced to it.

## Setup

From the `cadre-amigo` repository root:

```sh
git clone https://github.com/OpenMDAO/CADRE.git ../CADRE-OpenMDAO
git -C ../CADRE-OpenMDAO checkout 05f008489e34a123637b213419fd980162ccd303
```

This keeps the reference source outside `cadre-amigo` and its GitHub repository.
For a checkout somewhere else:

```sh
export CADRE_ROOT=/absolute/path/to/OpenMDAO-CADRE
```

`CADRE_ROOT` takes priority over `../CADRE-OpenMDAO` and the sibling `../CADRE`
fallback. The requested repository stores its package in `CADRE/`. The old
`src/CADRE/` layout is now rejected with setup instructions. No tables or
benchmarks fall back to the old checkout.

New integrated runs check the pinned source and input hashes before preparing
cases. Their summaries, preparation reports, joint-model reports, and newly
generated figure measurements include `cadre_source` metadata. Existing saved
optimization summaries retain their historical contents.

## What changed, and what matched

The [migration report](../results/validation/cadre_source_migration.json) records
the comparison between the two checkouts.

| Comparison | Result |
| --- | --- |
| Saved `data1346.pkl` benchmark | Different pickle bytes, but all 337 fields have identical numerical values and shapes |
| Solar angle and exposure tables | Identical bytes |
| Solar-cell voltage table | Identical bytes |
| Communication gain table | Identical bytes |
| Six-case launch table | Identical bytes |
| Orbit `f_dot` method | Identical parsed code |
| Other component implementations | Reorganized for newer OpenMDAO APIs; numerical forward checks described below |

The six-case design variables, shared surface/angles, summed final-data objective,
and separate SOC-periodicity constraints were checked against the requested
repository's `CADRE/CADRE_mdp.py` and `examples/mdp.py`. The paper configuration
is still 1,500 nodes, 300 controls per profile, and six cases. Its smaller default
example uses 150 nodes, six controls per profile, and two cases.

The existing AMIGO results remain results of our formulation. Matching input
arrays does not turn them into new OpenMDAO executions. AMIGO still uses
trapezoidal trajectory constraints; CADRE uses RK4. AMIGO enforces battery bounds
at each node; the upstream optimization uses KS-aggregated battery constraints.
Those formulation differences remain and must be stated when comparing runs.

## Checks

Using the AMIGO Pixi environment from the repository root:

```sh
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.cadre_paths
pixi run --manifest-path ../amigo/pixi.toml python -m orbit.orbit_source_validation
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.cadre_source
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.prepare_cadre --check-lookups
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.shared_angles
pixi run --manifest-path ../amigo/pixi.toml python -m integrated.validation.joint_model
```

`cadre_source.py` reads the upstream numerical methods rather than maintaining
another copy of their equations. For these checks it replaces OpenMDAO framework
base classes with a container, supplies NumPy and upstream kinematics helpers,
and calls the forward methods directly. It does not test OpenMDAO setup, total
derivatives, driver execution, or native MBI installation.

The [forward-equation report](../results/validation/cadre_source_validation.json)
contains 97 comparisons per case across all six cases. It covers orbit,
attitude, Sun geometry, communication geometry/rate, thermal rates, solar/net
power, battery current/SOC rates, and reaction-wheel torque/power/rates.
The maximum absolute difference was `1.82e-12` across the quantities checked;
the report keeps separate errors because those quantities have different units.
The lookup and analytic-derivative checks are separate from this forward audit.

No optimization was run for this migration. Historical objectives and timings
were not replaced or presented as fresh runs of the requested source.

A subsequent short single-case optimization (`openmdao_smoke_600`, now in the
local archive described in the [result index](../results/README.md)) used this pinned source with 600 nodes, 120 controls per profile, and a 4.8-hour
window. Communication power, solar currents, and initial SOC were free; roll and
surface allocation were fixed, with design physics enabled. It converged in
38 iterations and 167.64 seconds of solver time. Independent replay, battery
bounds, and SOC periodicity passed. This checks the reduced single-case run;
the full-resolution six-case joint optimization has not converged.

The [full default single-case rerun](../results/openmdao_default_1500/summary.json)
then used 1,500 nodes, 300 controls per profile, and the full 12-hour window.
It reached acceptable convergence in 73 iterations and 827.91 seconds of solver
time, with a final NLP residual of `1.53e-7`. Independent replay, battery limits,
and SOC periodicity passed. Downloaded data was `5264.456876690892`, a `4.0821%`
improvement over the feasible baseline, reproducing the earlier default result.
This is the default formulation without `--design-physics`; it is separate from
the restored-physics reference and the six-case joint formulation.
