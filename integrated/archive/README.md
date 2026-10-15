# Historical formulation

`cadre_optimize_pcomm_1500.py` is the original SOC-bounds-only optimization.
It does not enforce battery-current limits or SOC periodicity. It is retained
unchanged to explain the historical result in `../results/pcomm/`.

Use `python -m integrated.cadre_optimize_pcomm` for the maintained formulation.
