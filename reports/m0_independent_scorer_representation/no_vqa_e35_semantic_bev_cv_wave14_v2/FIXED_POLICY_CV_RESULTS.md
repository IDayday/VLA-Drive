# No-VQA E35 risk-stratified five-fold scorer audit

This audit uses the predeclared final epoch across disjoint Navtrain physical logs. Navtest is not read.

- Locked epoch: `7`
- Validation coverage: `103288` scenes / `162` logs
- Scene-weighted delta: `+0.00498339`
- Worst-fold delta: `+0.00394078`
- Worst-fold bootstrap lower: `+0.00255078`
- Robust all-log refit gate: `PASS`

| Fold | Scenes | Logs | Base PDMS | Selected PDMS | Delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20082 | 33 | 0.947490 | 0.953399 | +0.005909 | [+0.003891, +0.007872] |
| 1 | 20776 | 33 | 0.952166 | 0.957348 | +0.005182 | [+0.003226, +0.007199] |
| 2 | 20774 | 32 | 0.950925 | 0.956123 | +0.005198 | [+0.003634, +0.006986] |
| 3 | 20842 | 32 | 0.951791 | 0.956512 | +0.004721 | [+0.002924, +0.006512] |
| 4 | 20814 | 32 | 0.954195 | 0.958136 | +0.003941 | [+0.002551, +0.005383] |

The five validation-log sets are pairwise disjoint and cover every available Navtrain physical log exactly once. The ordinary per-fold best epochs are not used by the gate.
