# No-VQA E35 risk-stratified five-fold scorer audit

This audit uses the predeclared final epoch across disjoint Navtrain physical logs. Navtest is not read.

- Locked epoch: `7`
- Validation coverage: `103288` scenes / `162` logs
- Scene-weighted delta: `+0.00453616`
- Worst-fold delta: `+0.00357851`
- Worst-fold bootstrap lower: `+0.00209910`
- Robust all-log refit gate: `PASS`

| Fold | Scenes | Logs | Base PDMS | Selected PDMS | Delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 20082 | 33 | 0.947490 | 0.952408 | +0.004918 | [+0.003373, +0.006523] |
| 1 | 20776 | 33 | 0.952166 | 0.957011 | +0.004845 | [+0.003231, +0.006374] |
| 2 | 20774 | 32 | 0.950925 | 0.955526 | +0.004602 | [+0.002697, +0.006386] |
| 3 | 20842 | 32 | 0.951791 | 0.955370 | +0.003579 | [+0.002099, +0.004973] |
| 4 | 20814 | 32 | 0.954195 | 0.958948 | +0.004753 | [+0.003774, +0.005833] |

The five validation-log sets are pairwise disjoint and cover every available Navtrain physical log exactly once. The ordinary per-fold best epochs are not used by the gate.
