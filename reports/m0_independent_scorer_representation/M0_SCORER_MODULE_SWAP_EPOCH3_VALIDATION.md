# M0 scorer module-swap audit

The proposal bank is fixed. PDM labels are joined only after each module combination selects a candidate.

Scenes: 18179; physical logs: 61

| Context | Attention | Head | Selected PDMS | Delta vs Base | Pairwise | Regret | Base-index agreement |
|---|---|---|---:|---:|---:|---:|---:|
| epoch3 | epoch3 | epoch3 | 0.911369 | -0.040243 | 0.545152 | 0.077364 | 0.1264 |
| epoch3 | epoch3 | public | 0.729189 | -0.222423 | 0.685377 | 0.259545 | 0.0157 |
| epoch3 | public | epoch3 | 0.856220 | -0.095392 | 0.413595 | 0.132513 | 0.0427 |
| epoch3 | public | public | 0.911967 | -0.039645 | 0.604307 | 0.076767 | 0.1178 |
| public | epoch3 | epoch3 | 0.883126 | -0.068486 | 0.444102 | 0.105607 | 0.0619 |
| public | epoch3 | public | 0.694833 | -0.256779 | 0.673732 | 0.293900 | 0.0131 |
| public | public | epoch3 | 0.857906 | -0.093706 | 0.386346 | 0.130828 | 0.0451 |
| public | public | public | 0.951610 | -0.000002 | 0.714384 | 0.037124 | 0.9962 |

## Interpretation boundary

A context swap changes the current-observation representation. An attention swap changes trajectory/context interaction. A head swap changes only the six factor classifiers. This audit diagnoses checkpoint calibration; it does not train on or tune against Navtest.
