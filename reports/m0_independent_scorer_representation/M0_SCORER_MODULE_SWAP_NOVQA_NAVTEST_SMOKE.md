# M0 scorer module-swap audit

The proposal bank is fixed. PDM labels are joined only after each module combination selects a candidate.

Scenes: 16; physical logs: 12

| Context | Attention | Head | Selected PDMS | Delta vs Base | Pairwise | Regret | Base-index agreement |
|---|---|---|---:|---:|---:|---:|---:|
| no_vqa | no_vqa | no_vqa | 0.873554 | -0.001158 | 0.625611 | 0.053774 | 0.0625 |
| no_vqa | no_vqa | public | 0.781885 | -0.092826 | 0.000000 | 0.145443 | 0.0000 |
| no_vqa | public | no_vqa | 0.744572 | -0.130139 | 0.081819 | 0.182756 | 0.0000 |
| no_vqa | public | public | 0.768772 | -0.105939 | 0.375652 | 0.158556 | 0.0000 |
| public | no_vqa | no_vqa | 0.681492 | -0.193220 | 0.585580 | 0.245836 | 0.0000 |
| public | no_vqa | public | 0.713593 | -0.161119 | 0.373126 | 0.213735 | 0.0000 |
| public | public | no_vqa | 0.771469 | -0.103243 | 0.000000 | 0.155860 | 0.0000 |
| public | public | public | 0.874712 | +0.000000 | 0.701444 | 0.052617 | 1.0000 |

## Interpretation boundary

A context swap changes the current-observation representation. An attention swap changes trajectory/context interaction. A head swap changes only the six factor classifiers. This audit diagnoses checkpoint calibration; it does not train on or tune against Navtest.
