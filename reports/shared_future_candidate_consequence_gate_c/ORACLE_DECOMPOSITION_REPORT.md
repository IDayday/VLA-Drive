# Oracle Dynamic-value Decomposition

## Gate C1 (all eligible trainval logs): FAIL

| Group | Pairwise mean ± std | Worst fold | Top-1 regret | Collision AUROC | TTC AUROC |
|---|---:|---:|---:|---:|---:|
| O0 | 0.7413 ± 0.0016 | 0.7394 | 0.0740 | 0.8382 | 0.8071 |
| O1 | 0.7300 ± 0.0012 | 0.7281 | 0.1486 | 0.8408 | 0.8002 |
| O2 | 0.7607 ± 0.0014 | 0.7584 | 0.0796 | 0.8778 | 0.8447 |
| O3 | 0.8476 ± 0.0014 | 0.8452 | 0.0413 | 0.8889 | 0.8534 |
| O4 | 0.8574 ± 0.0016 | 0.8557 | 0.0333 | 0.9643 | 0.8959 |
| O5 | 0.9022 ± 0.0027 | 0.8974 | 0.0199 | 0.9933 | 0.9353 |
| O6 | 0.8466 ± 0.0014 | 0.8442 | 0.0406 | 0.8875 | 0.8519 |
| O7 | 0.8748 ± 0.0034 | 0.8694 | 0.0257 | 0.9917 | 0.9204 |
| O8 | 0.8748 ± 0.0029 | 0.8713 | 0.0263 | 0.9917 | 0.9218 |
| O9 | 0.8751 ± 0.0027 | 0.8710 | 0.0256 | 0.9917 | 0.9204 |
| O10 | 0.8347 ± 0.0009 | 0.8336 | 0.0384 | 0.9233 | 0.8657 |
| O11 | 0.8082 ± 0.0022 | 0.8053 | 0.0514 | 0.8317 | 0.7978 |
| O12 | 0.7921 ± 0.0026 | 0.7886 | 0.0656 | 0.8462 | 0.8101 |
| O13 | 0.8413 ± 0.0013 | 0.8390 | 0.0388 | 0.8712 | 0.8340 |

- Scenes/logs/K: 45,377 / 1,192 / 16
- Dynamic gain O8−O3: 0.0271
- Equal-log dynamic-gain point estimate: 0.0327
- Equal-log bootstrap 95% CI: [0.0301, 0.0352]
- Top-1 regret reduction: 36.38%
- State/recomputed-risk retention R_state: 1.011
- Held-out-family gain mean/worst: 0.0492 / 0.0146
- MLP parameter max/min ratio: 1.013
- Interpretation: oracle dynamic evidence is incomplete under Gate C1

O9 never receives collision/TTC labels: those risk features are recomputed from
actor-relative state and mask. O10/O11/O12/O13 are within-scene shuffle,
cross-scene shuffle, random-dimensional and repeated-static controls. Official
PDM aggregate/factor columns are isolated as targets and never enter O0–O13.
