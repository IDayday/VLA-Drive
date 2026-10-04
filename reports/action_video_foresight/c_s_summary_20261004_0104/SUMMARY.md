# C/S results: complete canonical evaluations

2026-10-04T01:04:31.585639+00:00

Same per-split scene/cache/scoring/sampling contracts verified against every complete CSV. PDMS units0–100; all failures retained. No new inference or training.

## navtest C series

| updates | C0 | C1 | C4 |
|---:|---:|---:|---:|
| 18400 | — | — | 82.8002 |
| 25000 | 84.8514 | 85.3063 | — |
| 26000 | — | — | 86.7090 |
| 31600 | 87.6001 | 87.1887 | — |
| 40200 | — | — | 87.1357 |
| 51400 | 88.3269 | 88.1848 | — |
| 66200 | 88.6393 | 88.3947 | — |
| 70000 | 88.9967 | 89.1057 | 88.9264 |
| 80000 | 89.0116 | 89.1966 | 89.2493 |
| 90000 | 88.9124 | 89.1594 | 89.0044 |
| 100000 | 89.1512 | 89.2395 | 89.2776 |

## navtest S series

| updates | S0 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|
| 50000 | 88.2431 | 88.1847 | 88.3295 | 88.5037 | 87.8784 |
| 60000 | — | — | — | — | 88.4261 |

## dev C series

| updates | C0 | C1 | C4 |
|---:|---:|---:|---:|
| 5000 | 78.3895 | 79.8939 | 78.1454 |
| 10000 | 84.2001 | 84.4492 | 82.9499 |
| 25000 | 89.6822 | 89.9494 | 87.7313 |
| 50000 | 90.7598 | 91.5464 | 91.2806 |
| 75000 | 90.9487 | 91.2517 | 91.4501 |
| 100000 | 91.0690 | 91.5147 | 91.3257 |

## dev S series

| updates | S0 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|
| 5000 | 78.8561 | 79.0481 | 79.5465 | 80.2036 | 79.5559 |
| 10000 | 83.5736 | 84.8536 | 84.9043 | 85.0683 | 82.3019 |
| 25000 | 89.8752 | 90.1452 | 89.9099 | 89.4702 | 89.2794 |
| 50000 | 91.0018 | 90.9594 | 91.1810 | 90.9723 | 90.5111 |

## Matched-checkpoint Navtest differences

| Pair | Delta PDMS points | Log-cluster95% interval |
|---|---:|---|
| navtest_C1-C0_100000 | +0.0884 | [-0.1782, +0.3589] |
| navtest_C4-C1_100000 | +0.0381 | [-0.2736, +0.3431] |
| navtest_C4-C0_100000 | +0.1265 | [-0.1512, +0.4261] |
| navtest_S1-S0_50000 | -0.0584 | [-0.3997, +0.2943] |
| navtest_S3-S2_50000 | +0.1742 | [-0.2142, +0.5614] |
| navtest_S2-S0_50000 | +0.0864 | [-0.3171, +0.4759] |
| navtest_S3-S1_50000 | +0.3190 | [-0.0416, +0.6619] |
| navtest_S4-S3_50000 | -0.6253 | [-1.0219, -0.2207] |

C0/C1/C4 use144W and legacy single-future supervision; target sizes128×96,256×192/pool2,512×384/pool4. S0–S4 fix C1 current targets: S0 frame sequence; S1 frame sequence+GT auxiliary action; S2 video; S3 video+GT auxiliary action; S4 S3+directW planner. All retain currentDINO and frozenGTMAE. C2/C3/C5 have no formal model results in these campaigns.

C/S future target normalization, exposure and calibrated weights differ. Do not attribute a cross-series difference to a single mechanism. All figures here use one training and one inference seed42, original10FMsteps, no oracle/scorer. Historical89.41 is external unverified context and excluded.
