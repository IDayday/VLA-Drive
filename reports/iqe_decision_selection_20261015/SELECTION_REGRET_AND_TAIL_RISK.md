# Fixed-pool selection and tail risk

All reported policies use the same Query70k Base and fixed road/interaction/safe-progress pool. CSV scores are [0,1]; tables use percent PDMS and percentage-point differences. Entire fixed manifests are retained. No newly executed Navtest or simulation calls.

## stage_val

| Policy | Scenes | Logs | PDMS /100 | Gain pp | 95% log CI pp | Regret pp | High→0 | New NC/DAC/TTC |
|---|---|---|---|---|---|---|---|---|
| expert_0 | 1024 | 53 | 91.3915 | +0.0000 | [0.0000, 0.0000] | 5.1301 | 0 | 0/0/0 |
| expert_1 | 1024 | 53 | 77.7063 | -13.6852 | [-16.6186, -10.7317] | 18.8152 | 125 | 111/46/182 |
| expert_2 | 1024 | 53 | 87.7218 | -3.6697 | [-5.6829, -1.4931] | 8.7997 | 37 | 26/18/53 |
| expert_3 | 1024 | 53 | 81.5784 | -9.8130 | [-12.3546, -7.7351] | 14.9431 | 88 | 59/51/167 |
| Oracle | 1024 | 53 | 96.5215 | +5.1301 | [4.0327, 6.4296] | 0.0000 | 0 | 0/0/2 |
| R00 | 1024 | 53 | 91.8226 | +0.4311 | [-0.6485, 1.3772] | 4.6989 | 10 | 10/6/31 |
| M_scene | 1024 | 53 | 89.5467 | -1.8448 | [-4.0769, -0.3358] | 6.9749 | 23 | 26/8/72 |
| M_set | 1024 | 53 | 91.6345 | +0.2430 | [-0.5831, 0.9747] | 4.8871 | 10 | 15/2/36 |

Paired comparisons, same scenes/logs, 2000 bootstrap repetitions:

| Difference | Mean pp | 95% CI pp |
|---|---|---|
| M_scene − R00 | -2.2759 | [-4.2179, -0.8719] |
| M_set − R00 | -0.1881 | [-1.2248, 0.8471] |
| M_set − M_scene | +2.0878 | [0.4776, 4.3925] |

## dev_report

| Policy | Scenes | Logs | PDMS /100 | Gain pp | 95% log CI pp | Regret pp | High→0 | New NC/DAC/TTC |
|---|---|---|---|---|---|---|---|---|
| expert_0 | 1696 | 16 | 90.5302 | +0.0000 | [0.0000, 0.0000] | 4.8215 | 0 | 0/0/0 |
| expert_1 | 1696 | 16 | 77.5541 | -12.9761 | [-16.0496, -10.0204] | 17.7976 | 210 | 123/119/230 |
| expert_2 | 1696 | 16 | 85.1406 | -5.3896 | [-8.0150, -3.3282] | 10.2111 | 96 | 47/57/93 |
| expert_3 | 1696 | 16 | 81.4423 | -9.0879 | [-12.2328, -4.4694] | 13.9094 | 128 | 82/85/246 |
| Oracle | 1696 | 16 | 95.3517 | +4.8215 | [4.1137, 5.7819] | 0.0000 | 0 | 2/0/2 |
| R00 | 1696 | 16 | 90.6960 | +0.1658 | [-1.1025, 1.2360] | 4.6557 | 11 | 14/4/44 |
| M_scene | 1696 | 16 | 88.3982 | -2.1319 | [-4.9045, 0.2085] | 6.9534 | 41 | 33/23/106 |
| M_scene_margin | 1696 | 16 | 90.5302 | +0.0000 | [0.0000, 0.0000] | 4.8215 | 0 | 0/0/0 |
| M_set | 1696 | 16 | 90.4538 | -0.0764 | [-1.2542, 0.9505] | 4.8979 | 16 | 15/11/60 |
| M_set_margin | 1696 | 16 | 90.5304 | +0.0003 | [0.0000, 0.0009] | 4.8212 | 0 | 0/0/0 |

Paired comparisons, same scenes/logs, 2000 bootstrap repetitions:

| Difference | Mean pp | 95% CI pp |
|---|---|---|
| M_scene − R00 | -2.2977 | [-4.5345, -0.5874] |
| M_set − R00 | -0.2422 | [-0.9111, 0.3103] |
| M_set − M_scene | +2.0556 | [0.3620, 4.1331] |

Development has already been observed in previous studies and is exploratory, not blind testing. One seed is not a multi-seed reliability result. R00 used full D_S and historical checkpoint selection; it is not an unseen control on the new internal validation partition.

High→0: Base ≥0.8 and selected ≤1e-8. New component failures compare valid full-score Base to selected below full score, using the locked tolerance. All rate denominators are the full fixed role. DDC has zero weight in this PDMS protocol. Positive and negative switch costs are separate in the CSV. Negative Oracle capture is preserved. Raw argmax is always reported even if calibration returns Base.
