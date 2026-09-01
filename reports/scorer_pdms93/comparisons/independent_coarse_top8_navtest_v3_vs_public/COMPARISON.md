# Paired Navtest 64-candidate comparison

- Scene tokens: 12146
- Logs: 136
- A: `independent_coarse_top8`
- B: `public_open_weight`
- Delta convention: A - B
- Confidence intervals: paired log-cluster bootstrap, 95%

| Metric | A | B | Delta | 95% CI |
|---|---:|---:|---:|---:|
| selected_pdms | 0.896769 | 0.909594 | -0.012825 | [-0.015661, -0.009960] |
| best_of_64_pdms | 0.984112 | 0.984112 | +0.000000 | [-0.000000, +0.000000] |
| scorer_regret | 0.087343 | 0.074518 | +0.012825 | [+0.009933, +0.015649] |
| mean_candidate_pdms | 0.795276 | 0.795276 | -0.000000 | [-0.000000, +0.000000] |
| mean_pairwise_endpoint_distance_m | 4.529268 | 4.529268 | +0.000000 | [-0.000000, +0.000000] |
| mean_pairwise_ade_m | 1.877294 | 1.877294 | +0.000000 | [-0.000000, +0.000000] |

## Released selected-trajectory evaluator parity

- Matched scenes: 12146
- Mean absolute error: 2.678758106344874e-15
- Maximum absolute error: 8.777423232686488e-12
