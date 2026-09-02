# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `3` / `3`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| combined_top8_reference_q50_strict_actor_all_logs__residual_hybrid | +0.003812 | 0.911415 | -0.000079 | [-0.000952, +0.000905] | 0.071714 | no |
| combined_top16_reference_q50_strict_actor_all_logs__residual_hybrid | +0.006212 | 0.910048 | -0.001445 | [-0.003578, +0.000995] | 0.073080 | no |
| combined_top16_reference_q10_balanced_actor_all_logs__residual_hybrid | +0.004338 | 0.907417 | -0.004076 | [-0.006331, -0.001791] | 0.075712 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
