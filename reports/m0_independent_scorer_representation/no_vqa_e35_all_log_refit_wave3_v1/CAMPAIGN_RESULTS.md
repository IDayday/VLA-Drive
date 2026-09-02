# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `3` / `3`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| combined_top16_hybrid_safety5_all_logs__residual_hybrid | +0.004793 | 0.910348 | -0.001145 | [-0.003195, +0.000996] | 0.072780 | no |
| candidate_only_top16_factor_safety5_all_logs__residual_factor | +0.007206 | 0.910204 | -0.001289 | [-0.003863, +0.001479] | 0.072925 | no |
| factorized_top16_cv_hybrid_safety5_all_logs__residual_hybrid | +0.004168 | 0.909893 | -0.001600 | [-0.003596, +0.000406] | 0.073236 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
