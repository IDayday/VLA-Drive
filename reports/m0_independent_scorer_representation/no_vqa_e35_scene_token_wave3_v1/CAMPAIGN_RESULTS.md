# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `9` / `9`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| factorized_top8_cv_hybrid_safety5_seed2__calibrated__residual_hybrid | +0.003836 | 0.909577 | -0.001916 | [-0.003807, -0.000154] | 0.073551 | no |
| combined_top16_hybrid_safety5_seed2__calibrated__residual_hybrid | +0.004793 | 0.909438 | -0.002056 | [-0.004147, -0.000045] | 0.073691 | no |
| factorized_top16_cv_hybrid_safety5_seed2__calibrated__residual_hybrid | +0.004168 | 0.909334 | -0.002160 | [-0.004109, -0.000277] | 0.073795 | no |
| candidate_only_top16_factor_safety5_seed2__calibrated__residual_factor | +0.007206 | 0.909197 | -0.002297 | [-0.004802, +0.000273] | 0.073932 | no |
| factorized_top8_cv_hybrid_safety5_seed2__residual_hybrid | +0.003151 | 0.908569 | -0.002924 | [-0.006021, +0.000038] | 0.074560 | no |
| factorized_top16_cv_hybrid_safety1_seed2__calibrated__residual_hybrid | +0.003832 | 0.908220 | -0.003273 | [-0.005511, -0.000976] | 0.074908 | no |
| candidate_only_top16_factor_safety5_seed2__residual_factor | +0.002828 | 0.905435 | -0.006058 | [-0.009578, -0.002424] | 0.077694 | no |
| factorized_top16_cv_hybrid_safety5_seed2__residual_hybrid | +0.002747 | 0.904907 | -0.006586 | [-0.009703, -0.003233] | 0.078221 | no |
| combined_top16_hybrid_safety5_seed2__residual_hybrid | +0.002808 | 0.903708 | -0.007786 | [-0.011049, -0.004392] | 0.079421 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
