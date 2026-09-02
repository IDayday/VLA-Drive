# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `7` / `7`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| risk4_fullactor_rawcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006062 | 0.912089 | +0.000596 | [-0.001380, +0.002450] | 0.071040 | no |
| risk4_fullactor_rawpointcontextcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.005951 | 0.911583 | +0.000090 | [-0.002160, +0.002520] | 0.071546 | no |
| risk4_fullactor_rawpointcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.005798 | 0.911443 | -0.000050 | [-0.002638, +0.002189] | 0.071686 | no |
| risk8_fullactor_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006103 | 0.910400 | -0.001094 | [-0.004441, +0.001863] | 0.072729 | no |
| risk4_fullactor_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.005829 | 0.909327 | -0.002166 | [-0.004908, +0.000258] | 0.073801 | no |
| risk2_fullactor_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006243 | 0.908670 | -0.002823 | [-0.005484, -0.000296] | 0.074458 | no |
| risk4_fullactor_cv_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006168 | 0.907770 | -0.003723 | [-0.006856, -0.000544] | 0.075359 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
