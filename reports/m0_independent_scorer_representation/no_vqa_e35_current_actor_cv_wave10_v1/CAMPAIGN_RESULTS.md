# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `8` / `8`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| fullactor_cv_rawpointcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.005164 | 0.910868 | -0.000626 | [-0.002761, +0.001475] | 0.072261 | no |
| fullactor_cv_rawpointcontextcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.006235 | 0.910597 | -0.000896 | [-0.003398, +0.001547] | 0.072532 | no |
| fullactor_cv_rawcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.005913 | 0.910532 | -0.000961 | [-0.003183, +0.001348] | 0.072597 | no |
| fullactor_cv_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006090 | 0.910238 | -0.001255 | [-0.004089, +0.001322] | 0.072890 | no |
| fullactor_cv_rawcontextcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.006133 | 0.909296 | -0.002198 | [-0.004472, +0.000101] | 0.073833 | no |
| fullactor_cv_rawcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006615 | 0.908510 | -0.002983 | [-0.005362, -0.000722] | 0.074618 | no |
| fullactor_cv_rawpointcombined_top16_hybrid_standard_seed2__residual_hybrid | +0.003350 | 0.905170 | -0.006324 | [-0.009760, -0.002741] | 0.077959 | no |
| fullactor_cv_rawcombined_top16_hybrid_standard_seed2__residual_hybrid | +0.002654 | 0.905040 | -0.006453 | [-0.009806, -0.002998] | 0.078088 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
