# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `6` / `6`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| fullactor_rawcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.006452 | 0.910377 | -0.001117 | [-0.003411, +0.001019] | 0.072752 | no |
| fullactor_rawpointcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.005787 | 0.910155 | -0.001338 | [-0.003828, +0.001303] | 0.072974 | no |
| fullactor_rawcontextcombined_top16_reference_q50_strict_seed2__residual_hybrid | +0.006196 | 0.910039 | -0.001455 | [-0.003736, +0.000842] | 0.073090 | no |
| fullactor_rawprivate_top16_reference_q50_strict_seed2__residual_hybrid | +0.004324 | 0.908519 | -0.002974 | [-0.005350, -0.000608] | 0.074609 | no |
| fullactor_rawcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006424 | 0.908309 | -0.003185 | [-0.005611, -0.000971] | 0.074820 | no |
| fullactor_rawpointcombined_top32_reference_q50_strict_seed2__residual_hybrid | +0.006769 | 0.908216 | -0.003278 | [-0.005768, -0.000877] | 0.074913 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
