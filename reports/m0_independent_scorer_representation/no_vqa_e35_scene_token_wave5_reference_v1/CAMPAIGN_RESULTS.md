# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `8` / `8`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| private_top16_reference_q10_strict_actor_seed2__residual_hybrid | +0.000019 | 0.911499 | +0.000006 | [+0.000000, +0.000021] | 0.071629 | no |
| combined_top16_reference_q10_strict_actor_seed2__residual_hybrid | +0.000003 | 0.911493 | +0.000000 | [+0.000000, +0.000000] | 0.071635 | no |
| combined_top32_reference_q10_strict_actor_seed2__residual_hybrid | +0.000213 | 0.911462 | -0.000032 | [-0.000214, +0.000092] | 0.071667 | no |
| combined_top8_reference_q10_strict_actor_seed2__residual_hybrid | +0.000215 | 0.911447 | -0.000047 | [-0.000239, +0.000085] | 0.071682 | no |
| candidateonly_top16_reference_q10_strict_seed2__residual_hybrid | +0.000038 | 0.911436 | -0.000057 | [-0.000197, +0.000023] | 0.071693 | no |
| combined_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.006212 | 0.910079 | -0.001415 | [-0.003466, +0.000453] | 0.073050 | no |
| combined_top8_reference_q50_strict_actor_seed2__residual_hybrid | +0.003812 | 0.909331 | -0.002163 | [-0.003745, -0.000734] | 0.073798 | no |
| combined_top16_reference_q10_balanced_actor_seed2__residual_hybrid | +0.004338 | 0.906180 | -0.005313 | [-0.007615, -0.003006] | 0.076949 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
