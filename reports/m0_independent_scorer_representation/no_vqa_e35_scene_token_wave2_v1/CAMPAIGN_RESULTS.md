# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `11` / `11`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| combined_top4_hybrid_actor050_seed2__calibrated__residual_hybrid | +0.003710 | 0.910824 | -0.000669 | [-0.002237, +0.000650] | 0.072305 | no |
| candidate_only_top8_hybrid_seed2__calibrated__residual_hybrid | +0.004559 | 0.910437 | -0.001057 | [-0.003357, +0.001272] | 0.072692 | no |
| combined_top4_hybrid_actor050_seed2__residual_hybrid | +0.002931 | 0.909887 | -0.001607 | [-0.003556, +0.000158] | 0.073242 | no |
| combined_top8_hybrid_actor050_seed2__calibrated__residual_hybrid | +0.002634 | 0.909580 | -0.001913 | [-0.003792, -0.000130] | 0.073549 | no |
| candidate_only_top16_hybrid_seed2__calibrated__residual_hybrid | +0.006809 | 0.909137 | -0.002357 | [-0.004540, -0.000073] | 0.073992 | no |
| candidate_only_top8_hybrid_seed2__residual_hybrid | +0.002605 | 0.909059 | -0.002435 | [-0.005698, +0.000744] | 0.074070 | no |
| combined_top16_hybrid_actor050_seed11__calibrated__residual_hybrid | +0.002761 | 0.908562 | -0.002932 | [-0.005012, -0.000922] | 0.074567 | no |
| combined_top16_hybrid_actor050_seed2__calibrated__residual_hybrid | +0.003249 | 0.908549 | -0.002944 | [-0.004603, -0.001208] | 0.074580 | no |
| candidate_only_top64_hybrid_seed2__calibrated__residual_hybrid | +0.004614 | 0.908316 | -0.003177 | [-0.005479, -0.001017] | 0.074813 | no |
| combined_top8_hybrid_actor050_seed2__residual_hybrid | +0.002146 | 0.908021 | -0.003472 | [-0.006498, -0.000534] | 0.075107 | no |
| combined_top16_factor_actor050_seed2__calibrated__residual_factor | +0.003864 | 0.906954 | -0.004539 | [-0.007195, -0.001881] | 0.076175 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
