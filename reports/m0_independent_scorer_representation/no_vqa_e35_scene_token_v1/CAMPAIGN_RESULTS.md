# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `7` / `7`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| direct_actor050_seed2__calibrated__residual_direct | +0.004943 | 0.909441 | -0.002053 | [-0.004087, -0.000272] | 0.073688 | no |
| factor_actor050_seed2__calibrated__residual_factor | +0.002260 | 0.909125 | -0.002368 | [-0.004300, -0.000646] | 0.074004 | no |
| hybrid_actor050_future025_seed2__calibrated__residual_hybrid | +0.002810 | 0.909052 | -0.002441 | [-0.004546, -0.000421] | 0.074076 | no |
| primary_hybrid_actor050_seed2__calibrated__residual_hybrid | +0.005190 | 0.908523 | -0.002971 | [-0.005149, -0.000864] | 0.074606 | no |
| control_hybrid_no_actor_seed2__calibrated__residual_hybrid | +0.002662 | 0.908208 | -0.003286 | [-0.005361, -0.001382] | 0.074921 | no |
| hybrid_actor050_deep_seed2__calibrated__residual_hybrid | +0.002333 | 0.908148 | -0.003346 | [-0.005584, -0.001073] | 0.074981 | no |
| primary_hybrid_actor050_seed11__calibrated__residual_hybrid | +0.003793 | 0.908011 | -0.003482 | [-0.006272, -0.000751] | 0.075118 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
