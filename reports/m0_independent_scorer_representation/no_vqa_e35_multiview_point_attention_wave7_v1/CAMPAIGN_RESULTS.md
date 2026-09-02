# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `7` / `7`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| rawpointcontextcombined_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.006218 | 0.910623 | -0.000871 | [-0.003166, +0.001291] | 0.072506 | no |
| rawpointcombined_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.006033 | 0.910546 | -0.000948 | [-0.003229, +0.001260] | 0.072583 | no |
| rawpointprivate_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.003969 | 0.909225 | -0.002269 | [-0.004457, -0.000178] | 0.073904 | no |
| rawpointcombined_top8_hybrid_topregret_actor_seed2__residual_hybrid | +0.003323 | 0.908703 | -0.002790 | [-0.005386, -0.000165] | 0.074426 | no |
| rawpointcombined_top32_reference_q50_strict_actor_seed2__residual_hybrid | +0.006372 | 0.908633 | -0.002861 | [-0.005652, -0.000284] | 0.074496 | no |
| rawpointcombined_top16_reference_q50_balanced_actor_seed2__residual_hybrid | +0.004319 | 0.907239 | -0.004254 | [-0.006635, -0.001687] | 0.075889 | no |
| rawpointcombined_top16_hybrid_standard_actor_seed2__residual_hybrid | +0.003112 | 0.904107 | -0.007386 | [-0.010965, -0.003636] | 0.079021 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
