# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `8` / `8`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| rawcontextcombined_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.005956 | 0.910470 | -0.001024 | [-0.003084, +0.001130] | 0.072659 | no |
| rawcombined_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.005833 | 0.910391 | -0.001102 | [-0.003263, +0.001111] | 0.072738 | no |
| rawcombined_top16_reference_q50_strict_noactor_seed2__residual_hybrid | +0.006236 | 0.910165 | -0.001328 | [-0.003394, +0.000680] | 0.072964 | no |
| rawprivate_top16_reference_q50_strict_actor_seed2__residual_hybrid | +0.003598 | 0.909699 | -0.001795 | [-0.003987, +0.000186] | 0.073430 | no |
| rawcombined_top32_reference_q50_strict_actor_seed2__residual_hybrid | +0.006912 | 0.909344 | -0.002149 | [-0.005008, +0.000458] | 0.073784 | no |
| rawcombined_top8_hybrid_topregret_actor_seed2__residual_hybrid | +0.002844 | 0.908988 | -0.002506 | [-0.005276, +0.000188] | 0.074141 | no |
| rawcombined_top16_reference_q50_balanced_actor_seed2__residual_hybrid | +0.005017 | 0.907531 | -0.003962 | [-0.006518, -0.001409] | 0.075597 | no |
| rawcombined_top16_hybrid_standard_actor_seed2__residual_hybrid | +0.003948 | 0.906327 | -0.005167 | [-0.007810, -0.002219] | 0.076802 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
