# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `5` / `5`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| combined_top8_hybrid_safety5_allfactor_topregret1_seed2__calibrated__residual_hybrid | +0.005398 | 0.911414 | -0.000080 | [-0.002089, +0.001740] | 0.071715 | no |
| combined_top8_hybrid_safety5_allfactor_topregret0_seed2__calibrated__residual_hybrid | +0.004690 | 0.910356 | -0.001137 | [-0.003131, +0.000849] | 0.072772 | no |
| combined_top8_hybrid_safety5_allfactor_topregret0_seed2__residual_hybrid | +0.003531 | 0.908643 | -0.002850 | [-0.005902, +0.000003] | 0.074485 | no |
| combined_top8_hybrid_safety5_allfactor_topregret1_seed2__residual_hybrid | +0.003066 | 0.908263 | -0.003231 | [-0.006224, -0.000342] | 0.074866 | no |
| combined_top8_hybrid_safety5_topkfactor_topregret1_seed2__residual_hybrid | +0.001926 | 0.907470 | -0.004023 | [-0.007369, -0.000995] | 0.075659 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
