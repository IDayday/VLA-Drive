# No-VQA scene-token scorer campaign

- Strict coverage: **PASS**
- Validation-promoted / Navtest-evaluated: `1` / `1`
- Matching No-VQA Base PDMS: `0.911493`
- Matching candidate-bank Best-of-64: `0.983129`
- Models above 0.93: `0`

| model | val delta | Navtest PDMS | Navtest delta | 95% CI | regret | >0.93 |
|---|---:|---:|---:|---:|---:|:---:|
| rawpointcombined_top64_reference_q50_strict_noactor_seed2__residual_hybrid | +0.006601 | 0.908253 | -0.003240 | [-0.006108, -0.000686] | 0.074876 | no |

All rows use FP32, 12,146 scenes, 136 logs, 64 fixed No-VQA
proposals, zero invalid scenes, and join official PDM factors only after
the scorer has frozen its selected index. Best-of-64 is an offline oracle
candidate-bank upper bound, not deployable PDMS.
