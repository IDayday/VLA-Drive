# All Effective Scorers: Complete Navtest Audit

Every artifact promoted by a positive held-out-log bootstrap lower bound was evaluated on the same complete Navtest cache. Official PDM factors were joined only after candidate selection.

## Coverage and outcome

- Promoted and tested artifacts: 35 / 35
- Navtest scenes / logs / candidates: 12146 / 136 / 64
- Public Base PDMS: 0.909593878
- Best tested scorer PDMS: 0.908850600
- Best tested delta: -0.000743278
- Positive test deltas: 0
- Positive test 95% CI lower bounds: 0
- Validation-positive to test-negative sign flips: 35
- Methods above 0.93 PDMS: 0

## Ranked results

| Method | Validation delta | Navtest PDMS | Navtest delta | 95% log-bootstrap CI | Switch rate | Status |
|---|---:|---:|---:|---:|---:|---|
| full_local_residual_top16_basepairw2_seed1_v1 | +0.003598 | 0.908851 | -0.000743 | [-0.002145, +0.000664] | 0.093 | TEST_NEGATIVE_INCONCLUSIVE |
| residual_partial12k_factor_top8_seed2_v1 | +0.001600 | 0.908560 | -0.001034 | [-0.002003, -0.000215] | 0.089 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_basepairw2_topksafetyw200_seed2_v1 | +0.001163 | 0.908033 | -0.001561 | [-0.002529, -0.000699] | 0.235 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_residual_top16_basepairw2_seed0_v1 | +0.004784 | 0.908024 | -0.001570 | [-0.003258, +0.000119] | 0.178 | TEST_NEGATIVE_INCONCLUSIVE |
| residual_partial12k_hybrid_top8_compositew5_seed2_v1 | +0.003195 | 0.907616 | -0.001977 | [-0.003360, -0.000823] | 0.113 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_local_top8_seed2_v2 | +0.002547 | 0.907581 | -0.002013 | [-0.003001, -0.001072] | 0.385 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_scene_set_top8_seed2_v1 | +0.002465 | 0.907511 | -0.002083 | [-0.003312, -0.000991] | 0.237 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_scene_top8_seed2_v1 | +0.002702 | 0.907282 | -0.002312 | [-0.003610, -0.001125] | 0.291 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_factor_topk_seed2_v1 | +0.003630 | 0.907231 | -0.002363 | [-0.003958, -0.000979] | 0.134 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_hybrid_top16_basepairw0_seed2_v1 | +0.003939 | 0.907197 | -0.002397 | [-0.003999, -0.000920] | 0.176 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_hybrid_top16_basepairw2_seed2_v1 | +0.003567 | 0.907150 | -0.002444 | [-0.003909, -0.000979] | 0.233 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_residual_top8_safetyw10_seed2_v1 | +0.001634 | 0.907022 | -0.002572 | [-0.003998, -0.001334] | 0.316 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_relativew0_seed2_v1 | +0.003716 | 0.906864 | -0.002730 | [-0.004323, -0.001304] | 0.120 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_hybrid_top16_basepairw2_seed0_v1 | +0.004312 | 0.906770 | -0.002824 | [-0.004622, -0.001118] | 0.212 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_anchored_basepairw1_seed2_v1 | +0.004269 | 0.906628 | -0.002966 | [-0.004522, -0.001581] | 0.239 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_residual_top16_basepairw2_seed2_v1 | +0.003616 | 0.906595 | -0.002999 | [-0.004511, -0.001553] | 0.240 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_relativew1_seed2_v1 | +0.003476 | 0.906571 | -0.003023 | [-0.004502, -0.001781] | 0.252 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_seed2_v1 | +0.004669 | 0.906291 | -0.003303 | [-0.005100, -0.001690] | 0.147 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_factor_topk_seed2_v1 | +0.003670 | 0.906148 | -0.003446 | [-0.004881, -0.002157] | 0.262 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_relativew0_seed2_v1 | +0.003918 | 0.906100 | -0.003494 | [-0.005262, -0.002013] | 0.429 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_anchored_basepairw0_seed2_v1 | +0.003918 | 0.906099 | -0.003495 | [-0.005262, -0.002014] | 0.429 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_seed2_v1 | +0.003912 | 0.906085 | -0.003509 | [-0.005358, -0.002043] | 0.246 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_anchored_basepairw5_seed2_v2 | +0.003287 | 0.905872 | -0.003722 | [-0.005085, -0.002493] | 0.379 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_relativew5_seed2_v1 | +0.004114 | 0.905835 | -0.003759 | [-0.005447, -0.002184] | 0.201 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_factor_topk_safetyw200_seed2_v1 | +0.004188 | 0.905647 | -0.003947 | [-0.005502, -0.002563] | 0.458 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_safetyw10_seed2_v1 | +0.004038 | 0.905634 | -0.003960 | [-0.005849, -0.002347] | 0.467 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_factor_topk_safetyw50_seed2_v1 | +0.003982 | 0.905456 | -0.004138 | [-0.005709, -0.002725] | 0.467 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_hybrid_top16_basepairw2_seed1_v1 | +0.004135 | 0.904974 | -0.004620 | [-0.006328, -0.002921] | 0.414 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_anchored_basepairw2_seed2_v1 | +0.004514 | 0.904849 | -0.004744 | [-0.006529, -0.003159] | 0.399 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top8_compositew1_seed2_v1 | +0.004671 | 0.904589 | -0.005005 | [-0.006440, -0.003643] | 0.407 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial12k_hybrid_top16_anchored_basepairw2_seed2_v1 | +0.004906 | 0.904575 | -0.005019 | [-0.006761, -0.003422] | 0.287 | TEST_NEGATIVE_SIGNIFICANT |
| full_local_hybrid_top8_basepairw2_seed2_v1 | +0.004606 | 0.904122 | -0.005471 | [-0.007244, -0.003661] | 0.604 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial8k_local_top16_seed2_v1 | +0.005716 | 0.900356 | -0.009238 | [-0.011532, -0.007029] | 0.766 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial8k_local_top8_seed2_v1 | +0.005912 | 0.896905 | -0.012689 | [-0.015421, -0.010042] | 0.875 | TEST_NEGATIVE_SIGNIFICANT |
| residual_partial8k_set_top16_seed2_v1 | +0.006912 | 0.895663 | -0.013931 | [-0.016596, -0.011245] | 0.862 | TEST_NEGATIVE_SIGNIFICANT |

## Interpretation

A validation improvement is treated only as a promotion signal. It is not reported as a planning improvement unless the complete Navtest result is also positive. The current campaign shows a systematic validation-to-test sign reversal, so none of these fine-rankers is deployable as an improvement over the released scorer.
