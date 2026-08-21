# Gate 3 decision

**Gate 3: FAIL.**

This decision uses the predeclared joint criteria; a consequence-vector-only gain cannot pass.

- AEE vs absolute: `{"effect_alignment_improves_each_seed": false, "effect_alignment_scene_ci_excludes_zero": false, "false_safe_reduction": false, "heldout_family_improvement": false, "pair_or_safety_auprc_improvement": true}`
- AEE vs global: `{"action_gap_retention": 0.9688224593349287, "action_gap_retention_at_least_90_percent": true, "equivalence_leakage_reduction": 0.1226549641920196, "leakage_reduction_at_least_20_percent": false, "separation_ratio_significant_improvement": true, "structured_target_not_significantly_degraded": true}`
- Structured action-dependent channel success: True.
- Failed conditions: ['AEE vs absolute: effect_alignment_improves_each_seed', 'AEE vs absolute: effect_alignment_scene_ci_excludes_zero', 'AEE vs absolute: false_safe_reduction', 'AEE vs absolute: heldout_family_improvement', 'AEE vs global: leakage_reduction_at_least_20_percent'].
- Recommendation: Do not attach the current AEE objective. Select Direction B for the next probe-only study: partially identified / uncertainty-aware world supervision. This is a provisional direction because confidence weighting significantly reduces false-safe errors relative to unweighted AEE, but does not significantly improve alignment and lacks reactive-model validation/test coverage.

## Paired scene-bootstrap deltas

Positive values favor AEE for alignment/AUPRC/separation ratio; negative values favor AEE for false-safe rate and structured error.

| Comparison | Point | 95% CI |
|---|---:|---:|
| aee_vs_absolute_alignment | -0.006208 | [-0.012633, 0.000037] |
| aee_vs_absolute_pair_auprc | 0.003021 | [-0.001313, 0.007319] |
| aee_vs_absolute_safety_auprc | 0.031796 | [0.018978, 0.044461] |
| aee_vs_absolute_false_safe | 0.103456 | [0.062700, 0.145373] |
| aee_vs_absolute_heldout_alignment | -0.024692 | [-0.036533, -0.012637] |
| aee_vs_global_separation_ratio | 0.436317 | [0.255822, 0.623323] |
| aee_vs_global_structured_error | -0.002487 | [-0.003112, -0.001820] |

## Confidence-AEE paired deltas

Positive values favor confidence-AEE for alignment/AUPRC; negative values favor it for false-safe rate.

| Comparison | Point | 95% CI |
|---|---:|---:|
| confidence_vs_aee_alignment | 0.002802 | [-0.000277, 0.005987] |
| confidence_vs_aee_pair_auprc | 0.001178 | [-0.001102, 0.003516] |
| confidence_vs_aee_safety_auprc | 0.000805 | [-0.006280, 0.007911] |
| confidence_vs_aee_false_safe | -0.137653 | [-0.175060, -0.098111] |
| confidence_vs_aee_heldout_alignment | 0.003949 | [-0.003563, 0.010898] |
| confidence_vs_absolute_alignment | -0.003406 | [-0.010119, 0.003336] |
| confidence_vs_absolute_safety_auprc | 0.032601 | [0.018993, 0.045847] |
| confidence_vs_absolute_false_safe | -0.034197 | [-0.074892, 0.005747] |
| confidence_vs_absolute_heldout_alignment | -0.020743 | [-0.033962, -0.006134] |

## Three-seed alignment stability

| Seed | Multi-candidate absolute | AEE | Delta |
|---:|---:|---:|---:|
| 20260821 | 0.307361 | 0.290094 | -0.017267 |
| 20260822 | 0.321824 | 0.333622 | +0.011798 |
| 20260823 | 0.303180 | 0.290026 | -0.013154 |

## Structured action-dependence checks

A channel passes only if AEE beats scene-only, train-mean, and zero controls under scene bootstrap and within-scene action shuffling significantly harms its primary metric.

| Channel | Metric | Beats all controls | Shuffle gap | 95% CI | Pass |
|---|---|---:|---:|---:|---:|
| candidate_relative_dynamic_occupancy | balanced_bce | False | 0.002398 | [0.001883, 0.002946] | False |
| drivable_area_sdf | normalized_l1 | True | 0.000098 | [0.000072, 0.000126] | True |
| lane_sdf | normalized_l1 | True | 0.000098 | [0.000071, 0.000127] | True |
| route_sdf | normalized_l1 | False | 0.000085 | [0.000053, 0.000118] | False |
| relative_longitudinal_velocity | masked_l1 | False | 0.000030 | [0.000008, 0.000054] | False |
| relative_lateral_velocity | masked_l1 | False | 0.000011 | [-0.000002, 0.000023] | False |
| dynamic_clearance | normalized_l1 | False | 0.000048 | [0.000037, 0.000061] | False |
| dynamic_collision_field | balanced_bce | False | 0.000270 | [0.000207, 0.000340] | False |
| ego_swept_footprint | dice | False | 0.000485 | [0.000381, 0.000606] | False |

Confidence-AEE significant false-safe benefit: True; significant alignment benefit: False; significant safety-AUPRC benefit: False; broad benefit: False.

Development stops here. Qwen+DiT world loss is not implemented, planning training is not run, and PDMS/EPDMS are not populated.
