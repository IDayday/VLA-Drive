# Phase 6 world-probe ablation

All trained methods use the same probe, normalized effect-latent dimension, effect decoder, consequence decoder, steps, 16-scene batch, and three seeds. Factual-only contributes its one unique expert candidate per scene; multi-candidate methods draw four candidates and average within scene before the scene batch is averaged. Pair distances are computed only after L2 normalization. Scenes—not pairs—are the bootstrap unit.

## Data protocol

- Scene-disjoint train/validation/test: 4200 / 500 / 500.
- Accepted training candidates after validity filtering and held-out-family exclusion: 55379.
- Candidate family held out from training: `turn_inner_outer_offset`.
- Pair-bearing training scenes: 4200; each sampled scene receives equal weight and missing pair categories remain masked.
- Pair thresholds and target normalization use train scenes only; validation is used only for risk-threshold calibration, and test remains held out until evaluation.

## Representation and risk results

| Method | Per-scene alignment | Pooled alignment | AG | EL | AG/EL | Eq/Div AUPRC | Safety AUPRC | False-safe | Shuffle gap | Held-out alignment | Reversal | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| aee | 0.3046 | 0.1115 | 0.1294 | 0.0435 | 2.975 | 0.7022 | 0.0472 | 0.3855 | 0.00006 | 0.2529 | 0.1993 | 85.7 |
| confidence_aee | 0.3074 | 0.1133 | 0.1398 | 0.0473 | 2.959 | 0.7028 | 0.0491 | 0.2543 | 0.00007 | 0.2569 | 0.2107 | 89.0 |
| factual_only | 0.2513 | 0.1324 | 0.0196 | 0.0116 | 1.687 | 0.6771 | 0.0548 | 0.5699 | -0.00001 | 0.1724 | 0.2500 | 68.0 |
| global_separation | 0.3130 | 0.1068 | 0.1336 | 0.0495 | 2.720 | 0.6968 | 0.0454 | 0.2340 | 0.00006 | 0.2665 | 0.1943 | 107.7 |
| mean_control | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00000 | N/A | N/A | 0.0 |
| multi_candidate_absolute | 0.3108 | 0.1182 | 0.0872 | 0.0257 | 3.393 | 0.6964 | 0.0425 | 0.2579 | 0.00003 | 0.2776 | 0.1823 | 76.3 |
| scene_only_control | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.5900 | 0.0326 | 0.4833 | 0.00000 | 0.0000 | 0.0000 | 67.7 |
| zero_control | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.00000 | N/A | N/A | 0.0 |

## Attribution checks

- **Action branch engineering failure:** ruled out by the separate Gate-2.5 synthetic fit, candidate overfit, gradient, Jacobian, variance, and optimizer-membership audit.
- **Factual-only statistical shortcut:** retained as a distinct diagnosis; the factual-only test alignment is 0.2513 and its structured action-shuffle gap is -0.00001.
- **Action-invariant target:** raw-map channels remain diagnostic and never enter the main collapse/Gate criteria; the nine-channel effect tube is evaluated channel by channel.
- **Multi-candidate data benefit:** absolute minus factual alignment is +0.0595; absolute minus factual false-safe rate is -0.3121.
- **AEE-specific benefit:** AEE minus absolute alignment is -0.0062; AEE minus global-separation equivalence leakage is -0.0061.

## Calibrated candidate risk

Thresholds are selected on validation scenes only and frozen for test. Values below are three-seed means; run-level scene-bootstrap intervals are in `phase6_artifacts/*.json`.

| Method | Unsafe prevalence | Balanced accuracy | AUROC | AUPRC | False-safe rate |
|---|---:|---:|---:|---:|---:|
| aee | 0.0879 | 0.6188 | 0.6679 | 0.1537 | 0.3855 |
| confidence_aee | 0.0879 | 0.6187 | 0.6729 | 0.1568 | 0.2543 |
| factual_only | 0.0879 | 0.5663 | 0.6144 | 0.1297 | 0.5699 |
| global_separation | 0.0879 | 0.6287 | 0.6734 | 0.1490 | 0.2340 |
| mean_control | N/A | N/A | N/A | N/A | N/A |
| multi_candidate_absolute | 0.0879 | 0.6103 | 0.6567 | 0.1346 | 0.2579 |
| scene_only_control | 0.0879 | 0.5991 | 0.6494 | 0.1329 | 0.4833 |
| zero_control | N/A | N/A | N/A | N/A | N/A |

## Structured effect channels

Per-channel rows, control comparisons, and scene-bootstrap intervals are saved in `phase6_artifacts/channel_metrics.csv`. Binary fields use balanced BCE/AUPRC/IoU; SDF and clearance use Huber/normalized L1; occupied velocity uses masked Huber/L1; footprint uses Dice/IoU.

Confidence-weighted AEE required in the matrix: **True**. It is retained in config but disabled when the LR/IDM disagreement subset is insufficient.

No Qwen/DiT parameter was updated and PDMS/EPDMS remain N/A.

## Structured-channel controls

The table reports the declared primary metric for each channel. Lower is better except for footprint Dice. Formal paired scene-bootstrap control and shuffle checks are reported in `gate3_decision.md`.

| Channel | Metric | AEE | Scene-only | Mean | Zero |
|---|---|---:|---:|---:|---:|
| candidate_relative_dynamic_occupancy | balanced_bce | 1.04371 | 1.04436 | 2.78507 | 12.64590 |
| drivable_area_sdf | normalized_l1 | 0.27457 | 0.27707 | 0.32187 | 0.41569 |
| lane_sdf | normalized_l1 | 0.26049 | 0.26268 | 0.30748 | 0.39554 |
| route_sdf | normalized_l1 | 0.24114 | 0.24012 | 0.31124 | 0.46316 |
| relative_longitudinal_velocity | masked_l1 | 0.19041 | 0.19309 | 0.18537 | 0.18628 |
| relative_lateral_velocity | masked_l1 | 0.18437 | 0.18437 | 0.17034 | 0.17038 |
| dynamic_clearance | normalized_l1 | 0.21513 | 0.21207 | 0.26657 | 0.41963 |
| dynamic_collision_field | balanced_bce | 0.85287 | 0.85231 | 1.22878 | 10.20346 |
| ego_swept_footprint | dice | 0.93062 | 0.93133 | 0.84089 | 0.01491 |

## Latent anti-collapse audit

| Method | Raw norm | Normalized variance | Covariance rank |
|---|---:|---:|---:|
| aee | 32.53169 | 0.0034251 | 85.7 |
| confidence_aee | 32.02829 | 0.0034099 | 89.0 |
| factual_only | 43.68415 | 0.0029629 | 68.0 |
| global_separation | 20.09452 | 0.0038006 | 107.7 |
| mean_control | N/A | N/A | 0.0 |
| multi_candidate_absolute | 41.89915 | 0.0029928 | 76.3 |
| scene_only_control | 39.14376 | 0.0029716 | 67.7 |
| zero_control | N/A | N/A | 0.0 |

## LR-to-IDM transfer

N/A in the scene-disjoint test split: all 128 scenes with reactive-model labels belong to the training partition used for the agreement/confidence audit. No reactive test label is copied across scenes, and this report does not invent a transfer score.
