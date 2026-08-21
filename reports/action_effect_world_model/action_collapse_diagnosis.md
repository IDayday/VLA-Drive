# Action-conditioning collapse diagnosis

## Scope and isolation

This Phase 5 pilot freezes Qwen and DiT, caches only current-observation action-query tokens, and trains lightweight consequence and structured-future probes on one factual expert trajectory per fit scene. Candidate futures/consequences are used only for held-out evaluation.

- Fit / held-out scenes: 406 / 102 (scene-disjoint deterministic split).
- Seeds: 3.
- Bootstrap: 1000 scene-clustered resamples at 95% confidence.
- Target statistics: factual anchors in the fit scenes only.
- World input: current images, navigation instruction, current ego state, and the candidate trajectory; no logged future enters Qwen or the probe condition.

## Quick consequence-probe results

Distances below are computed on predicted consequence outputs (hard probabilities plus normalized soft predictions), so arbitrary unsupervised trajectory-encoder variation cannot masquerade as action sensitivity.

| Method | Factual error ↓ | Shuffle gap ↑ | Candidate sensitivity ↑ | Action gap ↑ | Equivalence leakage ↓ | Effect alignment ↑ | False-safe ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| constant_mean_control | 0.3151 [0.3057, 0.3258] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| random_untrained_probe | 0.4821 [0.4680, 0.4955] | 0.0001 [-0.0000, 0.0002] | 0.0235 [0.0221, 0.0249] | 0.0278 [0.0253, 0.0307] | 0.0175 [0.0159, 0.0190] | 0.1875 [0.1463, 0.2330] | 0.0000 [0.0000, 0.0000] |
| same_parameter_no_action | 0.1686 [0.1560, 0.1810] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| scene_action_probe | 0.1660 [0.1535, 0.1787] | 0.0003 [0.0003, 0.0004] | 0.0092 [0.0085, 0.0098] | 0.0114 [0.0101, 0.0127] | 0.0062 [0.0055, 0.0069] | 0.2083 [0.1677, 0.2488] | 1.0000 [1.0000, 1.0000] |
| scene_only_probe | 0.1686 [0.1560, 0.1810] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| shuffled_action_probe | 0.1669 [0.1539, 0.1793] | -0.0000 [-0.0000, 0.0000] | 0.0023 [0.0022, 0.0024] | 0.0029 [0.0026, 0.0032] | 0.0018 [0.0016, 0.0020] | 0.2216 [0.1810, 0.2615] | 1.0000 [1.0000, 1.0000] |
| trajectory_only_probe | 0.1789 [0.1654, 0.1920] | 0.0065 [0.0055, 0.0073] | 0.1004 [0.0947, 0.1067] | 0.1214 [0.1116, 0.1310] | 0.0561 [0.0524, 0.0596] | 0.3625 [0.3380, 0.3884] | 1.0000 [1.0000, 1.0000] |

## Structured-future confirmation

The second probe predicts a candidate-aligned `[3,7,32,32]` tube at 1, 2, and 4 seconds. Channels are drivable area, lane/connector, route, log-replay dynamic occupancy, relative longitudinal/lateral velocity, and dynamic clearance. All future fields are target-only.

The class-balanced structured objective is learnable: scene-action reaches 0.708 versus 1.265 for the fit-only per-cell mean control. Scene-only is slightly better at 0.699, which is direct evidence that the learned factual future is scene-predictive but action conditioning adds no factual benefit. Unweighted map MAE goes the opposite way (0.235 versus 0.196 for the mean control) because the trained objective deliberately upweights sparse dynamic occupancy; this limitation is retained rather than hidden.

| Method | Balanced objective ↓ | Map MAE ↓ | Shuffle gap ↑ | Candidate sensitivity ↑ | Action gap ↑ | Equivalence leakage ↓ | Effect alignment ↑ | False-safe ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| constant_mean_control | 1.2653 | 0.1955 | 0.000000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scene_only_probe | 0.6987 | 0.2296 | 0.000000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| scene_action_probe | 0.7077 | 0.2347 | -0.000046 | 0.0095 | 0.0139 | 0.0061 | 0.1066 | 0.5056 |
| shuffled_action_probe | 0.7110 | 0.2196 | -0.000001 | 0.0007 | 0.0010 | 0.0005 | 0.1806 | 0.7191 |
| trajectory_only_probe | 0.7437 | 0.2590 | -0.000135 | 0.0417 | 0.0494 | 0.0275 | 0.1975 | 1.0000 |

Structured metrics use one diagnostic seed with 1,000 scene-clustered bootstrap samples. For `scene_action_probe`, Effect Alignment has 95% CI `[0.033, 0.183]`, the shuffle gap CI is `[-0.000079, -0.000019]`, and false-safe is `0.506 [0.303, 0.711]`. The three-seed consequence probe above carries the multi-seed direction evidence.

## Gate 2

**Decision: PASS.** Both the three-seed consequence probe and the single-seed structured confirmation support the action-collapse pattern required before implementing multi-candidate/AEE training.

- Factual probe learnable versus low-information controls: True.
- Small action-shuffle gap: True.
- Low effect alignment: True.
- Divergent outputs compressed relative to equivalence leakage: False.
- High unsafe-candidate false-safe rate: True.
- Structured factual objective learnable versus fit-only mean: True.
- Structured scene-only objective better than scene-action: True (collapse evidence, not a world-model quality win).
- Structured unweighted map MAE improves over the mean: False; sparse-risk weighting is an explicit remaining limitation.

The numerical gate thresholds are exploratory diagnostics recorded in `action_collapse_artifacts/gate2.json`; they are not changes to NAVSIM evaluator semantics.

## Artifacts

- `action_collapse_artifacts/metrics_by_run.csv`
- `action_collapse_artifacts/metrics_summary.csv`
- `action_collapse_artifacts/collapse_metric_comparison.png`
- `action_collapse_artifacts/predicted_vs_true_effect_distance.png`
- `action_collapse_artifacts/false_safe_examples.jsonl`
- `structured_collapse_artifacts/metrics_by_run.csv`
- `structured_collapse_artifacts/metrics_summary.csv`
- `structured_collapse_artifacts/structured_collapse_comparison.png`
- `structured_collapse_artifacts/structured_predicted_vs_true_effect.png`
- `structured_collapse_artifacts/false_safe_examples.jsonl`
