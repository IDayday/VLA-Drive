# Oracle Planning-Utility Probe

- Scope: **500 scenes / 6000 candidates**
- Split: complete `log_name`; train **120 logs**, validation **35 logs**, overlap **0**
- Feature leakage audit: **PASS**

| Probe | Pairwise accuracy | NDCG | Per-scene Spearman | Top-1 accuracy | Top-1 regret |
|---|---:|---:|---:|---:|---:|
| Probe_A_trajectory_only | 0.5604 | 0.9862 | 0.1776 | 0.5208 | 0.05894 |
| Probe_B_current_scene_plus_trajectory | 0.5928 | 0.9925 | 0.2758 | 0.5625 | 0.01812 |
| Probe_C_candidate_relative_future | 0.5728 | 0.9901 | 0.2152 | 0.4792 | 0.03150 |

## Probe C gains

- pairwise_ranking_accuracy: `0.012479001679865598`
- ndcg_mean: `0.00384011619297131`
- spearman_per_scene_mean: `0.037561120404610404`
- top1_accuracy: `-0.041666666666666685`
- top1_score_regret_mean: `-0.027443014085292816`

Probe C adds only candidate-relative future effect-tube summaries: dynamic occupancy/relative velocity/clearance/collision fields and map/lane/route SDF. It excludes the trajectory-derived ego-footprint channel and every official aggregate or factor score.

## Interaction-only inverse probe

- Accuracy: **0.2248** (majority chance **0.1667**)
- Macro F1: **0.1658**
- Interpretation: 当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。

This is an oracle sufficiency audit, not a deployable predictor: Probe C consumes logged-future-derived information that is unavailable online unless a learned future model predicts it.
