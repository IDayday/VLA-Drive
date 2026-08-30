# Predicted Candidate-Consequence Probe

- Scope: **2000 scenes / 24000 candidates**.
- Outer split: complete logs, **433 train / 110 validation**, overlap **0**.
- Training consequence features: **5 log-group folds, out-of-fold**.
- Train-OOF-selected primary predictor: **mlp_delta_strong**.
- Primary optimization convergence: **PASS**.
- Candidate-specific fidelity gate: **FAIL**.
- Overall prediction gate: **PREDICTOR_FIDELITY_NOT_MET**.
- Leakage audit: **PASS**.
- Fixed-seed repeat: **NOT VERIFIED**.

## Dynamic-consequence prediction quality

| Predictor | Validation NRMSE | Candidate-delta NRMSE | Candidate-delta Spearman | Pairwise-distance Spearman | Candidate-variance recovery |
|---|---:|---:|---:|---:|---:|
| extra_trees | 0.6923 | 1.9156 | 0.0771 | 0.3666 | 0.1636 |
| mlp_delta | 0.7879 | 1.1141 | 0.1665 | 0.5007 | 0.0501 |
| mlp_delta_strong | 0.8757 | 1.0230 | 0.1880 | 0.5190 | 0.0422 |

Only dynamic channels are predicted. Drivable-area, lane, and route SDF channels are exact candidate/map geometry and are supplied to every fair online baseline.

## Predictor convergence

| Predictor | Optimization gate | Median monitor improvement | Median best epoch | Parameters |
|---|---|---:|---:|---:|
| mlp_delta | PASS | 47.1% | 78 | 443241 |
| mlp_delta_strong | PASS | 49.4% | 58 | 443241 |

Epoch selection uses only an inner log-disjoint monitor split inside each OOF training fold. The OOF fold itself and the outer validation logs are not used for early stopping.

Small-subset overfit capacity sanity: **PASS** on 8 scenes / 96 candidates. Train loss reduction **100.0%**; candidate-delta Spearman **0.8829113816475511**; pairwise-distance Spearman **0.9970388340491613**; variance recovery **0.9927963103139412**.

The overfit check is an implementation/capacity diagnostic only. It is never used as held-out evidence or for model selection.

## Planning utility

| Probe | Pairwise | NDCG | Scene Spearman | Top-1 | Regret | Score RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Probe_B_original_current_plus_trajectory | 0.5561 | 0.9569 | 0.2007 | 0.5587 | 0.03414 | 0.2307 |
| Probe_D_fair_online_direct | 0.5875 | 0.9595 | 0.2141 | 0.5979 | 0.02671 | 0.2249 |
| Probe_E_predicted_dynamic_extra_trees | 0.5807 | 0.9602 | 0.1979 | 0.6136 | 0.02365 | 0.2235 |
| Probe_E_predicted_dynamic_mlp_delta | 0.5848 | 0.9604 | 0.2105 | 0.6057 | 0.02227 | 0.2265 |
| Probe_E_predicted_dynamic_mlp_delta_strong | 0.5963 | 0.9602 | 0.2295 | 0.5953 | 0.02581 | 0.2222 |
| Probe_F_oracle_dynamic_ceiling | 0.5999 | 0.9608 | 0.2404 | 0.6084 | 0.01973 | 0.2221 |

Primary predicted-vs-direct judgement: **CONDITIONAL_PASS**.

Paired scene-bootstrap deltas (predicted minus fair direct):

- ndcg_mean: **+0.000683**, 95% CI [-0.000580, +0.002181]
- pairwise_ranking_accuracy: **+0.008758**, 95% CI [-0.005786, +0.022863]
- spearman_per_scene_mean: **+0.015379**, 95% CI [-0.015636, +0.045070]
- top1_accuracy: **-0.002611**, 95% CI [-0.046997, +0.041775]
- top1_score_regret_mean: **-0.000906**, 95% CI [-0.009751, +0.006139]

## Interpretation boundary

The direct baseline sees exactly the same current structured actor state, candidate trajectory, constant-velocity candidate/actor interactions, and exact map channels as the consequence predictor. Therefore any predicted-consequence gain is an intermediate-representation/inductive-bias gain, not extra online information.

Logged-future consequences are supervised labels only. Downstream outer-train features are log-group OOF predictions and outer-validation features are predictions from a model trained only on outer-train logs; oracle future values are never input to the predicted planner.

Optimization convergence, candidate-specific prediction fidelity, and downstream planning utility are separate gates. A downstream no-gain result does not reject the method unless prediction fidelity first passes.

Current actors come from planning-instant annotations in this minimal probe. Dynamic future consequences are predictions, but this is still a structured-perception upper bound rather than an end-to-end camera-to-consequence result.

The oracle-dynamic probe is only a ceiling. It is never used as an online feature and validation oracle targets never train either model.
