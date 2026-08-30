# Predicted Candidate-Consequence Probe

- Scope: **500 scenes / 6000 candidates**.
- Outer split: complete logs, **120 train / 35 validation**, overlap **0**.
- Training consequence features: **5 log-group folds, out-of-fold**.
- Train-OOF-selected primary predictor: **mlp_delta**.
- Primary optimization convergence: **PASS**.
- Candidate-specific fidelity gate: **FAIL**.
- Overall prediction gate: **PREDICTOR_FIDELITY_NOT_MET**.
- Leakage audit: **PASS**.
- Fixed-seed repeat: **PASS**.

## Dynamic-consequence prediction quality

| Predictor | Validation NRMSE | Candidate-delta NRMSE | Candidate-delta Spearman | Pairwise-distance Spearman | Candidate-variance recovery |
|---|---:|---:|---:|---:|---:|
| extra_trees | 1.0242 | 1.9009 | 0.0545 | 0.3417 | 0.1149 |
| mlp_delta | 1.0980 | 1.1797 | 0.1272 | 0.2971 | 0.0209 |
| mlp_delta_strong | 1.1724 | 1.0979 | 0.1262 | 0.3657 | 0.0090 |
| mlp_raw | 1.0134 | 2.6366 | 0.0463 | 0.2377 | 0.2377 |
| ridge | 1.1656 | 4.2129 | 0.0288 | 0.3077 | 0.8887 |

Only dynamic channels are predicted. Drivable-area, lane, and route SDF channels are exact candidate/map geometry and are supplied to every fair online baseline.

## Predictor convergence

| Predictor | Optimization gate | Median monitor improvement | Median best epoch | Parameters |
|---|---|---:|---:|---:|
| mlp_delta | PASS | 37.8% | 44 | 443241 |
| mlp_delta_strong | PASS | 41.3% | 27 | 443241 |
| mlp_raw | PASS | 33.2% | 46 | 443241 |

Epoch selection uses only an inner log-disjoint monitor split inside each OOF training fold. The OOF fold itself and the outer validation logs are not used for early stopping.

Small-subset overfit capacity sanity: **PASS** on 8 scenes / 96 candidates. Train loss reduction **100.0%**; candidate-delta Spearman **0.8770506302605902**; pairwise-distance Spearman **0.9958043287496663**; variance recovery **0.9876388754313509**.

The overfit check is an implementation/capacity diagnostic only. It is never used as held-out evidence or for model selection.

## Planning utility

| Probe | Pairwise | NDCG | Scene Spearman | Top-1 | Regret | Score RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Probe_B_original_current_plus_trajectory | 0.5928 | 0.9925 | 0.2758 | 0.5625 | 0.01812 | 0.2117 |
| Probe_D_fair_online_direct | 0.5870 | 0.9933 | 0.2287 | 0.5521 | 0.01818 | 0.1892 |
| Probe_E_predicted_dynamic_extra_trees | 0.5930 | 0.9928 | 0.2412 | 0.5833 | 0.01823 | 0.1850 |
| Probe_E_predicted_dynamic_mlp_delta | 0.5652 | 0.9904 | 0.1656 | 0.5833 | 0.02456 | 0.1885 |
| Probe_E_predicted_dynamic_mlp_delta_strong | 0.5546 | 0.9918 | 0.1309 | 0.5521 | 0.02614 | 0.1950 |
| Probe_E_predicted_dynamic_mlp_raw | 0.5870 | 0.9918 | 0.2098 | 0.5312 | 0.02866 | 0.1857 |
| Probe_E_predicted_dynamic_ridge | 0.5887 | 0.9936 | 0.2189 | 0.5833 | 0.01789 | 0.1881 |
| Probe_F_oracle_dynamic_ceiling | 0.6033 | 0.9913 | 0.2504 | 0.4896 | 0.03372 | 0.1848 |

Primary predicted-vs-direct judgement: **INCONCLUSIVE**.

Paired scene-bootstrap deltas (predicted minus fair direct):

- ndcg_mean: **-0.002918**, 95% CI [-0.006772, +0.000217]
- pairwise_ranking_accuracy: **-0.021838**, 95% CI [-0.052099, +0.007505]
- spearman_per_scene_mean: **-0.063050**, 95% CI [-0.126312, -0.000994]
- top1_accuracy: **+0.031250**, 95% CI [-0.031250, +0.093750]
- top1_score_regret_mean: **+0.006385**, 95% CI [-0.003359, +0.021765]

## Interpretation boundary

The direct baseline sees exactly the same current structured actor state, candidate trajectory, constant-velocity candidate/actor interactions, and exact map channels as the consequence predictor. Therefore any predicted-consequence gain is an intermediate-representation/inductive-bias gain, not extra online information.

Logged-future consequences are supervised labels only. Downstream outer-train features are log-group OOF predictions and outer-validation features are predictions from a model trained only on outer-train logs; oracle future values are never input to the predicted planner.

Optimization convergence, candidate-specific prediction fidelity, and downstream planning utility are separate gates. A downstream no-gain result does not reject the method unless prediction fidelity first passes.

Current actors come from planning-instant annotations in this minimal probe. Dynamic future consequences are predictions, but this is still a structured-perception upper bound rather than an end-to-end camera-to-consequence result.

The oracle-dynamic probe is only a ceiling. It is never used as an online feature and validation oracle targets never train either model.
