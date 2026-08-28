# NAVSIM Candidate-relative Future Supervision — Final Feasibility Report

## 1. 本地环境

- Repository / branch / source base commit: `/mnt/project/DriveVLA-M0` / `feature/navsim-candidate-relative-feasibility-audit` / `d84bf2b39696050f715fe41c5f005d0d1115c0c1`
- Runtime NAVSIM: `1.1.0` from `/mnt/project/DriveVLA-M0/navsim/__init__.py`
- Additional v2 devkit: `2.0.0` at commit `9fe1459b8f6ab69a15274450ec301d541209bedd`; it is not the runtime import.
- Dataset split: `trainval`; audited Scene sample `500`, cache-matched candidate scenes `500`.
- Log coverage: general sequential statistics sample `2` logs; cache-matched candidate/oracle sample `6` logs. Oracle train/validation is split by complete log, never by adjacent scene.
- Metric cache: `/mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full`; deployed runtime class is the local `train_metric_chache.MetricCache` adaptation documented in `FIELD_AUDIT.md`.
- Synthetic data: `True` (204 warmup synthetic scenes).
- Reactive policy code / eligible training cache: `True` / `False`.
- Tests: `pytest -q tests` passed 26/26. Root-level collection additionally reaches vendored `nuplan-devkit` and is blocked by missing optional upstream test dependencies; see `TEST_AUDIT.md`.

## 2. 数据支持矩阵

The authoritative Q1–Q20 table is in `SUPPORT_MATRIX.csv` and `SUPPORT_MATRIX.md`. Counts are observed, not inferred from public NAVSIM documentation. Summary class counts: `{'A_DIRECT': 9, 'C_NONREACTIVE_ASSUMPTION': 4, 'B_EXACT_DERIVATION': 3, 'D_REACTIVE_OR_SYNTHETIC_ONLY': 3, 'E_UNAVAILABLE': 1}`.

## 3. 数据覆盖率

- GT 4 s: 100.00% (500/500)
- future CAM_F0: 100.00% (2000/2000)
- future annotations: 100.00% (500/500)
- track continuity: 1.0000
- map: 100.00% (500/500)
- route: 100.00% (500/500)
- traffic light active-any: 58.20% (291/500)
- general-sample metric-cache overlap: 23.00% (115/500)
- cache-matched scoring: 500 scenes / 100.00% candidates

The general Scene sample and cache-matched candidate sample are reported separately. The 500-scene general sample is sequential and covers only the log count stated above, so its percentages are deployment evidence rather than a claim about the full trainval distribution. Its cache overlap does not replace the measured 100.00% Gate-A cache load rate on selected training-cache tokens.

## 4. 候选构造

- Source: deterministic smooth fallback because no existing multi-trajectory dump was found; these are controlled perturbations, not real futures.
- Scenes × K: `500 × 12`; GT candidate is an additional immutable candidate at index 0.
- Official scoring success: `100.000%`; repeated and batch-vs-single maximum errors: `0.0` / `0.0`.
- Consequence diversity: non-zero pair ratio `100.000%`, mean unique consequences `12.00`, hard-negative pairs `5692`.
- Trajectory→consequence and consequence→score-difference Spearman: `0.5763` / `0.3180`.

## 5. Candidate-relative consequence

- Direct logged quantities: ego state/GT trajectory, future sensor paths, future annotation boxes/velocity/token, traffic lights, map and route.
- Exact derivations: SE(2) alignment, candidate dynamics rollout, static map/route relations and locally supported PDM factors.
- Non-reactive-assumption quantities: candidate-relative actor state, actor clearance/corridor/collision/TTC and combined structured risks against the shared logged future.
- Reactive/synthetic only: candidate-conditioned vehicle response in v2 IDM and warmup synthetic follow-up scenes.
- Unavailable: non-GT ground-truth images, causal effects, true multi-agent response in v1, and local v1 TLC/lane-keeping/EPDMS/extended-comfort fields.

`C_full` contains `9` trajectory-derived plus `15` environment fields. `C_environment_only` excludes waypoint copies, candidate identity/type and official score/factor columns; it is the oracle input. Target success is `100.000%`.

## 6. Soft contrastive label

- 0.5 s: effective positives `9.569`, false negatives under one-hot `6.538`, GT weight `0.154`
- 1.0 s: effective positives `9.775`, false negatives under one-hot `6.750`, GT weight `0.159`
- 2.0 s: effective positives `9.944`, false negatives under one-hot `7.518`, GT weight `0.166`
- 4.0 s: effective positives `10.126`, false negatives under one-hot `8.746`, GT weight `0.192`

Same-prefix/different-tail short-positive and long-separation rates are `1.0` / `1.0`. Hard one-hot would therefore create measurable false negatives. The K×K consequence label uses stable actor hashes, masks, standardized mixed units and only each horizon's prefix.

## 7. Oracle planning utility

| Probe | Pairwise | NDCG | Spearman | Top-1 | Regret |
|---|---:|---:|---:|---:|---:|
| A trajectory-only | 0.4801 | 0.9047 | -0.0407 | 0.1875 | 0.2059 |
| B current+trajectory | 0.4721 | 0.8894 | -0.0261 | 0.1875 | 0.2862 |
| C candidate-relative future | 0.7642 | 0.9807 | 0.6396 | 0.4602 | 0.0532 |

- Probe-C planning gain decision: `True`; Δpairwise(C−A) `0.2841`, Δregret(A−C) `0.1527`.
- Feature-name leakage audit passed: `True`. `C_environment_only` contains no official final or component score, candidate type/index or direct candidate waypoint copy.
- C vs A AUROC: collision `0.9506` vs `0.4817`, TTC `0.9022` vs `0.5263`, DAC `0.9985` vs `0.6319`, DDC `0.9845` vs `0.4851`; progress Spearman `0.6724` vs `0.0084`. Thus the gain is not collision/TTC-only; map compliance and progress also carry signal, while route-specific and red-light-specific attribution remains unidentifiable with this scorer.
- Factor attribution is limited to A/B/C prediction deltas for collision, TTC, DAC, DDC, comfort and progress. The deployed scorer exposes no TLC/lane-keeping/EPDMS target, so route/red-light utility cannot be separately claimed.
- Interaction-only inverse result: 当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。
- Interaction-only candidate-ID / semantic accuracy versus majority: `0.2505` / `0.0833` and `0.3878` / `0.3333`; Δtrajectory R² `-0.7288`.

This is an oracle association test: those future relations are training targets and must be predicted from the current scene plus candidate at inference.

## 8. GT future visual anchor

- Logged CAM_F0 future path / full synchronized-chain coverage: `100.000%` / `100.000%` across `2000` scene-horizon records. Image dimensions/decodability were opened on the bounded field-audit sample and the 12 rendered anchor scenes; the remaining count is path existence, not bulk decode.
- Supported: `logged I_GT(t+h) ↔ C_GT,h` for GT-only visual semantic alignment.
- Unsupported: `I_candidate_i(t+h)` for every non-GT candidate; no such real sensor viewpoint exists in the logs.
- Saved scene evidence: `figures/visual_anchor/` plus the twelve global audit figures.

## 9. Reactive 与 synthetic 扩展

- Reactive v2 code exists and simulates `VEHICLE`; all remaining object types are merged from log replay.
- No reactive response metrics were fabricated: the only deployed v2 cache records split `navtest`, which is excluded.
- Warmup synthetic: `204` scenes mapping to `16` originals; follow-ups/original min/median/max `9/11.5/19`.
- Synthetic referenced camera / LiDAR files: `100.000%` / `0.000%`; annotations and at least eight extended-track steps are audited separately in `V2_EXTENSION_REPORT.md`.
- Same-original groups with non-identical synthetic current states: `100.000%`. Therefore they are synthetic follow-up scenes / weak neighborhood supervision, not same-current real counterfactuals.

## 10. 最终五项判定

- F1 唯一 logged future → K 个 candidate-relative structured consequences: **PASS**
- F2 GT future image → GT structured future 的视觉语义锚定: **PASS**
- F3 prefix-aware soft contrastive supervision: **PASS**
- F4 使用 structured consequence 训练独立 inverse verifier: **CONDITIONAL_PASS**
- F5 为每条非 GT candidate 提供真实 future image supervision: **FAIL**

F4 的 `CONDITIONAL_PASS` 只支持风险/一致性 verifier：当前 interaction-only 分类有部分信号，但 Δtrajectory 回归未恢复候选运动，因此不等价于强 interaction inverse dynamics。F5 的失败来自实际文件/视点链路审计。

## 11. 推荐的下一步方法版本

**Plan A：完整候选相对世界模型（仅 GT logged future image 作视觉 anchor；非 GT 使用 structured targets）。**

Primary blocker: 没有任何非 GT candidate-specific ground-truth future image；合法训练 split 上也没有可运行的 v2 reactive metric cache。

## 12. 实现建议

建议的最小下一阶段接口（不在本任务中训练完整模型）：

- `current_scene_representation`: 当前相机/BEV、当前 actors、traffic lights、map/route；推理可用。
- `candidate_trajectory`: `K×8×3` current-ego-local poses + mask；包括模型候选和单独 GT。
- `candidate_relative_target_schema`: 预测 `C_environment_only[K,8,15]` 与 `candidate_relative_actor[K,8,N,10]`，使用 mask，不输入 token string。
- `GT_visual_anchor`: 只在 GT row/horizon 对齐 logged CAM_F0 embedding 与 `C_GT,h`；非 GT 无 image loss。
- `soft_contrastive_target`: horizon-prefix factual `q[K]` 和 consequence `Q[K,K]`。
- `utility_target`: 离线 official aggregate + supported factor targets，明确 unavailable factor mask。
- `inverse_verifier_target`: 输入预测的 environment interactions，输出 candidate consistency/risk ranking；若 interaction-only probe 近随机，不要求恢复完整 trajectory。

Terminology throughout: logged future, non-reactive candidate-relative consequence, candidate-conditioned relabeling, reactive-policy simulated consequence and synthetic follow-up scene. No result is described as a real counterfactual future.
