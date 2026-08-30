# NAVSIM Candidate-Relative Future Supervision: Final Feasibility Report

> Scope: legal `train` scenes from deployed NAVSIM trainval data. No test/navtest/navhard annotations were used to construct training targets. NAVHARD synthetic files were inspected only for deployment metadata.

## 1. Local environment

- Repository: `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit`
- Branch: `feature/navsim-candidate-relative-feasibility-audit`
- Audited base commit: `1482f1da87e31907b549f09836a38f99fd18f200`
- NAVSIM runtime: `2.0.0` from `/mnt/workspace/project/DriveDreamer-Policy-navsim-candidate-relative-audit/navsim/navsim/__init__.py`
- Also present but not imported: vendored NAVSIM 1.1 source tree.
- Dataset split: `train` selected from deployed trainval logs; field sample **500 scenes**.
- Official MetricCache: `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/metric_cache/pilot_small/train_phase6_v1`.
- Candidate cache: `/mnt/workspace/project/DriveDreamer-Policy/action_effect_cache/candidates/pilot_small/expert_phase6_v1`.
- Synthetic root: `/mnt/data_and_weight/Public_Space/navsim/navhard_two_stage/synthetic_scene_pickles` (NAVHARD; metadata-only, not training eligible).
- Reactive policy: **True**, cached on 128 scenes.

## 2. Data support matrix

The full evidence table is in `SUPPORT_MATRIX.md` / `SUPPORT_MATRIX.csv`: A_DIRECT=7, B_EXACT_DERIVATION=6, C_NONREACTIVE_ASSUMPTION=3, D_REACTIVE_OR_SYNTHETIC_ONLY=3, E_UNAVAILABLE=1.

| ID | Class | Conclusion |
|---|---|---|
| Q1 | `A_DIRECT` | Raw Frame/Scene 直接含 ego pose、velocity、acceleration。 |
| Q2 | `B_EXACT_DERIVATION` | 由逐帧 logged ego pose 经官方 absolute-to-relative SE(2) 转换得到；Gate A 与逐帧构造零误差。 |
| Q3 | `B_EXACT_DERIVATION` | 由 GT pose/timestamp 差分并 wrap heading 精确推导；原始 ego future frame 也提供动态状态。 |
| Q4 | `A_DIRECT` | Sparse audited horizons 的 CAM_F0 是 logged future 的实际图像文件。 |
| Q5 | `A_DIRECT` | Sparse audited horizons 的 logged future LiDAR blob 存在。 |
| Q6 | `A_DIRECT` | 每个 logged future raw frame 直接含 boxes；MetricCache 提供 10 Hz tracked objects。 |
| Q7 | `A_DIRECT` | Raw annotations 和 official tracked object 均含 velocity。 |
| Q8 | `A_DIRECT` | Raw annotations 直接含 track_tokens；连续率单独量化。 |
| Q9 | `A_DIRECT` | Future raw frames 直接含 lane connector 与 is_red；MetricCache observation 含 red-light occupancy。 |
| Q10 | `B_EXACT_DERIVATION` | 地图/route IDs 直接给出，几何关系通过官方 map API/centerline 精确查询。 |
| Q11 | `B_EXACT_DERIVATION` | 合法候选可经官方 PDMSimulator 确定性 rollout 到 10 Hz state array。 |
| Q12 | `C_NONREACTIVE_ASSUMPTION` | 地图/运动学因子可精确求，完整动态评价中的 collision/TTC/TLC 依赖 logged traffic replay；官方 scorer 路径可运行。 |
| Q13 | `C_NONREACTIVE_ASSUMPTION` | 把统一 global logged actor world 转到每个 candidate ego frame 后得到候选相关张量。 |
| Q14 | `B_EXACT_DERIVATION` | 候选 footprint/center 与静态 map、centerline、route 的几何关系精确查询。 |
| Q15 | `C_NONREACTIVE_ASSUMPTION` | collision/clearance/TTC/corridor/route/light 等逐 horizon 后果可由 one logged future 形成 K 份。 |
| Q16 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | NAVSIM v2 IDM 可产生车辆的 candidate-dependent simulated response；非车辆仍 replay。 |
| Q17 | `E_UNAVAILABLE` | 唯一 logged future 只给 GT camera view；candidate-conditioned relabeling不能改变记录像素。 |
| Q18 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | 部署中有 follow-up metadata/camera/extended tracks，但只解析到 NAVHARD two-stage 路径，当前 train 审计不得用其标注；且不支持 same-current-state claim。 |
| Q19 | `D_REACTIVE_OR_SYNTHETIC_ONLY` | 本地 NAVSIM v2 IDM 实现和 reactive consequence cache 均可用；只模拟 VEHICLE。 |
| Q20 | `B_EXACT_DERIVATION` | Oracle 与预测后果必须分开回答：oracle 结果混合；500→2000 场景的 log-safe OOF 预测显示候选差异 fidelity 和规划点估计改善，但候选方差仍严重塌缩且规划置信区间跨零，因此尚未证明稳健增益。 |

## 3. Data coverage

- GT future 4 s: **100.00%**
- Sparse future front camera: **100.00%**
- Sparse future LiDAR: **100.00%**
- Future actor/velocity/track fields: **100.00% / 100.00% / 100.00%**
- Raw / MetricCache adjacent track continuity: **93.93% / 99.18%**
- Map / route: **100.00% / 100.00%**
- Traffic-light field / nonempty: **100.00% / 50.20%**
- MetricCache load success: **100.00%**

Timestamp gaps were measured, not assumed: at least one smoke scene contained a ~1.0 s gap, so every horizon is resolved by nearest timestamp rather than fixed raw-array index.

## 4. Candidate construction

- Source: **Pre-existing deterministic expert-anchor perturbation cache; not model multi-sample output**.
- Scope: **64 scenes × 12 candidates**.
- Kinematic validity: **100.00%**; GT max anchor mismatch **1.8264839074145932e-06 m**.
- Official score success: **768 / 768 (100.00%)**; factor-diverse scenes **55**.

This is a deterministic expert-anchor perturbation bank, not a model multi-sample dump and not a collection of real futures. GT is inserted explicitly at candidate index 0.

## 5. Candidate-relative consequence

- Target coverage: **100.00%**, actor-slot valid mask coverage **90.84%**.
- Nonzero candidate-pair consequence ratio: **100.00%**; hard negatives: **651**.
- O(K²) directed relations: **9216**.
- Trajectory/consequence Spearman: **0.25699575934548347**; consequence/score-difference Spearman: **0.12751451415465964**.

The schemas remain separate: `trajectory_derived` is recoverable from the candidate, `shared_logged_future` is candidate-independent, `C_environment_only` requires candidate/world interaction, and `reactive_response` is populated only by an actual reactive-policy run. Official PDM scores/factors and waypoint copies are excluded from `C_environment_only`.

## 6. Prefix-aware soft contrastive labels

- Prefix-only construction: **True**; any tail-after-horizon use: **False**.
- Probability rows sum to one: **True**.
- Same-prefix/different-tail examples: **62**, pass rate **100.00%**.

| Horizon [s] | Mean GT weight | Effective positives | One-hot false negatives |
|---:|---:|---:|---:|
| 0.5 | 0.13458420883398503 | 9.650796340051922 | 7.046875 |
| 1.0 | 0.13504954369273037 | 9.853225007877139 | 6.90625 |
| 2.0 | 0.1414201979059726 | 10.096786222111485 | 6.640625 |
| 4.0 | 0.14585917140357196 | 10.229519298181646 | 6.625 |

The effective-positive counts and false-negative counts show that hard one-hot treats many prefix-compatible candidates as equally negative. Both GT-factual q and candidate-consequence K×K Q are non-degenerate.

## 7. Oracle and predicted planning utility

### 7.1 Oracle sufficiency check

- Scope: **500 scenes / 6000 candidates**, split by complete `log_name`; overlap **0**.
- Leakage audit: **PASS**.

| Probe | Pairwise ranking | NDCG | Per-scene Spearman | Top-1 | Regret |
|---|---:|---:|---:|---:|---:|
| A trajectory-only | 0.5603551715862731 | 0.9862203806905908 | 0.17763168476581512 | 0.5208333333333334 | 0.058943704391519226 |
| B current+trajectory | 0.5927525797936165 | 0.9924553789100642 | 0.275759344835762 | 0.5625 | 0.018122744436065357 |
| C + candidate-relative future | 0.5728341732661387 | 0.9900604968835621 | 0.21519280517042552 | 0.4791666666666667 | 0.03150069030622641 |

Probe C − A: `{"ndcg_mean": 0.00384011619297131, "pairwise_ranking_accuracy": 0.012479001679865598, "spearman_per_scene_mean": 0.037561120404610404, "top1_accuracy": -0.041666666666666685, "top1_score_regret_mean": -0.027443014085292816}`.

Against Probe B, Probe C changes pairwise ranking by **-0.019918406527477783** and score RMSE from **0.21173424847855934** to **0.1809522739206602**. Thus the result is mixed: candidate-relative future improves score regression and the trajectory-only ranking baseline, but does not beat current-scene+trajectory on every ranking metric.

The clearest factor gains over Probe B are DAC AUROC **0.8010414163262035 → 0.9297284306657054** and progress Spearman **0.5614356692971542 → 0.622220089465132**. Collision AUROC changes **0.714150668993601 → 0.5938772542175683** and TTC AUROC **0.8110745614035088 → 0.8142543859649124**. DDC/TLC validation labels are effectively constant, so those factors cannot support a positive utility claim here.

Interaction-only inverse accuracy is **0.2248263888888889** versus majority chance **0.16666666666666666**. 当前数据可支持候选相对风险重标注，但不足以支持强 interaction inverse dynamics。

Probe C uses non-reactive effect-tube relations (dynamic occupancy, relative velocities, clearance/collision fields, map/lane/route SDF) and explicitly excludes official PDM aggregate/factors, candidate type/ID, waypoint copies, and the trajectory-derived ego-footprint tube channel.

The oracle result answers only whether the target contains useful information. It does not establish that a model can predict that information from the current scene.

### 7.2 Predicted-consequence convergence and fidelity

- Scope: **500 scenes / 6000 candidates**; outer split **120 train logs / 35 validation logs**, with overlap **0**.
- Dynamic consequence predictor selection used outer-train OOF fidelity only; selected **mlp_delta**. Outer validation was not used for model/strength selection: **True**.
- Optimization convergence: **PASS**. Candidate-specific fidelity: **FAIL**. Overall gate: **`PREDICTOR_FIDELITY_NOT_MET`**.

| Predictor | Optimization | Median held-out-log monitor improvement | Median best epoch | Parameters |
|---|---|---:|---:|---:|
| mlp_delta | PASS | 37.78% | 44.0 | 443241 |
| mlp_delta_strong | PASS | 41.33% | 27.0 | 443241 |
| mlp_raw | PASS | 33.25% | 46.0 | 443241 |

For the selected predictor, train-OOF candidate-delta Spearman is **0.13293298038412757**. On unseen outer-validation logs it is **0.12723538190737874**, pairwise consequence-distance Spearman is **0.2970739017766492**, and candidate-variance recovery is only **0.02087384080803929**.

The MLP therefore did optimize: inner-log monitor loss improved and early stopping selected nonzero epochs. But optimization convergence is not prediction success. The low candidate-centered correlations and severe variance collapse show weak cross-log generalization of candidate-specific effects, even though scene-shared/global reconstruction can look strong.

Small-subset capacity sanity is **PASS** on **8 scenes / 96 candidates**: train loss reduction **99.96%**, candidate-delta Spearman **0.8770506302605902**, pairwise-distance Spearman **0.9958043287496663**, variance recovery **0.9876388754313509**. This deliberately in-sample check tests implementation/capacity only and is not generalization evidence.

A 2,000-scene scale control tests whether the 500-scene result is merely data-starved:

| Scenes | Selected predictor | Validation candidate-delta Spearman | Pairwise-distance Spearman | Variance recovery | Pairwise planning delta [95% CI] | Regret delta [95% CI] | Fidelity gate |
|---:|---|---:|---:|---:|---|---|---|
| 500 | mlp_delta | 0.12723538190737874 | 0.2970739017766492 | 0.02087384080803929 | -0.021838252939764824 [-0.052098810366812125, 0.0075049059707743785] | 0.0063845332091053315 [-0.0033585072339822864, 0.0217649265192449] | FAIL |
| 2000 | mlp_delta_strong | 0.1880460424557925 | 0.5190308616711734 | 0.04215598494077408 | 0.0087582483503299 [-0.005785969276786476, 0.022862691786223504] | -0.0009058548949717228 [-0.009750799826946953, 0.006139304556983566] | FAIL |

Scaling from 500 to 2,000 scenes improves candidate-delta correlation and reaches the pairwise-distance threshold; the planning point estimate also changes from negative/mixed to **+0.88%** pairwise with slightly lower regret. However both planning confidence intervals cross zero and candidate-variance recovery remains only **4.22%**, far below the 50% gate. This is a positive data-scaling trend, not yet a robust gain claim.

### 7.3 Fair downstream planning comparison

The direct planner and the consequence predictor receive exactly the same online inputs: candidate trajectory, planning-instant structured actors, constant-velocity interaction features, and exact map/route geometry. The predicted model receives no logged future at validation time.

Logged-future consequences are used only as supervised training/evaluation labels. Outer-train planner features are OOF predictions, and outer-validation planner features are predictions from a model fit only on outer-train logs; no oracle future consequence is fed to the predicted planner.

The detailed table below is the 500-scene run with a successful fixed-seed repeat. The 2,000-scene scale result is summarized in the preceding table and retained as a separate artifact.

| Probe | Pairwise ranking | NDCG | Per-scene Spearman | Top-1 | Regret | Score RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Fair direct, same online inputs | 0.5869930405567555 | 0.993309166046493 | 0.22868413295480644 | 0.5520833333333334 | 0.018180223802725475 | 0.18915950990984648 |
| Predicted dynamic consequence (mlp_delta) | 0.5651547876169907 | 0.9903912254327706 | 0.1656339604095518 | 0.5833333333333334 | 0.024564757011830807 | 0.18849750712802602 |
| Predicted dynamic consequence (ExtraTrees diagnostic) | 0.5929925605951524 | 0.992806542537164 | 0.24115526653236755 | 0.5833333333333334 | 0.018231218059857685 | 0.18495500778080642 |
| Oracle dynamic ceiling, same downstream learner | 0.6033117350611951 | 0.9913270502456096 | 0.250433842041415 | 0.4895833333333333 | 0.03372189433624347 | 0.18479553966416032 |

Formal primary predicted-vs-direct judgement: **INCONCLUSIVE**. Pairwise delta is **-0.021838252939764824**, with scene-bootstrap 95% CI **[-0.052098810366812125, 0.0075049059707743785]**; top-1 delta is **0.03125**, and regret delta is **0.0063845332091053315**.

Current predicted consequences do not establish a robust planning gain. In the 500-scene run, the ExtraTrees diagnostic has small point improvements on pairwise/top-1/RMSE, but it was not selected by held-out validation planning performance and does not pass candidate-fidelity gates. At 2,000 scenes the train-OOF-selected MLP has favorable pairwise/regret point estimates, but uncertainty crosses zero and variance recovery still fails. Most importantly, because prediction fidelity failed, these downstream results are **inconclusive about the method**, not evidence that consequence modeling is ineffective.

This minimal predictor uses planning-instant GT actor annotations, so it is a structured-perception upper bound rather than an end-to-end camera experiment. A subsequent model must first pass candidate-delta, pairwise-distance, and variance-recovery gates on unseen logs; only then should downstream ranking be used for a method-level decision.

## 8. GT future visual anchor

- Front-image file coverage: **100.00%** across 12 scenes / 48 horizon rows.
- Same-timestamp image + pose + annotations + traffic light + tracks + structured target: **100.00%**.
- Figures: **12** under `figures/visual_anchor/`.

Supported: `I_GT(t+h) <-> C_GT,h`. Unsupported: a real `I_candidate_i(t+h)` for non-GT candidates. Thus visual alignment is GT-only; structured candidate targets provide the multi-candidate supervision.

## 9. Reactive and synthetic extensions

- Reactive cache: **128 scenes / 1894 candidates**.
- Captured actor-track rerun: **32 / 32 scenes**; candidate-dependent response nonzero rate **0.010670731707317074**.
- IDM simulates vehicles only; pedestrians and static objects remain logged replay.
- Synthetic: **5462 NAVHARD two-stage files**, metadata sample **512**, legal-train eligible **False**.

Synthetic follow-up scenes are not treated as same-current-state action alternatives. They can at most be neighborhood-state augmentation/weak multi-future supervision when a legal training split is explicitly deployed.

## 10. Final five judgements

- **F1 candidate-relative structured consequence: PASS**
- **F2 GT visual anchor: PASS**
- **F3 soft contrastive supervision: PASS**
- **F4 inverse verifier supervision: INCONCLUSIVE**
- **F5 non-GT future image supervision: FAIL**

F4 remains INCONCLUSIVE because the predicted-consequence fidelity gate failed. It is not downgraded to FAIL on the basis of downstream ranking while the predictor has not recovered candidate-specific variation.

F5 fails because no non-GT observed future pixels exist, not because candidate-relative structured consequences fail.

## 11. Recommended next method version

**Plan B：结构化候选相对世界模型**

Primary blocker: 预测器优化与小样本容量检查均通过，扩大到 2,000 场景后 fidelity/规划点估计有所改善，但 held-out log 上候选方差仍严重塌缩且规划置信区间跨零；因此当前证据不足，不能归因于方法无效。

Plan B is a gated next experiment, not a positive utility claim. First improve prediction fidelity with richer current BEV/actor-token features, direct masked actor-future supervision, candidate-centered losses, and more train logs. Do not advance to a planning conclusion until the unseen-log fidelity gate passes; if it passes and planning still shows no gain with the fair same-input baseline, only then is a negative method conclusion justified.

Do not call the non-reactive targets true counterfactual futures. Preserve provenance per channel and train GT visual anchoring separately from candidate-relative structural prediction.

## 12. Minimal next-stage interface

```text
inputs:
  current_scene_representation: current cameras + current structured actors/map/route
  candidate_trajectory: [K, 8, 3] in current rear-axle frame
targets:
  candidate_relative_target_schema: C_environment_only [K, H, D] + actor [K, H, N, F] + masks
  gt_visual_anchor: frozen/learned embedding of logged I_GT(t+h), GT candidate only
  soft_contrastive_target: q_GT_prefix [H,K] and Q_consequence [H,K,K]
  utility_target: offline PDM factors/ranking, never an input feature
  inverse_verifier_target: coarse action relation from interaction-only consequence
outputs:
  predicted candidate-relative structured future + calibrated utility/risk + verifier logits
```

The first training prototype should predict masked environment relations at 0.5/1/2/4 s, use the GT image embedding only on the GT row, and use q/Q for prefix-aware contrast. Reactive-response heads remain optional and vehicle-only until broader legal reactive data is available.
