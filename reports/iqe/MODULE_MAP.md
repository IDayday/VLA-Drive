# Module map

| 功能 | 实现与真实调用入口 |
|---|---|
| 框架锁定/输入契约 | `preflight.audit`, `query_base.bind_source/build_query_framework`, CLI preflight |
| 新 Query 基础训练 | `training.base_trainer.train_base`, CLI train-base，原 ActionVideoForesightDataset/forward |
| 真实 S0 外层适配 | `s0_adapter.S0Adapter` 的 load/encode/predict/clone/IL/official conversion |
| 独立专家/冻结/推理 | `expert.IndependentExpert`, `model.IQEModel.set_trainable_stage/forward_candidates/predict` |
| Query-only/Adapter/residual | `expert.py`, `model.append_expert`，对应 E4/E5/E6 配置 |
| 完整策略副本 E1 | `training.policy_copy`, CLI train-policy-copy/evaluate-policy-copy |
| typed schema/atomic storage | `contracts.py`, `io.py` |
| 数据划分/接入/目标冻结 | `data.splits/manifests/sources`，build-splits/build-round |
| 缺口审计/目标评分/四桶 | `data.mining/sampler/support_retrieval`, `Pipeline.build_round/target_score` |
| 完整特征缓存/候选银行 | `data.feature_cache/candidate_bank`, cache-features/export-candidates |
| 官方上下文重建 | `scoring.prepare/prepare_worker`, prepare-metric-contexts |
| 单轨迹官方评分与不变量 | `scoring.reference/worker/audit`, score-candidates/verify-reference |
| 专家/Scorer/Router trainer | `training.trainer`, `Pipeline.train_expert/train_selector`，全局分母在 losses.py |
| exact resume | `training.checkpoint`, `data.sampler.ConsumedSampler`，三个 trainer --resume |
| Oracle/冻结/选轨评估 | `evaluation.oracle/retention/selection/diversity`, evaluate-oracle/audit-frozen/evaluate |
| Scorer/Router | `scorer.TrajectoryScorer`, `router.SceneRouter`, `model.forward_prerouted` |
| 校准/回退 | `evaluation.calibration`, `selector.SelectionRule/RouterRule`，calibrate |
| 多轮/selector-only | `rounds.run_round`，CLI run-round [--selector-only] |
| 部署/回滚/Agent | `export.py`, `agent.IQEAgent`, export-bundle/load-bundle/rollback |
| final_test 独立入口 | `final_test.evaluate_final`, CLI final-test |
| 成本、时延、效率 | `evaluation.latency`, `training.efficiency`, benchmark / qualified_base_training.sh |
| 32样本学得动诊断 | `overfit.overfit_32`，overfit-32 |

原 NAVSIM/S0 入口没有改写。实际训练与离线官方评分在阶段边界解耦；GPU batch loss 不创建仿真/Ray。
