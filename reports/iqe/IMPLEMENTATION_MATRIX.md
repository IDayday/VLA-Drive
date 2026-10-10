# IQE requirement / implementation / verification matrix

解释优先级：用户后续明确“原轨迹 S0 框架、DiT/flow matching 改为 Query、使用框架而不使用驾驶权重”。因此工程先训练新的 Query 基础策略，再冻结、增量克隆。没有假称原 DiT 与新 Query 数值等价。正式 GPU 训练已被用户后续授权，但必须先通过资源与吞吐资格检查。

`COMPLETE` 指有可执行实现和列出的行为证据；真实执行规模单列。`PARTIAL`/`BLOCKED` 不会因单元测试通过而升级。最终硬件/双轮状态见 REAL_SMOKE_REPORT 和 LATENCY_REPORT。下面路径相对本工作区。

| 需求 | 实际文件/函数与入口 | 测试及实际证据 | 状态 |
|---|---|---|---|
| 0 / 24 工作区保护、提交边界 | WORKSPACE_SNAPSHOT.json、BASE_DIRTY.patch；隔离worktree | 原始2个tracked改动及104个untracked快照保留；无push/PR/merge | COMPLETE |
| 1 / 2 真实框架与动作契约 | preflight.audit、query_base.build_query_framework、S0Adapter.load_and_validate_s0；preflight/train-base/freeze-base | 真实三图Qwen前向、原辅助目标1步训练、hash/错误架构拒绝；S0_CONTRACT | COMPLETE框架；GPU八卡profile通过；正式基础训练状态见报告 |
| 3 独立专家/冻结/模式/旧函数 | expert.IndependentExpert、IQEModel.set_trainable_stage/train/audit_optimizer；train-expert/audit-frozen | test_models；REAL_FRAMEWORK_PROBE；真实逐轨迹frozen_audit | COMPLETE；GPU真实图像/梯度/冻结探针PASS |
| 4 数据角色和来源组 | splits.split_training_logs/validate_isolation、sources.import_current；build-splits | test_splits_log_parent_and_observation_leakage；128真实清单/65组审计 | COMPLETE；全量接入未自动执行 |
| 5 官方独立单轨迹评分 | scoring.ReferenceBackend、worker、audit.verify_reference；prepare-metric-contexts/verify-reference/score-candidates | 96真实上下文重建；原正式入口完全对齐；23次追加/删减/排序/复制/chunk检查；选中轨迹直接重评 | COMPLETE：锁定v1.1非reactive协议 |
| 6 全旧池缺口与GT审计 | mining.diagnose_scene、Pipeline.target_score/build_round | test_gap_separation、test_mining_uses_actual_served_winners；第一轮31真实合格coverage gaps | COMPLETE；无阈值下调 |
| 7 H/A/N/G及检索采样 | sampler.mixture/sampling_plan、support_retrieval.retrieve；build-round | 缺桶/显式anchor0/组配额/曝光/固定targets单测；真实resampling_only计划 | COMPLETE；短smoke整数配额偏差单列 |
| 8 新数据/合成/替代目标 | sources.validate_incoming、manifests.freeze_targets；build-round | 严格schema、unverified隔离、来源父组/图像/context证据绑定和冲突target拒绝 | COMPLETE接入代码；真实E3/外部目标数据BLOCKED |
| 9 IL/缓存/梯度/选checkpoint | expert_il_terms、FeatureCache、Pipeline.expert_validation、training.trainer；cache-features/train-expert/overfit-32 | 真实缓存44场景；真实expert更新、stage_val官方Oracle选checkpoint；真实32场景/32步overfit诊断通过 | COMPLETE主路径；overfit_32仅作学习能力诊断 |
| 10 Oracle/状态/gate | evaluation.oracle/diversity、registry；evaluate-oracle | 固定旧池逐场景Oracle不减、paired group bootstrap、真实checkpoint报告 | COMPLETE；科学接纳UNTESTED |
| 11 / 12 轨迹条件Scorer及损失 | scorer.TrajectoryScorer、losses.scorer_terms/scorer_denominators | K1/可变K/置换/重复/追加/NaN/mask/软分项/全平局/梯度方向/DDP归约；真实Scorer backward | COMPLETE |
| 13 全候选银行/重放/缓存 | CandidateBank、Pipeline.export_candidates/score_candidates/train_selector | stale ID/hash拒绝；第二轮全旧新候选重放与warm start；单候选labels复用 | COMPLETE；全量银行未自动运行 |
| 14 校准/双基准保护 | SelectionRule、calibration.calibrate、selection_report、export.release_gate | Base/tie/无效/margin/safety测试；真实selector_cal选回退Base；vsS0/vs上一部署单列 | COMPLETE；没有正式发布收益证据 |
| 15 scene-only Router | SceneRouter、router_terms、RouterRule、IQEModel.forward_prerouted；train-router/evaluate/benchmark | refresh soft targets、输出层扩展、混合分组单测；REAL_ROUTER实际银行训练/校准/评估、真实图像hooks只跑被选Decoder | COMPLETE CPU真实路径；Router A800真实预路由时延已测 |
| 16 状态机/恢复/增量 | rounds.run_round、registry；run-round/--selector-only/--resume | 完整合成双轮状态机；真实双轮执行记录；部分阶段和校验和复用 | COMPLETE主入口；真实最终状态另表 |
| 17 严格记录/原子IO/exact resume | contracts、io、training.checkpoint、ConsumedSampler | world/data/config变化拒绝；双Gloo进程exact；真实Query头中断恢复逐tensor一致 | COMPLETE CPU/Gloo；NCCL实际两卡测试PASS |
| 18 真实接线/Agent/DDP | S0Adapter、query_base、agent.IQEAgent、Pipeline、trainer | 真实NAVSIM AgentInput从原日志构建，像素/prompt/ego一致；rank缓存失败共同退出；真实bundle Agent输出检查 | COMPLETE CPU主路径；GPU真实前向/八卡基础训练profile PASS |
| 19 严格配置 | config.load_config/resolve、configs/iqe | 所有消融配置校验、未知字段/错误类型/未解析/禁止操作拒绝；resolved_config落盘 | COMPLETE |
| 20 CLI/脚本/预算/final_test隔离 | cli、scripts/iqe、final_test.evaluate_final | 每命令help、JSON错误退出、零预算拒绝、角色隔离；真实CLI执行记录；实际checkpoint恢复路径 | COMPLETE；GPU/full命令未伪称已执行 |
| 21 自动化测试 | tests/iqe/*.py | tests-final.log；合成行为测试和真实资源测试分别记录 | PASS CPU；CPU套件85通过/1显式skip；GPU单独执行NCCL测试1通过 |
| 22 真双轮/时延/训练效率 | real_cpu_two_round_probe.py、real_resume_probe.py、latency.benchmark、efficiency | 真图像/真GT/真官方cache CPU执行；三台vlawm初检阻塞；后续vla-zt2 GPU/NCCL/真实前向/八卡profile通过；无未知进程被杀 | PARTIAL：GPU/NCCL/时延通过；正式训练执行状态另表 |
| 23 E0–E12及统计 | configs/iqe/ablations、policy_copy、expert variants、evaluate-baselines/diversity | E1完整策略副本在线IL；E4/E5/E6初始化测试；E7/E8/E10/E11独立配置/入口 | COMPLETE E0–E11实现；E3真实新增数据、E12可运行对齐基线BLOCKED |
| 24 交付和证据 | docs/iqe、reports/iqe、task-only commits | 无权重/原图/metric cache/密钥进入commit | COMPLETE文档；主闭环完成；E12对齐基线仍BLOCKED，科学状态单列 |

## 自动化验收组 21.1–21.14

| 验收组 | 行为测试/真实证据 |
|---|---|
| 21.1 S0/兼容 | test_wrong_checkpoint_architecture_rejected_before_model_loading、test_config_strict_unknown_unresolved_wrong_types、test_shared_once_and_k1_identity；真实REAL_K1 |
| 21.2 专家复制/更新/模式 | test_new_query_decoder_heads_update_old_buffers_frozen、test_storage_alias_rejected、test_ablation_init_and_modes；真实梯度及frozen_audit |
| 21.3 特征/缓存/共享 | test_feature_cache_hash_checksum_augmentation、test_inference_tensor_normalization_backward、test_feature_shards_partition_named_scenes_without_extra_encoding；真实在线/缓存比较 |
| 21.4 坐标 | test_coordinates_periodic_roundtrip；原normalizer与NAVSIM Agent输入实际核对 |
| 21.5 数据 | test_splits_log_parent_and_observation_leakage、test_strict_manifest_and_targets、test_buckets_missing_and_cumulative_exposure；真实split audit |
| 21.6 挖掘 | test_gap_separation、test_new_round_uses_updated_coverage_not_original_failure_list、test_mining_uses_actual_served_winners_not_last_rejected_selector；实际两轮gap_diagnoses |
| 21.7 评分 | SCORE_CONTEXT_EVIDENCE.json；reference worker每候选独立实例，seed/repetition显式记录；锁定非reactive版本，未声称支持其他协议 |
| 21.8 Oracle | oracle_report运行时逐场景不减断言；test_candidate_bank_join_stale_score_reject；真实oracle_stage_val |
| 21.9 Scorer | test_scorer_independent_candidates_and_detach、test_scorer_padding_and_nan、test_ties_rank_graph_zero、test_ranking_direction_tie_filter_gap_weight、test_masks_soft_components_and_scenewise_rank；真实重放identity |
| 21.10 选择 | test_selector_tie_margin_safety_invalid_base、test_calibration_role_and_no_fake_release、test_selection_capture_na_negative_and_regression；真实calibration/evaluation |
| 21.11 Router | test_router_refresh_expansion_and_preroute、test_router_no_valid_backward；完整pipeline Router训练/校准对照 |
| 21.12 registry/bundle | test_registry_append_only_hash_and_wrong_order、test_unit_bundle_save_restore_gate_smoke_rollback_and_corruption；真实bundle reload与ACTIVE未创建 |
| 21.13 DDP/恢复 | test_real_two_process_gloo_denominators_gradients_and_collective_exit、test_two_process_ddp_optimizer_boundary_exact_resume、test_exact_resume_world_change_rejected；REAL_EXPERT_EXACT_RESUME；NCCL独立GPU入口已通过 |
| 21.14 双轮真实 | REAL_SMOKE_REPORT与REAL_TWO_ROUND_RESULT，真实数据/图像/GT/Qwen/官方metric；不使用合成单测冒充 |

边界：vlawm-zt2/3未知容器PID未终止；后续在允许的vla-zt2取得空闲8卡并完成真实资格验证。没有final_test清单、真实新观测、合格LoRA-DiT对照时明确BLOCKED。没有科学收益、显著抗遗忘或部署安全保证的声明。
