# 本轮新增合同的针对性验证

这些证据验证执行和数值合同，不代表方法有效，也不是全仓审计。旧 Flow-GRPO 的转移、dimension-mean surrogate、优化器分组和评分内核继续复用。已发现的部署精度差异保留为失败证据，不能写成缓存与原导出完全等价。

| 用户要求 | 已执行证据及边界 |
|---|---|
| 1. reference ODE 与原导出一致 | [适配资格](evidence/action_adapter_qualification.json)验证同 BF16 模型内接口一致；它没有覆盖原导出默认模块 dtype。后续[原始观测隔离](evidence/deployment_numerical_isolation.json)发现并保存这一差异。主表统一重新使用原 VLAAgent，每个实际采样 worker 的首场景逐值核对 `predict_action_infer_1d`；不再以旧缓存分数作部署主结果。 |
| 2. 旧 SDE、dimension-mean 不变 | [适配资格](evidence/action_adapter_qualification.json)核对旧 G16/K10/rho0.8 chain/logprob；独立 RNG 只分离初始噪声和 SDE 创新项。D2 的 joint-sum 仅作离线诊断，未替换训练定义。 |
| 3. 逐候选评分独立 | [真实 v1 评分资格](evidence/v1_reward_qualification.json)检查候选顺序、切分和其他候选；另有[实际批形状](evidence/ODE_actual_search_shapes_qualification.json)的请求及安全标签核对。 |
| 4. v1/v2 与缺失语义 | 独立 v1 backend 和 v2 one-stage 入口、协议 hash；[v2 完整实跑](evidence/P2_epoch8_DEV_seed2101_v2.json)保留有效 two-frame 计数和官方聚合。v2 的 `pdm_score` 不作 v1。 |
| 5. reward 不进入 actor 条件 | [原始观测部署守卫](evidence/P0_epoch1_uncached_deployment.json)实际阻断 teacher、特征/候选/奖励缓存，访问尝试为0；`ExtractionActor` 仅对 observation-prefix states 调用速度场，reward/advantage 进入目标与 loss。没有另行宣称做过整网 reward 扰动不变性实验。 |
| 6. 外部候选无伪概率 | `test_external_and_ode_candidates_cannot_claim_sde_likelihood`、`test_nft_rejects_search_reference_and_wrong_behavior`；NFT 只接收当前 behavior 的真实 ODE 样本。 |
| 7. 请求/模拟区分及编码往返 | [动作适配资格](evidence/action_adapter_qualification.json)的外部请求往返最大误差4.77e-7；bank 分存 normalized action、requested trajectory 和 simulated states，FM 标签取请求动作。 |
| 8. 去重后的 q_ref/u | `test_duplicate_reference_and_sample_masses_merge_without_changing_prior`；重复请求不同反馈直接报错。非法物理提案的固定 reference 回退另有测试，原提案仍计入预算并保留。 |
| 9. 非负、归一化及线性约束 | FP64 dual 与独立 Clarabel 指数锥 oracle 对照；[三组实际目标库](evidence/P_fixed_target_summary.json)最大 primal 残差≤2.53e-10，逐场景数值诊断留在私有目标库。 |
| 10. 零预算/等分/全失败/唯一可行 | `test_zero_budget_has_zero_violating_mass_and_preserves_original_prior`、常量奖励参数化测试、`test_only_one_safe_candidate_and_reward_floor`。零预算通过排除支持集处理。 |
| 11. 不伪造可行改进 | 等分与全零目标 alpha=0，回到 reference；非法/异常结果不能变 reward0。D3 区分完全修复、部分改善和未找到，不将有限未发现解释为不可能。 |
| 12. 混合保持线性约束 | `test_nonzero_cost_constraints_are_separate_and_nontrivial` 检查 w 和 q_target；实际目标库保留混合前后成本。 |
| 13. 微小 EP 不被 std 放大 | `test_tiny_absolute_gains_cannot_force_large_mixture`；所有 P 组使用固定 eta、alpha_max 和原始分单位 delta_useful，无每场景 std。 |
| 14. 抽样 FM 对照精确求和 | `test_scene_weighted_sampling_matches_exact_enumeration` 对两场景各6万次抽样与独立精确期望比较；它验证抽样估计，不是神经网络收敛证明。 |
| 15. 无负权重 MSE | 非负候选概率检查拒绝负权重；NFT 正负预测采用[官方算术测试](evidence/contracts_tests.txt)，不是负 advantage 乘普通 MSE。 |
| 16. FM 噪声独立 | RNG namespace 测试；`fm_draws` 只接收标签和独立训练 seed，不接收候选 root noise。 |
| 17. reference/上游冻结及完整动作头 | [FM/NFT 真 GPU 资格](evidence/full_head_learning_qualification.json)和[NFT 恢复资格](evidence/NFT_exact_resume_qualification.json)：359 个张量/819,503,620 参数有实际合法 FP32 optimizer 梯度，上游 hash 不变。短窗口335张量有可见 BF16 改变，24张量没有，不能写成全部都有可见改变量。 |
| 18. 固定 behavior/权重身份 | `test_resume_rejects_changed_behavior_bank_and_objective`；所有轮次写入独立 bank/targets/source identity，刷新仅发生于外层边界。取消多seed时接管实际子任务，未重抽 behavior。 |
| 19. scene/repeat/维度归约 | `test_interpolation_direction_and_repeat_scene_reduction`；三项 loss 复用同一 FM 维度均值，reference velocity 用相同场景/8次repeat归约；日志分别记录三种场景曝光。 |
| 20. optimizer/RNG/数据恢复 | [FM 连续与4→8恢复](evidence/full_head_learning_qualification.json)及[NFT 同样恢复](evidence/NFT_exact_resume_qualification.json)对模型、8份optimizer、8份RNG/数据游标及scheduler逐文件/逐值一致。正式控制器的外层接管保留已有封存候选与训练恢复点；这不等于逐一注入所有故障组合。 |
| 21. 官方 NFT 核心对齐 | 两个 beta 的小张量 loss/梯度与固定上游文件 AST 算术对照通过，旧 behavior detach；已真实运行完整动作头与重新采样的在线窗口。源码与时间/速度符号映射见主报告。 |
| 22. 部署无 teacher/cache/search | [实际文件访问守卫](evidence/P0_epoch1_uncached_deployment.json)和后续完整 DEV 原始观测推理；actor 不加载 bank，官方 scorer 在推理之后独立运行。 |
| 23. 异常/缺场景保留分母 | bank 拒绝 worker_error、缺分项、错协议；单seed读取测试拒绝异常和缺场景。真实中断/失败目录保留，只有完整匹配人口才发布评价 COMPLETE。 |
| 24. DEV/Navtest 不用于更新 | `PopulationContract` 的 log 隔离和 update/calibration 边界测试；targets/replay/保持训练输入均要求 RL_TRAIN。Navtest 入口要求预先锁定权重、人口和源码，拒绝扩展 seed。全部24个DEV窗口完成后锁定六组导出；原单候选Navtest正在实际执行。 |

测试源码：[合同](../../tests/safe_flow_improvement/test_bank_contracts.py)、[求解器](../../tests/safe_flow_improvement/test_constrained_weights.py)、[FM/NFT](../../tests/safe_flow_improvement/test_objectives.py)、[搜索/回退/安全暂停](../../tests/rl/safe_flow_improvement/)、[单seed边界](../../tests/safe_flow_improvement/test_single_seed_override.py)、[最终锁定](../../tests/safe_flow_improvement/test_final_evaluation_lock.py)。早期33项与随后37项包含重复运行，不能相加称70个独立测试。

NFT 恢复资格沿用最初短训封存的 `train_fixed_bank.py`（SHA256 `d736f73bd95ca9dc9ee8e7b7e6b0f393fdb82a00eb32fd965857dca6d5509637`）。当前正式 trainer 相比它只增加 host-local checkpoint 路径/身份记录；训练算术、optimizer、RNG 和 loss 模块保持一致。资格输出不是正式方法分数，也不是第二训练 seed。

最终聚合入口`summarize_final_confirmation`已实际验证[拒绝不完整人口](evidence/final_aggregate_incomplete_gate.json)：只要一个锁定模型没有完整v1/v2封存结果，就拒绝发布全量报告。此检查没有新增模型采样或seed。
