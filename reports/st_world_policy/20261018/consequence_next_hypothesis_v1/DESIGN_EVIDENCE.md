# 依据与真实适配边界

状态：`EVIDENCE_VERIFIED_FOR_BOUNDED_ANALYSIS`，`FORMAL_TRAINING_APPROVAL=false`。

问题是后果训练实际收到什么信号。原合同§8–10与§14要求把终局反馈、过程信息、真实执行方向、梯度和规划收益分开。已完成的32场景分析新增的是描述性统计，没有改变FPO、reward、部署或网络。

**FPO：论文全文方法与固定源码已主动复核。** [Flow Matching Policy Gradients v2](https://arxiv.org/html/2507.21053v2) §3.1–3.5、Eq.5–9、16–20、Algorithm1与Appendix A.1.1–A.1.4说明以保存的同一CFM noise/time构造old/current代理ratio，再由优势加权。动作采样数G与CFM蒙特卡洛数Nmc是不同量；论文不证明将稀疏安全reward的G4改G8必然有效，也不提供物理事件时间至Flow步的对应。Eq.20讨论负CFM梯度的期望，不应扩展成任意共享参数、裁剪和有限样本完整策略梯度无偏保证。

官方仓库SHA `418c2554f7cd22d52e14c07d951280929d73bf2f`。已读取并逐字节比较本地git对象、本地工作文件与官方固定SHA raw内容：

- [gridworld/models/diffusion_policy.py](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/diffusion_policy.py) `sample_action_with_info` / `compute_cfm_loss`：真实rollout动作detach，另采CFM噪声/时间，坐标取mean。
- [gridworld/models/fpo.py](https://github.com/akanazawa/fpo/blob/418c2554f7cd22d52e14c07d951280929d73bf2f/gridworld/models/fpo.py) `FPO.learn`：MC逐项差值先clamp±3，再mean、clamp、exp，再clip/min优势目标；它的GAE/标准化不等于本项目scene内center-only。

可复用：同一真实轨迹与保存样本的奖励加权结构。新适配：完整驾驶轨迹contextual bandit、scene内center-only、原Beta/repeat8 BC锚点、实际DiT MC50/eval-with-grad合同。固定3584的优势来源统计只依赖这些已存在语义；不把它标成原论文新增的过程算法。

**DFQL：方法与附录边界复核，原早期“待读推导”卡已由后续证据补全。** [Direct Flow Q-Learning官方全文](https://proceedings.mlr.press/v306/cao26f.html) §3.1–3.3、Eq.10–21、Appendix A.1 Eq.22–27与A.2 Eq.28–39：在同一参数snapshot处，stop-gradient移位目标与局部内积目标的梯度相同；终端梯度按 `1/(K-k)` 注入各Flow速度。附录讨论函数空间局部速度扰动的渐近方向，不保证K10共享DiT参数更新与BPTT同向。Flow时间权重也不是驾驶物理事件时间权重。

官方SHA `8b6b15803cc1e5605ad1524553a2c95e103b0bad` 的 [agents/dfql.py::DFQLAgent.actor_loss](https://github.com/2012060/Direct-Flow-Q-Learning/blob/8b6b15803cc1e5605ad1524553a2c95e103b0bad/agents/dfql.py) 已本地git对象/文件/固定SHA官方raw三方逐字节复核。原实现逐步detach状态、robot末端clip、dense BC、Q/Bellman流程及实际mean归约均不能直接移植。现有独立CPU参考报告核验了K10/D32下源码Q项与论文注入式的尺度因子1/160，并有零状态Jacobian但共享参数方向相反的反例。此证据支持数学参考，不证明驾驶代理或训练有效。

采用的本地证据卡与边界：

- `reports/st_world_policy/20261016/evidence/DESIGN_EVIDENCE_CARDS.json` 的 `FPO_GRIDWORLD_REFERENCE`；其CPU参考已完成，当前没有重复调用模型。
- `reports/st_world_policy/20261018/consequence_proxy_evidence_v1/{REPORT.md,SOURCE_MANIFEST.json}`，以及 `consequence_proxy_reference_correctness_v1/REVIEW.json`（`PASS_REFERENCE`，navigation_adapter_validated=false）。
- `road_direction_actual_v3`：32原轨迹DAC均1、道路修复分母0；仅7定义方向且代理负向7/7下降，PDMS小幅下降来自EP。这没有提供安全失败修复方向，也没有验证DFQL参数更新。

本项目实际调用链（本报告源基线336e41cd）：

| 文件 / 符号 | 核实事实 | 新分析使用 / 不越过的边界 |
|---|---|---|
| `flow_posttraining/fpo_objective.py::group_centered_advantages` | float32 scene内中心化，无标准差归一化 | 复现真实非零判定；常数组保留 |
| 同文件 `fpo_ratio` / `fpo_surrogate` | 坐标mean、MC50、old/current coupling、动作全局分母 | 奖励方差不等于梯度范数；不改目标 |
| `flow_posttraining/rollout_contract.py::RolloutBatch.validate` / `sample_rollout_group` | 原生合同固定G4、10步、float32轨迹残差 | G8不是无语义成本的配置切换 |
| `tools/st_world_policy/train_flow_policy.py::feedback_from_records` | C1 score、C2 -any(D)、C3 NC-1、C4 -any(T) | 所有事件时间在现有policy项中未被使用 |
| 同文件 `OfficialFeedbackLedger.score` / `save_update_feedback` | 完整候选记录与四rank原reward身份均持久化 | 可以只读分析，不需要重新评分 |
| `st_world/consequence.py::observe_canonical` / `road_summary` / `collision_summary` / `ttc_summary` | 41原生时点、右闭8段、0.5碰撞严重性、TTC state/offset分离 | 先检归约与空事件分母，再判断信息支持 |
| 同文件 `ConsequenceEngine.score` / `_reference_evidence` | 每次[固定PDM参考,单候选]、完整原始进度、input/output身份分离 | 不重算候选池EP，不读metriccache |
| `navsim_v1.1/.../scoring/pdm_scorer.py::PDMScorer._aggregate_scores` | multiplier与5/5/2加权项、原EP分支 | 仅重建保存分数，未调用scorer |
| `tools/st_world_policy/audit_flow_gradient_balance.py::derived_statistics` | BC与policy范数/内积的一阶目标诊断 | 同batch结果待Lead；不冒充精确ZeRO2/Adam更新 |

新适配的明确反例/验收：同终局异时、空事件、先0.5后0、TTC预测offset、分箱边界、float32量化tie、协方差负份额/大于1、物理时间非一一对应均有合成数值控制。真实账本全量保留原32/128分母、hash和原排序。当前执行通过只表示该只读分析可运行，独立审阅和训练审批仍未签署。
