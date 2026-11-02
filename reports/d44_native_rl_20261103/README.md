# D44 native high-performance RL

本轮使用真正的 D IL44k，从同一父权重、新 Adam 状态独立启动各算法。当前处于资格验收和正式调度建设阶段，**尚无本轮新 RL 完整 Navtest 分数，不是训练完成报告**。

父模型完整 Navtest：**89.0601875684 PDMS**，12,146 scenes、44 physical logs、invalid=0；B1、clean-x ODE、K10、shift5、seed2101、canonical NAVSIM v1.1。

| 路线 | 当前真实证据 | 长训练状态 |
|---|---|---|
| D0 native SafetyEP PPO | 八卡10次真实更新；从 inner1 恢复到10，与连续训练的 LoRA/Adam/scheduler/8rank RNG及loss/梯度一致 | B64/G16/M2真实资格已通过，正式登记中 |
| D2 verified native improvement | 八卡G16十次更新，存在合格教师与非零辅助梯度；最新版本采用部署相同的TRAIN场景B1固定root与clean-x逐步算术 | 最新B64/G16资格已通过，正式登记中 |
| D5 expanded scope | 八卡G16十次更新；text/FiLM/action-I/O/position实际有梯度；reference不共享可训练张量 | B64/G16资格已通过，正式登记中 |
| D1 two-stage | 共同 native PPO 基础已实现；阶段II训练GT endpoint质量 | 待有效标签/阶段资格 |
| D3-FPO / D3-AWM | 各两卡G16十次真实更新；冻结参数hash不变 | 尚未八卡长训；AWM是正权重稳定化适配 |
| D4 post feedback | 八卡十次真实更新，恢复原IL readout，post任务和text LoRA均有梯度，冻结参数hash不变 | 未据此宣称事件理解或部署收益 |
| D6 performance max | 复杂阶段/历史教师生命周期尚未完成，入口明确拒绝调度 | BLOCKED_IMPLEMENTATION，不用简单D0冒充 |

共享底座复用现有 native D actor、官方评分、LoRA wrapper、原子checkpoint与精确恢复。没有新独立DiT，没有套用PostCoT动作专家，没有训练测试标签。

## 证据与文档

- [父模型](IL44_PARENT_AUDIT.md)、[架构与来源审计](ARCHITECTURE_AUDIT.md)、[代码身份逐文件hash](source_audit.json)。
- [算法边界](ALGORITHM_DESIGNS.md)、[训练配置与命令](TRAINING_CONFIGS.md)、[资源与成本](RESOURCE_BUDGET.md)。
- [真实恢复对照](D0_REAL_MID_INNER_EXACT_RESUME.json)、[完整模型执行profile](D0_PROFILE_EXECUTION.json)。
- 运行资产根：`/mnt/project/d44-native-rl-20261103`；资格权重与正式初始化严格分开，不上传权重或原始数据。
- 最新状态由 `python -m tools.d44_native_rl.report --output ...` 从有效边界和invocation账本生成，不能把进程退出或资格更新记作长训完成。

当前只能证明核心链路可执行且部分等价性通过。是否突破自身IL、是否超越历史Qwen-World、能否达到92，等待真正完整B1评测。
