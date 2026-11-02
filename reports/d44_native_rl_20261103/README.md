# D44 native high-performance RL

本轮使用真正的 D IL44k，从同一父权重、新 Adam 状态独立启动各算法。D0/D2/D5已从同一父模型正式训练，其他路线已登记并等待剩余资格与容量，**尚无本轮新 RL 完整 Navtest 分数，不是训练完成报告**。

父模型完整 Navtest：**89.0601875684 PDMS**，12,146 scenes、44 physical logs、invalid=0；B1、clean-x ODE、K10、shift5、seed2101、canonical NAVSIM v1.1。

| 路线 | 当前真实证据 | 长训练状态 |
|---|---|---|
| D0 native SafetyEP PPO | 八卡10次真实更新；从 inner1 恢复到10，与连续训练的 LoRA/Adam/scheduler/8rank RNG及loss/梯度一致 | B64/G16/M2正式运行，4epoch目标 |
| D2 verified native improvement | 八卡G16十次更新，存在合格教师与非零辅助梯度；最新版本采用部署相同的TRAIN场景B1固定root与clean-x逐步算术 | B64/G16/M2正式运行，6epoch目标 |
| D5 expanded scope | 八卡G16十次更新；text/FiLM/action-I/O/position实际有梯度；reference不共享可训练张量 | B64/G16/M2正式运行，6epoch目标 |
| D1 two-stage | 共同 native PPO 基础已实现；阶段II训练GT endpoint质量 | 待有效标签/阶段资格 |
| D3-FPO / D3-AWM | 各两卡G16十次真实更新；冻结参数hash不变 | 尚未八卡长训；AWM是正权重稳定化适配 |
| D4 post feedback | 八卡十次真实更新，恢复原IL readout，post任务和text LoRA均有梯度，冻结参数hash不变 | 未据此宣称事件理解或部署收益 |
| D6 performance max | 阶段/历史教师已实现，八卡短程已走完I–V，正在验收inner恢复 | 尚未正式长训；入口要求同源码真实恢复证明 |

共享底座复用现有 native D actor、官方评分、LoRA wrapper、原子checkpoint与精确恢复。没有新独立DiT，没有套用PostCoT动作专家，没有训练测试标签。

## 证据与文档

- [父模型](IL44_PARENT_AUDIT.md)、[架构与来源审计](ARCHITECTURE_AUDIT.md)、[代码身份逐文件hash](source_audit.json)。
- [算法边界](ALGORITHM_DESIGNS.md)、[训练配置与命令](TRAINING_CONFIGS.md)、[资源与成本](RESOURCE_BUDGET.md)。
- [真实恢复对照](D0_REAL_MID_INNER_EXACT_RESUME.json)、[完整模型执行profile](D0_PROFILE_EXECUTION.json)。
- 运行资产根：`/mnt/project/d44-native-rl-20261103`；资格权重与正式初始化严格分开，不上传权重或原始数据。
- 最新状态由 `python -m tools.d44_native_rl.report --output ...` 从有效边界和invocation账本生成，不能把进程退出或资格更新记作长训完成。

当前只能证明核心链路可执行且部分等价性通过。是否突破自身IL、是否超越历史Qwen-World、能否达到92，等待真正完整B1评测。


## 20:39 UTC阶段快照

D0/D2/D5八rank共同有效更新分别为116/202/174，真实新PG候选59,392/103,424/89,088。原始序列连续，第一inner sampler/recompute联合ratio最大误差均为0；这不表示后续inner的ratio也应为1。完整记录见[进度快照](experiments/PROGRESS_20261102.json)。快照不是最终预算或新的Navtest。

同一IL与RNG下，三个scope在首次正式collection生成的64个不同scene、1024条候选及官方反馈逐值一致；D0与D2在rescue尚未启用前的独立100更新权重完全一致，见[P0初始化证据](P0_FORMAL_INITIAL_BEHAVIOR_IDENTITY.json)、[D0/D2早期对照](D0_D2_INDEPENDENT_100_IDENTICAL.json)。因此早于0.5epoch不能把D2与D0差异归因于rescue。

本轮CPU小型验收33项通过，包含两rank显式全batch梯度归约和恢复发布失败边界；真实GPU验收另列，不将CPU测试数当成算法有效性。独立官方评分对照104次全部一致，见[评分一致性](CANONICAL_SCORING_CONSISTENCY.json)。实验性flat-W3重排未通过fresh-ratio门槛，已拒绝用于正式训练，没有放宽1e-4阈值。

运行中的科学源955289f保持不变。CPU调度/统计修复与资格操作分别冻结；新的报告入口按完整run身份核验评测覆盖，不能因其他算法多测了几个点就把未评测算法记为完成。最终胜者、RL净增益和92目标目前均未判定。

正式冻结前的小改动逐项保留于[资格→正式源码审计](QUALIFICATION_TO_FORMAL_RECEIPT.json)和[实际diff](qualification_to_formal.patch)：费用记录、padding KL分母、拒绝flat-W3、D1 endpoint confidence，以及D2追加教师条件补齐Comfort。核心actor/Gaussian/LoRA/codec文件hash相同；不能把历史GPU资格说成所有正式文件逐字相同。正式权重从第一步就使用955289f，未从资格权重恢复。

后续资源修复已真实启动D1/FPO/AWM的两卡B32/G16独立资格，实际更新和正式长训状态分别见[最新资格/正式快照](experiments/QUALIFICATION_AND_FORMAL_20261102.json)。D5非零权重的[完整B1裁剪与compact导出对照](D5_DEPLOYMENT_PARITY.json)通过，三个完整K10前向逐值一致、无新评分；这不是新的Navtest或算法提升证据。

后续运维与分析验收扩展至40个CPU测试通过；对应实际GPU资格、部署对照、阶段II真实标签桥接分开记录。没有将一次短资格更新或loss下降当成Navtest增益。
