# 版本、科学源码和代码入口

各实验保留独立源码与运行身份。本索引代码基于已推送1724e45，新增的是文档和有限评测恢复脚本。所有旧训练源码和已注册run没有热修改、reset、stash、force push或合并。

**科学源码SHA**表示实际训练/验收锁定计算；**分支HEAD**可能只是更新报告/调度的后续提交，不能混为一谈。优化标签专用训练代码在独立分支，不宣称当前索引分支一个CLI包含所有训练配置。

| 版本ID | 状态 | 科学源码 | 模块与分支 |
|---|---|---|---|
| DDP_FLOW_GRPO | legacy_stopped | `25f9148eb75ddf171313720e758a280b82c24f7f` | [代码](https://github.com/IDayday/VLA-Drive/tree/25f9148eb75ddf171313720e758a280b82c24f7f/starVLA/rl/flow_grpo)；`fix/ddp-action-transition-batch` |
| DDP_CONSTRAINED_NOISE | legacy_stopped | `90782b6b5a10a77541c22b29f90b83d7309d7081` | [代码](https://github.com/IDayday/VLA-Drive/tree/90782b6b5a10a77541c22b29f90b83d7309d7081/starVLA/rl/constrained_flow_noise)；`feature/ddp-learned-noise-constrained-grpo-20261014` |
| DDP_SIMWAM_RECIPE | legacy_stopped | `12b8e0d75691d7bed04ab02072020e130f711a3b` | [代码](https://github.com/IDayday/VLA-Drive/tree/12b8e0d75691d7bed04ab02072020e130f711a3b/starVLA/rl/simwam_recipe)；`feature/ddp-simwam-recipe-transfer-20261015` |
| DDP_SIMWAM_FM | legacy_stopped | `b7162de9ad33440e80c2431525daf80daa9ba391` | [代码](https://github.com/IDayday/VLA-Drive/tree/b7162de9ad33440e80c2431525daf80daa9ba391/starVLA/rl/fm_transfer)；`feature/ddp-simwam-fm-transfer-20261016` |
| DDP_SOURCE_PROCESS | legacy_stopped | `901e3b41b3123c6f45280cbc800febcdf5e4b226` | [代码](https://github.com/IDayday/VLA-Drive/tree/901e3b41b3123c6f45280cbc800febcdf5e4b226/reports/ddp_simwam_source_process_20261018)；`feature/ddp-simwam-source-process-20261018` |
| NATIVE_QWEN_FLOW | stopped_by_user | `7ff501ba86e9a6e8fa6965c3644495583129f9d0` | [代码](https://github.com/IDayday/VLA-Drive/tree/7ff501ba86e9a6e8fa6965c3644495583129f9d0/starVLA/model/modules/qwen_sourceflow)；`feature/qwen-sourceflow-native-vla-20261018` |
| TEMPORAL_DINO | paused_by_user | `69a9958746dc9addb70776f5d8fcafe6c3915e18` | [代码](https://github.com/IDayday/VLA-Drive/blob/69a9958746dc9addb70776f5d8fcafe6c3915e18/scripts/analysis/qwen_sourceflow/train_temporal_il.py)；`source-lock/qwen-flow-temporal-dino-formal-v4-20261029` |
| NATIVE_SAFE_GRPO | legacy_training_complete | `ee200bc11f03b026ee54c02e1997eb7db8a57861` | [代码](https://github.com/IDayday/VLA-Drive/blob/ee200bc11f03b026ee54c02e1997eb7db8a57861/starVLA/rl/qwen_sourceflow/safe_runtime.py)；`source-lock/qwen-flow-safe-grpo-formal-v3-20261029` |
| WORLD_IL_RAW | training_100k | `d54617b3ff02d714c067ce809a50f50cd19bceb6` | [代码](https://github.com/IDayday/VLA-Drive/tree/d54617b3ff02d714c067ce809a50f50cd19bceb6/starVLA/model/modules/qwen_world_postcot)；`由不可变snapshot绑定到Git源码；已在当前已推历史中` |
| WORLD_RL_ATTENTION | training_complete_50k | `9a57b54d46b6f84b2000b0d4d43b5adfbd2995d2` | [代码](https://github.com/IDayday/VLA-Drive/blob/9a57b54d46b6f84b2000b0d4d43b5adfbd2995d2/starVLA/rl/qwen_world_postcot/rl_actor.py)；`feature/qwen-world-rl-il40k-20261027` |
| WORLD_RL_MEMORY | training_complete_50k | `367bffd27aff5785fce6352481b89e2a433a9ee1` | [代码](https://github.com/IDayday/VLA-Drive/blob/367bffd27aff5785fce6352481b89e2a433a9ee1/starVLA/rl/qwen_world_postcot/scope_actor.py)；`feature/qwen-world-rl-memory-time-20261027` |
| WORLD_RL_TIME | training_complete_50k | `367bffd27aff5785fce6352481b89e2a433a9ee1` | [代码](https://github.com/IDayday/VLA-Drive/blob/367bffd27aff5785fce6352481b89e2a433a9ee1/starVLA/rl/qwen_world_postcot/scope_contracts.py)；`feature/qwen-world-rl-memory-time-20261027` |
| WORLD_LOSS_PHYSICAL | training_complete_2k | `b3f7fc7cc54d6ecd588a531c642e06bca72f656e` | [代码](https://github.com/IDayday/VLA-Drive/blob/b3f7fc7cc54d6ecd588a531c642e06bca72f656e/starVLA/rl/qwen_world_postcot/coordinate_objective.py)；`experiment/qwen-world-coordinate-loss-20261028` |
| WORLD_LOSS_FAMILY | training_and_evaluation_complete_2k | `3e344dff880a7cc3fcfc3638102e604749648aed` | [代码](https://github.com/IDayday/VLA-Drive/blob/3e344dff880a7cc3fcfc3638102e604749648aed/starVLA/rl/qwen_world_postcot/loss_designs.py)；`experiment/qwen-world-il-loss-designs-20261028` |
| WORLD_IL_V12 | training_100k | `8315c9a25c872c690e739132e900deaf9f8d3f65` | [代码](https://github.com/IDayday/VLA-Drive/blob/8315c9a25c872c690e739132e900deaf9f8d3f65/tools/qwen_world_postcot/optimized_il_owner.py)；`experiment/qwen-world-optimized-labels-il-20261028` |
| WORLD_LABEL_MATCH | evaluation_complete | `1724e456d3ea4cffa57ea178d5f226d2eb845bb4` | [代码](https://github.com/IDayday/VLA-Drive/blob/1724e456d3ea4cffa57ea178d5f226d2eb845bb4/tools/qwen_world_postcot/finish_label_source_match.py)；`eval/qwen-world-original-vs-optimized-labels-20261029` |

## 设计边界与历史结果

- **DDP_FLOW_GRPO**：旧DDP完整动作头GRPO，历史PDMS+0.1722/EPDMS+0.2126；安全恢复并不全面。
- **DDP_CONSTRAINED_NOISE**：固定C_SDE与learnednoise两线，旧DDP研究；未混入当前World。
- **DDP_SIMWAM_RECIPE**：源配方R_SOURCE/R_CANONICAL，旧DDP8x4与LoRA适配。
- **DDP_SIMWAM_FM**：动作FM迁移B路线；不是当前20层clean-x World。
- **DDP_SOURCE_PROCESS**：公开困难人口和源过程对照，历史保留。
- **NATIVE_QWEN_FLOW**：此前Qwen+30层源ActionDiT、velocity FM；IL98k88.004393历史参考。
- **TEMPORAL_DINO**：历史两帧+当前三视图与DINO；独立旧模型，不能当World预测memory。
- **NATIVE_SAFE_GRPO**：旧Qwen-Flow的WLA_ADV/SCADV3；当前World三组均不含此dual目标。
- **WORLD_IL_RAW**：当前Qwen+160scene/32predictive memory+20层clean-x actor；原GT、完整IL。
- **WORLD_RL_ATTENTION**：只训练160个动作注意力投影的LoRA，320张量/5242880参数，r16/alpha32。
- **WORLD_RL_MEMORY**：动作LoRA+完整scene/predictive memory，458张量/52707840参数；Qwen冻结。
- **WORLD_RL_TIME**：Time MLP/20block与末端AdaLN LoRA，r32/alpha32，46张量/4816896参数。
- **WORLD_LOSS_PHYSICAL**：同63k模型/Adam续训2k；物理容差固定坐标权重。
- **WORLD_LOSS_FAMILY**：TIME_RESIDUAL/DELTA/HUBER三个独立63k+2k臂。Huber原协议补测已完成全量评分。
- **WORLD_IL_V12**：通用初始化完整100k；原L1/四aux不变，只替换规划轨迹标签。执行恢复源码905c5ef另列。
- **WORLD_LABEL_MATCH**：新增原GT10k/15k全Navtest，复用V12同更新点；不拿70k对15k。

## 现在主要使用的代码

| 组件 | 代码 |
|---|---|
| Qwen／两类memory／20层actor／四readout | [model目录](../../starVLA/model/modules/qwen_world_postcot) |
| IL/共同optimizer与scheduler | [train_il.py](../../tools/qwen_world_postcot/train_il.py)、[il.yaml](../../configs/qwen_world_postcot/il.yaml) |
| attention RL | [rl_actor.py](../../starVLA/rl/qwen_world_postcot/rl_actor.py)、[rl_runtime.py](../../starVLA/rl/qwen_world_postcot/rl_runtime.py)、[train_rl.py](../../tools/qwen_world_postcot/train_rl.py) |
| memory/time RL | [scope_actor.py](../../starVLA/rl/qwen_world_postcot/scope_actor.py)、[scope_runtime.py](../../starVLA/rl/qwen_world_postcot/scope_runtime.py)、[train_scope_rl.py](../../tools/qwen_world_postcot/train_scope_rl.py) |
| scope配置 | [attention＋memory](../../configs/qwen_world_postcot/rl_attention_memory.yaml)、[time](../../configs/qwen_world_postcot/rl_time_modulation.yaml) |
| 物理尺度与独立损失 | [coordinate_objective.py](../../starVLA/rl/qwen_world_postcot/coordinate_objective.py)、[loss_designs.py](../../starVLA/rl/qwen_world_postcot/loss_designs.py) |
| 数据／checkpoint／bank身份 | [data.py](../../starVLA/rl/qwen_world_postcot/data.py)、[checkpoint.py](../../starVLA/rl/qwen_world_postcot/checkpoint.py)、[bank_versions.py](../../starVLA/rl/qwen_world_postcot/bank_versions.py) |
| 真实部署与原生导出 | [deployment.py](../../starVLA/rl/qwen_world_postcot/deployment.py)、[export.py](../../tools/qwen_world_postcot/export.py) |
| 单候选评价／配对统计 | [evaluate.py](../../tools/qwen_world_postcot/evaluate.py)、[summarize.py](../../tools/qwen_world_postcot/summarize.py) |
| 原GT与V12同预算评测 | [finish_label_source_match.py](../../tools/qwen_world_postcot/finish_label_source_match.py)、[登记](../qwen_world_original_vs_optimized_labels_20261029/README.md) |
| V12训练专用分支 | [905c5ef源码树](https://github.com/IDayday/VLA-Drive/tree/905c5ef00a9f9602eecd3e0c3a16f79dea5f3db6)、[训练owner](https://github.com/IDayday/VLA-Drive/blob/905c5ef00a9f9602eecd3e0c3a16f79dea5f3db6/tools/qwen_world_postcot/optimized_il_owner.py)、[配置](https://github.com/IDayday/VLA-Drive/blob/905c5ef00a9f9602eecd3e0c3a16f79dea5f3db6/configs/qwen_world_postcot/optimized_labels_il.yaml) |

本次把三帧DINO正式v4和旧安全GRPO正式v3作为独立source-lock分支保存；归档只保留代码/小型记录，不恢复训练。新增对象均经大小和敏感资产扩展名审查，见[source_archive_review.json](source_archive_review.json)。

## 上游与复现边界

SimWAM上游锁定1433ee7e06a62bd8b89da74455df0d82bda7b726。相关vendor/许可证与原source-oracle记录沿用原代码，没有重新标注许可。当前World更换了观测/动作架构、clean-x目标和有效clip1，是明确适配，不能称原SimWAM完整视频—动作模型逐字节复现。

旧DDP分项约束、可学习噪声、外部搜索/候选吸收/NFT及三条已暂停研究仍只属于其历史目录。它们没有被偷偷加入当前World baseline。post-head辅助监督也不能被描述为已实现独立轨迹修复器。

大型权重与原始场景仍在资产存储。索引中的Git SHA、目录和报告可定位实现，复现还需要合法的数据/模型/teacher/cache资产及当前资源授权。
