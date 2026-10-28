# Qwen-World：补测IL检查点并复核RL是否回退

用户要求“测一下其他Qwen-world的ckpts的IL分数”，并追问“目前的RL训练的分数有回退吗”。本轮锁定8个完整原生检查点，尚未产生新Navtest结果。

| 类型 | 锁定更新数 | 对照 |
|---|---|---|
| IL | 30k、35k、45k、50k、53k | 已有37k/40k/52k全量结果 |
| Attention LoRA RL | 32.5k | 同组26.6k：90.567244 |
| Attention LoRA＋Memory RL | 22.1k | 同组18.4k：90.661658 |
| Time/AdaLN LoRA RL | 43.8k | 同组35.6k：89.369663 |

三组已测RL相对自身IL40k87.972382均有正向增益，但这些点本身不能判断后续训练是否回退。此次每条RL都与自身上次评测做全量配对，分别报告分数差、物理log-bootstrap区间、零分和NC/DAC/TTC转移；同时保留相对共同IL40k的比较。不会把不同RL步数的分数直接归因方法优劣。

所有新评测统一12146场景、canonical NAVSIM v1.1、seed2101、单候选clean-x十步shift5 ODE，无best-of-N、teacher、搜索或scorer选轨。5个新IL点选自40k附近和最新完整检查点，已测37/40/52k直接复用；5k到25k不属于此次有限补测。Test已经历多轮观察，不能称盲测。此请求不改变固定IL40k RL父模型或训练配方，也不自动重训/重置任何模型。

预算为8×12146=97168次逻辑单候选评测，另8次独立原生恢复一致性预测。完整原生库存已硬链接锁定（不复制大权重），防止滚动保留影响53k；导出和native一致性阶段仍执行原严格hash/加载验证。pin准备遇到RL库存已包含identity的重复路径，在任何worker启动前拒绝；唯一化路径、逐文件同inode核验后完成准备，证据保留，无受影响策略样本。

仅使用固定许可五台：vla-zt、tsinghua-zt/zt3/zt4/zt5。实际UUID/内存/共用者在每次派发时重新验证，未知PID拒绝，不停止其它项目。临时rl-zt2、tsinghua-zt2、tsinghua-zt6继续禁用。GPU资源允许共享；既有三RL与World IL继续原计划。

复用已通过全量评测的SnapshotRequest、原生加载/导出、概率无关的ODE推理和canonical评分。改动只在CPU调度登记、最多8个有预算的model请求并行及结果报告中加入“同组前一检查点”比较。18项相关合同测试通过，CLI --help实际执行；没有改变actor、sampler、动作归一化、评分或训练代码。

实际启动入口（本次唯一owner；运行后不可重复执行）：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/mnt/project/VLA-Drive-qwen-world-il-checkpoint-sweep-20261028 \
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -u \
-m tools.qwen_world_postcot.latest_snapshot_owner \
--root /mnt/project/qwen-world-postcot-20261025/additional_IL_RL_NAVTEST_20261028T013728Z \
--report-dir /mnt/project/VLA-Drive-qwen-world-il-checkpoint-sweep-20261028/reports/qwen_world_checkpoint_sweep_20261028
```

运行期间不编辑此source worktree或HEAD：原生worker每次启动验证gitclean/sourceSHA。实时证据写在资产根；全部worker完成并退出后唯一owner才提交全量报告并push。基础结果、同组后续变化和训练指标分开解释。失败worker保留证据、不将缺失场景从分母删除，不用部分均分发布完整PDMS。
