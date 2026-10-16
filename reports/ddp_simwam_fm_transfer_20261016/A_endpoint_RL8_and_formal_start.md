# A20k SFT端点八卡验收与正式RL启动

2026-10-16 20:02 UTC，A_DDP_TIME完成基于自身20k SFT和完整TRAIN困难清单的实际八卡验收，随后在tsinghua-zt4启动正式R_SOURCE。运行名为`A_DDP_TIME_R_SOURCE_seed42_v1`，固定终点20,000 optimizer updates。没有使用验收权重作为初始化，没有更改已登记配方。

机器可读证据见[A_endpoint_RL8_and_dispatch.json](A_endpoint_RL8_and_dispatch.json)。完整TRAIN清单来源见[筛选报告](A_SFT_full_TRAIN_screen.md)，自己的固定SFT部署基线见[20k报告](final_A_SFT_update020000.md)。

## 实际验收

| 检查 | 已执行结果 |
|---|---|
| 8卡真实TRAIN条件、G8、官方独立评分 | 两条执行路径均完成；每条路径8场景、64候选 |
| 同一fresh rollout四次inner optimizer更新 | 连续路径4次；另一条2次保存后恢复再做2次 |
| 连续4次 vs 2+恢复2次 | 全部LoRA、Adam、scheduler、RNG完全一致 |
| behavior、保存链、old概率、优势、评分 | 两路径一致；恢复未重新采样候选 |
| 实际可训练范围 | 96个attention投影、192个LoRA A/B张量、4,718,592参数 |
| 更新/冻结 | 八rank均确认全部adapter更新；原动作头和永久reference未变 |
| 第一inner ratio数值误差 | 八rank最大绝对误差0；不将其解释为后续没有更新 |
| 非零LoRA合并完整ODE | native/shift5最大绝对误差均≤2.38419e−7 |
| 合并后原始观测部署 | 两个真实TRAIN场景、两个ODE网格，共4次正式候选预测和评分；额外1次一致性诊断预测 |

这些结果证明该有限真实路径的更新、恢复和部署适配通过，不证明后续RL一定提升PDMS。原样源概率裁剪、部分随机步骤及dimension-mean策略代理的理论边界不变。

## 验收成本与正式预算分开

验收端到端354.290秒。执行阶段分别为：continuous 63.973秒、split2 78.964秒、resume4 78.345秒、状态比较1.954秒、完整ODE合并检查40.697秒、导出50.011秒、原始观测部署38.637秒。阶段时间包含各自启动、加载和退出开销，不能全部当作optimizer计算。

两条相互对照的执行路径合计8次物理optimizer更新、128条实际评分的训练候选；它们对应同一组64条匹配behavior样本的两次执行，不能写成128条独立发现。另有4条原始观测部署评分和1次额外诊断预测。所有这些开销均属于资格验收，不计入正式RL的场景/候选预算。

正式预算仍为20,000次更新、5,000份fresh全局rollout、40,000次场景曝光、320,000条候选。在20:05 UTC左右已观察到118次真实更新、30份collection、240次场景曝光、1,920条新候选；首个100步原生恢复点已完整发布，phase=READY、world=8、next_inner=0。日志中的累计计数来自实际运行，不以计划数字代替完成量。

正式运行的rank0在100步记录91.521秒本次执行wall；其中rollout 13.130秒、评分等待5.655秒、old概率重算4.102秒、更新51.263秒。该窗口仍很短，包含预热与保存，暂不据此承诺精确完成时间。评分worker CPU时间另计，不能重复加成端到端wall。

## 正式身份与不影响训练的文案更正

- 正式和已验收执行源码：`225b9af660cb0210208e2c46fc4216d0f151275b`。
- 正式run identity：`78f81211d3da81bc86acf27d509b4cc7b6e58b849a13ed92fc1f5ceae80e72eb`。
- 自己的SFT模型SHA256：`ac52e9346080ce0e6bf27441935ac9aa6acbff239330b316bee36b560486579a`。
- 固定困难清单文件SHA256：`e490778ca017a4e90571d2e66c61f0bea1c4d8d64ef6f14d74d8c0874b294d10`。
- native100步恢复点files SHA256：`f101c585e31592a8ceae5f9f84afd536fc17f28d059c9f332ee9efedb7fd5482`。

读取正式身份时发现一个描述字段问题：共享runtime的`source_action_forward`字符串在A路线也写成了B的`standalone ActionDiT.forward`。该字符串不参与模型构建、前向、采样或loss；实际A的`architecture=ddp`，执行`TransferPolicy.velocity_sigma → -head.predict_velocity`，实际参数合同仍是原24层DDP动作头，96个attention投影。

本工作树只将该描述改为按架构区分，已做语法检查并核对差异只涉及身份文案。**运行中和验收过的225b源码目录未修改**，正式训练继续使用它；不能将本工作树的新描述版本直接用于无迁移收据的跨源码恢复。机器可读报告记录了两个runtime文件hash和明确解释，不重写已发布run identity。该问题没有改变轨迹、概率、梯度、优化器或任何既有结果，无需重训。

## 已执行入口与恢复

以下命令均在锁定源码目录运行：

```bash
cd /mnt/project/fm-transfer-20261016/source/225b9af660cb0210208e2c46fc4216d0f151275b

/mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.qualify_rl_endpoint \
  --sft-checkpoint /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_SFT_seed42_v1/checkpoints/update020000 \
  --manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json \
  --output /mnt/project/fm-transfer-20261016/qualification/A_DDP_TIME/own_SFT_update020000_RL8_v1

/mnt/project/flow-safe-improvement-20261013/venv/bin/python \
  -m scripts.analysis.fm_transfer.dispatch_rl \
  --sft-checkpoint /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_SFT_seed42_v1/checkpoints/update020000 \
  --manifest /mnt/project/fm-transfer-20261016/hard_manifests/A_DDP_TIME_SFT_update020000.json \
  --qualification-root /mnt/project/fm-transfer-20261016/qualification/A_DDP_TIME/own_SFT_update020000_RL8_v1 \
  --output /mnt/project/fm-transfer-20261016/runs/A_DDP_TIME_R_SOURCE_seed42_v1
```

验收入口拒绝覆盖已有目录；正式dispatcher拒绝重复启动。只有确认原trainer已退出、授权和资源仍符合要求时，才可使用同一dispatcher加`--resume`指向已核验的原生checkpoint，例如实际存在的`checkpoints/update000100_inner0`。运行期间不能重复执行恢复命令；故障发生时应选择当时最新合法提交边界，而非盲目使用这里的100步快照。

新评价controller继续按5k/10k/15k/20k里程碑分别评价native和shift5完整DEV及固定TRAIN诊断。B路线继续自身100k SFT，随后建立自己的筛选与验收；不把A的困难人口或验收权重借给B。当前还没有新A/B的RL部署收益或Navtest结论。
