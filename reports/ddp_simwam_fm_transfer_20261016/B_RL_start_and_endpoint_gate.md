# B自身100k SFT验收通过，正式Flow-GRPO已启动

2026-10-17 07:09 UTC，B_SOURCE_ACTION_DIT已在tsinghua-zt6的八卡上开始正式R_SOURCE训练，实际八个rank均已完成152次更新；固定终点仍为20,000。首个100步原生恢复点已单独核验。这里的进度不等于已有RL规划收益，首个正式RL开发集观察点仍是5k。

全量TRAIN单候选筛选、固定困难清单、完整SFT基线和真实八卡验收均已完成。CPU衔接控制器v3以`FORMAL_B_RL_STARTED_HANDOFF_COMPLETE`正常退出；其中`formal_RL: COMPLETE`只表示派发交接完成，不表示20k训练完成。后续评价由原里程碑控制器负责，不能重启旧v1/v2或重复派发B。

## 实际验收与身份

[只读核验记录](B_RL_start_execution.json)绑定以下结果和hash，未产生新rollout、评分或参数更新。

- 连续四次更新与两次更新后恢复到第四次一致：八rank的LoRA、Adam、scheduler、RNG以及真实chain、old概率、优势和评分均相同。全部适配器实际改变，冻结动作专家和永久reference保持不变。
- 非零LoRA合并前后，两条TRAIN condition、native/shift5完整ODE的最大绝对误差为`2.384185791015625e-7`。
- 从真实原始观测运行部署入口通过。两场景、两网格共4条正式资格预测，另有1条入口一致性诊断预测；这不是DEV效果评价。
- 正式初始化来自自己的100k SFT，未使用资格训练权重，没有resume或额外预训练。初始LoRA B严格为零、Adam状态为空；100步时480个LoRA张量均已有实际更新，480份Adam状态均为step100、FP32 moments。
- B实际覆盖240个self/cross-attention投影，共480个LoRA张量、15,728,640个可训练参数。Qwen、原生text embedding及reference冻结；不能把A的96个投影数量套到B的30层ActionDiT。
- 正式数值源码保持`225b9af660cb0210208e2c46fc4216d0f151275b`，64份训练源码hash逐项核验。实际torchrun及八个worker的工作目录均为该不可变快照。

父SFT model SHA256为`4e71b1fcf963d38456174de1f8d17762a9a4ccf1597c22b7cec262026d9201b7`；困难manifest文件SHA256为`cc6017f0740c080dfae73f36ddc24bb657ae2996a8440b7c035ad4821612ec4a`。正式run identity为`471f3a53c0ea91e21ac842ca2ef4c3164dbc2155ec1be9ec1e804fff7dbae86f`。验收权重与正式权重分开保存。

## 首个合法恢复边界及早期更新

`update000100_inner0`的11个payload重新核验hash，并复用原恢复阶段校验器检查八rank一致性：阶段READY、100次optimizer updates、25份fresh全局rollout、200次scene曝光、1,600条真实新候选。四次inner的chain、old概率、优势及behavior身份保持不变；scheduler完成100步warmup，LR为5e-5。这里没有将四次inner重复算成四份新候选。

首100步对应200个场景组，其中1组无PG优势（0.5%），没有全满分或全零组；140组仅progress产生差异，51组安全分项存在差异。这只是最初的固定困难人口样本，不能外推完整训练流，也不是部署成绩。困难人口总体见[完整TRAIN筛选](B_SFT_full_TRAIN_screen.md)：21,836场景中96.0570%在筛选样本下NC/DAC/TTC全通过。

| 同一fresh rollout的inner | 最大ratio偏离1 | 平均PPO clip fraction |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 0.00241947 | 0 |
| 3 | 0.00676155 | 0.00083333 |
| 4 | 0.00999749 | 0.00562500 |

第一inner未更新时ratio严格为1；后续inner出现真实更新后的clipping，不据此放宽0.005。没有修改源reward、anchor、部分随机窗口、概率clamp或有效gradient clipping=0的执行合同。

## 开销及当前效果边界

整个八卡资格链耗时298.129秒。连续4步63.629秒、split2为59.763秒、resume4为51.166秒、比较2.501秒、合并ODE34.022秒、导出45.068秒、原始观测部署40.821秒；各phase包含其启动与加载开销。两次试验实际生成并评分128条训练候选，逻辑上是同一64候选恢复对照，不能算128个独立探索发现。它们不计入正式320,000候选预算。

正式100步rank0累计update耗时122.426秒、rollout 30.388秒、old概率重算9.218秒、评分等待5.984秒。未结束invocation的累计wall不是完整训练成本；这些项目也不包含全部启动、保存和同步，不以它们伪造精确20k ETA。资格总wall、各worker耗时与GPU计算时间不能混为一谈。

B自身SFT开发集基线为native **93.354165**、shift5 **93.428631**，来自完整1696场景，不能算RL收益。A已经完成20k RL，其DEV native为93.607389，相对自身SFT **−0.112703个百分点**；shift5为94.195811，增益**+0.735501个百分点**，但分别新增62/25例TTC违规、仅恢复1/2例。A有效PG比例99.5525%，已有证据的主要问题是安全与进度权衡，以及收益对部署网格的依赖，不能继续仅归咎于零优势或宣布换ActionDiT已经解决问题。

固定阶段评价已完成28/36项，剩余8项为B的5k/10k/15k/20k各自DEV与TRAIN诊断。最终Navtest控制器仍等待两组固定终点和完整观察，尚未打开最终测试；DEV超过92不代表Navtest目标完成。旧C/N及暂停的旧follow-up没有恢复。

## 已执行入口

以下为实际执行命令，当前正式trainer在运行，不得重复执行。所有命令位于上述225b不可变快照；父权重、清单、gate、基线与dispatch证据已关联。

```bash
python -m scripts.analysis.fm_transfer.qualify_rl_endpoint --sft-checkpoint /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_SFT_seed42_v1/checkpoints/update100000 --manifest /mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json --output /mnt/project/fm-transfer-20261016/qualification/B_SOURCE_ACTION_DIT/own_SFT_update100000_RL8_v1
python -m torch.distributed.run --standalone --nproc_per_node=8 --module scripts.analysis.fm_transfer.train_rl --config configs/flow_rl/simwam_source_ddp.yaml --sft-checkpoint /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_SFT_seed42_v1/checkpoints/update100000 --manifest /mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json --output /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1
```
