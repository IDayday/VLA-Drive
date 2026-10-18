# IL 5000步中期资格与后续执行

本页记录中期实现检查和费用测量。**没有正式DEV/Navtest成绩，也没有RL规划净收益结论。** 这两条16步资格训练不代替登记的两条50k正式训练。

## 正式IL的实际状态

2026-10-18约23:26 UTC核实时，八个rank均在完整训练阶段达到5198次更新，累计166336次scene曝光；最新完整恢复点为update005000。完整Qwen语言、视觉相关投影/视觉主干、新provider和全部动作专家均已进入登记的训练阶段。单步约4.3秒，峰值75.014GB。一次FM loss约0.0067，这是训练误差，不能转写成规划分数。

5000步恢复点在rolling保留期到期前被完整hardlink固定为私有资格父资产 `qualification/IL_formal005000_parent`，保持原始模型、Adam、RNG和文件hash。固定资产避免后续rolling退休后丢失资格实验来源；原正式训练不暂停、不修改。该父模型尚未通过五个DEV观察点选择，正式RL入口不接受它作为正式父模型。

## 两组真实有限资格

CANONICAL在 `training-vlawm-tsinghua-zt6`、SOURCE在 `training-vlawm-tsinghua-zt4` 各用8卡。两者从相同5000步IL和相同LoRA零初始化开始，每组16次optimizer更新、4批fresh全局rollout、32次scene曝光、256条新候选。

| 实测 | CANONICAL | SOURCE |
|---|---:|---:|
| 有效PG组 / 全部组 | 31/32 | 31/32 |
| 无PG组 | 1 | 1 |
| 各fresh collection第一inner ratio最大误差 | 0 | 0 |
| 有非零Adam任务一阶动量的LoRA张量 | 480/480 | 480/480 |
| 实际Adam与scheduler计数 | 16 / 16 | 16 / 16 |
| 采样mean/std与重算逐值对照 | 96项全通过 | 96项全通过 |
| 峰值GPU显存 | 38.284GB | 38.285GB |
| 最慢rank阶段时间 / fresh全局rollout | 8.677秒 | 8.926秒 |

这证明当前训练过的原生模型、真实NAVSIM反馈和源GRPO能形成合法更新信号。32组来自固定公开hard TRAIN数据流，不能当整个TRAIN的无偏统计；也不能与旧全人口、不同G及不同IL程度下的等分率直接相减。

两组的固定SFT reference、Qwen和provider hash训练前后保持一致；采样/重算对照覆盖全部8rank×4批×3随机步骤。资格权重不进入正式初始化。详见 [完整费用与组统计](interim_IL005000_RL_profile.json)、[概率/optimizer核对](qualification_IL005000_RL_checks.json)。

训练随机策略在这32组上的canonical均分约0.5523、联合安全候选比例约0.6211。这是G8随机采样、多个行为版本和定向困难人口的描述，**不是单候选ODE部署成绩**。SOURCE和CANONICAL在16步内的细微差值没有方法优劣意义。

按4批实测阶段时间外推，每组50k约30.1/31.0小时；另加模型加载、保存、评价、故障重算和未来共享波动。这个短窗口还包含初次评分初始化，不能视为稳定速度或论文加速倍数。最终预算仍各50000次optimizer更新、12500批fresh rollout、100000 scene曝光和800000新候选。

## 后续执行链已部署

CPU owner `campaign_v2` 已实际运行，当前等待完整IL及五个登记DEV。训练/评价仍使用已验收的不可变 `d4d0012`；调度、导出验收和派生分析使用不可变 `b784e82`。它们的身份分别记录，不覆盖正式训练快照。

1. 既有 `milestone_queue_v2` 完成20k/40k/60k/80k/100k全量DEV，锁定共同IL父模型。
2. 对选定父模型分别做有限16步资格和费用确认；资格权重丢弃出正式初始化路径。
3. 两组各用一台当时授权且满足容量的8卡机器，从共同IL与相同LoRA初始化训练到50k。
4. 0为共同IL基线；5k/10k/20k/35k/50k做完整DEV。各导出都先对原未合并模型检查raw-image完整10步ODE一致性，再允许评分。
5. 主比较为固定50k终点。额外DEV选择仅在5k/10k/20k/35k/50k已训练检查点中取canonical PDMS最高者，精确同分选较晚步数。这个规则在任何正式RL结果出现前登记。
6. IL基线在IL选择锁定后独立做Navtest。两个RL固定终点和全部DEV结果锁定后，才做RL终点及DEV选择模型的完整Navtest；相同检查点只评一次。全部seed2101、单候选shift5 ODE。
7. 验证真实Adam/scheduler计数、两组实际scene流hash，并产生配对G+/G−、安全转移和按log bootstrap。达到92与RL相对自身IL增益分别报告。

调度不包含算法选择、额外seed、动态课程或失败自动重训。worker退出码0仍须核对完整原生50k边界及optimizer计数；模型加载阶段即保留GPU UUID，避免多个任务因显存暂未分配而抢同一卡。CPU导出一次一个，并检查实际内存和CPU容量。

[调度测试](qualification_campaign.json) 覆盖12项CPU测试，包括部分派发、故障拒绝重试、未完成资格阻断正式训练、未锁定终点阻断RL Navtest、错误Adam计数、GPU加载期预留及导出等待数值验收。实际CLI `--once` 和重复owner拒绝也已执行。旧的仅等待owner在尚无任何子任务时被显式替换，收据保留；正式IL未中断。

实际注册见 [campaign_execution_registration.json](campaign_execution_registration.json)。恢复、原始运行和预算入口见 [execution.md](execution.md)。这条执行链使当前长IL完成后可以继续正式RL和最终评价；它本身不构成已完成研究的证据。
