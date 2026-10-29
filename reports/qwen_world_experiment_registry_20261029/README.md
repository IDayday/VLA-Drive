# Qwen-World 与前序 Flow 实验总索引

本目录把当前IL、RL、损失设计、优化标签和同预算补测整理为一份可核查索引。代码仍保留各自版本和科学身份；没有自动合并分支，没有修改活跃训练源码，没有把暂停的研究重新启动。快照时间见[registry.json](registry.json)。

## 当前可作出的结论

- 当前Qwen-World原GT IL已测最高是**70k／88.792910 PDMS**；最新已测83k为88.500809。
- 当前World RL已测最高是**动作attention LoRA＋scene/predictive memory，33.3k／90.741310**。其真实父模型是IL40k／87.972382，净增**2.768928点**。
- 50k固定终点：attention89.905612、memory90.628892、time89.508000，分别高于同一IL父模型1.933230、2.656510、1.535618点。三条曲线均不单调，当前仍未达到92。
- 损失初筛中，PHYSICAL88.336938、TIME_RESIDUAL88.146292、DELTA88.131260，均未超过匹配CONTROL88.381806的点估计。Huber训练已完成，评分尚未完整；不能填入部分均值。
- 优化V12标签IL已测10k79.904274、13k66.369502、15k81.359718。13k有明显暂时回退；native/export一致性通过不等于其原因已查明。完整100k仍在训练，不能拿原GT70k与优化标签15k相减归因标签优劣。
- 原GT10k/15k配对Navtest已完成：原GT80.531938/84.337362，对应优化标签79.904274/81.359718，净差−0.627663/−2.977643点。15k配对区间[−5.821223,−0.146684]；10k区间跨零。两组前15k八rank的数据顺序、LR、更新计数、辅助曝光和实际bank完全匹配，sigma_mean差0，见[匹配核验](label_match_audit.json)。这不冒充逐元素噪声验收，也不外推完整100k的异步bank一致性。

以上当前分数全部来自**完整12146 Navtest、invalid0、canonical NAVSIM v1.1、单候选clean-x十步shift5 ODE、seed2101**。训练seed42。ODE是生成方式，PDMS是评价指标。没有候选oracle或scorer部署选轨。Test历史已反复观察，最高点只是已测最高点。

## 当前实验矩阵

| ID | 改变什么 | 训练人口与预算 | 当前状态／结果 |
|---|---|---|---|
| WORLD_IL_RAW | 原始GT；完整Qwen＋memory＋actor＋四辅助任务 | 全TRAIN90890；100k更新 | 训练继续；已测最高70k88.792910 |
| WORLD_RL_ATTENTION | 冻结Qwen/memory，只训练动作attention LoRA | 公开hard清单与当前TRAIN交集15171；50k | 训练完成；峰值20k90.580893，终点89.905612 |
| WORLD_RL_MEMORY | attention LoRA＋完整scene/predictive memory | 同一IL40k、同一人口、50k | 训练完成；峰值33.3k90.741310，终点90.628892 |
| WORLD_RL_TIME | Time MLP/AdaLN目标LoRA，Qwen/memory冻结 | 同一IL40k、同一人口、50k | 训练完成；峰值40k89.838968，终点89.508000 |
| LOSS_CONTROL | 原归一化L1；精确恢复63k | 新增2k、64000次scene曝光 | 完整Navtest88.381806 |
| LOSS_PHYSICAL | 固定物理容差构造坐标权重 | 同63k恢复、同2k | 完整Navtest88.336938 |
| LOSS_TIME_RESIDUAL | TRAIN恒加速残差尺度构造未来时刻权重 | 同63k恢复、同2k | 完整Navtest88.146292 |
| LOSS_DELTA | 原L1＋匹配GT的XY增量项 | 同63k恢复、同2k | 完整Navtest88.131260 |
| LOSS_HUBER | 物理容差定义分段Huber | 同63k恢复、同2k | 训练完成；评测分片准入冲突后继续补齐 |
| WORLD_IL_V12 | 仅规划轨迹监督替换为V12优化标签／合法fallback | 通用初始化；全TRAIN；完整100k | 训练继续；10k/13k/15k已测 |
| WORLD_LABEL_MATCH | 补原GT10k/15k；不新增训练 | 每模型12146场景；同seed/协议 | 全部完成；优化标签两个点估计较低，15k区间低于0 |

RL三组的共同50k预算是12500份fresh全局rollout、100000次scene曝光、800000条新候选；4次inner是复用同一批链的4次真实optimizer更新。不能把50k当作50k份新rollout，不能只以update数量宣称总计算成本相同。

当前World三组RL均使用canonical5:5:2训练反馈，没有混入旧SCADV3、learned-noise、拉格朗日乘子、GT replay、BC或NFT。

## 文档与机器可读记录

| 文件 | 内容 |
|---|---|
| [SCORES.md](SCORES.md) | 47个完整模型读出、近期IL、RL共同步数曲线和历史参考 |
| [all_completed_scores.csv](all_completed_scores.csv) | 每点更新数、PDMS、NC/DAC/TTC/EP/Comfort/DDC、来源路径与hash |
| [complete_scores.json](complete_scores.json) | 已验证结果身份、实际RL父模型、端点、峰值和快照状态 |
| [ARCHITECTURE_AND_PROTOCOL.md](ARCHITECTURE_AND_PROTOCOL.md) | 实际20层clean-x模型、输入、四辅助任务、IL/RL与部署区别 |
| [VERSIONS_AND_CODE.md](VERSIONS_AND_CODE.md) | 16项版本/历史尝试的科学源码SHA、模块和独立分支 |
| [RUNBOOK.md](RUNBOOK.md) | 真实入口、恢复与资源边界、故障分类及费用口径 |
| [registry.json](registry.json) | 版本状态、代码链接、协议、数据数量及log隔离 |
| [formal_IL_source_check.json](formal_IL_source_check.json) | 原始IL不可变snapshot的50个文件与Git源对象逐hash一致 |
| [loss_screen_results.json](loss_screen_results.json) | 2k窗口相对CONTROL的log配对区间与收益/损失 |
| [loss_stage_gate.json](loss_stage_gate.json) | 用户确认的预实验→筛选→100k流程；尚未伪造选择或启动 |

## 损失预实验如何解释

当前2k窗口检验的是“已训练63k后切换损失”的效果，不是“从通用初始化一直使用该损失”的排名。64000次场景曝光约为完整TRAIN的0.704次遍历，仅占100k预算的2%；继承了已形成的表征与后期LR。

五组均核对了八rank完整63001..65000、optimizer计数匹配、skipped0、loss/grad有限、scene/LR/候选bank/辅助曝光一致。没有观察到数值失稳；不能仅凭末端分数把结果归因于Adam遗留状态。其betas为0.9/0.95，旧动量的直接贡献衰减较快。输出层梯度校准也不代表全模型梯度或有效更新幅度完全相同。

| 相对CONTROL | ΔPDMS点 | 44个物理log配对95%区间 | 当前证据 |
|---|---:|---|---|
| PHYSICAL | −0.044867 | [−0.273579, +0.196037] | 未证明提升，也未证明等价 |
| TIME_RESIDUAL | −0.235514 | [−0.474440, +0.009149] | 本窗口没有提升证据 |
| DELTA | −0.250546 | [−0.460934, −0.018016] | 支持此窗口回退，不能否定整类从头设计 |
| HUBER | 未完成 | — | 不作提前判断 |

按用户要求，先完成预实验，再筛选少量方法做完整100k确认。Huber分数未齐，当前尚未锁定赢家或启动新损失100k。完整确认应从相同通用Qwen和相同随机World初始化开始，保持数据、teacher、bank日程、训练范围和预算匹配；新初始化下的尺度需要TRAIN校准，不能机械搬用63k尺度。若复用现有原L1控制，必须核对这些身份；不能把63k续训改名为从头训练。已有结果不足时，短程通用初始化预实验的预算须先profile并登记。

## 已知限制与仍未完成的问题

1. 当前RL有真实单候选增益，但仍有新增违规/零分，且继续训练不保证继续涨分。约束型安全RL尚未作为当前World新训练线运行。
2. 四辅助任务学习不等于逐项造成PDMS提升；缺少完整匹配去辅助任务消融。对象定位与稀有安全小分识别仍需改进。
3. post-head是训练期后果监督，不是部署critic/选轨器，也尚不是能给出逐时刻轨迹修复量的完整修复策略。
4. 优化标签13k回退原因未定位；标签历史98分并不保证模型部署达到98。
5. 一个训练seed和一个推理seed，区间只覆盖物理log采样不确定性。不同峰值步数不是匹配预算的方法优越性证据。

## 原始报告入口

- [World三scope的冻结范围、Time适配与真实八卡验收](https://github.com/IDayday/VLA-Drive/blob/ed41c4f72ed280ca5d3829d0af37a809f10ab080/reports/qwen_world_scope_RL_20261027/README.md)
- [RL共同10k/20k/40k/50k完整曲线](https://github.com/IDayday/VLA-Drive/blob/93489a2fc5fe243554c07435392cc5bc82e10305/reports/qwen_world_RL_midpoints_20261028/README.md)
- [原GT65k—83k补测](https://github.com/IDayday/VLA-Drive/blob/472214bd30b9e271ab997012e64d24c44247424f/reports/qwen_world_IL_late_navtest_20261029/README.md)
- [优化标签13k/15k完整结果](https://github.com/IDayday/VLA-Drive/blob/be12e67248f3282f870c5e17d0ba64ce638cc6b4/reports/qwen_world_optimized_IL_intermediate_20261029/README.md)
- [跨IL模型的同GT几何误差核算](https://github.com/IDayday/VLA-Drive/blob/bee4b4ed1d4f15e783581eb26085cfc2886e890d/reports/cross_il_trajectory_loss_20261028/README.md)

大型权重、optimizer、原始场景、图片、cache和密钥不进入本目录；原资产索引只引用真实已有路径。历史报告中的主机授权已过期，不能据此派发新任务。

## 前序尝试如何接到当前版本

旧DDP Flow-GRPO、约束/learned-noise C/N、SimWAM式DDP配方、源ActionDiT与公开困难人口迁移，均作为历史研究保留在代码表中。此前外部搜索、固定候选提取与NFT未形成合格的单候选部署方案，证据见[旧诊断报告](../flow_safe_improvement_20261013/README.md)。不能把候选best-of-N、搜索教师分数或训练reward列入当前策略主表。

随后原生Qwen-Flow已得到真实RL增益，但存在计算开销和安全退化；三时刻DINO版本被用户暂停。当前World重做了memory、20层clean-x actor和四个辅助任务，建立原GT IL与三条参数范围RL。旧潜噪声策略／物理搜索／成功存档三条研究线仍处于暂停状态，不是当前三条scope RL，也没有据本次整理自动重启。

历史DDP＋DINO IL100k89.527868高于当前World IL已测峰值；由于架构、输入、训练与推理协议不同，保留作端点参考，不据此归因某一辅助任务。当前World RL相对自己的IL40k有净收益，这是与此前局部迁移阶段不同的已验证事实。

- [原GT与V12标签同预算完整结果](https://github.com/IDayday/VLA-Drive/blob/7ca648d4a5feb0aa6f4d4bd0a7a8dcaaa1e83172/reports/qwen_world_original_vs_optimized_labels_20261029/README.md)
