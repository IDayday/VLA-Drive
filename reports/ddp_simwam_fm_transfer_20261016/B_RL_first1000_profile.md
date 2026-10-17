# B正式Flow-GRPO首1000步：训练信号与实际成本

2026-10-17，B_SOURCE_ACTION_DIT从自身100k SFT开始的R_SOURCE训练已完成首1000步，并继续向固定20k终点运行。**有效PG组为99.5%；主要耗时是参数更新，评分等待约占rank0窗口wall的2%。** 本文是训练与成本统计，不是新增的单候选ODE效果评价。正式DEV观察仍为5k/10k/15k/20k。

训练源码仍为`225b9af660cb0210208e2c46fc4216d0f151275b`，没有改变reward、LoRA、精度、时间网格、概率clamp、anchor或预算。只读汇总使用既有`d58485b08078a6319b2835793d8c79ffbbd3e9d5`，人口与链身份核对使用既有`8dbd2bb2c82026d3a8a0b4b96c9b0ba211248f81`。新增模型采样、评分和optimizer更新均为0。

## 完整窗口与数据计数

| 首1000步的实际项目 | 数量 |
|---|---:|
| 全局optimizer updates | 1000 |
| fresh全局rollouts | 250 |
| scene曝光 / 不同scene | 2000 / 2000 |
| 不同logs | 683 |
| 新G8候选 | 16000 |
| 发生有效PG的场景组 | 1990 / 2000（99.5%） |
| 全局仅anchor的更新 | 0 |
| 全满分 / 全零分 / 其他canonical等分组 | 1 / 5 / 4 |
| progress-only组 | 1434 / 2000（71.7%） |
| NC/DAC/TTC存在差异的组 | 461 / 2000（23.05%） |

局部零优势组保留anchor，其他rank仍可贡献PG，不能把10个局部零优势组写成10次全局无RL更新。每份rollout四次inner固定同一chain、old概率、优势和behavior身份，只算一次fresh采样。人口核对复用了真实源sampler，前2000个场景没有重访或epoch padding，journal中有2000个不同chain摘要；额外核验24份边界behavior payload。

困难人口仍是B自身固定的21,836个场景，其source epoch补齐长度为21,840。完整epoch将产生的4次padding曝光不能提前写成首1000步已经发生的重复。此窗口有683个logs，不代表困难人口的所有1056个logs都已覆盖。

progress-only的定义是canonical分数有差异、其他保存分项全部相同；分母是全部2000个场景组。不能与旧全量训练、不同基模、不同人口或以有效组为分母的比例直接相减，归因于ActionDiT。

## 优势与概率诊断

正优势候选中，8632条联合通过NC/DAC/TTC，23条未联合通过；后者中17条所在组还存在完全通过的候选。联合通过要求三项均为1，NC=0.5不算通过。这说明源scalar reward仍可能给违规候选正相对优势，但数量、梯度权重与实际ODE后果是不同问题；不能把23条写成23次新增部署事故，也不能据此判定B终点会退化。全失败组中的正相对优势本身不自动构成实现错误。

| inner | 最大dimension-mean ratio偏离1 | 平均PPO clip命中率 |
|---|---:|---:|
| 1 | 0 | 0% |
| 2 | 0.00964159 | 0.018750% |
| 3 | 0.01856977 | 0.477083% |
| 4 | 0.02600217 | 1.831250% |

第一inner未更新时数值ratio误差为0，后续clipping来自实际参数更新。全窗口逐维source relative clamp平均命中率为0.031396%，它与表中的PPO clip不是同一事件。这仍是源dimension-mean随机转移代理，不是完整物理轨迹概率比，最大ratio偏移也不能解释为最终ODE分布变化上限。

原始reward共同std均值0.168072，范围0.028269–0.403476。没有添加旧group std floor或分项约束。训练随机策略均分与有限候选最高值保留在JSON中，本文不把它们列入部署主成绩表，也不据前后窗口不同场景的均分判断RL增益。

## 真实耗时与边界

1000步封存边界记录的本invocation训练窗口为**1766.970秒（29.450分钟）**，起始update为0，实测速率**0.565940 updates/s**。由这一窗口线性外推，到20k尚需约**9.326小时**；这是当时的训练估计，不包含后续DEV/Navtest，也不是保证完成时间。

| rank0累计计时项目 | 秒 |
|---|---:|
| 参数更新区间 | 1217.547 |
| rollout | 298.053 |
| old概率重算 | 91.273 |
| 评分等待 | 35.498 |
| 评分器执行，另列 | 15.273 |

更新区间与评分等待分别约为同段rank0 wall的68.91%和2.01%。计时包含相应阶段的同步等开销，不能据此区分所有GPU kernel和CPU开销；评分器执行可能与其他阶段重叠，不与等待简单相加。主要耗时已在模型更新，继续优化评分等待的总提速空间有限。当前不为提速中途更换已锁定的数值或概率路径。

采集统计时训练未暂停，须明确三个时间边界：

1. 本文主表及checkpoint成本固定到1000步：250 fresh rollouts、2000 scene、16000候选。
2. 原始汇总抓取到1020个完整updates，JSON中最后一个20步窗口另存，不能悄悄并入“首1000步”。
3. 更晚的未关闭执行账本已记录16384条实际生成/评分候选，包含1000步之后继续运行的collection；多出的384条不能记成失败重算开销。

当时8个invocation均未关闭，累计worker wall含已观测准备时间为14612.540秒，累计保存worker wall为66.527秒；这些是八worker求和，不能当成端到端wall或独占GPU时长。未见aborted invocation，未关闭记录的尾部仍只能作为已观测成本。完整成本将在20k结束后按关闭账本核算。人口只读核对的主函数wall为1.077秒、CPU为0.788秒，不含Python导入和最终发布；未另造精确的全部分析进程总耗时。

## 当前判断和后续

这轮已经形成有效更新，早期零优势比例低；同时超过七成场景组的canonical差异仍只来自progress。它支持继续观察源目标对进度与安全的权衡，不支持宣布换ActionDiT已解决DDP RL，也不支持把所有问题归结为“没有梯度”。

B自身SFT完整DEV基线仍为native93.354165、shift5 93.428631。下一项效果证据是固定5k checkpoint在同人口、同seed2101、单候选native/shift5下的重新推理，以及相对自身SFT的新增违规与恢复。固定20k终点、后续完整评价和最终Navtest门槛不变，没有新增训练seed或算法组。

## 实际证据与已执行命令

[训练与成本原始汇总](B_RL_first1000_training.json)、[人口核验公开摘要](B_RL_first1000_population.json)、[报告计数/hash核对](B_RL_first1000_validation.json)。公开摘要省略私有scene标识和单条behavior记录，完整证据保留在项目资产目录。原生恢复点为`update001000_inner0`，阶段READY，文件清单SHA256为`d7983d609d68a9cad25587bee8568832c2608a38c1159ed33bfda5af19f1ad86`。

以下两条命令已在对应不可变分析快照、CPU-only环境实际执行，既有输出不能覆盖。它们没有触碰运行中的225b训练源码或模型。

```bash
# d58485b：原始抓取到1020步，首个窗口严格为1000步，成本边界为checkpoint1000
python -m scripts.analysis.simwam_recipe.summarize_training --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_RL_interim_first1000_window_20261017.json --window-updates 1000
# 8dbd2bb：明确指定不可变1000步前缀
python -m scripts.analysis.fm_transfer.analyze_rl_population --run /mnt/project/fm-transfer-20261016/runs/B_SOURCE_ACTION_DIT_R_SOURCE_seed42_v1 --manifest /mnt/project/fm-transfer-20261016/hard_manifests/B_SOURCE_ACTION_DIT_SFT_update100000.json --updates 1000 --output /mnt/project/fm-transfer-20261016/analysis/B_SOURCE_ACTION_DIT_RL_update001000_population.json
```
