# SimWAM式DDP RL：恢复复核与20k最终结果

两组20k、固定阶段DEV/TRAIN诊断和最终Navtest均已完成。本轮完成隔离修复与真实验收，**未达到92，也未证明安全保持下的稳定提升**。正式源码、权重、轨迹、评分与人口没有被修改或重跑。

- 修复分支：`fix/ddp-simwam-audit-recovery-20261015`，不合并原分支。
- 审阅起点：`ae8ac5a6e2a25d0b8a91e67a9047d86b93edf0b4`。
- 正式训练源码：`12b8e0d75691d7bed04ab02072020e130f711a3b`，保持不可变。
- 恢复补丁八卡验收源码：`73dd0f7c4ea799c5d81420c168943187baa83c86`。
- SimWAM上游：`1433ee7e06a62bd8b89da74455df0d82bda7b726`。
- 训练seed42、评价seed2101；主成绩均为单候选ODE、canonical NAVSIM v1.1。没有best-of-N或best-of-two。

## 最终Navtest

每项均为12,146场景/136 logs完整结果。差值相对**同网格SFT**，单位为0—100分上的分数差。区间按log聚类，不包括训练seed/推理seed不确定性。

| 模型 | E_NATIVE | 差值 / 95%区间 | E_SHIFT5 | 差值 / 95%区间 |
|---|---:|---|---:|---|
| SFT | 88.9116 | — | 88.7641 | — |
| R_SOURCE 20k | 87.9125 | −0.9991 [−1.5519, −0.4562] | 88.5802 | −0.1840 [−0.6266, +0.2482] |
| R_CANONICAL 20k | 88.9517 | +0.0401 [−0.2985, +0.3736] | 89.1116 | +0.3474 [−0.0075, +0.7026] |

R_SOURCE提高EP和DAC，但NC/TTC下降更大。R_CANONICAL减少了相对R_SOURCE的退化，尚未建立相对SFT的稳健净收益。shift5有所缓解，不能替代原生ODE结论，也不能把改网格的基线变化全部计作RL收益。

| 原生ODE | NC | DAC | TTC | EP | Comfort | DDC |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 98.1640 | 96.9702 | 95.0519 | 83.0034 | 100.0000 | 98.2834 |
| R_SOURCE | 97.1719 | 97.2007 | 91.2399 | 84.4850 | 100.0000 | 98.2216 |
| R_CANONICAL | 97.6741 | 97.3572 | 93.5699 | 84.3186 | 100.0000 | 98.2669 |

“新增/恢复”按官方分项是否完全通过统计，NC=0.5也属于未完全通过。固定SFT-safe集合由seed2101定义，不能称所有噪声下绝对安全。

| 模型 / 网格 | NC 新增/恢复 | DAC 新增/恢复 | TTC 新增/恢复 | SFT-safe新增联合违规 | 非零→零 / 零→非零 |
|---|---:|---:|---:|---:|---:|
| SOURCE / NATIVE | 133 / 15 | 36 / 64 | 502 / 39 | 530 / 11213 | 168 / 70 |
| CANONICAL / NATIVE | 70 / 12 | 23 / 70 | 219 / 39 | 242 / 11213 | 93 / 74 |
| SOURCE / SHIFT5 | 83 / 17 | 25 / 87 | 348 / 39 | 366 / 11184 | 106 / 96 |
| CANONICAL / SHIFT5 | 54 / 12 | 17 / 94 | 190 / 38 | 205 / 11184 | 69 / 100 |

原生ODE的G+ / G−分别为SOURCE **186.1776 / 307.5333**、CANONICAL **155.9835 / 151.1147**，是原始0—1 score的场景求和，不是百分点。SOURCE有7,542场景出现任意正差，但少量大幅退化使总分下降；只数“提升场景”会误导。

JSON中的mean为0—1，mean_percent为乘100；G+/G−保持原始score求和。完整分项、低尾、转移、运动量和区间见[results.json](results.json)。[21份完整评价复核](evidence/evaluation_audit.json)逐文件验证payload与score seal，所有metric与旧摘要相同，没有新增全量推理或评分。

## 固定阶段与泛化

![DEV过程与固定终点Navtest](learning_and_navtest.svg)

| 更新 | DEV SOURCE native/shift5 | DEV CANONICAL native/shift5 | 定向TRAIN SOURCE native/shift5 | 定向TRAIN CANONICAL native/shift5 |
|---|---|---|---|---|
| SFT | 93.6192 / 93.5747 | 同左 | 80.3316 / 79.1099 | 同左 |
| 5k | 94.0763 / 93.9778 | 94.3121 / 94.3214 | 83.1492 / 82.2102 | 84.4715 / 83.9018 |
| 10k | 94.4791 / 94.4361 | 94.5018 / 94.3856 | 86.1072 / 86.1952 | 86.7514 / 86.9100 |
| 15k | 94.3706 / 94.2974 | 94.6291 / 94.4625 | 86.0192 / 87.0240 | 86.1652 / 86.5451 |
| **20k固定终点** | **93.8193 / 94.2939** | **94.6719 / 94.5781** | **84.1979 / 83.7905** | **85.6523 / 85.1714** |

DEV完整1,696场景/16 logs；TRAIN诊断512场景有定向难例偏好，不是总体无偏估计。各阶段安全转移均在JSON中。SFT可能见过DEV，仅称“RL更新未见”；历史Navtest曾多次使用，不能称完全盲测。

没有提前删除SOURCE或改reward。最终Navtest锁定20k终点，锁记录 `cad8a9124a9c8b0fecd80087682db77b3229dcba20ab9145846b1335cb712265`。DEV进步没有可靠迁移至Navtest，SOURCE出现终点回落，不能只展示10k峰值。

## 更新与行为证据

每组核实：20,000 optimizer updates、5,000 fresh全局rollout、40,000场景曝光、320,000新候选；192个Adam状态和scheduler均到20,000。八rank日志连续，场景顺序符合固定清单，各有7次登记的epoch padding，见[终点审计](evidence/endpoints.json)。

| 全程训练统计 | R_SOURCE | R_CANONICAL |
|---|---:|---:|
| 有效PG组 | 99.590% | 99.635% |
| progress-only组 | 65.860% | 70.553% |
| 正优势且仍违规候选 | 859 | 460 |
| 有安全候选却给违规候选正优势 | 657 | 340 |
| 第1 inner最大ratio误差 / clip fraction | 0 / 0 | 0 / 0 |
| 第2/3/4 inner平均clip fraction | 0.412% / 2.633% / 6.371% | 0.396% / 2.338% / 5.625% |

困难清单主要是安全但低进度场景。当前信号充足，不能再以零优势过多为主因，也没有证据说明clip完全阻止更新。共享LoRA的泛化漂移也可能造成新违规，不能将所有新失败都归因于少量正优势违规候选。

随机策略和部署分表：SOURCE四个5k窗口canonical随机均分为82.2068、82.9142、83.1414、83.1573，其训练reward为78.0446、78.8834、79.1511、79.1852；CANONICAL为82.1064、82.7486、83.0642、83.0436。窗口场景组成不同，不能与另一人口的ODE均分相减后证明sampler迁移问题。详见[R_SOURCE](R_SOURCE_training.json)、[R_CANONICAL](R_CANONICAL_training.json)。

原生Navtest请求终点平均前向位移为SFT 19.3036m、SOURCE 19.7127m、CANONICAL 19.5413m；模拟前向位移为19.4678m、20.1014m、19.8545m。SOURCE更向前、EP提高更多，也有更多NC/TTC新违规。两组只改变训练reward，支持源10:5:2权衡更激进；尚不能证明anchor某通道是原因。

请求segment-jerk RMS诊断为0.9462、2.3872、1.7060 m/s³。它不是官方Comfort，不能互换；官方native Comfort仍为100。到自身行程90%的时间随路径长度改变，不是同一固定终点的到达时间。

复用5k并补算10k/15k/20k已保存batch，无新候选/评分。7/8步PG norm通常大于beta50 anchor，9步anchor接近PG。SOURCE的9步cosine在10k/15k为−0.417/−0.331，20k变为+0.236；CANONICAL为−0.040/−0.161/+0.183。冲突随batch变化，不能凭5k一批认定anchor始终过强或无效。[逐步与通道诊断](gradients.json)每阶段仅8个场景，是条件证据，不是总体估计。

## 修复与有效性判断

旧异常路径可以把局部/未finalize rollout发布为普通COMPLETE并推进latest。现将COLLECTING失败保存在独立证据目录；合法恢复点检查阶段、跨rank计数/behavior/advantage、Adam/scheduler进度。发布失败不推进latest、不覆盖旧checkpoint，遗留incomplete用新attempt区分。原失败文件保留。

新增日志invocation和恢复祖先；回滚保留原日志及aborted后缀凭据，仍拒绝重复有效update。behavior不静默覆盖。旧checkpoint须只读分类和显式迁移receipt；已在旧inner1检查点的隔离副本完成真实八卡迁移，没有动正式run。

| 判断维度 | 结论 |
|---|---|
| 核心算法实现 | 保留完整已核实配方；无新actor数学错误证据。部分随机步骤/clamp仍是经验代理。 |
| 正常训练数值 | 正常路径与正式源码基准精确一致；实际mean/std和八卡恢复通过。未放宽clip或改anchor。 |
| 异常恢复 | CPU与有限真实GPU故障测试通过；旧未分类状态不自动恢复。未人为制造物理GPU损坏。 |
| 评价正确性 | 21份完整评价metric不变；新账本真实原始观测smoke的动作与score和旧SFT精确一致。 |
| 成本统计 | 新账本有真实验收；历史load/save/重试缺失处UNKNOWN，历史总成本不完备。 |
| 既有结果 | 未发现正式路径触发该恢复漏洞。结果有效不等于提分成功：未达到92，安全仍有退化。 |

正常路径精确对照包含权重、Adam、scheduler、RNG、后续链、old probability、advantage、loss、梯度norm和LR。两组正式run未发现failure检查点或恢复启动。不能因此否定正常结果，也不能用工程PASS推导RL一定提分。[validation.md](validation.md)列出所有执行证据和已保留失败。

## 成本与实际状态

正式训练最后一段rank0约SOURCE 4.985h、CANONICAL 7.403h；八worker相加39.883h和59.223h，**是共享GPU worker wall，不是独占GPU计算小时**。累计update阶段为94,854/145,789 worker-seconds，评分等待5,427/6,779。不同共享负载下不能直接归因算法速度差异。旧加载、保存、不可追溯重试不编造，见[costs.json](costs.json)。

新吞吐用本次end−start update；启动计数或窗口不足时ETA未定义。评价每次invocation分别保存new/reused、加载、前向、评分等待、job跨度与worker wall。真实重入smoke第一次new=1、第二次reused=1，旧耗时保留；共2条网格预测与1条入口诊断预测，单列，不混入主成绩。失败invocation总耗时与其中真正被回滚的耗时不混为一谈，后者缺证据时保持未知。

实查两组训练、milestone与最终Navtest队列均结束，旧controller PID退出。旧progress的RUNNING是快照错误，新代码在20k写COMPLETE。仅在授权Tsinghua主机、zt2、zt3进行有限资格测试和已有batch分析；未使用撤销主机，未改zt4上的G任务，未重启C/N或重复正式训练。

## 下一轮只建议一个变化

以R_CANONICAL为固定控制，**仅将纵向anchor权重0.2改为1**；其他reward、预算、清单、LoRA、sigma、网格、种子保持不变，检查原生ODE的NC/TTC新增违规是否减少且EP收益保留。当前前进偏移、TTC退化及晚步anchor证据支持研究保持方式，但尚未证明纵向权重是病因；困难人口偏好与DEV覆盖差异仍是混杂。

这是待检验的单因素假设，本轮未启动。没有加LR、噪声、seed或预算。只有安全转移与固定人口净收益共同成立，才能称为改进。

补充的[八份已有batch通道梯度复核](anchor_channel_supplement/README.md)进一步限定了该建议：源实现按权重和归一化，0.2→1会令纵向系数变为3.6667倍，同时横向/朝向系数降至0.7333倍。它是配置上的单因素，不是只加强纵向且其余保持强度不变。各阶段PG与纵向anchor的方向关系不同，不能据此承诺安全或92分。补充只做已保存链的只读分析，没有新增训练或评分。

源码差异和hash见[source_audit.json](source_audit.json)、[正式→审阅diff](patches/formal_to_review.patch)、[审阅→修复diff](patches/review_to_repair.patch)。可执行入口见[commands.md](commands.md)。不上传权重、缓存、原始数据或场景图。
