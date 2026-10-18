# 唯一后续建议：C1完整既存TRAIN反馈的信号支持审计

状态：`PROPOSED_NOT_IMPLEMENTED_NOT_EXECUTED_NOT_APPROVED`。本文件仅为Lead决策准备预算和执行边界；本次32scene读取上限没有扩大。没有新评分、GPU或训练提案。

固定3584的16个非零优势组全为EP-only，且D/C/T事件为空。它对BC/policy梯度解读有价值，但无法评估稀有安全信号。同人口真实梯度现已提供：policy/BC范数比0.0613169%，仅支持这一batch的尺度差异；数学结果与监督退出异常分开记录在 `GRADIENT_AND_SCALE_ADDENDUM.md`。官方FPO整批GAE标准化没有被本项目明确注册的center-only适配采用，因此完整账本还需描述原优势幅值，不能只数非零组。

唯一下一假设为：**在已执行的C1 TRAIN流中，多数非零优势组只由进度区分；完整既存账本的分项、原优势幅值和同终局异时计数，能确定继续研究梯度尺度还是过程后果信号缺少哪一项证据。** “多数”按非零组中的EP-only比例大于1/2解释，未用全量结果选择门槛。该假设不是已成立的总体结论；本诊断不产生建议训练系数。

在C5、G4/G8或DFQL之前补该缺口最合理：现有完整反馈已经付出采样和评分成本；C2–C4全程的非零优势曝光率低，却来自其他策略轨迹，不能替代C1自身事件分布。G8会额外采样/评分且自然32scene对稀有事件欠功效。C5需要跨分项支持与进度权衡证据；道路代理当前没有修复失败场景的分母，不能直接启动DFQL对照。

严格固定范围仅为完成4096更新的 `C1_fulltrain_seed42_official_4096_v1`，同一原始DiT/Flow、Qwen冻结、G4、原reward与原评分。已有小汇总给出131,064次scene呈现、524,256次候选评分呈现；这是重复曝光分母，不是独立scene数，也不是4096×32。独立资源Agent的 `c8819df8:reports/st_world_policy/20261018/full_train_feedback_capacity_v1/CAPACITY_REPORT.json` 已逐个解析小COMPLETE并stat结果：4096唯一完整ledger、4096attempt，无重复/失败/不完整记录。**这只是元数据完整性，尚未读取/核验RESULTS内容及payload hash，也没有证明每个ledger均被训练实际消费。** 独立scene、log、版本覆盖和可消费完整性仍由将来审计核对。

建议将来实施的单次流程：

1. 冻结C1 run identity、原训练注册/结果、既存完整训练统计、4096更新的消费边界。用 `rollouts/boundary_*/rank_*_COMPLETE.json` 和原完成更新记录绑定实际消费的rollout/reward身份；不是把每个账本目录一律算作已消费训练。最终native状态/登记仍只读小回执，不载入checkpoint。原记录可能含重试或未消费attempt，分别保留，绝不双计。
2. 协调进程创建一次小文件索引：每boundary的policy_version、request_hash、REQUESTS/ADMITTED/COMPLETE/RESULTS的路径、size/hash、四rank receipt、实际候选数与适用schema。这一索引是反馈索引，不是训练全manifest。按完整boundary分片，保持scene的四候选同worker；每boundary只由一个worker读取与计数。不同policy snapshot不得拼成G8。
3. 复用此次已核实的归约/分项来源/事件时间定义；让固定32假设参数化为receipt实际人口，并对这项改变独立审查。有效零分、全失败组、未消费/无效记录都有分母。每条候选仍用保存的固定参考EP，绝不在四候选或全部候选间重归一化。
4. 全程统计同一套C1/C2/C3/C4诊断reward的非零组数、分项变化、EP-only/混合组、D/C/T事件非空量；这是在C1轨迹上描述其他已存在分项的信号，不是额外训练组。EP-only要求C1实际float32优势非零、EP变化、NC/DAC/TTC/Comfort均不变且乘法门控为正；门控变化不能因其出现在gated_EP代数项里就归成EP原因。记录每组原float32 A的sum、sum-square、max-abs、reward range、有效候选数，归约成每policy窗口的曝光加权幅值和组间分布；不计算新标准化奖励、候选权重或反事实梯度。对终局相同且事件非空的候选对计数物理时间与严重性差异，分别报告scene/log/group/pair分母。此前仅在128条验证过过程字段；全量每条须检查相同schema与事件至终局归约，缺字段不能计成空事件，空事件条件分母0返回undefined。
5. 预先按policy版本0–1023、1024–2047、2048–3071、3072–4095报告时间窗，并报告全程曝光加权结果、唯一scene覆盖与每scene重复次数；同scene跨snapshot出现不是独立重复。唯一scene描述为“在已执行训练期间至少一次发生”，不能伪装成某个冻结策略的事件概率。保留每log汇总，候选pair不作独立统计单位。
6. 一次归约结束即止，不依结果追加C2–C4全量、DEV、G8、代理梯度或训练。既有C1总体非零组80,933/131,064作为消费/计数交叉核对；不一致先定位receipt、尾batch或重试，不修改分母凑数。只存紧凑的每boundary/scene/log统计与稀有事件身份，不复制完整轨迹、maps或原始cache。

资源提案现以实测目录容量替代先前等大外推。RESULTS总量 **14,088,142,976B（13.1206GiB）**，单个最大3,477,984B、最小2,578,336B。资源Agent读取COMPLETE共1,687,552B，用时17.6843s，payload读取0B；该资源报告SHA256为 `473927f99e96bb30922425465c3214209ebd65e65dbedcab9aeb85bf7adfc3cc`。一次全文件hash加一次nominal load的payload总逻辑读取 **28,176,285,952B（26.2412GiB）**；资源报告含COMPLETE的对应值为28,177,973,504B。另给REQUESTS、ADMITTED、四rank回执、消费小日志及重复核验预留空间，所有实际读取计入总限额。torch.load不能绕开hash；也可在一次预付费读取后以内存bytes完成hash和decode，但不能提前假定此节省。

固定3584的CPU1.1434s只来自一个ledger/32scene，含单次import，不是32个ledger，也不是全量吞吐实测。资源Agent提案支持以下有限拓扑；本Agent没有派发或运行它：

| 项目 | 预建议硬上限 / 拓扑 |
|---|---|
| CPU | 1协调进程＋8 worker，4个获准host各2 worker，每worker线程1；建议child-inclusive CPU合计≤9600s，独立登记后才能执行 |
| GPU / sampler / scorer / optimizer | 全部0 |
| wall | 整个调用≤1200s，含索引/启动/归约；达到限额写明已完成和待完成索引后退出，不自动重试 |
| 输入逻辑读取 | 所有host合计≤32GiB（34,359,738,368B），hash检查与load重读均计入；统一预算预付，不能每worker各领32GiB |
| RSS | 全进程合计≤8GiB，按完整boundary流式读取，bounded queue |
| 输出 | 所有host合计≤128MiB（134,217,728B）：紧凑索引、聚合/唯一scene统计、失败与成本；不复制原记录 |
| I/O | 完成两遍payload读取所需最低平均23,481,645B/s；建议总读取限速≤32MiB/s，实际存储压力由Systems核对，不争用真实训练 |

按已stat的字节大小确定性平衡成8个不相交分片，每片512个ledger、约1.761GB payload，保持完整boundary与G4。索引排序规则、每片ledger列表/预计字节数/hash在读取payload前冻结；给每片发放读/输出额度且保留协调份额，保证总预算。最初每worker的首个ledger作为同一次分析中的吞吐窗口，包含在总预算和结果人口中，读过即保留统计，不重复读取profile。1200s是停止界限，容量实测不保证反序列化和事件分析能在该时限完成。身份/hash变化、消费映射不一一对应、事件归约不符、未知schema、超预算或资源监督失效均停止；保留部分完成状态，不跳过异常继续声称全量完成。

必要反例与解释边界：

- 若全C1出现大量安全分项变化、EP-only并不占主要非零组，H被反驳；不能继续用3584单batch解释全程梯度。
- 若事件非空但终局同分对不存在，不能检验过程信息损失；若有时序差异，仅证实标签有区分能力，不证实晚事故更好，更不能按物理时刻直接分配Flow步。
- 若完整C1仍极少事件，说明这个已执行训练流的支持不足；不是原策略全训练分布或真实世界安全保证。保留曝光、唯一scene与policy时间窗，不从历史均值反推iid事件概率。
- 即使支持H，不自动更改BC权重、扩大batch/G、解冻Qwen、加新reward或训练预测器。C5唯一组合需要单独数学定义与进度权衡，过程方法需要实际执行的动作扰动方向证据，DFQL还需要共享参数适配对照。

后续交接只能给出一项研究选择。若完整账本存在足够的同终局非空异时例，最小下一个证据缺口是对这些TRAIN例验证局部命令扰动如何改变原执行后的事件时序；不能直接把异时当成训练目标。若支持仍不足，报告信息不足并终止这一升级。本文件不启动任何后续操作。
