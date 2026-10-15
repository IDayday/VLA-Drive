# 场景条件下的局部保持与必要修正：实际实现

本轮冻结原始GT训练的S0 100k。共享基础训练源码为1493deda247107efe076b9294863a6753391298a，重核17个checkpoint文件；不从旧E/P/R继续训练，不更新S0，不恢复G或nuScenes。

**版本状态：** 九个旧v4正式头已安全暂停并保持正确性hold；其标签、权重和结果仅保留为历史。
样条起始速度曾把实际向量投成 `[norm(v),0]`，不满足本轮边界合同；现已在
`0471af96e4f769eb2b31b787a68c57be24bebee0` 中增加 `measured_vector_v2`，
新目标单独使用v5路径。新模式的实际速度向量
与t=0车身heading分别精确保留，曲线、query、求解器与密集验收共用同一模式。
真实64场景验证初始误差为零、原生q0不变、旧模式逐位复现。不能将旧头承接到
新输入分布后混作公平主比较；全部对照已从共同随机头起点重新训练并完成共同24epoch。

corrected v5完整B折（fullB）48,881场景及开发集dev1696均已COMPLETE，见
[fullB完成收据](/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5/COMPLETE.json)
和 [dev1696完成收据](/mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5/COMPLETE.json)。
9个v5 formal小头已完成共同24epoch与9168updates：seed42的LOG_NONE/CV/FUT、MIN_NONE/CV/FUT六组，
seed43的LOG_FUT/MIN_CV/MIN_FUT关键三组；共同终点和72项开发评价已完成，见[阶段A结果](ROUND1_STAGE_A_RESULTS_ZH.md)。
九组正式训练曾通过原生optimizer安全边界迁入字节相同RAM特征镜像，现均已完成，
见 [RAM迁移请求](/mnt/project/minimal-repair-artifacts/20261015/RAM_RESUME_REQUESTS_v11.json)。
[RAM_CACHE收据](/mnt/project/minimal-repair-artifacts/20261015/RAM_CACHE_training-vla-zt_v10.json)
记录镜像字节及元数据完全相同；
[resume_RAM_v11/ACCEPTANCE.json](/mnt/project/minimal-repair-artifacts/20261015/resume_RAM_v11/ACCEPTANCE.json)
记录真实64场景、48updates、第7步中断后共享存储→RAM恢复的全部原生状态逐位一致。
这是小集恢复验收，正式迁移的逐run起点另见 [原生承接收据](evidence/STAGE_A_RAM_NATIVE_CONTINUATIONS_v11.json)。
跨主机恢复和共享标签→RAM标签恢复均通过48步完整状态逐位对照；三组本地seed43使用标签RAM镜像，
已读热的远程任务不再中断。原始数据、逻辑特征身份、数值模式
与上述冻结训练源码保持不变。

## 输入与目标

`CurrentFeatures`仅含原HA/W、合法当前状态和车身；`PlanInputs`显式提供真正待修复的物理计划和编码计划。所有query及残差基准使用它。旧ModelInputs的q0字段仅在受检包装中承载plan_in，原始干净q0不另传。线上plan_in就是原S0的十步采样结果。

每scene有原生、左右平滑法向偏移、两种时间偏移共5种输入，两个场景绑定q0种子。24epoch每原始scene恰好12次原生、6次横向、6次时间输入；新增变体不增加统计独立场景数。偏移幅度0.25/0.75/1.25m的训练侧预登记概率0.5/0.4/0.1。训练输入时间warp固定首尾，最大中间偏移0.5s；不声称它覆盖任意终点进度错误。求解器另允许0至0.8s的受限终点延迟，始终不越过4s日志范围。

复用CubicHermiteSpline、原query生成、原动作编码及两层512维CrossReadout。日志框经过原坐标与类别转换到ego(t0)，道路使用原地图提取。与已标注可靠对象的距离/重叠、传感器支持的已观测空闲分别保存。缺轨迹和高度不可靠对象保留unknown；未把无回波当空闲。中间时刻的框采用中心、尺寸、模π方向插值，修复了直接插值顶点导致旋转框缩小的错误。

目标求解仅含位置、姿态和时间改动代价，不读取ego未来GT或PDMS。SLSQP最多3起点、每起点100迭代、每输入3秒；只在明确违反且标签充分时运行。求解内0.1秒车身多边形检查，返回8点计划重新样条后在0.05秒独立检查。KEEP原物理轨迹直接返回，不调用求解器。REPAIR必须通过独立验证，成功status本身不够。无解、歧义或不确定不伪造KEEP。

这里的KEEP只是本轮已标注局部约束可接受，不是交通规则、路线、未标注障碍或连续时间安全认证。负道路距离使用多边形Hausdorff代理，负障碍距离使用交叠面积平方根代理；符号和可行性用精确GEOS covers/intersects核验，不宣称精确穿透深度。

实现中的 `route_scope_ok` 是相对输入计划的局部改动范围：最大位置变化2m、heading变化
0.4rad、沿原计划终点方向的进度比例至少0.65。它不是车道图或导航路线验证器。
这些边界限制大幅换模态和无理由停车，但不能证明所有近距离岔路都保持正确路线；
原生输入本身的路线正确性也没有独立GT认证。本文据此只报告所选标注条件下的局部
几何可接受性，并将canonical方向、进度和新增失败另行评价，不能把KEEP等同完整驾驶正确。

## 固定比较与真实人口

沿用1176个训练logs的原两折清单，A为52711 scenes，B为48881 scenes。阶段A六组按完整B折24epoch配方训练，仅共同target_eligible计划贡献目标loss；其余scene仍计曝光，最终开发评价保留全部1696场景。LOG和MIN共用同一个有效人口与final_trajectory_loss，分别使用日志GT和真实未来局部约束生成的KEEP/REPAIR目标；同一目标族内切换NONE/CV/FUT不会更换目标或资格mask。

必须区分目标资格与关系support：前者只进入监督loss，后者在CV/FUT特权探针中作为输入。
`common_valid`取CURRENT、CV、FUTURE三种关系有效性的交集，因而包含未来标注支持信息；
CV和FUT使用完全相同的support，NONE则不接收GT关系或support mask。因此FUT−CV估计的是
共同特权support条件下关系数值的增量；FUT−NONE同时包含support信息增量，不能当作
纯粹未来关系价值的识别。CV不应被描述为无未来信息的可部署基线，CV/FUT均为特权探针。
这是当前比较的解释边界，不构成六组目标资格不一致的证据。

六组是LOG_NONE、LOG_CV、LOG_FUT、MIN_NONE、MIN_CV、MIN_FUT。seed42全部比较；seed43确认LOG_FUT、MIN_CV、MIN_FUT。只将新增头seed称为头复跑，S0仍是同一个底座训练seed。

旧v4头源码为58efe649，现已暂停；修正版本的执行源码为上述0471af96完整SHA。配方保持FP32、TF32关闭、batch128/micro16、lr1e-4、AdamW wd0.001、clip1，一epoch warmup后cosine到0.1LR。每组计划9168个数据更新边界，模型有效更新及KEEP/REPAIR曝光另记。全无资格batch不调用AdamW，避免制造有效监督；scheduler仍按共同数据曝光前进。

阶段B的P_ABS_A/P_DIFF_A只可从A折随机小头起点训练，R_B第一步开始读取相应冻结P_A的真实预测。入口拒绝全量P、错误标签版本、同fold以及未完成24epoch的P。正式B在完成阶段A和一次有限训练修复后记为NOT_RUN，依据见[处置记录](evidence/STAGE_B_DISPOSITION_v15.json)；已完成的小集P→R链路不作为正式结果。

## 截至当前的corrected v5执行证据

固定4096训练诊断场景共40960个输入计划：KEEP5914、REPAIR163、UNCERTAIN34631、UNSOLVED230、TASK_UNCERTAIN22；资格比例14.8364%，653场景至少有一个有效计划。163项修复中，道路违反109、障碍交叠17、运动学违反39，原因可重叠。修复输入包括5个原生、113个横向、45个时间扰动。没有新增终点位移跨1m阈值的情况；该指标不是基于速度的完整停车检测。KEEP与日志GT的平均ADE约0.756m，说明目标不是把所有可接受计划复制成GT；REPAIR目标对GT平均ADE约0.901m。数值见 [TARGET_QUALITY_4096_v5.json](evidence/TARGET_QUALITY_4096_v5.json)，仅描述固定4096诊断集，不代表fullB总体覆盖统计。

覆盖不足和修复信号稀疏是本研究的实质限制，尤其仅17项修复涉及已标注障碍交叠。阶段A只能检验这一有限局部约束合同，不能把FUT无增益或置信区间跨零外推为未来信息没有价值。全部1696开发场景保留在评价分母中，未知或未解场景不伪造KEEP。正式24epoch训练和canonical评价已完成，见[最终报告](FINAL_REPORT_ZH.md)；本文这些目标诊断数值仍不是部署收益，也不证明路线、信号或完整交通正确性。

完整B折的488,810条计划中，KEEP76,836、REPAIR1,930，目标资格16.1138%；8,456个scene至少有
一个合格目标。修复原因道路1,255、已标注对象交叠194、运动学502，可重叠。原生q0的97,762条
计划中仅21条REPAIR，其余1,909条来自合成输入；必须把实际自然错误稀少与合成恢复学习分开。
KEEP目标对原GT的平均ADE为0.757m，REPAIR为0.835m，目标没有全部复制GT。
修复平均最大位置改动0.395m、平均终点位移变化−0.062m；1个目标跨过终点位移1m阈值。
该计数名义字段虽是`repair_new_full_stop`，实际算子只是终点位移阈值，不表示速度严格为零；
不得据此夸大停车或安全结论。数据见 [TARGET_QUALITY_FOLDB_v5.json](evidence/TARGET_QUALITY_FOLDB_v5.json)。

23项相关测试通过，见 [TESTS_CURRENT_VECTOR_v10.log](evidence/TESTS_CURRENT_VECTOR_v10.log)。真实小集连续/中断恢复48updates结果逐位相同；4图在线与缓存q0/q_final逐位相同，每场景一次视觉、一次语言前向、一次原十步proposal。标签替换不改变camera-only输出。4图热运行模型时延中位约1.048秒，图像读取不在此计时内；这是smoke小头、非最终24epoch部署性能。恢复及在线证据见 [RESUME_ACCEPTANCE_v10.json](evidence/RESUME_ACCEPTANCE_v10.json) 和 [ONLINE_CACHE_PARITY_v10.json](evidence/ONLINE_CACHE_PARITY_v10.json)。

## 资源与成本

代码与持久产物均在/mnt/project，字节相同的临时RAM特征镜像位于各主机/dev/shm；原始共享特征库保留。允许共享GPU；只从当前授权清单派发。完整B标签由vla-zt、rl-zt2、tsinghua-zt5、tsinghua-zt4生成，未使用撤销限时授权主机，未干扰vla-zt2的G2。固定有限上限64 GPU-process hours、160 CPU-process hours、40GiB新存储。历史v4的100步真实训练profile读热后中位0.113秒/batch128、P90约0.130秒；首次读入、checkpoint和完整评价成本独立记录，不能将其当corrected v5完整B折实际耗时。

当前正式训练与评价注册使用共享产物目录中的
[STAGE_A_V5_TASKS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json)、
[STAGE_A_V5_RUNNERS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_RUNNERS.json)、
[STAGE_A_V5_EVAL_NATURAL.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json)、
[STAGE_A_V5_EVAL_RECOVERY.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json)
及 [STAGE_A_V5_EVALUATION_RUNNERS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVALUATION_RUNNERS.json)。
结合各run原生状态、RAM迁移收据及对应 `COMPLETE.json` 判断实际进度；共同终点和72项开发评价已完成，见[阶段A结果](ROUND1_STAGE_A_RESULTS_ZH.md)。
运行命令参见 [CURRENT_VECTOR_COMMANDS_ZH.md](CURRENT_VECTOR_COMMANDS_ZH.md)。旧v4注册、标签、权重、成本及失败验证作为历史保留，不作为corrected v5正式结果。

## 覆盖不足的具体来源

对固定前256个训练scene的原生seed42计划进行只读归因：224个scene被已消失track的未知可达区域影响，3个scene路面高度不足，1个缺有效帧，3个受对象首次出现的区间影响；类别可重叠，聚合证据见 [SUPPORT_ATTRIBUTION_TRAIN256_v10.json](evidence/SUPPORT_ATTRIBUTION_TRAIN256_v10.json)。保留的未知区域采用最后提供速度及固定加速度/转向界，消失时间中位3秒、P90为3.75秒，因而较远时间的支持明显减少。本轮没有为了覆盖率把这些区域标为free或修改目标版本。该保守合同及真正的动态修复稀疏性必须和规划结果一起解释，不能把小头无收益直接归因于未来信息没有价值。

历史v4的固定32个分层训练案例（16项REPAIR，另有KEEP和未知/未解案例）经canonical检查，目标使4个原零分变为非零，没有新增零分；6个案例分数轻微下降。该人为分层旧版本诊断仅为历史证据，不作为corrected v5目标质量或正式模型成绩。v5另有固定32训练案例验收，见 [TARGET_REVIEW32_ACCEPTANCE_v5.json](evidence/TARGET_REVIEW32_ACCEPTANCE_v5.json)；同样只说明有限训练案例中的离线目标质量，不代表总体收益或完整驾驶正确性。


## 一次限定的训练充分性修复

原配方有效训练曝光中KEEP190062、REPAIR2969。固定训练诊断显示稀少REPAIR仍明显欠拟合。三个MIN组新增一个独立、匹配的固定类别权重对照，权重完全来自上述B训练曝光；详见[登记与实现](BOUNDED_REPAIR_BALANCE_ZH.md)。不改变输入、目标资格、网络、部署或原主六组结果。它只检验稀少修复梯度这一故障假设，不是新增风险loss或在线门控。这一有限复核已完成，见[结果](BOUNDED_REPAIR_RESULTS_ZH.md)；正式P/R按条件规则NOT_RUN。原六组、平衡对照及最终Navtest保持独立身份。

最终三组正式camera-only权重均完成真实当前图像与缓存逐位一致性验证；全部新研究队列已退出，RAM副本已释放，持久原件保留。完整成本、计时缺口、时延边界和最终结论见[完成报告](FINAL_REPORT_ZH.md)。
