# 场景条件下的保持／必要修复

本轮从归档 `e588fdc0b19a92922c1ec59aa5acba48a4683769` 建立分支
`feature/scene-conditioned-minimal-repair-20261015`。冻结原始GT S0 100k，
复用已经核验的紧凑W/H_A和FP32十步q0；17个原生checkpoint文件已重新核对哈希。
S0训练源码是 `1493deda247107efe076b9294863a6753391298a`，
原v4标签的小头训练源码是 `58efe6493ecde48aa14543278f96a41b79980819`，
现已因正确性问题保存并暂停，不再作为正式结果。修正源码冻结为
`0471af96e4f769eb2b31b787a68c57be24bebee0`。报告提交不替代实际执行身份。

## 最终状态

[完整结论与交付](FINAL_REPORT_ZH.md)汇总了12个共同24epoch正式小头、78项开发观察、12项关系干预、24项局部行为评价和最终Navtest。原六组及关键seed43共9头；另3头是保留独立身份的一次限定KEEP/REPAIR平衡对照。

最终12,146场景Navtest：S0 **89.52787**，LOG_NONE **89.29093**，原MIN_NONE **89.53612**，平衡MIN_NONE **89.58957**。最后一项相对S0的95% log-cluster区间为[−0.19539,+0.34547]pp，且自然开发下降；没有稳定净收益结论。[最终Navtest](FINAL_NAVTEST_ZH.md)、[限定对照](BOUNDED_REPAIR_RESULTS_ZH.md)、[数值结果](final_navtest_results/SUMMARY.json)与[逐场景评分索引](score_rows/INDEX.json)均已归档。

未来GT未显示稳定超过CV的价值；一次训练充分性修复后结论未改变。正式阶段B按条件规则记为 **NOT_RUN**，其P_A/R_B代码和真实小集链路已完成，不将smoke当正式DIFF结果。全部新研究队列退出、临时RAM镜像释放；共享原数据和权重保留。最终源码身份见[SOURCE_VERSIONS_FINAL.json](SOURCE_VERSIONS_FINAL.json)，最终命令见[FINAL_COMMANDS_ZH.md](FINAL_COMMANDS_ZH.md)。

以下保留实现演进与验证细节。

## 正确性修复与重跑证据

继承的样条起始导数使用 `[norm(v), 0]`，没有保留合法当前速度的横向分量。
固定256训练场景的向量误差P95为0.3993m/s，最大0.7251m/s。因此九个旧v4
小头全部在optimizer边界保存并暂停；旧标签、权重、成本及负结果保留，禁止把它们
直接恢复后称为修正版本。旧开发评价调度器已停止，未改动G2或无关任务。

修正使用显式版本 `measured_vector_v2`，保留完整初始速度向量与当前车身姿态，
旧默认分支用于历史复现。23项相关测试通过；真实64场景、640条计划复核中，旧查询
及有效性mask逐位复现，新版初始速度/姿态误差为零，原生q0不变。
新目标使用独立 `targets_*_v5`。corrected v5完整B折（fullB）48,881场景与开发集
dev1696均已COMPLETE，见 [fullB完成收据](/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5/COMPLETE.json)
和 [dev1696完成收据](/mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5/COMPLETE.json)。
fullB收据记录全量身份及SHA256核验完成。另有64、256、4096场景诊断；4096场景的
40,960条计划中KEEP5914、REPAIR163，目标资格14.8364%，没有新增终点位移低于1m的阈值跨越；这不是完整停车定义。
这些覆盖数值来自固定4096训练诊断集，不是fullB总体统计；KEEP仍可与日志GT显著不同。
9个v5 formal小头均已从对应seed的共同随机初始化完成24epoch、9168updates：seed42六组
LOG_NONE/CV/FUT、MIN_NONE/CV/FUT，seed43关键三组LOG_FUT/MIN_CV/MIN_FUT。
实际训练源码仍为 `0471af96e4f769eb2b31b787a68c57be24bebee0`，九组共同终点与72项开发评价现已完成，见[阶段A结果](ROUND1_STAGE_A_RESULTS_ZH.md)。

九组曾在原生optimizer边界保存完整状态，从字节相同的RAM特征镜像恢复并完成训练。
rl-zt2的两组迁至tsinghua-zt4；其余保留原主机。共享存储中的原始文件没有改变。
镜像字节与元数据一致性见 [RAM_CACHE收据](/mnt/project/minimal-repair-artifacts/20261015/RAM_CACHE_training-vla-zt_v10.json)，
迁移请求见 [RAM_RESUME_REQUESTS_v11.json](/mnt/project/minimal-repair-artifacts/20261015/RAM_RESUME_REQUESTS_v11.json)。
[resume_RAM_v11/ACCEPTANCE.json](/mnt/project/minimal-repair-artifacts/20261015/resume_RAM_v11/ACCEPTANCE.json)
已验证真实64场景、48updates及第7步中断后共享存储→RAM恢复的模型、optimizer、scheduler、
RNG、数据位置和曝光计数逐位一致；该小集先验证执行等价；九组实际完成证据另见共同终点审计。
跨主机48步恢复也逐位一致，见 [跨主机恢复验收](evidence/RAM_CROSSHOST_RESUME_ACCEPTANCE_v11.json)。
三组本地seed43任务进一步使用相同字节的标签RAM镜像；标签位置切换的完整恢复验收见
[标签RAM验收](evidence/RAM_LABEL_RESUME_ACCEPTANCE_v12.json)。其余已读热的任务不再中断。
迁移保持原始数据、逻辑特征/标签身份、数值模式及冻结训练源码，逐run进度以原生状态收据为准。

完整B折统计已完成：488,810个预构造计划中KEEP76,836、REPAIR1,930、UNCERTAIN406,460、
UNSOLVED3,180、TASK_UNCERTAIN404；有效比例16.1138%，8,456个原场景至少有一个有效计划。
修复原因包括道路1,255、已标注对象交叠194、运动学502，类别可重叠；原生q0仅21项REPAIR。
这些是计划池统计，不是实际训练曝光，后者由各run逐步计数。
另有1项目标终点位移从1.0205m变为0.9972m；原计划最后半秒速度已约0.033m/s，目标约0.031m/s。
这是既有低速计划的阈值跨越，不能称为新增完全停车，见[专项核查](evidence/LOW_PROGRESS_TARGET_AUDIT_v12.json)。
见 [完整B折目标质量](evidence/TARGET_QUALITY_FOLDB_v5.json)。A折同版本52,711场景也已完成并通过SHA核验；其CPU进程成本39.785小时单列，
没有提前放行阶段B正式P/R训练。

新版11个真实小集训练链路、48步中断恢复与4图在线/缓存对齐均已通过；在线每场景
一次S0视觉、一次语言前向和一次原十步proposal，q0/q_final逐位一致。
此前的v4训练和目标诊断仅为保留的工程历史，不是修正后方法的主结果。

本轮实际新增了待修正 `plan_in`、独立恢复输入/标签schema、局部约束检查、
有预算上限的SLSQP目标生成、六组匹配修正训练、P_A→R_B接口、自然/恢复双评价、
camera-only部署和失败转移分析。全部query及残差基准由实际输入计划决定；
原始干净q0没有作为额外答案输入。旧E/P/R、G、nuScenes及其缓存没有改动或恢复。

## 已完成的旧v4执行证据（已被正确性修复取代）

- 旧v4几何的64、256、4096训练诊断及1696开发场景标签均已生成。完整B折
  48,881场景的四个CPU分片及全量SHA核验已完成；这些旧标签现有明确hold。
- 4096诊断场景的40,960个输入中，KEEP 5915、REPAIR 159、UNCERTAIN 34631、
  UNSOLVED 233、TASK_UNCERTAIN 22。共同目标资格约14.8291%；未知场景保留在评价中。
- 固定32训练案例和另一个分层32训练案例均通过原canonical执行检查。后者修复4个
  零分、没有新增零分，也有6个小幅下降；这是离线目标质量检查，不是模型开发成绩。
- 21项相关测试通过。真实小集连续48updates与第7步中断恢复的模型、optimizer
  moments和数据边界逐位一致。标签替换不改变camera-only结果。
- 六个阶段A、两个预测器及三个预测关系修正器已经完成真实小集24epoch链路测试。
  这些smoke不代替正式完整人口训练。
- 4个真实当前图像场景的在线与缓存q0/q_final逐位相同；每场景只有一次S0视觉、
  一次语言前向和一次原十步proposal。FP32热模型时延中位约1.048秒，不含图像加载；
  这是smoke头验证，尚不是正式终点部署性能。

## 正式比较和状态解释

沿用原1176训练logs两折：A=52711、B=48881；开发1696场景独立。阶段A六组
LOG_NONE/CV/FUT、MIN_NONE/CV/FUT共用B、输入分布、目标资格、结构与24epoch配方。
seed42六组、seed43的LOG_FUT/MIN_CV/MIN_FUT共9个v5正式头已完成；6/12/18/24epoch的
自然输入和恢复输入各自评价，共72项固定开发观察均已完成。旧v4九组保持正确性hold，v5使用独立run身份。
[共同终点审计](evidence/STAGE_A_COMMON_ENDPOINT_AUDIT_v13.json)核对九组原生COMPLETE、初始化、9168次更新边界及同seed曝光完全一致。

共同目标资格仅控制哪些计划贡献监督loss；关系support是另一条输入通道。CV/FUT的
`common_valid`是CURRENT、CV、FUTURE有效性mask的交集，包含未来标注支持信息，
两者使用相同support；NONE不接收该mask。因此FUT−CV只识别共同特权support条件下
关系数值的增量，FUT−NONE还混有support信息增量，不能解释成纯粹的未来关系价值。
CV/FUT均为特权探针。局部可接受性不保证导航路线、信号或完整交通正确性。

正式阶段B依据完整阶段A与一次有限复核，记为NOT_RUN；具体证据见[阶段B处置](evidence/STAGE_B_DISPOSITION_v15.json)。全部P/R接口已有真实smoke，不作为完整P_A/R_B结果，也不由区间跨零直接否定未来信息。最终三组camera-only Navtest均已完成。
[限定的KEEP/REPAIR平衡学习对照](BOUNDED_REPAIR_RESULTS_ZH.md)亦已完成，训练源码`d2996601429044e3ef3b71055a4ca546a3ca928e`。
它不是主六组定义的一部分，也不覆盖原结果。历史实际注册与调度见
[STAGE_A_V5_TASKS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json)、
[STAGE_A_V5_RUNNERS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_RUNNERS.json)、
[STAGE_A_V5_EVAL_NATURAL.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json)、
[STAGE_A_V5_EVAL_RECOVERY.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json)
及 [STAGE_A_V5_EVALUATION_RUNNERS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVALUATION_RUNNERS.json)。
完成状态以对应run/评价的 `COMPLETE.json` 和后续共同终点报告为准。

资格低的主要已知原因是消失track之后的保守未知可达区域。固定256训练场景中224个
受到该项影响，类别可与其他未知原因重叠。没有为了覆盖率把未知区域改为空闲。
v5固定4096诊断集仅163项REPAIR，其中17项涉及已标注障碍交叠，未来修复信号稀疏，
见 [TARGET_QUALITY_4096_v5.json](evidence/TARGET_QUALITY_4096_v5.json)。全部开发场景仍保留在评价分母中；
低覆盖及区间跨零均不能支持“未来信息无价值”的结论。本轮KEEP仅表示所选标注条件下的
局部可接受，不是路线、完整交通规则或现实安全保证。

## 身份、资源和记录

实现及局限见 [METHOD_AND_IMPLEMENTATION_ZH.md](METHOD_AND_IMPLEMENTATION_ZH.md)。
聚合数值、恢复、真实部署、标签和来源收据在 [evidence](evidence/)。
修正版本运行命令参见 [CURRENT_VECTOR_COMMANDS_ZH.md](CURRENT_VECTOR_COMMANDS_ZH.md)。
实际无损承接、临时缓存与终点诊断命令见 [EXECUTION_AND_RESUME_ZH.md](EXECUTION_AND_RESUME_ZH.md)。
原数据、权重、地图图示和cache均不上传Git。

持久产物只在 `/mnt/project/minimal-repair-artifacts/20261015`。标签和训练使用已授权的
vla-zt、rl-zt2、tsinghua-zt5、tsinghua-zt4；开发评价另接入tsinghua-zt/zt2/zt3，
相应Qwen/评分环境与nuPlan源码哈希已核验。允许按余量共享GPU，分片数由实际余量决定。
G2所在vla-zt2不受本轮调度干扰，已撤销限时授权的服务器不连接。
CPU标签和GPU导出/CPU评分并行；等待必要依赖的评价进程不占GPU。
有限上限为64 GPU进程小时、160 CPU进程小时、新存储40GiB，历史S0的483.481 GPU小时
作为共享基础来源单列，不计作本轮新增开销。

历史失败均保留：早期空框、时间/heading连续性、旋转框顶点插值问题已有对应修复；
不合格v3标签有明确hold。B折收尾等待脚本曾因缺少右括号退出，未放行任何训练，
原脚本和日志保留；修正版本先通过语法校验，再调用不变的全量SHA核验入口。

最新资源变更：用户撤销vla-zt2，已更新全局禁用和任务派发记录；既有G2按原生89,574步检查点退出并单独迁移，不改变本轮实验结论或计入小头成本。见[撤销记录](evidence/VLA_ZT2_WITHDRAWAL_REQUEST_v16.json)。
