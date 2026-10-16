# Codex多Agent启动指令

将配套研究计划 `WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md` 提供给Codex，然后发送下面整段指令。它要求实际调用独立Agent，而不是在单线程里模拟角色。

```text
你现在是 IDayday/VLA-Drive 本轮研究的主Agent与唯一实验调度者。

执行配套文档：
WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md

先在当前上传文件/工作区定位真实文档；读取全文，核对版本为2026-10-17 v1.0。
建议将原文原样归档到：
docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md
找不到文件时请求提供，不凭聊天摘要重建另一套方案，不猜挂载路径。

本轮主线：
A. 当前/历史三视图中的细粒度时空表征学习。
B. 先用PDMS真实反馈，直接改进原Flow Matching / DiT策略。
C. 两者分别建立证据后，进行联合对照。
部署仍由VLA直接生成唯一轨迹，不增加Residual Refiner、在线scorer择优或真值规划器。

必须严格执行两条规则：

规则1：方法必须有依据。
提出非平凡架构、损失、训练或代码适配前，主动搜索相关论文全文、作者项目页、官方源码与固定版本文档。
为每项设计建立Design Evidence Card：问题、来源、章节/公式、repo SHA、文件/符号、可复用部分、新适配、风险、对照与验收。
只读摘要或README不能放行关键数学/代码实现；创新适配明确标为假设，不冒充文献已证结论。
发现原论文/代码和我们的设想不一致时，先解释并修订，不硬搬名称。
不因成熟设计有人做过而回避复用，也不把拼装组件当成已成立创新。

规则2：正式训练前必须独立双审。
未经过独立正确性审阅和实测多进程/多GPU效率审阅，不允许正式启动。
正式pilot也属于正式训练；仅允许有明确上限、权重丢弃的SMOKE/PROFILE先行。
批准必须绑定最终整合SHA、配置、数据/标签/teacher、精度、批量、设备拓扑和资源租约。
代码/语义/精度/批量等相关变更使对应批准失效。
不要只检查approve.json存在，也不要让实现者给自己签三项PASS。

一、建立隔离环境和真实多Agent工作流

1. 读取AGENTS.md、当前资源授权/禁止清单、仓库状态和配套文档第2节的源码来源。
2. 建立本轮隔离worktree和任务分支。保留旧代码、数据、权重、负结果，不改main、不force push。
3. 核实当前Codex原生subagent能力、版本、上下文和权限，按官方文档实际创建子Agent。
   不硬编码未经验证的CLI选项或模型名；工具可用时使用真实spawn/delegate/wait/close机制。
   若不可用，记录BLOCKED_MULTI_AGENT并请求启用；可继续只读准备，不能模拟独立审批。
4. 默认同时运行最多4个子Agent；所有角色不必同时存在。
5. 每个实现Agent用隔离worktree、明确owned_paths。共享trainer/框架文件只由你整合patch。
6. 子Agent不得自行租用GPU、启动正式任务或无上限递归派生Agent。

按以下角色实际分工：

A1 Evidence：
阅读全文与官方代码；优先DINO-Foresight、DINOv3、FPO、DFQL、RFO和现有PDMS。
审阅DINO空间压缩、历史输入、时间差分和policy-gradient的理论/实现边界。
输出逐项证据卡、复用清单和未核实内容。它不负责主要代码修改。

A2 SpaceTime：
实现高分辨率原生网格、稀疏位置读出、W48/96、绝对未来与时间差分。
保留teacher空间位置，不把输出query数与Qwen W数强绑定。
负责st_world模块及对应测试，不改官方评分语义。

A3 HistoryData：
实现-1.0/-0.5/0秒真实历史、view/time/ego变换、缺帧mask和teacher/cache身份。
先验证原生历史输入，再实现有预算约束的历史融合；不缓存可训练H_A/W。
负责数据与预处理接口，给出磁盘、CPU和编码成本。

A4 Consequence：
从原执行器和scorer提取道路、责任碰撞、TTC过程和原始进度。
事件记录前后必须重现原canonical指标。
解决EP候选集合依赖、事件严重性/首次时间、TTC观察时域和每worker独立scorer。
不创建新风险大全、不做MIN整轨迹资格筛选。

A5 FlowPolicy：
先核验并迁移FPO更新器，用不可微真实反馈训练原DiT，原FM锚点保留。
明确old/ref策略、时间采样、CFM权重、动作编码、代理ratio和on-policy版本。
可微GT代理/DFQL仅在方向验证后推进；不复制机器人Bellman流程。
不使用含GT的bridge端点冒充真实生成轨迹，不引入后修正器。

A6 Correctness：
不主要编写被审模块。审阅真实调用链、输入/标签隔离、数学与梯度、分布式归约、恢复和部署。
必须给出具体代码位置、反例测试、实测命令与结果。
对最终集成SHA出具CODE_CORRECTNESS_REVIEW，未解决实质问题不得PASS。

A7 Systems：
核对授权资源；profile完整模型、教师提取、rollout→CPU仿真→更新与评价流水线。
比较8卡一任务和2×4卡并发的真实总吞吐，小头可采用单卡多任务。
检查DDP/DeepSpeed累积、数据切分、CPU/BLAS过订阅、缓存、日志同步和checkpoint开销。
不得减少10步、标签、数据或冻结研究对象来假装等价加速。
输出RESOURCE_EFFICIENCY_REVIEW与唯一可执行资源计划。

A8 Evaluation：
预登记主要差值和共同终点；保持完整人口与分项、零分双向转移、log级统计。
区分时空能力、后果预测、梯度方向和实际规划；关注高分退化和停车捷径。
不根据Navtest挑选超参数或事后只报最佳epoch。

你作为A0负责整合、取舍和正式launch，不替代A6/A7独立审查。

二、工作波次与交接

第一波并行：证据检索、代码映射、历史/缓存资产、PDMS事件合同。
收集结果后，先锁定公共ModelInputs/TrainingTargets、shape、时间和梯度合同。
每项交接包括任务ID、起点SHA、owned_paths、证据卡、测试、资源限制、输出与阻塞。

第二波并行：时空模块、历史数据、事件提取、FPO迁移。
及时交给独立审阅Agent检查，不等全部写完才发现公共合同错误。
第三波：整合到不可变SHA，执行正确性测试、真实多GPU/多进程profile和审批。
第四波：仅由你按已批准注册文件启动正式实验；其他Agent分析已完成产物和下一项证据。
运行期间不热改源码、teacher缓存或reward定义。

三、按文档执行的最小实验序列

W线：
1. S-P0…S-P3：教师分辨率/目标网格/W容量逐项拆分。
2. 固定空间配置后T0…T4：当前、历史、绝对未来、时间差分。
3. 视频或几何teacher只有诊断支持时选一个，不做全组合。

C线：
1. canonical后果提取及其与原分数的等价性；先不训练预测器。
2. C0-BC、C1-PDMS、C2-D、C3-C、C4-T，使用同一已审查更新器。
3. 根据训练分布和分项结果预登记一个C5组合，明确进度保护与限制。
4. 验证真实生成轨迹上的可微代理方向后，才比较DFQL/RFO类更新。
5. 真值明确有效后才预测最少后果，保持部分真实仿真复核。

两线独立推进，一个分支阻塞不应让另一条停止。
联合阶段只做同输入/同结构的J00/J01/J10/J11，不把历史输入收益算成未来监督收益。
不得默认每组100k；严格遵循文档的分波次预算、入围和完整确认要求。

四、正式训练审批

每个run生成：
DESIGN_EVIDENCE
CODE_CORRECTNESS_REVIEW
RESOURCE_EFFICIENCY_REVIEW
RUN_APPROVAL

RUN_APPROVAL必须指向同一集成源码、参数和数据身份。
通过后自动按预算执行，无需无理由等待用户再次批准；仍遵守实际资源权限。
未通过时返回具体阻塞给owner，修复受影响范围，不重新做无关历史全仓审计。
仅文档更新不重做所有profile；标签/精度/计算路径变动必须重新验证相关部分。

正式入口用check_formal_approval阻断无批准启动。
任何Agent不得通过--skip-review之类后门绕过用户规定。
正式运行至少记录scene曝光、optimizer更新、候选数、policy_version、teacher身份、精度和成本。
SMOKE/PROFILE权重丢弃，不能充作正式起点。

五、资源与安全

先读当前授权，不能恢复vla-zt2或其他撤销主机。
不要恢复MIN、G、r1/V12、nuScenes、Bench2Drive或旧观察器。
两台8卡A800是历史资源信息，不代表此刻所有GPU可用。
GPU采样与CPU仿真应流水并行，但有队列背压和policy staleness限制。
共享缓存单写者，评分worker独享scorer，CPU-only阶段不得占住GPU context。
完整计入child CPU、GPU process与allocation、cache/编译/保存成本，不能用wall×worker伪造实测。
达到上限在安全optimizer边界保存并返回，不无限跑到超过旧分数。

六、交付与研究判断

阶段性提交代码、独立审阅、实测profile和真实结果；只推本任务分支，不合并main。
不上传数据、模型、cache、私有场景图和密钥。

最终回答：
- 细空间目标与更多W是否分别有价值？
- 历史本身与未来监督分别改善了什么？
- 时间差分是否保留更多运动信息并提高规划？
- 哪些真值后果真正改变了原DiT生成，而非只多了一个loss数值？
- 过程时间究竟通过什么机制被使用？
- 改善是否以安全/进度/高分场景退化交换？
- 后果预测替代真值后还有多少增益？
- 世界理解与后果训练是否互补，额外成本是否合理？

状态分开：IMPLEMENTED、VERIFIED、PILOT_COMPLETE、FORMAL_COMPLETE、NOT_RUN、BLOCKED、FAILED。
不把论文有先例作为否定理由，不把动机好作为效果证据。
现在先实际创建第一波独立Agent并开始核实证据和接口，再进入实现、双审与登记实验。
```
