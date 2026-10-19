# 后果反馈研究：实验设计、已完成结果与在训假设

快照：**2026-10-19 00:23:41 UTC**。本页汇总当前 ST 后果研究；机器可读状态见 [CURRENT_STATUS.json](CURRENT_STATUS.json)，来源与校验值见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。快照中的 RUNNING 不替代未来的现场核验。

**当前科学结论：五组共同4096步实验已完成，没有建立实用的留出PDMS增益。已定位分项反馈稀疏和一个真实批次上的policy梯度尺度问题；唯一新增归一化配方正在训练，效果尚未测定。**

## 1. 研究问题与不变部分

要检验的是：官方环境产生的不可微真实后果反馈，能否直接改善原Flow Matching / DiT的生成策略。训练保留原GT的BC/FM锚点，用FPO的CFM代理目标接收真实评分；不把NumPy分数直接当成可微loss。

部署保持原输入和十步Euler生成的**唯一轨迹**。四候选采样只发生在策略训练阶段，没有部署Scorer择优、生成后修正器或GT规划器。共享Qwen冻结，训练原动作侧参数。本研究不属于历史Query专家/Scorer或MOPD实验。

原始合同及文献/代码核验要求：

- [完整研究计划](../../../../docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md)。
- [多Agent执行指令](../../../../docs/research/CODEX_MULTI_AGENT_START_20261017.md)。

用户已暂停本campaign的特征、教师维度、历史/未来表示路线；历史产物保留，当前不自动续跑这些路线。

## 2. 已完成的C0–C4共同端点

五组实际训练源码为 `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`。从同一个已训练父模型开始，匹配样本顺序、学习率和数值合同：

| 项目 | 固定定义 |
|---|---|
| 训练人口 | 101592 unique TRAIN scenes，1176 logs |
| 实际曝光/组 | 131064 scene presentations，1048512 BC repeat presentations |
| 更新预算 | 4096 optimizer updates，seed42 |
| 批量 | world4、microbatch4、accumulation2、global32；一个24-scene尾batch |
| 学习率 | warmup100，peak1e-5，cosine到5e-7，共同4096时间轴 |
| 数值 | BF16执行，FP32 master/Adam，ZeRO2；绑定的FP32-master评估、TF32关闭 |
| 训练候选 | 每场景G4，old/current各CFM Monte Carlo50，chunk5；BC repeat8 |
| 固定评估 | DEV1696 scenes /16 logs，各组requested=valid=1696，invalid=unresolved=0 |
| 统计 | scene加权差值，按log配对bootstrap2000次，seed20261017；单训练seed |

原GT/BC在全部五组保留；C1–C4分别增加总PDMS、道路、责任碰撞或TTC反馈。每个反馈组实际产生524256条官方训练评分呈现，不将调用数或重复曝光称为unique场景数。

分数为百分制，差值为**百分点**：

| 组 | 反馈 | PDMS | 相对C0 | 95%配对区间 | 零分数 |
|---|---|---:|---:|---|---:|
| C0 | 仅BC/FM | 91.138335 | — | — | 45 |
| C1 | 总PDMS | 91.185294 | +0.046959 | [-0.003235,+0.092123] | 45 |
| C2 | 道路违规 | 91.134959 | -0.003375 | [-0.003756,-0.002901] | 45 |
| C3 | 责任碰撞 | 91.170085 | +0.031750 | [-0.003913,+0.064697] | 44 |
| C4 | TTC违规 | 91.136605 | -0.001729 | [-0.002674,-0.001076] | 45 |

C1修复2个TTC失败，C3修复1个DAC失败；五组NC逐场景不变。不能把C3的总分变化解释成责任碰撞能力改善。全部反馈组没有新增零分、高分到零分，亦未触发预登记停车代理；这不构成未知场景安全保证。

C1/C3区间跨0；C2/C4的负向差值虽区间不跨0，效应也很小。当前数据不支持宣称后果训练已经稳定有效，也不支持从一次筛选否定所有后果学习方法。上述评估是固定DEV，**不是Navtest**。

[机器可读主表](COMPLETED_C0_C4_RESULTS.csv)；[完整分项、退化、停止代理和成本报告](../../20261018/consequence4096_complete_results_v1/RESULTS.md)。

## 3. 已完成机制诊断与解释边界

### 3.1 反馈是否提供可比较的监督

原C1完整训练记录包含80933个非零优势scene呈现，其中79586个仅有EP/进度差异，占98.34%。分项组更稀疏：C3仅39/4096步有非零优势，C2为713步，C4为495步。这度量同一场景候选之间的差异，不能作为失败或碰撞总数；全部候选同样失败也可能产生零优势。

### 3.2 policy梯度相对BC有多强

固定旧C1 boundary3584的真实32场景，单独backward并做完整向量统计：BC L2为0.1620325、policy L2为0.0000993534，policy/BC约**0.0613169%**，余弦-0.0130164。这个批次的非零比较均为EP-only。

它支持继续检查尺度问题，不能代表全部训练梯度、Adam实际更新或安全梯度，更不能直接预测PDMS。该诊断数学结果完整且经独立复核；外围supervisor在进程退出观察窗口发生的 `Missing live RSS counter` 失败仍保留，不改写为整体执行PASS。

[数值结果和完整边界](../../20261018/gradient_balance_correctness_v1/COMPLETED_V3_RESULT_REVIEW.md)。

### 3.3 道路可微代理是否值得扩大

固定32个实际失败案例中，2个满足完整稳定方向定义，其余30个undefined。负方向在2个定义良好案例均修复DAC，正方向0个，随机方向1个。修复发生在毫米级执行道路边界，主要总分跳变来自DAC门控；不能据2/2宣称总体修复率100%。

该有限诊断已关闭，未批准扩大proxy/DFQL训练。[原始决定](../../20261018/road_failure_direction_lead_decision_v1/DECISION.md)。

## 4. 当前唯一新增正式配方：rollout尺度归一化

**假设：**保留场景内中心化，在同一old-policy完整optimizer rollout的所有rank上统一计算标准差，能否使原本幅值很小的真实反馈在学习中发挥作用。

定义：保留原FP32 `A = reward - scene_mean(reward)`，分母使用全rollout FP64两遍统计、correction1、epsilon1e-10；不再次对分子做全局中心化。零优势保持零，BC/policy权重均1。不改变原GT、四候选、MC50、repeat8、样本顺序或单轨迹部署。

这是一项训练定义变化。旧训练数据上的scale中位数约909.62；EP-only层和安全可区分层的scale中位数相差约78.33倍，可能改变不同update的相对重要性。因此不能称其为无损修复，也没有预先断言它改善安全。没有看到DEV结果后再改epsilon、cap或奖励权重。

[证据与风险决定](../../20261018/rollout_scale_evidence_decision_v1/DECISION.md)；[实际解析配置](ROLLOUT_STD_RESOLVED_CONFIG.json)。

正式run为 `C1_rollout_std_fulltrain_seed42_official_4096_v1`，运行源码 `751fcfa2b41549ae16f7b3204aef8dd18b912016`。原S0初始化及fresh matched optimizer；任何SMOKE/PROFILE权重均不进入正式初始化。已绑定独立研究依据、正确性和实测效率批准。

截至本快照：**1210/4096步**、38720次scene呈现、154880次官方评分，最近完整native为1024步。最近100步loss/gradient有限，抽样FP32 master更新全部验证通过；最慢rank常规update mean15.401秒、p50 15.327秒、p95 16.441秒，排除checkpoint、启动和外层调度。它们是训练执行数据，尚无该配方的DEV结果。

固定判断点仍是4096步、DEV1696及原C1/C0对照，同时检查NC/DAC/TTC、EP、零分、双向退化和停车行为。不用中间DEV选择系数或继续预算。实际终点checkpoint和新资源租约存在后再完成绑定，不提前填写评估成功。

## 5. 成本与恢复资格

已完成的独立PROFILE为240个真实更新，使用原完整TRAIN sampler的前7680个unique场景。两个20预热+100稳态窗口中，199个正常update mean15.350秒、p50 15.338秒、p95 16.286秒；周期checkpoint耗时14.182秒。完整driver占用4卡3931.423秒，合4.368 allocated GPU-hours；waited-child CPU5.919小时。allocation不是GPU kernel busy时间。

测得主要区域为官方反馈约5.68秒、当前BC/FPO backward+optimizer约3.49秒、old CFM约2.37秒。归一化约4.5毫秒，嵌套在反馈区域中，不相加成重复工作量。实测GPU显存峰值17.68GiB，进程RSS合计峰值118.10GiB。[完整效率报告](../../20261018/rollout_scale_formal_systems_plan_v1/final_efficiency_v1/README.md)。

真实四rank恢复对照：连续A4步，与新B2步→新进程恢复2步，得到 **EXACT_REPLAY**。四rank逻辑模型/buffer、FP32 master/Adam、client/cursor、RNG、后续IDs、LR、loss和rollout审计一致。累计8次更新、1024次评分、6次训练器启动（含历史失败），零重试。BC noise/time未直接存储，完整FP64 update delta/cosine未计算，不能追加这些项目的逐位声明。

历史冷启动、ownership、序列化和退出清理失败完整保留；没有重跑已完成A。最终实际owned进程清理检查为0残留。[真实恢复闭环](../../20261018/native_resume_completion_lead_v5/FINAL_OUTCOME.md)。

## 6. 发布与后续边界

本次提交发布实验设计、实现/审阅历史和小型证据索引。权重、原始图像、metric cache及训练缓存保留在项目存储，不上传。外部资产路径是实际证据定位，仓库链接与内容hash用于追溯，不能凭路径存在替代身份核验。

正式训练继续原run，未被本次文档整理暂停、迁移或重置。当前仅等待既定终点的科学判定；没有新开Navtest、没有更新ACTIVE、没有恢复特征研究，也不扩张seed或奖励配方搜索。
