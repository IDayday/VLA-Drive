> **2026-10-25用户更新：方法有效性与性能提升只依据完整Navtest。本文既有DEV/VAL与定向TRAIN数值仅作工程、拟合和机制诊断，旧阶段价值标签不再作为当前结论。测试结果未完整前，一律待确认。**

# Qwen-Flow三条独立探索路线

本轮由主agent统一协议与资源，恰好三个研究子agent：agent_latent_policy、agent_feedback_search、agent_preservation；均不得再委派。所有路线从同一成熟CANONICAL50k完整模型（原Navtest90.513164）独立开始，而非回退到较弱IL。部署输入为当前三视图、导航和合法ego状态，原生3维源ActionDiT，单候选10步shift5 ODE。潜策略只改变root分布；其他路线保持原ODE先验。

## 当前已完成与未完成

已完成公共评分门槛、三个路线的CPU数学/恢复及真实GPU短更新资格；共同候选池28416条已实际完成，F0–F4的512TRAIN/256选择集搜索亦全部完成。正式筛选和跨路线研究结论尚未完成，不能把资格PASS当成方法有效。

- 公共32TRAIN/16选择集、G4及候选增删/重排/官方单轨对照全部通过：原生分项与执行轨迹一致，评分最大差0；ODE batch1/chunk4归一化差约1e-6。G16真实源SDE四次更新，480个LoRA张量更新、第一inner ratio误差0。峰值28.52GB。[公共证据](common_smoke_v2.json)
- 潜策略：32TRAIN/128真实候选、四次小网络更新；全部生成器参数和buffers hash不变，联合Gaussian密度/KL及实际inner恢复通过；约15.48GiB峰值。[路线与证据](latent/README.md)
- 反馈搜索：原请求曲线的加速度预筛错误已在资格期定位并修正，动态量改在实际模拟轨迹上检查；不改变官方评分。v1未测试到外部teacher，不充当该机制证据；v2确实使用不同于base的可行外部轨迹完成四次原生FM更新。中性资格probe不是正式成功标签。[实际资格](feedback/real_profile_v2.json)
- 存档/约束：真实P5四次逻辑更新、七次物理计算（包括恢复重放），480张量Adam一阶矩非零；mid-inner、后续chain/old概率/RNG逐值恢复；全部冻结参数与buffers保持。[实际资格](preservation/real_profile_v1.json)

资格权重不用于正式初始化。共同源SDE使用CPU随机发生器后搬GPU，不能传CUDA generator；早期公共v1因此失败并保留成本，修正的是适配器接口，不改源公式。[失败记录](common_smoke_v1_failure_audit.json)

## 固定协议

TRAIN全人口原101592场景/1176logs，DEV1696/16logs，Navtest12146/136logs按log无重叠。共同512TRAIN为固定自然人口随机样本，不只选搜索成功或旧hard list；DEV按场景数平衡完整log拆成选择8logs/849scene与审计8logs/847scene。诊断选择集固定256scene。两半历史均有基线观察，只是本轮配置锁定前封存审计，不能称全新盲测。

独立reference种子11001–11005建立有限采样分层；部署评价种子2101–2105，逐scene先汇总seed后按完整log bootstrap。新指令允许选择性三训练seed确认（42/43/44），不改变原有50k实验的seed42。

主奖励始终canonical v1.1原始0—1；展示乘100，.005原分数即.5 PDMS点。官方NC/DAC/TTC完全通过才能称联合可行，NC=.5不算通过。过程反馈仅用于scorer/搜索/训练目标，不进入actor。

相比历史source配方，本轮短窗明确采用自然人口、B16/G16、共同RAW全batch std floor .01、P0-P5统一关闭advantage±5截断，PPO clip仍.005、均值anchor50不变；latent的24维联合PPO clip单独为.2，不冒称同一种ratio。八步warmup为256更新的3%预算适配。完整配置和有限预算见[协议摘要](common_protocol_summary.json)、[预算锁定](BUDGET_LOCK.json)。

每在线臂64round×16scene×16candidate=16384新评分，四inner共256真optimizer更新。共同K32加独立5reference为28416条实际评分，费用单列；F0和搜索前8复用同池时同时计逻辑成本与真实新评分。S1/S2/S3各256监督更新，teacher固定TRAIN梯度校准一次；搜索、GT replay、额外FM前后向不当免费数据。P4/P5预算来自固定128TRAIN×G16的初始同核SDE，不能把ODE均值预算冒称SDE。

## 三路线对照

|路线|实验|核心隔离|
|---|---|---|
|潜策略|L0，固定scale .7/1/1.3，L1条件+priorKL，L2全局+priorKL，L3条件无priorKL|完整24维潜变量密度；生成器始终冻结；q单采样与mu-only分开|
|反馈搜索|F0 IID、F1盲结构化、F2仅标量、F3物理反馈、F4合格新颖度；S0/S1/S2/S3学生|4D平滑proposal与相同CEM；外部轨迹只作正权重源FM，不虚构on-policy概率|
|能力保留|P0、P1保守更新（漂移校准未匹配）、P2单轨、P3多轨、P4约束、P5约束+多轨|固定初始同池存档；source anchor公共不变；独立成本、投影dual、once-per-fresh|

主结论比较重新生成的单轨输出，候选oracle另表。统计能力不足、原基线未退化或学生未学动时标证据不足；不凭短训判整类路线无效。零新增失败的bootstrap零宽不等于风险为零。选择集只有8logs，稀有失败与0.2个百分点差异的分辨率有限，需要封存审计和必要确认。

当前资源为允许永久主机的实际余量，尤其vla-zt共享显存；每张物理GPU最多一个本研究模型进程。原IL、原两组RL、安全50k和时序DINO保持，未知高占用任务不驱逐。Navtest专用的vlawm-zt/zt6资源没有用于本轮训练。

## 入口和资产

私有根目录`/mnt/project/qwen-flow-three-routes-20261024`保存完整protocol、token/log清单、候选记录、所有RNG与checkpoint。Git只保存源码、配置、聚合证据、身份与复现入口；不上传权重、cache或场景图。

已实际执行：`prepare_common`、`validate_common`，各路线`*_validate`和真实`*_profile`。共同bank已完成`build_common_bank --protocol .../common_protocol.json --output .../common_bank --rank N --world 7`。route train/evaluate入口均真实存在；正式派发与完成状态将记录在同一私有根的dispatch/账本，不能从本报告快照推断实时完成。

## 第一阶段已完成的发现结果

512TRAIN的K32等分组166个：163个全满分、3个全零；346个非等分组中315个仅EP变化。独立五reference均分94.0226，K32有限安全候选上界95.2183；282/512场景发现至少.005原分数的安全改善。稳定零分只有3个，本人口对全零修复的分辨力有限。

256选择场景K32 oracle：IID93.87686、盲结构化94.84325、标量CEM94.25631、物理反馈94.25450、新颖度94.28565。盲结构化相对IID +.96639点（log bootstrap95%CI[+.24865,+2.42259]）；物理反馈相对标量没有额外价值证据。F2/F3教师各237场景/748目标，511/512场景教师标签相同。因此后续S2/S3相近不能单独解释为“学生没吸收”。[搜索机制与全成本](feedback/discovery_v1.md)

L1/L2/L3与S1/S2/S3已完成正式筛选及重新采样部署评价；P0–P5仍在固定预算内训练。P0首inner因旧评价调度器第三进程抢入共享显存OOM，未发生optimizer更新；保存的256候选合法提交链仍可原样恢复。缺少failure_evidence目录导致的二次异常已最小修复并CPU注入验收；不更改算法。后续用原97不可变训练源码加预建目录恢复，不重采原链。旧调度器CPU owner只扩展资源预留，所有既有GPU训练不受信号。[失败证据](preservation/formal_failure_P0_initial.json)

主配对部署锁定为native_qwen_sourceflow_ODE命名空间、CPU FP32场景绑定root、每seed B1、原shift5 K10。97版本S1自动评价使用另一命名空间，只保留为辅助；主评价统一用无teacher/ref/archive依赖的合并纯ActionDiT入口。主协议及已完成轨迹不改。[评价执行锁](EVALUATION_LOCK.json)

## 完整学习窗口与新增机制限制

L1/L2/L3均已完成64份fresh collection、1024自然场景曝光（512各两次）、16384条真实新候选、256次连续Adam更新。全部冻结参数和buffers保持；每份old行为四inner固定，预算未用inner复用虚增。[终点审计](latent/terminal_training_audit_v1.json)

已保存TRAIN特征的CPU诊断显示L1/L3小MLP学到首Tanh饱和，输出近乎全局常量；初始网络不饱和、LayerNorm分母正常。这是已学到的条件塌缩，不是Gaussian密度或采样bug。L2的jointKL位移又明显小于L1，不能将未来L1>L2直接归因场景条件化。原结果保留；唯一的固定TRAIN均值中心化修正已另立L1_CENTER_V1，完成同预算训练与评价，正进行终点机制分析。[最小诊断](latent/conditionality_minimum_diagnosis_v1.json)

新增只读行为分析入口对已保存完整评价计算请求ADE/FDE、heading漂移、实际执行速度/jerk/越界、固定参考路径距离到达时间。后者是距离代理，未到明确保留，不能称官方到达时间。局部请求坐标和全局执行坐标明确分开，所有结果来自实际模拟而非8点请求差分估计安全。


## 已完成的单轨部署结果（2026-10-24，尚非最终研究结论）

以下为固定256 DEV_SELECT、8个完整log、每scene五个新seed；先按scene汇总seed，再按log配对bootstrap。它不是完整Navtest，不应将92以上的切片均分称为达成最终92目标。所有行无best-of、无搜索选轨。

|单轨策略|DEV_SELECT PDMS|相对同协议BASE|
|---|---:|---:|
|BASE/L0|92.14310|0|
|固定root scale1.3|92.32443|+0.18133|
|L1 条件潜策略+先验KL|92.47539|+0.33229|
|L2 全局潜策略+先验KL|92.14792|+0.00481|
|L3 条件潜策略无先验KL|92.44964|+0.30653|
|S1 继续GT FM|91.80622|−0.33688|
|S2 标量搜索教师FM|92.37017|+0.22706|
|S3 物理反馈教师FM|92.34461|+0.20151|

L1相对BASE的95%CI为[+0.26215,+0.39724]点，但相对固定scale1.3的CI跨零；原条件MLP又出现输出近乎全局常量，L2优化位移明显更小，尚不能证明条件策略的独立贡献。TRAIN上的L1均分变化CI跨零，且新增联合安全违规多于恢复。详见[潜策略完整结果](latent/terminal_results_v1.md)。

S3相对BASE的CI跨零，S3相对S1的约+0.54点部分来自S1自身退化，不能代替相对共同BASE净收益。S2/S3教师几乎相同，部署也相近。学生到教师的最近ADE从0.349降至约0.282米，说明部分形状已被吸收；但发现的唯一稳定零分修复在学生重新采样后仍五次全零，部分安全修复场景还退化。当前问题包括投影后的实际安全后果，不能只看FM loss或轨迹距离。[学生吸收与成本](feedback/terminal_results_v1.md)

P0–P5尚未完成，对存档、约束及其互补性暂不作结论。P1训练校准未达到预登记的漂移匹配容差，replay系数达到上限后梯度比例也低于目标；即使最终有增益，这些机制限制仍必须披露。新增加同x0的SDE/ODE终点均值审计，预算另封存26624条新评分，独立于主纯合并ODE单轨结果。

新增纯q部署包包含全部所需网络与固定TRAIN均值buffer；无需原训练checkpoint、optimizer或外部normalizer。分片评价只划分独立scene，保持原B1前向、场景绑定CPU随机数与评分入口。实际L1/S3纯包与已有主评价的160条逐值对照已全部通过：root、归一化动作、请求轨迹、模拟状态和全部原生分项最大误差均为0；CPU发布/缺片/错误root/授权撤销六项亦通过。[实际GPU证据](partitioned_evaluation_GPU_v1.json)。该入口尚未派发封存审计/Navtest，等待全部训练终点和配置锁定。

反馈路线费用已将共同缓存、73728条真实新搜索、三个学生、11520条主评价和S1旧3840条辅助评价分别计账。[完整成本账本](feedback/final_cost_ledger_v1.md)。


唯一CENTER修正也已完成：DEV单轨92.40899，相对BASE +0.26588（CI[+0.02283,+0.71032]），相对固定scale1.3仅+0.08455点；TRAIN +0.00882且CI跨零。q恢复跨场景变化，但KL位移也更小，不能据此证明条件性独立价值。DEV收益主要由4次联合安全恢复贡献，当前未观测新联合违规不等于真实新增风险为零。原L1与CENTER均保留，未依据审计集选择，暂不扩大训练seed或算法矩阵。[修正完整结果](latent/CENTER_terminal_results_v1.md)
