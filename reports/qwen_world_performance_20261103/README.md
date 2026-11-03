# Qwen-World：以真实单轨迹 PDMS 为目标的规模化 RL

本轮从 `987bc614aec5a042f2df8e8590d6f2479b3e1010` 开发。研究进行中，工程资格通过不等于算法提分。

## 当前优先：D46 六条专用 RL 路线

新增TSH4八卡已授权用于FPO双节点。FPO自身8卡1440_READY/EXIT0后，以同Adam/RNG/MC4材料进入新 `FPO_WORLD16_PRIORITY_v1`（TSH3+TSH4，science82d5f78）；资格first3+pending恢复1已PASS，资格权重不进入正式起点。**正式16rank现全部≥1464，从自身1440增加至少24次真实Adam、B1恢复零差，原6E和无stop-after保持。** 当前D0/FPO各16卡，D2/AWM/D5各8卡，共5路线/56逻辑卡位；后续数量仍活读状态。protected D IL60k评测不被停止，实际按shared65000MiB余量准入，原75/peer失败保留。[正式更新实证](/mnt/project/d46-priority-rl-20261103/operations/WORLD16_STAGES/FPO_WORLD16_PRIORITY_v1/ACTUAL_FORMAL_TRAINING.json)、[GPU资格审计](/mnt/project/d46-priority-rl-20261103/operations/FPO_WORLD16_TSH4_20261103/GPU_QUALIFICATION_AUDIT.json)、[新增TSH4授权](/mnt/project/d46-priority-rl-20261103/operations/ADD_TSH4_20261103/AUTHORIZATION.json)。

FPO正式短窗完整cycle独立实测 **1.900523×**：旧1409–1440/15完整cycle平均63.31731秒，新1445–1464跳前4warm/9cycle平均33.31574秒；纯更新中位18.66498→9.77126秒，scene/s1.010782→1.921014。B64/G16/M2/MC4不变；shared65k允许受保护IL共同驻留，不是长期保证或新分数，不能套用D0的2.0727×。**FPO当前完整分仍90.12278094@900，1464是训练进度。**[独立FPO正式计时](/mnt/project/d46-priority-rl-20261103/analysis/FPO_WORLD16_THROUGHPUT_20261103.json)

常规0.5E官方完整结果已新增：D0/Adam1614 **90.13428003**（IL46+0.63875906；修复266/新增98；固定高分净贡献约−1.5423），比其1400点90.17112884低0.03685，不宣称该小变化显著或触及上限。D5/Adam1614 **89.70890336** 同样完整12146/invalid0。原四个早评点与新常规点合并但不重复计等参引用；现D46最高仍D0/1400的90.17112884。FPO低曝光90.12278094与较少高分回退是加速研究的动机，不是最终胜利。[D0.5完整分数](/mnt/project/d46-priority-rl-20261103/evaluations/D0_epoch0.5/SUMMARY.json)、[配对分析](/mnt/project/d46-priority-rl-20261103/evaluations/D0_epoch0.5/PAIRED_ANALYSIS.json)、[D5.5完整分数](/mnt/project/d46-priority-rl-20261103/evaluations/D5_epoch0.5/SUMMARY.json)。

**正式D0双节点16卡已经启动并持续更新**：实际阶段为 `D0_WORLD16_PRIORITY_v2`，TSH5+WM4，从自身1714_READY继承Adam/RNG与原6E预算，16rank真实超过22次新Adam、B1原生恢复零差，无资格stepcap；旧v1 DEFERRED不是当前阶段。[正式启动实证](/mnt/project/d46-priority-rl-20261103/operations/WORLD16_STAGES/D0_WORLD16_PRIORITY_v2/ACTUAL_FORMAL_TRAINING.json)

实际完整cycle短窗吞吐 **2.072719×**：8卡独占1677–1714（17完整collection）43.77093秒/cycle，正式16卡跳过4次warm更新后1719–1796（38collection）21.11764秒/cycle，B64/G16/M2不变；纯更新中位10.6821→5.1880秒、新场景/小时5264→10910。按当前吞吐外推1E约19.6→9.5小时，不含启动/完整评测，不作长期保证或新PDMS声明。[真实计时输入](/mnt/project/d46-priority-rl-20261103/analysis/D0_WORLD16_THROUGHPUT_WORKING.json)

FPO在TSH3/world8、D2在TSHzt/world8继续；AWM曾1194_READY退出资源hold，D1/D5同样保留原预算。v7仅在WMzt/TSH2今后完整8卡可归因空闲且每卡≥79000MiB时按AWM>D5>D1自动恢复自己的CP，不触未知占用；是否实际恢复及当前训练数量只按实时PRIORITY_STATUS，不能固定沿用“三条”。旧I仍4877/inner1资源hold，原10000承诺保留。常规.5E队列按实际完成片、worker收据与排队显示；D1/D2等价别名已落账，不作额外独立结果。

新研究根 `/mnt/project/d46-priority-rl-20261103`；固定完整 IL46 部署策略 NAVTEST **89.49552098**（model SHA `386a0a27…`），各路线 fresh Adam0/RNG42，不接续 D44 RL 或资格权重。父合同 v2 允许六条 policy-only 路线，优先顺序如下；它们适配 D 原生 Qwen，不是把旧 IL70 Safety 配方替换父权重。

| 路线 | 方法与实际阶段门 |
|---|---|
| D3_FPO | MC4 CFM-energy ratio PPO代理；非精确轨迹似然 |
| D3_AWM | 正优势 matching 与参考保持；保守适配，非完整 signed AWM |
| D5 | text LoRA r32、last8 FiLM低秩及action I/O/position适配 |
| D1 | 2 exposure epochs后加入Comfort和TRAIN GT endpoint语义质量信用 |
| D2 | 0.5 exposure epoch后，真实B1对照的严格安全高分教师吸收 |
| D0 | joint24条件Gaussian PPO、clip .1与自适应native KL基线 |

共同 B64/G16/inner2；每条先完成真实八卡10更新资格，再从同一 IL46 策略和新空 Adam 正式开始。资格/模型初始化不算实际正式在训，当前数量、进度与分数只读 [`PRIORITY_STATUS.json`](/mnt/project/d46-priority-rl-20261103/PRIORITY_STATUS.json) 和 [`CAMPAIGN.json`](/mnt/project/d46-priority-rl-20261103/CAMPAIGN.json)。科学源冻结在 `runtime/formal_v1@ff3013c`。原完整 IL native 已清理，保留的部署策略经原生导出零差核对；缺失 readout 的 D4/D6 仍暂缓，不伪造恢复材料。[父合同v2](/mnt/project/d46-priority-rl-20261103/PARENT_CONTRACT_v2.json)

六台 wm4/rl2/tsh2/wmzt/tsh3/wm2 已针对 D46 **重新开放、无到期截止**；下文00:00截止与23:51归还属于原 IL70 任务历史，不是 D46 当前权限。其它任务授权不扩张。[实际新授权](/mnt/project/d46-priority-rl-20261103/operations/RESTORE_SIX_UNBOUNDED_20261103/APPLIED.json)。旧D44 D0/D2/D5/D1分别保存于763/637/495/9、inner1并正常退出，完整状态保留且未转入新研究；IL70保存让位同样不判方法无效。[D44保存收据](/mnt/project/d46-priority-rl-20261103/operations/USER_PRIORITY_20261103/D44_ACTUAL_NATIVE_SAVED.json)

### D46 速度与首次评测

2026-11-03 08:20:25 UTC 实测快照：每条全八rank最近100个完整collection/200次Adam、同一invocation，不跨暂停。

| 路线 | 纯更新 秒/Adam | 含采样、评分等端到端 秒/Adam |
|---|---:|---:|
| D0 | 10.794 | 22.410 |
| D1 | 11.010 | 22.891 |
| D2 | 10.750 | 22.369 |
| D5 | 11.234 | 23.165 |
| D3_FPO | 18.712 | 32.113 |
| D3_AWM | 22.430 | 36.023 |

该快照尚无D46 RL完整NAVTEST或已写评测请求；首评0.5 exposure epoch=1614次实际Adam，最快1198次/.371epoch，按同负载约2.6–3小时到门，**不含评测耗时**。D1 2E信用和D2 .5E救援尚未启用，后续速度需重测。[完整速度与首评审计](/mnt/project/d46-priority-rl-20261103/analysis/D46_THROUGHPUT_REVIEW_20261103.json)

最早IL12k父策略的D为B8/G8/M4，纯更新2.715秒；因没有逐行时间戳，端到端3.406秒取首完整1000次Adam invocation，不能冒充当前同200步窗口。D44首D0的B64/G16/M2为10.744/21.997秒。当前每Adam新场景/候选为32/512，最早为2/16；当前D0端到端每步约6.58×，但场景/候选吞吐约2.43×/4.86×。父模型、训练配方、窗口和共享GPU负载不同，这是实际工作负载比较，非严格同算力或纯执行加速消融。[历史原始证据](/mnt/project/d46-priority-rl-20261103/analysis/D_THROUGHPUT_ORIGINAL_BASELINES_20261103.json)

用户随后要求提前完整评测，再决定额外资源倾斜。独立有限批次已登记四个唯一参数策略：D0/D5各Adam1400、44800有效场景（.43374E），FPO Adam900、28800场景（.27883E），AWM Adam800、25600场景（.24785E）。每策略32逻辑分片，在TSHzt/RL2/WM2按策略轮转分波，逻辑分片数不等于同时物理GPU数；不修改训练owner或科学源。D1/D2的同Adam1400父策略、完整scope与封存optimizer payload签名均与D0一致，仅列“参数一致的早期参考”，不重复伪造独立结果；在该1400检查点各自2E/.5E机制尚未启用。[封印计划](/mnt/project/d46-priority-rl-20261103/operations/EARLY_EVAL_20261103/PLAN.json)、[实际分片状态](/mnt/project/d46-priority-rl-20261103/operations/EARLY_EVAL_20261103/STATE.json)

四份官方完整结果现已完成，均12146场景/44log/invalid0、B1/seed2101。当前D46最高为 **D0 Adam1400：90.17112884**，相固定IL46 **+0.67560786**；D5同1400为89.97889876，FPO900为90.12278094，AWM800为89.95115511。D0相IL46修复248个零分、新增87；其它方法与完整CI/高分保持见实时面板和各PAIRED_ANALYSIS。不同曝光预算不能作同预算方法排名，D1/D2早期等参数参考不另计独立点。这是 **NAVTEST_SELECTED** 的阶段资源选择证据，不是独立未选择测试。历史IL70最高91.45359756仍单列，不能作为D46父基线。

已嵌入两张现成全量图，未重新推理/评分/画趋势。[零分修复与gain/loss](/mnt/project/d46-priority-rl-20261103/analysis/plots/D46_zero_repair_and_gain_loss.svg)、[安全与进度变化](/mnt/project/d46-priority-rl-20261103/analysis/plots/D46_safety_and_progress_delta.svg)。固定IL46≥.9为同8722场景：D0该队列全局贡献−1.368811 PDMS、EP−1.551854pp，抵消部分修复；FPO少曝光的高分净损失较小，AWM保持较多但NC小降，不能称安全全面改善或因果排序。[封印事实与图口径](/mnt/project/d46-priority-rl-20261103/analysis/plots/D46_EARLY_FULL_FACTS.json)

2026-11-03 12:40 UTC资源快照为3条实际训练：D0在TSH5与已知旧I共享，FPO在TSH3，AWM在WM4；D1保留自身1614，D5保留1647/inner1，均为资源hold，不判方法无效。D2原attempt002在恢复前Gloo90秒超时、0新更新，已复核一次同source/同1614暖缓存重试；是否已推进按实时PRIORITY_STATUS，不把初始化算在训。此前D0/D1/D2/D5在1614_READY完整保存后因缺evaluation_requests目录退出1，完整原生/Adam/scheduler/RNG审计PASS，目录与.5门已补回；旧code1保留，不改造成成功。[恢复审计](/mnt/project/d46-priority-rl-20261103/operations/EVAL_DIRECTORY_RECOVERY_20261103/RECOVERY_AUDIT.json)、[回填](/mnt/project/d46-priority-rl-20261103/operations/EVAL_DIRECTORY_RECOVERY_20261103/DIRECTORIES_AND_GATE_BACKFILL.json)、[D2超时审查](/mnt/project/d46-priority-rl-20261103/operations/EVAL_DIRECTORY_RECOVERY_20261103/D2_STARTUP_TIMEOUT_REVIEW.json)。

后续D2同source暖重试已真实越过1614；实时在训数由PRIORITY_STATUS持续更新。旧IL70综合I已协作保存于 **4877/inner1，job EXIT0**，临时资源保留以优先D0；原 **10000次实际更新** 承诺保留，父模型、Adam/RNG与待更新材料不重置，未作算法终止。STOP_REQUESTED保留为资源恢复栅栏，后续须从自身最新完整原生状态恢复。D0实际主机仍以当前表为准，不从独占意向推断GPU pid核验结果。[实际保存退出](/mnt/project/d46-priority-rl-20261103/operations/OLD_I_RESOURCE_HOLD_20261103/NATIVE_SAVE_EXIT0.json)、[资源优先级与预算保留](/mnt/project/d46-priority-rl-20261103/operations/OLD_I_RESOURCE_HOLD_20261103/RESOURCE_PRIORITY_PAUSE.json)。

常规0.5E评测已在可用RL2/WM2派发；D1/D2自身1614与D0自身1614的完整payload/scope/parent/loader配置等价证明已准备。别名只在实际新控制器应用后按EQUIVALENT_REFERENCE展示，不新增独立分数或复制D0结果；当前实际引用状态仍以评测controller收据为准。

world16早期仅D0的GPU pending恢复工程资格通过时，短4次更新约5–6秒并不等于正式长训；D5只属引擎支持。后续正式D0 v2与更长实测吞吐已见最前状态，仍不保证长期2倍或部署提分。12:40资源核查中的未知占用未被停止，实际主机/资源以动态收据为准。[动态资格边界](/mnt/project/d46-priority-rl-20261103/operations/MULTINODE_NETWORK_20261103/GPU16_QUALIFICATION_STATUS.json)

128个GPU评测worker已正常EXIT0；随后只替换有限CPU聚合器，用四CPU并行、官方validator每个不可变数据只读一次并复用。canonical公式未改，该次聚合接管未向GPU发信号；后续执行恢复、资源hold和在训状态见实时表。HTML完整分数不依赖STATE是否已结束paired汇总：32片与identity/摘要封印通过即可发布分数，paired未齐明确“分析中”，不编修复/退化统计。[聚合接管证据](/mnt/project/d46-priority-rl-20261103/operations/EARLY_EVAL_20261103/AGGREGATION_OWNER_TRANSITION.json)。

HTML从每片最新job的RESULT/RUNNING读取完成、已启动待返回、准入、资源等待与异常，避免整轮派发尚未刷新的STATE计数掩盖已发任务。仅完整12146/44log/invalid0、seed2101的封印identity与指定检查点绑定后才发布分数；完整配对分析封印后再报告IL46相对修复/新增零分和固定IL46≥.9队列变化。四策略曝光预算不同，提前结果用于资源分配观察，不作同预算算法排名；不将NAV标签加入训练bank。

运营控制已实际升级为评测期间持续训练：仅替换CPU owner，原GPU job/科学源/Adam/RNG保持；有限早评批次运行时暂缓重复的常规gate派发，gate保留而非丢弃。HTML训练表从最新PRIORITY_STATUS读取Adam与有效曝光/epoch；提前批次显示已写场景文件数和逐分片实际收据，绝不发布partial均值。上文08:20计时仍是历史窗口，不代替当前进度。[CPU实际接管](/mnt/project/d46-priority-rl-20261103/operations/CONTINUOUS_EVAL_20261103/OWNER_TRANSITION.json)、[执行策略](/mnt/project/d46-priority-rl-20261103/operations/EXECUTION_POLICY.json)。

## 既有 IL70 研究的完整结果

**最高已确认完整部署成绩仍为 SafetyEP Control10k：91.45359756，尚未达到92。** 相同完整 IL70 为88.79290991。以下本轮结果均为真实单轨 B1、12146场景/44 physical logs/invalid0；不是候选 best-of 或 TRAIN 分数。

| 本轮完整检查点 | 完整 PDMS | 已启用机制与解释 |
|---|---:|---|
| S5 / Adam1616 | 90.93054358 | 仅phase1基础SafetyEP覆盖；rescue/history/额外B1均未启用 |
| S0 / Adam808 | 90.90006526 | 基础SafetyEP |
| S1 / Adam808 | 90.90006526 | 尚未到0.5epoch救援门，与S0相同目标 |
| FAST PACK4 / Adam808 | 90.87256484 | 一致合批实现、独立IL70初始化 |
| AWFM / Adam404 | 89.03478690 | MC4/inner2的早期点；不据此判长期上限 |

[原始完整结果索引](results/NEW_EVIDENCE_INDEX.json)。这些点的场景/候选/Adam预算不同，不能作等算力或单因子因果结论。

S5相IL70修复356个零分、新增56；相Control10k虽同为265个零分，却修复48又新增48。相Control的差距−0.523054中，固定IL70高分8724场景贡献−0.377945，其余−0.145109；EP差距较大而NC/TTC改善是描述性关联，不能直接推断参数或行为因果。八rank封存状态证明其仍为coverage：25856有效场景/413696主PG查询、无rescue/history查询。[S5全量分析与三图](analysis/S5_u001616_FULL_REVIEW/SUMMARY.json)、[实际阶段与预算](analysis/S5_u001616_FULL_REVIEW/PHASE_AND_BUDGET.json)。

FAST相IL70+2.079655、修复353/新增68/剩余零分280，仍比历史Control10k低0.581033。固定原IL高分群体解释差额−0.482956，其余−0.098077；早期0.25epoch不足以判断长期上限。[FAST完整ledger、统计与三图](analysis/FAST_u000808_FULL_REVIEW/SUMMARY.json)。

实时 HTML：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。原总报告 `/mnt/project/qwen-world-rl-optimization-20261030/report.html` 的最前方已接入本轮面板，沿用唯一原报告守护，每 30 分钟及阶段事件更新。

## 已经启动的训练与当前判断

实际训练数量及每条状态只从当前stage job、RESULT和八rank新日志读取，不将注册/加载/陈旧RUNNING当作训练。[最新归档状态（带统计时间）](results/LATEST_DOCUMENTED_STATUS.json)与实时HTML用于核对；进入23:45保存窗口后数量会继续变化。S0在update890/inner2、S2在404、S4在320/inner0、LowLR在30/inner2原生保留，为更高优先级研究让出资源，不判算法无效。旧20:59快照[保留作执行历史](results/PROGRESS_20261102T2059Z.json)，不能覆盖当前状态。

S1 从临时 wm4 保存于 update205/inner1，按同一实验完整状态恢复到长期可用 tsh5 共享资源。其与独立 S0 对照的前 221 个更新、八 rank 共 1768 条记录中，轨迹、old log-prob、优势、mask、参数摘要、loss、梯度和 LR 完全一致。S1 的 rescue 从 0.5 epoch 才启用，早期目标相同，不能在启用前判断机制无效；该一致性同时支持本次原生迁移未改变训练。

已另行登记固定 TRAIN2048 真实 B1 诊断，先评测同一 IL70，再与本实验自己的后续 checkpoint 逐场景配对。它属于训练分布诊断，不作为独立泛化验证，更不作为 NAVTEST 完整成绩。

该固定人口的完整配对现已完成：IL70 **93.98113**、S0u400 **93.89103**、S5u800 **94.03471**。两臂均修复同12个原零分、保留同3个、新增同2个；低分432场景的全局贡献分别为+.50124/+.54895，高分1616场景为−.59134/−.49538，收益几乎抵消。12个修复中11个尚未直接训练曝光，支持同TRAIN人口内跨场景参数迁移，不能说只是记住已见token。S5较S0的+.14368主要来自EP，零分集合相同不等于策略相同；两臂同12800场景/204800 PG查询，但B/Adam不同，且rescue/history均未激活。单次TRAIN诊断、历史采样见证与分解均不能证明因果、独立泛化或最终上限。[配对摘要](analysis/TRAIN2048_S0u400_S5u800_REVIEW/SUMMARY.json)、[研究简报](analysis/TRAIN2048_S0u400_S5u800_REVIEW/RESEARCH_READOUT.json)。

两项吞吐开发已经通过真实八卡原生恢复检查：冻结 IL 参考缓存的连续训练与开缓存保存/恢复，参数、Adam、scheduler、全部八 rank RNG/scene stream 逐位一致，参考调用 12→3；PACK4 的连续训练与保存/恢复也逐位一致。PACK4 首次 CPU 比较将行政 invocation UUID 误判为差异，原失败记录保留，另行严格复核仅排除该顶层 UUID 后通过。缓存已经受审计地迁移到同一 S1；PACK4 因 BF16 batch 形状改变，以独立 IL70/Adam0 实验运行，未接续小 batch 的 RL 权重。

旧综合 I 的完整 2k 点为 **91.02554897**，比 IL70 高 2.232639 分、比历史 Control 最高低 0.428049 分。逐场景全量分析显示，原 IL 高分组的额外下降贡献 −0.436213 分，基本解释其与 Control 的差距；同安全指标下 EP 下降是主要相关现象，不能据此声称已证明参数更新的因果机制。[全量分析](analysis/INTEGRATED_I_u002000_FULL_REVIEW/SUMMARY.json)

TRAIN 定向诊断：15 个固定 B1 零分场景中，G16、G32 各发现 12 个严格安全高分场景，并集 13 个；38 组都不是奖励全零或 SafetyEP 原始优势全零。独立采样种子、不同成本的小样本不能证明 G32 更优。另有真实合法 GT 的 2035 次评分，其中 245 条严格高分且比 B1 高至少 .005，但 255 条 GT 比 B1 低超过 .005，不能盲目模仿 GT。[采样简报](results/TRAIN19_G16_G32_REVIEW.json)、[GT 审核](results/GT_TEACHER_RESEARCH_READOUT.json)

新独立 **S6 Verified Quality** 从第一轮曝光同时学习低分修复和安全高分中的效率改善，真实 B1 条件的 ODE 末步与 CFM 联合监督，共享原 β50 约束预算。冷启动时原 IL anchor 梯度很小，现用真实 PG 梯度 EMA 提供辅助梯度校准下限；资格阶段已验证自然教师触发与八 rank 原生恢复，正式 native0 也核对了同一 IL70、空 Adam、零曝光。它尚无完整部署成绩。[算法设计](../qwen_world_verified_quality_20261103/ALGORITHM_DESIGN.md)、[资格实测](results/S6_GRADIENT_READOUT.json)

进一步的 TRAIN 诊断中，7 个追加 SDE16 仍无合格教师的场景，其 B1 和 GT 其实都严格安全、舒适，主要是进度不足。对同 7 场景的 112 条显式合成进度候选进行官方评分，获得 35 条严格合格教师，覆盖 7/7；GT 路径族覆盖 6/7、B1 路径族覆盖 5/7，并集覆盖全部。合成池也有不安全候选，必须逐条验证。独立S7已通过真实八rank资格并正式登记/启动，将它们用于训练期教师发现，不纳入PPO/std或测试时选择；当前是否实际在训以实时八rank日志为准。这是小型 TRAIN 诊断，尚未证明策略吸收或部署提分。[真实 GT roundtrip 审核](results/S6_GT64_ROUNDTRIP_READOUT.json)、[合成候选结果](results/S6_SYNTHETIC_PROGRESS112_READOUT.json)

S6前100实际更新已完成全8rank激活审计：1600个去重新鲜TRAIN场景，教师792（repair267、already-high525），全场景平均辅助系数份额1.490625%；PG25600+B11600+条件额外SDE16 2160=29360次查询，额外14.6875%。135次额外池中49场补出high，不能用首7场无high外推额外探索无效。这只证明门控与校准梯度路径激活，尚未测更新后同场景B1吸收或NAVTEST收益。[完整激活审核](results/S6_THROUGH_U100_ACTIVATION.json)。

**S7正式从相同IL70/Adam0开始，绝不使用资格权重。** mainG16/W3 SafetyEP、PG-EMA与α.02/.05预算不变；仅当B1、mainG16、原extraSDE16均无strict-high时，追加最多16条经过官方验证的结构化teacher-only候选，仍进入原CFM+真实ODE-last辅助。资格continuous4/split2/resume4的完整Adam/RNG/stream/辅助恢复通过；自然35教师，其中4个由structured补出，实际64次结构化查询。它尚无完整NAVTEST分数。[资格](results/S7_WORLD8_NATIVE_COMPLETE.json)、[正式登记](results/S7_FORMAL_REGISTRATION.json)、[真IL70/空Adam证明](results/S7_FRESH_IL70_ADAM0_PROOF.json)。

另两个真实B1零分、G16/G32均无high的固定TRAIN场景，32次CPU结构化候选评分找到10条合格GT-path教师，覆盖2/2，最佳teacher score为.971158/.944913；B1-path16条全部DAC失败。这是特权TRAIN教师发现，不是B1已修复或部署成绩。[ZERO2完整诊断](results/TRAIN_ZERO2_PROGRESS_READOUT.json)。

信用谱系核对复用了4416组有界真实TRAIN行为。旧最高Control10k本就使用global rawA std；五代局部SafetyA/λ_EP1/feasible3门一致。固定同G16/同θ/同轨迹的B/B8优势倍率中位约.94，不支持“新版突然数量级压缩EP”的解释；人口/G/Adam混杂仍在，rawA能量不是参数梯度。继续原至少3/首6epoch检验，并优先真实梯度方向审计，不能据早期点判上限。[完整谱系](results/SAFETYEP_CREDIT_LINEAGE_REVIEW.json)。

## 当前路线

| 优先级 | 路线 | 主要目标 | B / G / M |
|---|---|---|---|
| 0 | S1 Nominal Rescue | 将真实严格高分候选吸收到部署 B1：CFM + 真 ODE 末状态 SmoothL1 | 128 / 16 / 4 |
| 0 | S7 Structured Progress | 条件结构化教师发现→CFM与真实ODE-last；主PG不变 | 64 / 16 / 4 |
| 0 | S5 Performance Max | 全量覆盖→候选吸收→主动历史保持→低 LR 精修 | 64→128 / 16 / 4 |
| 0 | S6 Verified Quality | 低分修复与安全高分效率改善，PG-EMA 校准 CFM/真实 ODE 末步 | 64 / 16＋条件 teacher-only 16 / 4 |
| 资源保留 | S0 Large SafetyEP | 大场景覆盖、有效曝光调度与长训 | 128 / 16 / 4 |
| 资源保留 | S2 Conservative AWFM | 原生 FM 优化，负优势由 rollout 一致性项稳定 | 128 / 16 / 2 |
| 1 | S3 FPO | 逐 MC CFM energy-ratio PPO 代理 | 128 / 16 / 2 |
| 资源保留 | S4 Temporal Rescue | 真实时序安全代价 + 已验证候选吸收 | 128 / 16 / 4 |
| 资源保留 | S0 Low LR | 独立 IL70，action/memory 3e-5/3e-6 | 128 / 16 / 4 |
| 吞吐研究 | S0 FAST PACK4 | 一致合批真实 W3 采样和重算，独立 IL70 起点 | 128 / 16 / 4 |

每条独立路线从相同完整 IL70 初始化，Adam0、独立运行目录/优化器/scheduler/RNG/buffer；资格测试权重不进入正式训练。真实模型 SHA：`7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40`；父路径 `/mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT`。

## 已完成的真实资格

- B64/G16、B128/G16、B64/G32、B128/G32：完整真实多场景 collection，八 rank 全局有效候选 std、四次 Adam、冻结参考不变、当前 memory 更新。
- B128/G16 在第一次 inner 后恢复至第四次，与连续四次训练的参数、完整 Adam、scheduler 及全部八 rank RNG/scene stream 逐位相同。
- 完整 IL70 上的 S1 定向检查：68 次实际评分、4 个 TRAIN 场景找到严格合格教师，真实 ODE 最后一步与 clean-x 输出逐位一致；动作、scene memory、predictive memory 均有梯度。此项显式注入资格启用门，实际训练曝光为零。
- S1/S4 启用辅助项的真实 B128/G16/M4，各完成 4 Adam；S2/S3 各完成两次 B128 fresh collection、4 Adam；S5 完成两次 B64 fresh collection、8 Adam，并执行 8 次主动历史 B1 查询。均为独立资格目录，非正式成绩。

[资格与吞吐量](performance_qualification.json)、[算法设计及论文依据](ALGORITHM_DESIGNS.md)、[完整配置](CONFIGURATIONS.md)、[实际计算预算](TRAINING_BUDGET.md)。算法资格后续增补在 NAS `qualification_v5`，不把未覆盖的机制登记成实测通过。

## 数据与部署协议

103288 个合法唯一 NAVTRAIN scene；与 NAVTEST 的 token、segment log、physical log 交集均为零。逐文件检查了 309864 张当前图像和 103288 个评分缓存。缓存检查为存在性/XZ 结构筛查，未声称完整反序列化或逐场景评分重验。旧 DEV1696 已属于 TRAIN，仅作训练诊断。[审计](data/AUDIT.json)

首个完整曝光 epoch 均匀遍历，之后 60% uniform /25% 当前 B1 hard /15% 自身退化。只有本 run 的真实 TRAIN B1 反馈可形成优先级；未获得反馈时回退 uniform。每条记录按实际观测曝光判断陈旧，不能用发布快照的时间刷新旧标签。epoch 尾部 8 个 padding 不计有效曝光、不进 std、不产生任何 loss。

部署固定 B1、K10、shift5、seed2101，不重排、不选 oracle，官方 canonical NAVTEST12146/44 logs、invalid0。完整预定评测可用于选择最佳 checkpoint，明确 `NAVTEST_SELECTED=true`；不得称完全盲测，更不得将 NAVTEST 的场景标签/轨迹/priority 回流训练。

## 执行与预算

原生 trainer 继续复用 `tools/qwen_world_safety_exploration/train.py`，新入口是薄适配，不复制完整 Trainer。所有修改只在本研究工作树；旧实验冻结源码不变。

```bash
export WORLD_COMPARE_ROOT=/mnt/project/qwen-world-performance-rl-20261103
export WORLD_COMPARE_TASK=qwen_world_safety_exploration_20261031
python -m tools.qwen_world_performance.launch --help
python -m tools.qwen_world_performance.evaluate --help
```

正式源冻结、原生恢复和资源锁通过后才 dispatch。初始预算到 6 exposure epochs，0.25/.5/1/2/3/4/6 逐点检查；8/12/20 仅为可扩展预算，未自动声称训练完成。数值崩溃或严重遗忘可停止，早期低于其他算法不等于最终上限。

本次六台临时资源wm4、rl2、tsh2、wmzt、tsh3、wm2至UTC2026-11-03 00:00（北京时间08:00）硬截止。**UTC23:40是北京时间07:40，23:45是07:45。** 本任务原900秒保存余量已恢复，23:45开始保存退出；此前23:15提前禁派发已按原用户窗口纠正，历史收据保留。[窗口纠正](results/RESOURCE_WINDOW_CORRECTION.json)、[900秒恢复](results/RESOURCE_GUARD_RESTORATION.json)。本轮六台临时机上的自有任务已协作保存退出，23:51实际核对自有QwenWorld进程为0（不是所有项目进程为0）；[实际归还证据](results/ACTUAL_PRE_EXPIRY_RELEASE.json)。S7最新封存u59/inner3、960有效曝光，structured查询848，仍无部署效果。close_window/guardian按实际授权执行，后续是否转永久须看真实准入和原生恢复回执，不能凭计划声明完成。其他任务授权、未知进程和B/D保护不变。

S7永久槽位等待器只观察旧I自然完成已承诺10k，不提前停止旧I或占用B/D。之后仍需满足最多2个已知native与显存准入：root操作员在第三方D44 D0迁入后实测TSH5同卡已有oldI/S1/D0，可用仅14657–14963MiB，低于已测NAV推理peak21,370,439,680bytes。不能降低门槛硬塞或停止foreign任务；旧I完成也不保证立刻有足够容量。当前等待状态独立记录，评测未齐不发布均分，资源等待不判算法无效；未有实际分配回执前不称迁移成功。[等待快照](results/S7_PERMANENT_WAIT_SNAPSHOT.json)。

本分支已汇总原 S0–S5、冻结参考缓存、PACK4、S6与S7的已验证实现。在线训练仍固定各自已验证的源码，不因仓库整合而热切换：基础 `5e5220c`、S1 缓存执行 `c8a0440`、FAST `cc45f9b`、S6 `8067e13`、S7 `c5f1fd1`。CPU owners 已采用 `ops_v9`（`d0e33bb`），为 S2/S3 登记唯一日志检查差异的执行迁移；S3 原失败记录保留，恢复到资源等待，S2 原资源让位保留。真实八 rank CPU 原生检查通过，尚无修复后的 GPU 续训结果：[M2 恢复说明](M2_RECOVERY.md)。完整来源及实测证据见 [结果索引](results/NEW_EVIDENCE_INDEX.json) 与 [进度快照](results/PROGRESS_20261102T2059Z.json)。

CPU 调度随后更新为 `b09694b`：精确识别已登记共用任务、保持 32GiB 准入余量，并显式允许每卡最多两个已知评测，锁前后均复核。算法源码与 Adam/RNG/scheduler 不变。FAST 已从自身 u808 原生恢复，八 rank 首次 fresh ratio 均为 0。自有全 TRAIN IL70 B1 诊断暂存于 74935/103288，保留全部逐场景缓存，优先把资源用于正式 NAVTEST；未把该未齐诊断均值当作完整结果。[操作记录](results/OPS6_OPERATIONAL_ADOPTION.json)

首窗口测得约 144–155 GPUh/epoch（八卡，非成本上限，不含新增辅助、评测、checkpoint 和失败重放）。尚不能据此判断 B128、G32 或长期训练提高了部署上限；后续完整结果会分别回答。实际已完成任务成本见[审计](results/ACTUAL_COMPLETED_COST_AUDIT.json)：230个已准入完成invocation计301.773429 GPU-slot-hour，合并设备时间区间258.550642小时；不是独占GPU kernel用量，CPU核时未记录。
