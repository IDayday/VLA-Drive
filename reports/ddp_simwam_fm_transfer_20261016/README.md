# 两组 Qwen→Flow 动作头迁移：SFT→RL

## 状态与研究范围

2026-10-17 02:30 UTC前后，B的SFT已推进至约72.7k/100k，原八卡进程与72k恢复点正常；近期包含保存开销的速度约2.28 updates/s，仅剩余SFT约3.3小时，不含后续筛选、验收与RL。A已完成的20k结论不变。已完成[评价资源筛选顺序修复及控制器接管](capacity_evaluation_handoff_20261017.md)：先排除显存不足的主机，再分配固定任务，避免遗漏后面的空闲资源；13项针对性测试和真实资源对照通过。新CPU owner接管同一36项队列，24项完成、12项待办，97份既有分析文件hash不变，训练/推理/评分源码未修改。B75k评价仍等待正式检查点，B RL与最终Navtest尚未开始。

2026-10-17 00:55 UTC前后，[A组20k RL终点及完整DEV评价](final_A_RL_update020000.md)完成：原生PDMS **93.607389**，相对自己的SFT **−0.112703**个百分点、区间跨零；shift5 **94.195811**，相对SFT **+0.735501**、区间高于零。原生TTC新增62/恢复1，shift5新增25/恢复2，中期增益未完整保持，不能以最高点替代固定终点。20k Adam/scheduler、5000份fresh rollout、4万次scene曝光和32万新候选均实际核验；99.5525%的组有PG，主要问题已是安全保持和部署收益。B约59.4k/100k SFT，尚未进入RL；24/36项固定阶段评价完成，最终Navtest仍未启动。报告附全部曲线、终点安全转移、成本和进程退出证据，正式配方未变。

23:55 UTC前后，[B组50k SFT完整评价](interim_B_SFT_update050000.md)完成：DEV原生93.465822、shift5 93.602993，相对25k +0.598729/+0.441426个百分点，两个区间仍跨零；零分26/22→17/16，同时新增9/10个原联合安全场景违规。B继续100k SFT，尚未进入正式RL。两路线22/36项固定评价完成，16个本次评价worker已核实退出；没有改变配方或提前启动Navtest。

23:50 UTC前后，[A组15k RL完整评价](interim_A_RL_update015000.md)完成：DEV原生94.474666、shift5 94.513221，相对自身SFT +0.754574/+1.052911个百分点，两个配对区间均高于零。相对10k的差值均跨零；原生新增TTC减至13个，但DAC新增6个、零分从10k的4增至9，安全保持仍未解决。15k前缀99.52%的组有PG，完整困难人口已覆盖，训练配方不变。A实际训练约15.9k/20k，B约50.6k/100k SFT，两条进程正常；B50k双网格DEV与TRAIN诊断已派发。20/36项固定评价已完成，最终Navtest未启动。

23:21 UTC，[空闲本机接入原里程碑队列](local_evaluation_handoff_20261016.md)：rl-zt2出现不可见GPU进程，改用仍授权且实际空闲的vla-zt与tsinghua-zt承接下一阶段评价。旧controller已在空闲边界退出，新owner锁核验通过；7项测试和实际本机模块路径检查通过。18/36项已完成，A RL13,931/B SFT46,774，训练及数值评价源码不变。

23:04 UTC，[A组跨epoch人口核算](A_RL_epoch_population_accounting.md)完成：12k已提交前缀为24,000次场景曝光、21,367个独立困难场景、192,000条fresh候选；重复曝光2,633次中仅1次来自源末批补齐，其余为正常下一epoch重访。逐rank日志顺序、原生恢复点和56份边界behavior核验通过，14项CPU测试通过；没有新增策略评价或改变正式配方。

22:46 UTC，[完整RL终点预算汇总等待任务](RL_endpoint_summary_handoff.md)已启动，只在八rank正式20k和对应成本账本全部关闭后调用现有汇总器。实际等待和11项有限准入检查通过，尚未生成A/B最终摘要；训练与Navtest准入不变。

22:35 UTC，[A组10k RL完整评价](interim_A_RL_update010000.md)完成：DEV原生94.627902，相对自己的SFT +0.907809个百分点，相对5k再增+0.295554；两项配对区间均高于零。shift5为94.381735，相对SFT +0.921425，与5k基本持平。原生相对SFT新增TTC从28降至20，但仍仅恢复3个，固定SFT-safe中新违规22个；安全保持与最终Navtest目标尚未完成。八卡训练前缀、10k精确保存成本、真实批次梯度及两人口的安全转移分析均已核算，正式训练配方未变。

21:44 UTC，[B 的 SFT→筛选→八卡验收→RL 衔接控制器](B_endpoint_handoff.md)已启动，实际等待完整100k终点；27项针对性测试与实际重复owner拒绝通过。未提前启动B RL，训练/评分仍调用原225b锁定源码，原里程碑评价owner不变。

A的[首个5k RL完整ODE评价](interim_A_RL_update005000.md)已完成：DEV原生94.332347、shift5 94.384140，相对自己的20k SFT分别+0.612255/+0.923830个百分点，按log配对区间均高于零。但原生TTC新增28/恢复3、固定SFT-safe人口新增31个联合违规；shift5相应为8/2和10个。真实部署均分有积极增量，安全保持仍未解决。继续固定20k，不把DEV成绩当Navtest目标完成；梯度、训练预算与完整成本已随报告保存。

进一步的[保存轨迹安全转移分析](A_RL5k_safety_transition_analysis.md)发现，DEV原生28个新增TTC场景在首次失败时均前进更多、速度更高，27个首次违规发生于物理3.6–4.0秒；这些轨迹按源10:5:2奖励也全部比自己的SFT更差。原安全人口中新违规贡献了约99.19%的原生G−。这是已有轨迹的描述性证据，没有新推理/评分，不据此中途修改正式配方。

B的[25k完整DEV与TRAIN诊断结果](interim_B_SFT_update025000.md)已完成：DEV native/shift5为92.867093/93.161568，相对10k +1.178021/+1.473688个百分点，两个配对区间均跨零；零分从45降至26/22，但两个网格均有新增DAC/TTC违规。这是SFT中期结果，继续既定100k；尚无B正式RL或新Navtest结论。

20:40–20:43 UTC，[里程碑评价已改为两台主机并行](parallel_evaluation_execution_20261016.md)。B已封存25k SFT checkpoint，完整DEV与固定TRAIN诊断各8个实际worker已分别启动；A的RL和B的SFT持续运行。只是调度变化，没有新增方法/seed，也没有热修改训练或数值推理源码。

历史现场快照见[20:27–20:30 UTC进度与资源](progress_20261016T2030.md)：当时A完成20k SFT并进入正式RL（匹配汇总1,696/20,000），B继续SFT（23,137/100,000）；旧RL没有活动进程，三台授权八卡空闲。该快照时新RL尚未到5k观察点，后续结果见上方5k报告。[最终评价准入](final_evaluation_lock.md)已完成CPU测试和既有资产/证据核查，真实Navtest仍关闭。

2026-10-16 开始新实验。两条正式 SFT 已于17:39 UTC在 tsinghua-zt4/zt6各8卡启动；tsinghua-zt承担新路线的有限验收和评价。只有两条新方法路线，训练 seed42，正式单候选评价 seed2101。92 分是目标，不是已取得的结果。正式评价保留 native uniform10 和 shift5_10 两个独立读出，不取二者最好值。

| 路线 | SFT 初始化及动作专家 | SFT 变化 | 后续 RL |
|---|---|---|---|
| A_DDP_TIME | 原 F-SFT step100000；原24层 DDP DiT、qwen_proj、8×4编码 | 迁移源 shift5 时间采样与时间加权 FM；完整动作头继续监督训练 | 同一份 R_SOURCE Flow-GRPO，永久 reference 为新的 SFT |
| B_SOURCE_ACTION_DIT | 源 standalone ActionDiT；随机初始化 seed42 | Qwen 的8个归一化 action-query 进入原生 text_embedding/cross-attention；完整新动作头 SFT | 同一份 R_SOURCE Flow-GRPO，LoRA 映射到该专家真实 attention |

两组都冻结同一 Qwen/history/上游输入，维持原相机、processor、合法状态、导航、8个未来点、归一化及解码。B 不加载 Wan、MoT、VAE 或视频生成模型。B 不是公开 SimWAM 已发布权重的复现，也没有继承其 Wan-derived 初始化。

先完成各自 SFT，建立各自新基线，再计算 `自己的 RL − 自己的 SFT`。A 相对历史 SFT 的变化同时包含额外监督训练曝光，不能仅凭涨分将全部变化归因于时间加权。A/B 参数量、初始知识、监督预算和各自困难人口不同；这是两条迁移路线的比较，不是只改一个层的纯因果消融。

运行状态、正式源提交及验收证据以同目录 `registration.json`、`validation.json`、`dispatch.json` 为准；不存在的后续结果不以计划代替。

2026-10-16 18:08 UTC 后续执行快照见 [里程碑与资源状态](milestones_20261016T1808.md) 和 [机器可读进度](progress_20261016T1808.json)。两条旧追加实验已经无损暂停；两组新 SFT 正常运行，A 的5k双网格完整 DEV 评价已启动。快照不是当前实时进程状态。

A的首个5k完整DEV和固定TRAIN诊断已完成，见 [5k SFT配对分析](interim_A_SFT_update005000.md)：DEV native +0.245277个百分点、shift5 −0.023871个百分点，两者区间均跨零，TTC仍有新增违规。这是SFT阶段结果，尚不是RL增益或Navtest结论。

A的[10k完整结果与配对曲线](interim_A_SFT_update010000.md)已完成：DEV native +0.204636个百分点、shift5 −0.168002个百分点，区间仍跨零；两个网格均有新增安全违规。继续原20k终点，不以中期最高点替代终点，不将SFT增益当作RL收益。

B的[10k完整DEV与TRAIN诊断](interim_B_SFT_update010000.md)已完成：DEV native/shift5为91.689072/91.687880，独立ActionDiT已从随机初始化学到有效监督策略；仍低于原F-SFT的DEV水平，不能称为架构优越或RL成功。B继续既定100k SFT预算。

19:21 UTC左右，[A组20k SFT正式完成](A_SFT_complete_training.md)：八rank与640000次场景曝光已实际核算，覆盖全部101592个TRAIN场景。自己的单候选TRAIN筛选已在三台主机24卡启动，终点双网格评价并行进行；正式RL仍等待完整筛选与终点八卡验收。

A的[20k SFT完整部署基线](final_A_SFT_update020000.md)随后完成：DEV native为93.720093（相对初始化+0.100856个百分点），shift5为93.460310（−0.114439），区间均跨零。原生无新增零分但新增3个TTC违规。后续RL固定从此20k SFT出发并与它配对，不选择5k中期高点。

19:55 UTC，[A20k完整TRAIN筛选及hash核验](A_SFT_full_TRAIN_screen.md)完成：101,592个场景，单候选PDMS94.190124，固定困难清单21,367个场景。满分/零分人口是单候选统计，不能当作G8等分组比例。A的八卡真实RL恢复/合并部署验收已于19:56 UTC启动，验收权重不进入正式初始化。

20:02 UTC，[A端点八卡验收通过并启动正式RL](A_endpoint_RL8_and_formal_start.md)：连续4次与2+恢复2次的实际状态完全一致，非零LoRA合并及原始观测部署通过；正式100步原生恢复点已发布。正在运行的数值源码仍为225b，另有一个不影响执行的A/B身份描述文案更正在报告中说明。该启动记录本身不构成RL部署收益证据，后续结果见上方5k报告。

## 旧实验资源调整

最新核查见[20:27–20:30 UTC实际资源与旧实验处置](progress_20261016T2030.md)：旧RL均已停止或完成；A正式RL与B SFT各用八卡，三台八卡空闲；独立Base训练保持。[19:40 UTC筛选阶段快照](resources_20261016T1940.md)保留为历史记录，不代表当前状态。

2026-10-16 18:51–18:52 UTC重新核对了全部八台允许主机，见[实际进程与资源快照](resources_20261016T1852.md)。旧追加线保持暂停；新A/B训练活跃；vla-zt、rl-zt2、tsinghua-zt在该快照时八卡空闲。新评价控制器继续等待已登记里程碑，不恢复旧实验。

根据用户“腾出资源给新布置实验”的指示，停止两条旧追加实验及其自动训练/评价控制器：

| 旧实验 | 保存的真实 optimizer 更新 | fresh collections | scene 曝光 | 新候选 | 保存阶段 |
|---|---:|---:|---:|---:|---|
| R_CANONICAL_LAST_ODE | 15,553 | 3,889 | 31,112 | 248,896 | INNER_COMMITTED，next_inner=1 |
| R_CANONICAL_X1_SAFETY25 | 13,094 | 3,274 | 26,192 | 209,536 | INNER_COMMITTED，next_inner=2 |

八个 rank 均完成原生保存和退出，权重、Adam、scheduler、RNG、待更新链及剩余 inner 全部保留；每个 checkpoint 的11个原生文件已经核查。不是从 merged export 恢复。四个相关控制器已退出，调度策略记录了任务暂停，禁止自动重启旧跟进实验。

LAST_ODE 的15k DEV native 为93.36988078，低于原 SFT 的93.6192366；没有支持继续优先占卡的部署增益。SAFETY25 在10k相对同一步 X1 的 DEV native 仅+0.0968153个百分点，log-cluster CI仍跨0；困难 TRAIN 上较大的提升不能替代 DEV 或 Navtest 证据。两条均是用户重新分配优先级后暂停，**未完成原20k终点，不能写作完整阴性实验**。

原 R_SOURCE/R_CANONICAL/X1 已完成，原 C/N 保持暂停。本次没有恢复它们；没有停止、修改或抢占 G/Base/其他真实任务。身份不明或容器内不可见的 GPU 进程保持不动。

## 源码复用及必要适配

上游固定：[H-EmbodVis/SimWAM@1433ee7e](https://github.com/H-EmbodVis/SimWAM/tree/1433ee7e06a62bd8b89da74455df0d82bda7b726)。完整来源/hash及补丁在 [upstream_lock.json](../../starVLA/rl/fm_transfer/vendor/upstream_lock.json)。保留实际 MIT LICENSE。

- 原样内容：ActionDiT 类实现；实际 self/cross-attention、RoPE、RMSNorm、DiTBlock；continuous scheduler；已有源 sampler、clamped probability、advantage、anchor、PPO 和 LoRA 逻辑。
- 接口适配：ActionDiT `action_dim:3→4`、`text_dim:4096→2048`；原 DDP 的 t=1−sigma、velocity 取负；B 使用源 continuous sigma×1000，不人为做 DDP bucket 量化。仅提取独立动作专家依赖，没有迁移 WanVideoDiT/MoT。
- 执行适配：新 SFT 全头 FP32参数/Adam，BF16算子，FP32统计；缓存为已验证原观测执行的 qwen_proj 前输出。原 VLAAgent 通过 opt-in 工厂分支加载 B；旧默认分支不变。
- 新实验选择：A 继续20k SFT，B 从随机头训练100k SFT；源时间加权与各自新基线；两组接同一 R_SOURCE。它们不冒充 SimWAM 论文训练预算。

新训练/部署命名空间是 `starVLA/rl/fm_transfer/`。RL 入口通过显式 runtime 接口调用现有 `scripts/analysis/simwam_recipe/train.py`；没有复制另一个 PPO trainer，也没有全局 monkeypatch。旧入口的原 SFT、LoRA 和 recipe 断言仍在。

## FM 数学与条件边界

两组使用源 scheduler：`u~Uniform(0,1)`，`sigma=5u/(1+4u)`，`x=(1−sigma)y+sigma*epsilon`，源目标 `epsilon−y`。A 以原 DDP velocity 的负值预测它，送入原 bucket 的时间为 `t=1−sigma`。B 直接运行源 ActionDiT。

源时间权重原样调用 `training_weight(sigma*1000)`：以源1000点网格计算归一化常量，再对每个样本的 velocity MSE 加权。FM repeat8 表示同一场景的独立噪声/时间重复，先按场景归约；不把它算成8个独立场景。

训练条件仅来自原 Qwen 归一化 action-token states。GT动作由原数据预处理单独读取；标签、奖励、地图和评分结果不会进入 condition。101,592个 TRAIN 标签准备完成，约69.4秒 CPU；没有把 DEV/Navtest 放进更新人口。原基模可能见过 DEV，仍只称“RL 更新未见”。

缓存身份：`80a2dd4b417a3154cbbf575044daea586d945b54e014eee22269d7bd48d39e0b`。执行要求 Qwen BF16、history FP32、FlashAttention2、normalized action states、scene batch1。A 的 qwen_proj 在 SFT 可训练，所以绝不复用旧投影后的 condition；RL 冻结自己的新投影后才可缓存其结果。

验收入口曾漏设 attention 环境变量，落到 VLAAgent 的 SDPA 默认值。真实观测检查拒绝了该执行：没有放宽容差、没有改写缓存。锁定原 FlashAttention2 后两组 raw condition 和原单候选 ODE 均逐位相同。该失败发生在正式启动前，保留日志；不据此否定使用正确后端的旧实验。

## 已执行验收及边界

1. 直接加载锁定上游 ActionDiT 作为 oracle；3维/4维、B=1/2：前向、输入梯度、全部68个 tiny 参数梯度完全一致。完整4维适配的源加权 FM loss/梯度完全一致。没有实例化 Wan/MoT。
2. 实际完整模型各8卡：连续8步与4步保存后恢复至8步。所有完整模型参数、Adam、scheduler、8个 rank 的 RNG/cursor，以及后4步 loss/数据次序逐位相同。
3. A 的359个张量/819,503,620参数，B 的824个张量/1,018,782,724参数都得到有限非零梯度；每个张量的抽查坐标均发生更新。完整参数一致性见第2项，不将抽查说成所有元素都改变。
4. 实际原始观测→原 VLAAgent：与 cache 路径同 root 的 native ODE 逐位相同。包括 B 的新头，仍无搜索/teacher/scorer。
5. 实际全模型 G8/K10链、随机步骤[7,8,9]、四次 source PPO更新：A192个、B480个 LoRA 张量实际更新；永久 reference 冻结。第一 inner ratio误差为0；后续真实参数更新会产生 clipping，不强制ratio=1。merge velocity最大误差分别约9.54e−7、5.59e−7。
6. 第5项使用**明确标记的人工 reward 向量做算术测试**，未进行驾驶评分，不能作为 RL 提升证据。正式 RL 前仍需各自完成 SFT、自己的单候选 TRAIN screen，以及真实 reward 的多卡有限验收。
7. 随后两组均完成**真实 canonical 评分的两卡短训**：连续4次 optimizer 更新，与保存 inner2后恢复至inner4的全部 LoRA/Adam/scheduler/RNG、实际链、old logprob、优势和真实 score逐位一致。每组16个逻辑候选、32次实际资格评分；A两rank均有PG，B一rank全零组，验证了该rank仍保留anchor且正确同步另一rank梯度。原完整冻结头和reference hash不变，所有LoRA张量实际更新。该验收使用8步SFT资格模型，仍不能代表正式SFT终点的RL效果；正式RL启动资格与规划结果单列。
8. 两组正式step0 export均从原 VLAAgent成功加载，分别通过2个固定DEV场景×2种网格的原入口一致性和真实评分 smoke。B的step0完整DEV基线随后已派发；不将2场景smoke当完整DEV结果。
9. 对真实reward更新后的非零LoRA做完整10步ODE合并检查：A的native/shift5最大误差3.58e−7，B为4.77e−7；不是只比较某一层或单步velocity。

A的step0全模型张量与原F-SFT全部相同，resolved配置相同，实际raw-observation双网格输出及score也与历史原始记录相同。因此复用已有完整1696 DEV、512 TRAIN诊断的起点结果，并重新验证全部既有payload hash；不重新花GPU/评分预算。这只适用于A的初始化，后续新SFT和RL必须重新评价。B没有可复用的旧基线。

短训练实测每卡峰值约15.42GiB/19.11GiB。仅7个稳定 update 的数据+更新中位时间约0.301/0.435秒；粗算 A约1.7小时、B约12.1小时，尚不包含长期 I/O波动、checkpoint及评价开销。正式 ETA 应由正式进程近期完整窗口重新估计。

## 固定预算与后续顺序

| 项目 | A_DDP_TIME | B_SOURCE_ACTION_DIT |
|---|---:|---:|
| SFT optimizer updates | 20,000 | 100,000 |
| scene global batch | 32 | 32 |
| 逻辑 SFT scene 曝光 | 640,000 | 3,200,000 |
| 峰值 SFT LR | 1e−5 | 1e−4 |
| SFT milestones | 0/5k/10k/20k | 0/10k/25k/50k/75k/100k |
| 后续 RL optimizer updates | 20,000 | 20,000 |
| RL fresh全局rollout / scene / candidate | 5,000 / 40,000 / 320,000 | 同左 |

SFT共同 AdamW(.9,.95), eps1e−8, wd.01, warmup3%+cosine(min ratio.1), FMrepeat8，SFT clip1。RL仍为源有效 clip-gradient=0；不得混淆。两组RL共同：G8/K10、shift5、last3随机、innovation±5、relative logprob clamp、global raw reward std、inner4、PPOclip.005、attention LoRA r16/alpha32、mean-anchor beta50和[.2,1,.5,.5]、源10:5:2训练奖励，canonical5:5:2正式评价，无BC/GTreplay/dual/learned noise。

新的困难清单必须来自**各自完整SFT**的 TRAIN 单候选 native seed2101 canonical<.9；不能借旧 F-SFT 清单冒充自己的困难人口，更不能用 G8 mean/worst/best。固定后继续复用原 source epoch sampler。两组 native/shift5 的 IL→RL增益分开；不把训练随机策略成绩或有限候选上界当部署分数。

当前A已完成自己的20k RL及DEV终点评价，原生中期收益未保持；B继续自己的SFT。两组完整比较、最终Navtest、92目标及完整安全收益仍未完成，不因工程验收通过而宣称方法有效。

## 运行和恢复入口

命令必须在登记的不可变 source snapshot 下执行；每次派发重新读取当前授权和实时 GPU 占用。

```bash
python -m scripts.analysis.fm_transfer.prepare_data --help
python -m scripts.analysis.fm_transfer.analyze_sft_training --run COMPLETED_FORMAL_SFT_RUN --output NEW_BUDGET_JSON
python -m scripts.analysis.fm_transfer.analyze_screen --screen COMPLETE_TRAIN_SCREEN --hard-manifest VERIFIED_HARD_MANIFEST --output NEW_SCREEN_ANALYSIS
python -m scripts.analysis.fm_transfer.qualify_math --help
python -m scripts.analysis.fm_transfer.qualify_deployment --checkpoint NATIVE_QUAL_SFT --output QUAL_JSON
python -m scripts.analysis.fm_transfer.dispatch_sft --variant A_DDP_TIME --output NEW_FORMAL_RUN
python -m scripts.analysis.fm_transfer.dispatch_sft --variant B_SOURCE_ACTION_DIT --output NEW_FORMAL_RUN
# 同一正式run、相同source/配置，恢复完整Adam/RNG/cursor：
python -m scripts.analysis.fm_transfer.dispatch_sft --variant A_DDP_TIME --output EXISTING_RUN --resume SEALED_NATIVE_CHECKPOINT
python -m scripts.analysis.fm_transfer.export_policy --checkpoint NATIVE_SFT_OR_RL --output NEW_EXPORT
python -m scripts.analysis.fm_transfer.evaluate_policy --export EXPORT --export-sha256 SHA --output SCORES --cohort DEV --shard 0 --shards 8
torchrun --standalone --nproc_per_node=8 -m scripts.analysis.fm_transfer.train_rl --config configs/flow_rl/simwam_source_ddp.yaml --sft-checkpoint OWN_SFT_ENDPOINT --manifest OWN_FIXED_HARD_MANIFEST --output NEW_RL_RUN
```

这些 CLI 已实现；完整 RL/evaluation 路径的执行状态单列，不把 `--help` 通过当作真实学习验收。SFT完整初始状态、rolling里程碑和资格权重分开保存。中断时向已识别 rank0 发送 SIGUSR1，由全部rank在原生边界保存退出；不能用SIGKILL代替无损停止。旧实验停止回执与所有资格失败证据保存在项目私有资产目录，仓库只提交代码和汇总，不上传权重/缓存/场景数据。

SFT终点核算入口要求全部rank正式完成，逐rank检查更新日志连续、无重叠、cursor及TRAIN人口匹配；FM repeat不算独立场景，八rank的更新不乘成八倍逻辑更新。7项针对性核算测试及CLI smoke通过，正式全量核算待各自SFT终点执行。重复/回滚日志须明确审计祖先，不静默去重。累计worker wall、rank0训练wall和派发至最后训练边界分别报告；未单独记录的模型加载、逐次保存和退出时间明确列为统计限制，不补造精确成本。

全量TRAIN筛选分析复用已验证的完整清单，不改变canonical<0.9成员规则。它区分全人口与困难人口、精确零分/满分和安全分项，并单列失败/重试实际推理、worker累计时间与job wall。这里每场景仅一个固定ODE样本，不能把零分/满分场景比例写成G8等分组比例。4项统计/失败成本测试及CLI smoke通过；完整筛选结果须等所有分片完成后才能生成。
