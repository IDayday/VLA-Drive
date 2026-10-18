# QwenSourceFlowVLA：原生 Qwen＋SimWAM 动作策略

**2026-10-18 执行状态：正式8卡 IL 已开始实际 optimizer 更新；完整模型 IL 恢复、真实8卡 GRPO 中断恢复、独立原始图像部署验收通过。完整 IL、两组正式 RL、DEV/Navtest 成绩仍未完成。没有达到92或已有RL净收益的结论。**

本任务分支为 `feature/qwen-sourceflow-native-vla-20261018`。资产及实际运行收据位于 `/mnt/project/qwen-sourceflow-native-20261018`。报告只提交代码、身份及小型证据，不包含权重、原始数据、特征缓存或私有场景图。

旧 DDP C/N、SimWAM recipe、A/B、source-process 170k 已按用户指示停止；后两条170k停在25416和25109，后者保留inner1和旧链。原权重、恢复状态、失败证据不覆盖。见 [停止收据](old_experiment_stop.json)。其他项目不在本次停止范围。

## 正在执行的正式实验

| 项目 | 状态及固定合同 |
|---|---|
| 新 IL | `IL_seed42_v2`，`training-vlawm-tsinghua-zt` 单机8卡；实际开始更新 |
| IL 源码 | 不可变 `d4d0012769540af8378e013c072d6d8053d9f275`，后续部署/分析补丁不热修改该快照 |
| 初始化 | 官方通用Qwen＋官方通用Wan转换；重新初始化所有新接口，不使用资格验收或旧驾驶权重 |
| IL 数据 | 101592 TRAIN场景、1176 logs；global32、FM repeat1、seed42 |
| IL 预算 | 接口稳定1000步计入100000总步；随后完整Qwen/provider/action训练 |
| IL 观察与选择 | 固定20k/40k/60k/80k/100k全量DEV；全部完成后按DEV canonical PDMS最高选父模型，精确同分选较晚步数；Navtest不参与 |
| 两组正式 RL | 尚未开始；必须使用同一个完整IL父模型。CANONICAL 5:5:2、SOURCE 10:5:2，各50k，其他设置匹配 |
| 正式评价 | seed2101、单候选10步shift5 ODE、canonical NAVSIM v1.1；没有best-of-N、teacher选择或新增seed |

正式首个update的loss、梯度norm、数据顺序和LR与先前8卡global32资格profile**完全一致**，见 [首步核对](qualification_formal_first_update.json)。这验证启动修复没有改变正常IL计算，不是驾驶能力证据。

登记详情见 [formal_IL_registration.json](formal_IL_registration.json)。`IL_seed42_v1` 是一次**零optimizer更新**的启动失败：rank0发布身份文件后，迟到rank误认为已有旧run。隔离修复为rank0发布＋全rank一致性检查，实际两卡Gloo延迟rank测试通过，再以同配方新run `v2` 启动；失败目录保留。

## 原生模型与来源对应

锁定来源见 [upstream_lock.json](../../configs/qwen_sourceflow/upstream_lock.json)、[runtime_lock.json](../../configs/qwen_sourceflow/runtime_lock.json)。SimWAM固定 `1433ee7e06a62bd8b89da74455df0d82bda7b726`。Qwen固定官方2B Instruct权重revision `89644892e4d85e24eaac8bacfd4f463576704203`；Wan固定官方5B revision `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`。实际许可证随代码保留。

| 分类 | 本实现 |
|---|---|
| A：源计算复用 | ActionDiT、MoT缓存动作路径、连续时间scheduler、min/max codec、源FM时间权重、Euler符号、源LoRA和GRPO数学；vendor仅必要import适配 |
| B：必要观测接口 | Qwen28层到action30层的单调映射；完整视觉tokens到独立memory K/V；文本bridge到4096；当前8维proprio；padding/mask/RoPE适配 |
| B：执行/数值接口 | 原生3维动作/冻结condition descriptor接入已有事务trainer；FP32参数/动作概率、BF16冻结Qwen prefill后显式FP32条件转换；独立部署构造保持训练参数dtype |
| C：预登记研究设置 | Qwen异构IL LR、接口1000＋完整100k、global32、两种RL reward及各50k预算；明确不是已发布SimWAM驾驶权重的原始训练日志复现 |

当前三视图为left/front/right，导航和合法当前状态位于图像前；官方processor/chat template、mRoPE、DeepStack与因果forward保留。不调用语言generate或无用LM logits，不含未来标签/reward/GT地图。

官方smart resize固定目标面积384×672，实际该人口每视图grid为1×22×42，共**693个合并视觉tokens**。位置按实际grid计算，没有硬编码为756、压成8个摘要或复制黑图。有效非视觉文本也完整保留；缺失相机显式标记和mask。

Provider保留所有层的视觉memory：共享2048→1024 bridge、独立拷贝的逐层源LN/K/V/RMSNorm，time=0、横向相机tile位置；不把相机编号当未来时刻，不声称多相机已配准为BEV。观测K/V与action参数不alias；RL action LoRA不会改变冻结memory。文本2048→4096后使用源text embedding，合法proprio为vx/vy/ax/ay及四维导航one-hot（含unknown）。

ActionDiT保持30层、hidden1024、FF4096、text4096、24×128 attention空间和实际Linear输出head，输出8×3。在线没有Wan视频专家，没有旧DDP action query/history/qwen_proj/8×4编码/整数bucket/Beta时间或旧动作DiT。

## 通用初始化与真实数据

[Wan动作初始化证据](generic_action_initialization.json)：三个官方权重分片SHA全部核对；原转换严格加载825个Wan条目，动作backbone为820个条目，300个复制、520个按源规则插值，4个action encoder/head张量按源规则随机保留。完整轴和alpha规则及产物hash已保存，未以随机backbone或NAVSIM训练权重替代。Qwen身份见 [generic_qwen_identity.json](generic_qwen_identity.json)。

[TRAIN](data_TRAIN.json)、[DEV](data_DEV.json)、[Navtest](data_navtest.json) 均从原始Scene读取，分别101592/1696/12146场景、1176/16/136 logs。动作是当前rear-axle局部坐标x前/y左/heading弧度，8个0.5秒未来点，源absolute min/max归一化，不裁剪合法超范围标签。全TRAIN往返最大误差为 `[5.722e-6m, 2.861e-6m, 1.788e-7rad]`，见 [codec证据](qualification_real_TRAIN_codec.json)。

TRAIN678、DEV2、Navtest36场景存在原始帧时间间隔异常；保留官方Scene索引和0.5秒名义间隔，不自行插值或过滤。部分TRAIN原图路径缺失时仅使用另一已授权位置的同一原始JPEG；没有黑图或删场景。Navtest记录已准备，但尚未运行新模型正式Navtest。

RL固定作者公开hard list与TRAIN交集17037，排除352个DEV重叠；成员不由新模型分数动态变化。IL使用完整TRAIN。新模型训练不使用DEV标签。

## 已执行的资格证据

| 验收 | 实际范围和结果 |
|---|---|
| [源MoT/FM数学](qualification_math.json) | 相同非零外部KV，B1/B2输出和全部action梯度与源逐值一致；padding对逐样本reference一致；FM/weight/Euler符号通过 |
| [完整模型action loss](qualification_full_model.json) | 通用Qwen＋完整30层action，真实TRAIN图像；625 Qwen、156 provider、824 action张量有有限非零梯度 |
| [观测接口](qualification_observation.json) | 相机置换和导航影响condition；未来标签不进入actor；缺失相机mask；FP32 padding/逐样本相对RMS误差2.45e-6至1.69e-5，不声称bitwise |
| [完整IL恢复](qualification_full_IL_resume.json) | 实际两卡含完整训练阶段：连续/恢复模型1605张量、Adam4815张量、全部rank RNG和状态逐值一致 |
| [8卡profile](qualification_eight_GPU_profile.json) | 完整模型、FP32 Adam、global32；接口1步＋完整5步；测得真实吞吐和显存 |
| [身份发布](qualification_IL_identity_publication.json) | 实际两rank：迟到rank、重复启动、原身份恢复、跨rank错配及不覆盖检查 |
| [真实8卡GRPO](qualification_actual_RL.json) | 原始图像、64条真实策略候选和canonical反馈；四次实际Adam更新；240投影/480 LoRA张量；Qwen/provider/reference hash保持 |
| [实际任务梯度证据](qualification_actual_optimizer_moments.json) | 两组各240个A、240个B均有非零Adam一阶动量且step=4；AdamW weight decay不进入该动量，不能仅以参数数值变化冒充任务梯度 |
| [真实GRPO inner恢复](qualification_actual_RL_resume.json) | 在inner2停止再完成inner3/4，与连续4步的adapter/Adam/scheduler/RNG/旧链/旧概率/优势逐值一致 |
| [两奖励配置实际对照](qualification_actual_reward_pair.json) | 各自8卡、64条真实候选、4次inner；初始完整chain、old probability、官方反馈逐值一致，配置仅reward EP权重和variant不同 |
| [独立部署](qualification_standalone_deployment.json) | 不读取通用模型权重、teacher/训练cache/scorer，raw图像到完整10步ODE与原完整IL模型逐值一致 |
| [实际非零LoRA合并部署](qualification_actual_RL_merged_deployment.json) | 240个projection的B均非零；完整raw-image ODE合并前后归一化最大误差4.77e-6，物理x/y/heading最大误差0.000122m/0.000080m/0.00000763rad |
| [官方评分连通](qualification_ODE_scoring.json) | 2个真实TRAIN场景，原始图像→单候选ODE→官方canonical v1.1；完整分母、无异常填0 |
| [共享故障事务](qualification_recovery_shared.json) | 两rank CPU/Gloo 12种故障与inner恢复；不完整collection不发布latest |
| [实际两卡collection故障](qualification_actual_collection_failure.json) | 16条真实候选已评分，rank1在advantage提交前注入故障；两rank保留完整未finalize链，latest仍为READY0，未发布行为/假优势，物理费用保留 |

实际GRPO使用仅训练4步的资格IL父模型，8组中1组有真实PG，其他组按源语义保留anchor；同一全局梯度更新所有rank。两个TRAIN ODE评分均为合法0分，**只能证明数据/推理/评分链路运行，不能当新模型效果、正式基线或RL结论**。资格权重不进入正式IL/RL初始化。

首轮旧概率ratio误差为0；第二至第四inner产生真实clipping，未改0.005阈值。实际随机步骤[7,8,9]的sampling mean/std与保存链重算完全一致。源创新clamp、relative logprob clamp和部分随机步骤仍是经验代理，不称完整轨迹精确密度或无偏完整链PG。

## 实际失败与修复范围

失败日志、部分状态及原始检查点保留。没有把基础设施异常当reward0；下列失败没有产生被接纳的正式RL更新。

| 问题 | 修复及影响范围 |
|---|---|
| legacy framework import提前初始化NCCL | 原生CLI直接导入独立模型模块；实际两卡import验收通过，不改模型计算 |
| BF16冻结prefill直接进入FP32动作Linear | 明确condition FP32转换；采样、概率重算、部署统一；IL自身有梯度autocast路径不变 |
| 模型加载跨rank耗时超过短Gloo控制超时 | 全模型加载后先在既有主NCCL组完成启动barrier，再进入控制事务；未改概率或优化器 |
| 保存chain切片stride与实际rollout不同 | action输入显式同样contiguous；mean差约1e-6被严格资格检查发现，修复后逐值一致；未放宽clip或改写old概率 |
| evaluator缺invocations目录、官方NumPy不能直接JSON | 建目录、无损转换和记录hash；保留失败收据，不改轨迹或scorer |
| 独立Qwen from_config嵌套文本dtype仍BF16 | 完整训练权重strict加载也会被隐式舍入；部署构造显式FP32并逐张量检查dtype；修复后完整ODE误差0。正常IL/from_pretrained路径未受影响 |
| 正式IL首次启动身份发布竞争 | 零更新失败后隔离修复；新v2首步与已验收profile完全一致 |

未改变源SDE、时间网格、LoRA范围、loss、reward、anchor、PPO clip或正式预算。

分析工具另修复一处旧20k预算硬编码：ETA现在读取实际run的max_steps，适用于20k/50k/170k；资格run不输出正式训练ETA。已执行三预算检查及两条实际资格日志汇总，见 [RL资格成本](qualification_RL_cost.json)。这只改变派生统计，不改任何训练数学。

## 资源、成本与下一步执行

已核对当前授权八个主机、**64个不同物理GPU UUID**，没有把SSH别名算成额外GPU。正式IL占一台8卡；其余资源用于并行资格验收、合并部署验证及后续全量DEV/Navtest分片。vla-zt2与临时撤销主机不派发，其他项目真实进程不停止。压力占用脚本保持关闭。资源数量不改变单IL、两RL、单seed的研究范围。

8卡完整IL短profile的update中位数 **4.589秒**、峰值 **75.014GB**；完整恢复点保存78.76秒，model＋optimizer约38GiB。按该短窗口推算100k约 **5.38天、1033 GPU-hours**，不含评价/准备和未来共享扰动。不是已观测终点ETA。正式接口阶段约2.3秒/step，不能用它外推完整训练。完整IL保留全部五个里程碑及两个rolling恢复点，预计266GiB；旧实验/失败/资格权重不清理。

[managed_cost_snapshot.json](managed_cost_snapshot.json) 保留新managed worker各次启动、失败和完成的实际wall。它明确是费用子集，不含更早direct-launch资格、准备和仍运行的IL；不能把缺失时间填0，不能把共享worker wall当独占GPU时间。两组资格GRPO的真实候选数、按rank阶段时间、保存/恢复费用另列在上面的RL成本文件。

`milestone_queue_v2` 已以单CPU owner运行：每个登记IL检查点完成后，使用当时授权、可用且UUID去重的GPU分片跑完整1696 DEV；每scene噪声绑定seed2101，布局不影响候选数；缺失/异常不从分母消失。等待里程碑时每5分钟检查一次，不高频轮询训练日志。所有五次DEV后才封存IL父模型。

正式RL尚需在选定父模型上完成费用确认，再分别启动两台8卡的CANONICAL/SOURCE各50k；不能提前用欠拟合资格模型冒充正式父模型。RL每组12500份fresh全局rollout、100000 scene曝光、800000候选，四次inner不重复计新候选。最终锁定权重、按同一主入口完成IL和RL的12146全Navtest，报告RL−自身IL、G+/G−、NC/DAC/TTC转移和log-cluster bootstrap。

可执行入口与恢复边界见 [execution.md](execution.md)。本报告证明已实现、验收并启动新的训练链路；**尚不回答新模型RL能否获得净PDMS提升**，该判断必须等待充分IL及两条正式RL的固定单候选结果。
