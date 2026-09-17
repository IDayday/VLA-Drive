# Action-only Flow-GRPO 验收报告

**最终结论：NOT_READY。** 代码、配置、测试、两套初始化、真实短训、恢复、导出和原协议评估均已交付，但第 22 项的候选分块 BF16 参数梯度对照失败。未放宽容差、删除失败、冻结额外参数或改造奖励；暂不应启动长周期训练。工程结论与下述小集 EPDMS 变化相互独立。

## 实现与参数合同

实际策略为用户发布的 QwenOFT + GR00T FlowmatchingActionHead；minimal history/action prompt，Qwen normalized last_hidden_state，8×4 动作，10 步。用户 action-only 配置原本关闭辅助任务，两套 GRPO 都按后续指示冻结 visual，继续训练其余源 SFT 可训练参数。

| 初始化 | 源 SFT 可训练参数 | GRPO 可训练参数 | 真实 RL-only 非零梯度张量 |
|---|---:|---:|---:|
| frozen visual step100000 | 2,233,120,260 | 2,233,120,260 | 672/672 |
| unfrozen visual step120000 | 2,640,077,316 | 2,233,120,260 | 672/672 |

visual 冻结是用户明确授权的唯一新增例外。lm_head 与输入 embedding 共享参数，源 SFT 已冻结，继续冻结。unfrozen 权重多出的 6 个 inactive agent_dino_head 张量保留且冻结。没有 LoRA、额外冻结、分辨率降低或新增辅助任务。源配置冲突解析、参数 aliases、逐参数 shape/dtype/group/梯度来源和 optimizer 覆盖证据见 `preflight_action_*/`、`integration_action_frozen/`、`integration_action_unfrozen_deterministic/`、`action_run_evidence/`。

**IMPLEMENTED：** `starVLA/rl/flow_grpo/` 已包含 SFT 合同、共享条件/velocity、锁定参考的 score-corrected SDE、完整链/old log-prob/组优势、官方 NAVSIM CPU spawn 服务与缓存、独立完整策略 reference、GRPO + conditional transition KL + 原 SFT replay、Accelerate/ZeRO 分布式更新、完整 checkpoint/resume/export、日志与评测。配置与本机启动器在 `configs/flow_grpo/`、`scripts/flow_grpo/`；合同、公式和命令在 `docs/ddp_flow_grpo/`。沿用已有训练依赖；仅为权重下载补装系统 aria2。

## 25 项验收矩阵

TESTED 表示真实执行；结果列明确 PASS 或 FAIL，不能把执行失败理解为通过。

| # | 合同 | 状态 / 结果 | 证据与范围 |
|---|---|---|---|
| 1 | 原 ODE 等价 | TESTED / PASS | 两套 f9449d55 独立源码 oracle，输出 max_abs=0 |
| 2 | 原 SFT loss 等价 | TESTED / PASS | 两套原 forward/head 与共享内核全部 loss 分量相同 |
| 3 | 投影一次/token 语义 | TESTED / PASS | 真模型 hook，qwen_proj 调用一次；原源语义保留 |
| 4 | 导出原 evaluator 等价 | TESTED / PASS | frozen 989、unfrozen 995 张量逐位一致；原 VLAAgent 输出 max_abs=0 |
| 5 | 锁定上游 mean/std/logp | TESTED / PASS | `sampler_numerical_comparison.json`，全部 10 步 |
| 6 | Normal/sum/mean/KL | TESTED / PASS | `tests/flow_grpo/test_math.py` FP64 严格对照 |
| 7 | sqrt(dt)/边界/ODE | TESTED / PASS | 首末步、小噪声、noise=0 无伪 log-prob |
| 8 | 经验均值/方差 | TESTED / PASS | 固定种子 200,000 个 Gaussian 样本 |
| 9 | old/current ratio及扰动 | TESTED / PASS | 两套真模型不更新 max_abs_logratio=0；`action_*_perturbed_ratio.json` 分别扰动 Qwen/action 可检测 |
| 10 | 完整 reference KL | TESTED / PASS | 相同时 KL=0，分别扰动 Qwen/action 后 KL>0；reference 自己编码 |
| 11 | manifest/optimizer | TESTED / PASS | 两套实际权重 strict load、tied alias 值检查、源/actor 名单及唯一 visual 例外，optimizer 无遗漏/重复 |
| 12 | RL/SFT/ref/aux 梯度 | TESTED / PASS | 两套**首个实际训练组、真实官方 rewards/advantages** RL-only：语言309、history4、动作359均非零；见 `action_*_real_reward_gradient.json`。action-only 辅助不适用；原完整 SFT auxiliary-only 回归另存 |
| 13 | detach/ref 负例 | TESTED / PASS | 真模型拒绝 detached current；误送 current 条件给 reference 被检出 |
| 14 | 优势符号/全等组 | TESTED / PASS | 严格低维测试；真实短训全等组保留，优势为0 |
| 15 | G/K/累积缩放 | TESTED / PASS（分块失败见22） | 原生产 loss wrapper 的 replay 每场景一次；真实单卡累积2与双卡累积1的全部 Adam 一/二阶矩逐位相同 |
| 16 | rank 不等有效数/全等组 | TESTED / PASS（限定范围） | 真实双进程 Gloo 不等计数对照全局基准；真 DDP 模型训练使用每rank完整等大小组并实际包含全等/非全等组。生产异常全rank失败，不支持静默删候选或可变场景batch |
| 17 | 官方 reward 一致性 | TESTED / PASS | 实际 NAVSIM cache，逐条原 evaluator、串行/spawn、候选顺序及 batch 组合一致 |
| 18 | 零分/异常/timeout | TESTED / PASS | 真零分有效；缺缓存、worker错误和超时测试；50步评分异常0 |
| 19 | normalization/坐标/时间 | TESTED / PASS | 原1225变换往返与8×0.5s轨迹，原 evaluator实际评分 |
| 20 | future隔离/split | TESTED / PASS | 真模型污染未来标签后条件不变；RL train/val与navtest隔离 |
| 21 | 真模型单卡/多卡更新 | TESTED / PASS | 单卡、单卡CPU optimizer/reference offload、双卡50步25→50恢复；另一初始化1→2恢复 |
| 22 | checkpoint/分块梯度 | TESTED / **部分FAIL** | 确定性 checkpoint on/off 672张量通过；候选chunk1/2在同链全部10步上失败，两套均失败；见 `candidate_numerics.md` |
| 23 | BF16网络/FP32概率 | TESTED / PASS | 两套实际checkpoint的FP32概率/velocity梯度对FP64 Normal；见真实 integration报告 |
| 24 | 连续/恢复完整状态 | TESTED / PASS | 双卡连续4 vs 从2恢复到4；模型、2份optimizer、scheduler、2份Accelerator RNG、2份额外RNG/消费游标全部逐位相同；data_workers=2、prefetch=2 |
| 25 | 冻结/ref不变及actor更新 | TESTED / PASS | 各rank训练前后完整张量哈希不变；完整离线权重审计及真实RL梯度证明actor更新，非仅weight decay推断 |

轻量回归：`action_runtime_tests.log` **34 passed**；`baseline_regression_tests.log` **36 passed**。此前合并运行 `combined_precheckpoint_tests.log` 70 passed。后续新增实际奖励梯度、扰动ratio、完整状态和导出检查均独立保存结果；没有拿 toy tests 替代真实模型验证。

## 数值失败与限制

第22项未改动容差：`atol=2e-6, rtol=2e-3`。开启与训练相同的 PyTorch deterministic 算法修复了 checkpointing 对照；候选分块仍失败。冻结版首次失败的 Qwen q_proj 梯度最大绝对差为 1.90735e-5，解冻版为 6.91414e-6；完整探针还显示其他张量有误差，不能只看这两个首次断言值判断无害。

固定两候选探针中，forward mean/logp 最大差为 1.19e-7 / 4.77e-7，condition梯度相对L2差约2.11e-6；BF16参数反向/累加后的差异更大。试验性 FP64 条件梯度归约未解决问题，实验补丁和失败报告均保留，该变更已从交付运行内核撤回。**未通过这个 gate，即使默认chunk=1的短训、缩放、恢复、导出成功，也不能声明 READY_FOR_FULL_RUN。**

锁定上游 sampler 的10步 FP32对照：mean最大差0，std最大差1.86e-9，log-prob最大绝对差0.00293、最大相对差3.62e-7（随机离分布 x_next 的大幅负log-prob）；使用原定 atol+rtol 断言。真实网络/实际生成链的概率检查另见 integration：最大log-prob差约8e-6，未混用联合概率比与dimension-mean surrogate。

## 真实短训与评估

实际训练代码 HEAD：`9bcf3b310075ecd9ebc2bf144ed96db664ad4e25`，执行源文件指纹 `739fc4b888bec23b286a270cb29dfd400d47258d3b33884504ebe5f5d425fe35`。后续交付提交补充验证脚本、文档和报告，运行内核指纹保持一致。原工作区基线 `9fe1459b8f6ab69a15274450ec301d541209bedd`；用户原有未提交修改以 `inherited_dirty.patch` 单独保留，未认领为本次改动。

源 action-only 代码：`IDayday/VLA-Drive@f9449d55bea6895a7a0bd86d09d7ab85fd353f26`；Flow-GRPO：`879042cf5707f8b90daa98d147d7deac2317c5da`；SimWAM：`68b426c162827cb7701396895dbb3572d29f3420`；原官方 DDP 回归代码：`8cefcac46e5944add529e1be19cba78bc06cc2bd`。逐文件锁定及许可证见 `reference_lock.json`、`THIRD_PARTY_NOTICES`。

本机8×A800 80GB，主短训实际使用GPU1、5两卡，PyTorch2.5.1+cu124、Accelerate1.5.2、Transformers4.57.0、Diffusers0.35.2，现有DeepSpeed/FlashAttention。完整环境、argv、各次启动UTC、GPU、依赖版本、base processor SHA及dirty diff SHA在 `action_run_evidence/*/source_environment_*.json`。

固定原始三视角1024×576及原processor、G=8、K=10、全部672个可训练张量、reference系数0.01、原action SFT replay系数0.1、源LR×0.1、noise=0.1，未改变任务开关。navtrain10,000个token，与navtest12,146个token不相交；从RL训练流留出16个validation，余9,984用于RL。该16场景可能已被源SFT使用，不代表独立泛化测试。

- frozen：双卡50个optimizer updates，在25保存、恢复到50；每次全局2个场景，共100组800候选。峰值 **32.93 GiB/卡**。rollout/reward/reference合计约385.09秒，更新合计735.45秒，另有加载/保存时间。奖励均值0.92003，按update均值范围0.5625–1；真实零分候选2.75%、全等组32%，全部保留。ratio范围[1,1]，评分错误0，conditional KL平均0.01869、最大0.13452。没有NaN或程序评分异常。
- unfrozen初始化：按用户指示冻结visual，单卡1→2保存恢复，峰值50.66 GiB；独立导出/原评测通过。
- frozen单卡CPU optimizer+reference offload：1次真实更新，峰值13.15 GiB，保存与完整不变性检查通过。
- 单卡累积2/双卡累积1对照：实际所有Adam moments逐位相同；诊断clip设1e6以避免clip隐藏缩放错误，主短训仍为clip=1。实际命令、逐rank日志与摘要在 `action_run_evidence/` 和 `action_training_summary.json`。

| 初始化及固定16场景原单候选ODE评估 | SFT EPDMS | RL EPDMS | 变化（百分点） |
|---|---:|---:|---:|
| frozen，50 updates | 0.96287105 | 0.96759443 | +0.47234 |
| unfrozen，2 updates | 0.96262917 | 0.96257795 | -0.00512 |

这是当前NAVSIM v2 one-stage官方协议，非文件名中的发布成绩复现，也非全量Navtest。小集变化不证明统计显著提升或排除遗忘；没有按成绩改奖励、挑种子、替换候选或选择最佳checkpoint。部署评估均使用相同场景/seed和原10步ODE，无oracle选择。`eval_action_*`保存逐场景分项、轨迹和结果。

固定16个训练场景的noise校准：SFT ODE均值0.95221；noise 0.05/0.1/0.2的SDE group mean分别0.94471/0.93038/0.92309，XY std分别0.03961/0.06130/0.11097。全部设置均报告且ratio检查为0误差，保留预定0.1；oracle仅作诊断。见 `action_calibration_summary.json`、`calibration_action_frozen/`。

原完整SFT通用回归另有50步、auxiliary-only梯度、导出和连续/恢复证据，保留在 `integration_deterministic/`、`full_short_parameter_updates.json`、`export_full_sft_regression/`、`resume_deterministic_comparison.json`及`scheduler/RNG`补充 `resume_full_sft_supplement.json`。它们不代替本次action-only验收；历史失败报告也保留。

## 交付状态与入口

- **IMPLEMENTED**：上述全部10类职责、两套源配置、本机脚本、自动测试及文档。
- **TESTED**：25项均有执行证据与范围说明；第22项候选分块明确FAIL，其余结果如表。
- **NOT_RUN**：长周期训练、大规模搜索、8卡完整真模型训练、全量Navtest，以及真模型ZeRO1/纯torch DDP路径。它们没有被宣称通过。本次生产验收使用ZeRO2；没有自动启动长训练。
- **BLOCKED**：无权重/数据/GPU等外部条件缺失。第22项是已执行的工程失败，不是缺资源导致未运行。

可复制的preflight、单卡诊断、多卡完整训练、resume、export、原协议评估及下载命令均在 `docs/ddp_flow_grpo/README.md`。代码工作区 `/mnt/project/DriveDreamer-Policy-flow-grpo`，本地分支 `codex/ddp-full-sft-flow-grpo`；未push、未创建PR、未改写原checkpoint或停止其他训练。

导出产物：`runs/action_frozen_short/export_update50/`、`runs/action_unfrozen_short/export_update2/`。完整状态：各自 `checkpoints/update_000050/` 与 `update_000002/`。两个原始权重已下载且分片/整文件SHA验证通过，见 `download_completed.json`。所有测试/训练均已限定运行上限；当前不具备宣布完整工程就绪的依据。
