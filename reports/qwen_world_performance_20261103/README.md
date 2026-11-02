# Qwen-World：以真实单轨迹 PDMS 为目标的规模化 RL

本轮从 `987bc614aec5a042f2df8e8590d6f2479b3e1010` 开发。研究进行中，工程资格通过不等于算法提分。

**最高已确认完整部署成绩仍为 SafetyEP Control 10k：91.45359756。** 原始统一 IL70k 为 88.79290991。本轮新算法尚无完整 NAVTEST 结果。

实时 HTML：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。原总报告 `/mnt/project/qwen-world-rl-optimization-20261030/report.html` 的最前方已接入本轮面板，沿用唯一原报告守护，每 30 分钟及阶段事件更新。

## 已经启动的训练与当前判断

2026-11-02 20:59 UTC 附近快照：S1、S5、S0、S2、S3、FAST PACK4、S6 七条真实八卡训练正在进行，另有旧综合 I 继续训练；实时数量以 HTML 新鲜八 rank 日志为准。[逐路线实际更新与曝光](results/PROGRESS_20261102T2059Z.json)。本轮尚无完整 NAVTEST 结果，不能宣称已经改善部署成绩。S4 保存于 update320/inner0、LowLR 保存于 update30/inner2，资源让位于优先研究，均未被判定无效。

S1 从临时 wm4 保存于 update205/inner1，按同一实验完整状态恢复到长期可用 tsh5 共享资源。其与独立 S0 对照的前 221 个更新、八 rank 共 1768 条记录中，轨迹、old log-prob、优势、mask、参数摘要、loss、梯度和 LR 完全一致。S1 的 rescue 从 0.5 epoch 才启用，早期目标相同，不能在启用前判断机制无效；该一致性同时支持本次原生迁移未改变训练。

已另行登记固定 TRAIN2048 真实 B1 诊断，先评测同一 IL70，再与本实验自己的后续 checkpoint 逐场景配对。它属于训练分布诊断，不作为独立泛化验证，更不作为 NAVTEST 完整成绩。

两项吞吐开发已经通过真实八卡原生恢复检查：冻结 IL 参考缓存的连续训练与开缓存保存/恢复，参数、Adam、scheduler、全部八 rank RNG/scene stream 逐位一致，参考调用 12→3；PACK4 的连续训练与保存/恢复也逐位一致。PACK4 首次 CPU 比较将行政 invocation UUID 误判为差异，原失败记录保留，另行严格复核仅排除该顶层 UUID 后通过。缓存已经受审计地迁移到同一 S1；PACK4 因 BF16 batch 形状改变，以独立 IL70/Adam0 实验运行，未接续小 batch 的 RL 权重。

旧综合 I 的完整 2k 点为 **91.02554897**，比 IL70 高 2.232639 分、比历史 Control 最高低 0.428049 分。逐场景全量分析显示，原 IL 高分组的额外下降贡献 −0.436213 分，基本解释其与 Control 的差距；同安全指标下 EP 下降是主要相关现象，不能据此声称已证明参数更新的因果机制。[全量分析](analysis/INTEGRATED_I_u002000_FULL_REVIEW/SUMMARY.json)

TRAIN 定向诊断：15 个固定 B1 零分场景中，G16、G32 各发现 12 个严格安全高分场景，并集 13 个；38 组都不是奖励全零或 SafetyEP 原始优势全零。独立采样种子、不同成本的小样本不能证明 G32 更优。另有真实合法 GT 的 2035 次评分，其中 245 条严格高分且比 B1 高至少 .005，但 255 条 GT 比 B1 低超过 .005，不能盲目模仿 GT。[采样简报](results/TRAIN19_G16_G32_REVIEW.json)、[GT 审核](results/GT_TEACHER_RESEARCH_READOUT.json)

新独立 **S6 Verified Quality** 从第一轮曝光同时学习低分修复和安全高分中的效率改善，真实 B1 条件的 ODE 末步与 CFM 联合监督，共享原 β50 约束预算。冷启动时原 IL anchor 梯度很小，现用真实 PG 梯度 EMA 提供辅助梯度校准下限；资格阶段已验证自然教师触发与八 rank 原生恢复，正式 native0 也核对了同一 IL70、空 Adam、零曝光。它尚无完整部署成绩。[算法设计](../qwen_world_verified_quality_20261103/ALGORITHM_DESIGN.md)、[资格实测](results/S6_GRADIENT_READOUT.json)

## 当前路线

| 优先级 | 路线 | 主要目标 | B / G / M |
|---|---|---|---|
| 0 | S1 Nominal Rescue | 将真实严格高分候选吸收到部署 B1：CFM + 真 ODE 末状态 SmoothL1 | 128 / 16 / 4 |
| 0 | S5 Performance Max | 全量覆盖→候选吸收→主动历史保持→低 LR 精修 | 64→128 / 16 / 4 |
| 0 | S6 Verified Quality | 低分修复与安全高分效率改善，PG-EMA 校准 CFM/真实 ODE 末步 | 64 / 16＋条件 teacher-only 16 / 4 |
| 1 | S0 Large SafetyEP | 大场景覆盖、有效曝光调度与长训 | 128 / 16 / 4 |
| 1 | S2 Conservative AWFM | 原生 FM 优化，负优势由 rollout 一致性项稳定 | 128 / 16 / 2 |
| 1 | S3 FPO | 逐 MC CFM energy-ratio PPO 代理 | 128 / 16 / 2 |
| 2 | S4 Temporal Rescue | 真实时序安全代价 + 已验证候选吸收 | 128 / 16 / 4 |
| 可选 | S0 Low LR | 独立 IL70，action/memory 3e-5/3e-6 | 128 / 16 / 4 |
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

本次六台临时资源至北京时间 2026-11-03 08:00（UTC00:00）：wm4、rl2、tsh2、wmzt、tsh3、wm2。现行 worker 的 45 分钟保存余量会从 23:15 UTC 开始主动保存；独立 guardian 最迟 23:45 请求保存，并在硬期限前仅按已核验自有 PID 完成退出。永久 B/D IL 保护保持，其他任务和未知进程不回收。

本分支已汇总原 S0–S5、冻结参考缓存、PACK4 与 S6 的全部实现。在线训练仍固定各自已验证的源码，不因仓库整合而热切换：基础 `5e5220c`、S1 缓存执行 `c8a0440`、FAST `cc45f9b`、S6 `8067e13`、CPU 调度 `a1a4c3c`。完整来源及实测证据见 [结果索引](results/NEW_EVIDENCE_INDEX.json) 与 [进度快照](results/PROGRESS_20261102T2059Z.json)。

首窗口测得约 144–155 GPUh/epoch（八卡，非成本上限，不含新增辅助、评测、checkpoint 和失败重放）。尚不能据此判断 B128、G32 或长期训练提高了部署上限；后续完整结果会分别回答。
