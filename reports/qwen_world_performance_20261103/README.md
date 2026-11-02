# Qwen-World：以真实单轨迹 PDMS 为目标的规模化 RL

本轮从 `987bc614aec5a042f2df8e8590d6f2479b3e1010` 开发。研究进行中，工程资格通过不等于算法提分。

**最高已确认完整部署成绩仍为 SafetyEP Control 10k：91.45359756。** 原始统一 IL70k 为 88.79290991。本轮新算法尚无完整 NAVTEST 结果。

实时 HTML：`/mnt/project/qwen-world-performance-rl-20261103/report.html`。原总报告 `/mnt/project/qwen-world-rl-optimization-20261030/report.html` 的最前方已接入本轮面板，沿用唯一原报告守护，每 30 分钟及阶段事件更新。

## 已经启动的训练与当前判断

2026-11-02 18:44 UTC 快照：S1、S5、S0、S2、S3 五条真实八卡训练正在进行，S4 和 LowLR 等待合适资源。[逐路线实际更新与曝光](results/PROGRESS_20261102T1844Z.json)。所有路线尚不足 0.1 epoch，无本轮完整 NAVTEST 结果，不能宣称已经改善部署成绩。

S1 从临时 wm4 保存于 update205/inner1，按同一实验完整状态恢复到长期可用 tsh5 共享资源。其与独立 S0 对照的前 221 个更新、八 rank 共 1768 条记录中，轨迹、old log-prob、优势、mask、参数摘要、loss、梯度和 LR 完全一致。S1 的 rescue 从 0.5 epoch 才启用，早期目标相同，不能在启用前判断机制无效；该一致性同时支持本次原生迁移未改变训练。

已另行登记固定 TRAIN2048 真实 B1 诊断，先评测同一 IL70，再与本实验自己的后续 checkpoint 逐场景配对。它属于训练分布诊断，不作为独立泛化验证，更不作为 NAVTEST 完整成绩。

两项吞吐开发保留为独立证据：冻结 IL 参考项缓存在真实完整模型上四 inner 的梯度、参数、Adam、scheduler、RNG 逐位一致，参考调用 12→3；一致合批 PACK2/PACK4 的真实采样与更新首更 ratio 偏差均为零。两者的八卡原生恢复验证仍在推进。前者仅在受审计的执行迁移后用于既有实验；后者因 BF16 batch 形状改变，正式运行必须独立从 IL70/Adam0 开始。

## 当前路线

| 优先级 | 路线 | 主要目标 | B / G / M |
|---|---|---|---|
| 0 | S1 Nominal Rescue | 将真实严格高分候选吸收到部署 B1：CFM + 真 ODE 末状态 SmoothL1 | 128 / 16 / 4 |
| 0 | S5 Performance Max | 全量覆盖→候选吸收→主动历史保持→低 LR 精修 | 64→128 / 16 / 4 |
| 1 | S0 Large SafetyEP | 大场景覆盖、有效曝光调度与长训 | 128 / 16 / 4 |
| 1 | S2 Conservative AWFM | 原生 FM 优化，负优势由 rollout 一致性项稳定 | 128 / 16 / 2 |
| 1 | S3 FPO | 逐 MC CFM energy-ratio PPO 代理 | 128 / 16 / 2 |
| 2 | S4 Temporal Rescue | 真实时序安全代价 + 已验证候选吸收 | 128 / 16 / 4 |
| 可选 | S0 Low LR | 独立 IL70，action/memory 3e-5/3e-6 | 128 / 16 / 4 |

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

本次六台临时资源至北京时间 2026-11-03 08:00（UTC00:00）：wm4、rl2、tsh2、wmzt、tsh3、wm2；23:45 UTC 请求原生保存，随后仅按已核验自有 PID 做期限退出。独立 deadline guardian 已部署。永久 B/D IL 保护保持，未知进程不回收。

首窗口测得约 144–155 GPUh/epoch（八卡，非成本上限，不含新增辅助、评测、checkpoint 和失败重放）。尚不能据此判断 B128、G32 或长期训练提高了部署上限；后续完整结果会分别回答。
