# 共同 10k 回落：过程诊断和三组实际训练

用户报告入口保持为 `/mnt/project/qwen-world-rl-optimization-20261030/report.html`，按 30 分钟与阶段事件更新。本目录是 2026-11-01 的可复核静态证据，实时进度以该 HTML 和 H 的 journal 为准。

## 当前判断

8 条路线、全 8 rank 的指定连续窗口共 96,000 条更新日志、24,000 个场景组，以及 10k/12k 原生检查点元数据，没有发现共同的 10k 配置断点：LR 与 Adam step 连续，完整训练参数集合不变，数据 cursor/RNG 恢复和下一场景对齐，没有参考策略更换，也没有跨全量数据 epoch。审计不是对全部训练代码的无缺陷证明。

共同配置仍然可疑：原 50k cosine 在 10k 时 LR 仍为峰值的 92.7%，20k 时仍约 70.5%；每次只覆盖 8 个不同场景，连续更新 4 次。持续较大步长、有限场景梯度波动、已学行为遗忘是正在分别检验的假设，尚未证实因果。标量 anchor/PG loss 比例不能代替各自梯度分析。

## 已实现并实际部署

三组均从相同 IL70 开始，使用 TRAIN 全量 103,288 场景，保留动作 LoRA 和场景/预测 Memory 的 52.7M 可训练参数；不是只训练动作头。对照直接复用 F Control，不重复训练对照。

| 路线 | 主要改变 | 冻结科学源 | 实际资源 |
|---|---|---|---|
| Early decay | warmup 1000；cosine 在 20k 衰减至峰值 0.1，之后保持；预算身份仍 50k，初阶段到 20k | e13a4b6 | rl-zt2，8 GPU |
| Multi-scene32 | 同一个旧策略每 rank 收集 4 个不同场景；总 32 场景 × G8；真实梯度累积后才 Adam，仍 4 inner | dfa83e2 | tsinghua-zt6，8 GPU |
| Verified retention | 2k 后开始构建自身 TRAIN 真 B1 ODE 严格安全且 canonical≥0.9 的历史 reservoir；复评退化才使用原 β50 中的 25% 保持历史行为，PG 仍来自新采样 | df50bce | tsinghua-zt2，8 GPU |

Batch32 的调度按采样数据时钟同步：5k Adam 更新对应 Control 20k 的新鲜候选预算，不能直接以相同 Adam 步数比较。1250 的非完整 inner 边界有额外 128 次候选查询，证据中单列。Verified retention 在 2k 前不启用辅助项，之后额外官方 ODE 查询上限为 PG 查询的 25%，实际计数需汇总各 rank；不能声称与 Control 总查询预算相等。

## 如何判断改善

只发布全部 NAVTEST 场景完成的分数；部分评测不报总分。比较同预算的总分、零/低分修复、新增零/低分，以及固定 IL 高分场景保持。Early decay 和 retention 在 10k 后按 2k 间隔评测；Batch32 按等数据预算映射。初始 2k/5k 是观察点，不凭短训给有潜力的方法下最终无效结论。

训练快照见 `LIVE_SNAPSHOT.json`。三组仍处早期，尚无新算法的最终性能改善证据，不能用 CPU 检查、训练 loss 或事后 oracle 冒充 92 分结果。

## 资源与调度修复

现有 GLOBAL 撤回的 vla-zt3、vlawm-zt2/3/5 和 tsinghua-zt4 全部排除。五台临时资源最终截止 UTC 11 月 1 日 23:00（北京时间 11 月 2 日 07:00），合作保存 guard 已对齐现有独立 guard 的 UTC 22:15。未部署重复 guard，B/D IL 优先级、T 系列暂停、ABCD RL/HUBER 终止保持。

旧 worker 的共享训练检查只接受无限期许可，切换有限期许可后有 6 个续训阶段和 56 个评测分片在创建 CUDA 子进程前失败。已核实旧 worker 退出，保留原始失败与场景缓存，只重排这些确证的兼容失败；没有伪造退出结果或重置科学状态。F/G/H 仅 CPU 控制器采用 df50bce 运维实现，GPU 的各自科学源没有改变。续训 rank 已实际越过原检查点；详见恢复证据。当前 owner 读取 OWNER 与 /proc，不能重启旧 PID。

## 文献如何对应任务

`ALGORITHM_PROBLEM_MAPPING_20261101.json` 列出原论文链接、机制、实际采用状态与限制。REINFORCE++ 的全局优势归一化已有部分机制，Batch32 扩大全局场景人口；CLEAR 的行为保持提供 retention 方向；ScaleRL 支持认真研究批次与配置，但不证明 cosine 普遍最优。RLOO 单独不能解决全组同分，因为固定 G 时仅是组内减均值的常数倍。GSPO/CISPO/TOPR 的概率与离策略公式不能直接将现有 W3 似然代理当成精确流密度使用。未实现的设计均明确标为调研或条件后续，而非已启动实验。
