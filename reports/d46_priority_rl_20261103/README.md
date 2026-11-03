# D46：优先推进 D 框架的专用 RL 路线

用户指定 D 框架 IL46k 作为起点，并恢复六台临时节点、不设截止时间。
本轮从已发布的 D 专用研究实现 `e460f1a` 开发，保留原生 Qwen denoising、信用计算和 Flow Matching 方法。
实时报告仍在 `/mnt/project/qwen-world-performance-rl-20261103/report.html`，总报告仍在 `/mnt/project/qwen-world-rl-optimization-20261030/report.html`。

**D46 IL 起点的完整部署 PDMS 为 89.49552098，12146 场景、44 logs、invalid0。**
本轮尚无 RL 完整部署成绩；既有全局最高 SafetyEP Control 91.45359756 保留为历史成绩。

## 正在推进的方法

| 路线 | 实际优化机制 | 关键启用点 |
|---|---|---|
| D3_FPO | 原生 clean-x CFM energy 的 ratio 代理与 clipping，MC4 | 从更新0 |
| D3_AWM | 正优势加权 flow matching、参考保持 | 从更新0；不是 Gaussian PPO |
| D5 | 文本 LoRA r32、末8层 Time-FiLM LoRA、动作 I/O 与 position | 从更新0；分别记录梯度角色 |
| D1 | 两阶段信用：后期加入 TRAIN GT endpoint 与 comfort 质量 | 原正式2 epoch门保持 |
| D2 | 真正优于本场景 B1 的安全候选，clean-x 与真实 B1 末步吸收 | 正式0.5 epoch启用 |
| D0 | D 原生 joint24 Gaussian PPO、clip0.1、自适应 reference KL | 基础参照 |

这些方法复用已有 D Trainer 和原生架构，没有复制一份 Trainer。
W3 source transition joint density 以及 FPO energy ratio 都不能解释成精确的最终轨迹 likelihood。
各自从同一份 D46 策略、零 RL 增量、Adam0 开始，不继承 D44 RL 或资格权重。

正式默认 B64/G16/M2，完整 NAVTRAIN103288；以有效场景曝光调度 LR，
warmup .08 epoch、hold .60、cosine到4、floor .05。文本 LR 通常2e-6，AWM3e-6；
D5 的 FiLM1e-5、动作接口3e-6、position1e-6，沿用各方法原设计。
初始训练预算6 epochs，8/12/20须结合实际完整部署结果扩展。

每条先独立完成真正的八卡 B64/G16/M2 十次更新资格，检查原生零增量轨迹、current/reference、冻结参数；
资格与正式目录、Adam、RNG、checkpoint分开。D2资格显式提前开启辅助项测试，正式门仍为 .5 epoch。
运行状态和实际命令见 [归档快照](results/PRIORITY_STATUS.json)、[实际调用](results/ACTUAL_COMMANDS.json)，
实时状态以 NAS `CAMPAIGN.json`、`PRIORITY_STATUS.json` 和八 rank 日志为准。

## D46 权重究竟保留了什么

原 full native 46k 及阶段 pin 已被历史 IL 清理流程移除。
本轮保护的是经完整 NAVTEST 与 native/export 逐值一致检验的**完整部署策略**，
模型 SHA `386a0a27bd197cb002501bcb90c343659001d2e6d4f7bad0cbd5e31a91f8477d`。
11个文件均为硬链接，未额外复制8.76GB；实际模型字节、processor、运行库与源迁移均核对通过。

所有部署 C/P、Qwen、action、Time-FiLM 参数严格恢复；训练期 R16 和 readout 已裁剪，不能声称恢复了它们。
新配置绑定不可变 [v2父合同](results/PARENT_CONTRACT_v2.json)，v1单独归档。
依赖原训练头的 D4/D6暂不按旧版本启动。
另已审查显式新增随机训练辅助头的 `D4_FRESH_AUX/D6_FRESH_AUX` 可行性，
须使用新 identity、独立 scope/optimizer，以及真实 B1 无扰动与辅助激活后的中途恢复资格，尚未部署。
[父策略审计](results/PARENT_AUDIT.json)、[辅助路线边界](results/D46_FRESH_AUX_FEASIBILITY.json)。

## 资源和恢复

旧 D44 的 D0/D2/D5/D1 已分别在763/637/495/9更新、inner1完整保存、EXIT0，
旧代码/结果/Adam/RNG/rollout均保留，未将其作为 D46 初始化。
旧 IL70 队列保留状态并让位；B/D IL、未知或不可见进程不回收。
六台恢复节点按当前授权与真实 UUID/owner/显存/lease 准入。

FPO/AWM首次资格发生同机并发加载 OOM，尚无 Adam 更新。
原失败与 native0保留；`operations_v2` 修复了 CUDA 初始化尚不可见时的预留竞争，
以八 rank 当前调用的真实更新确认启动完成，并提高冷加载余量到48GiB。
两路均经明确资源复核后只从自身 qualification native0恢复；科学源码仍固定 `ff3013c`，
没有改变 group、batch、loss、采样或学习率来掩盖错误。

完整 B1 NAVTEST固定 K10/shift5/seed2101、无rerank，使用32分片并行；
完整评分前不发布局部均值。NAVTEST只用于已披露的选点，任何逐场景标签或轨迹不回流训练。

代码入口：`tools.d46_priority_rl.owner`；科学冻结源：`runtime/formal_v1@ff3013c`；
CPU调度冻结源：`runtime/operations_v2@d7dd6d5`。实际资源失败、恢复收据与调用全部独立保留。
