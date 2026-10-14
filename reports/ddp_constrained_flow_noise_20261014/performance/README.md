# 训练执行优化：保持同一真实链与更新结果

本记录只处理执行效率。两条正式训练线、完整动作头、G16/K10、两个 inner epochs、原始场景顺序、canonical v1.1、五项安全成本、乘子、replay、参考保持及 ODE 部署协议都沿用已锁定设置。它不解决零优势组缺少学习信号的问题，也不将训练加速当作规划性能提升。

## 已测出的时间去向

[原正式训练最近 64 个完整 collection](training_cost_20261014T1705Z.json)，使用八个 rank 的实际日志和已封存候选，未额外生成训练样本：

| 环节 | C_SDE | N_FLOW_NOISE |
| --- | ---: | ---: |
| 候选生成 | 37.15% | 32.06% |
| 等待实际评分 | 40.23% | 46.67% |
| 动作头更新 | 18.38% | 16.92% |
| reference | 1.24% | 1.35% |
| 其余 IO / 同步等 | 约 3% | 约 3% |

C 窗口为 1,664 个 raw scenes / 2,258 秒；N 窗口为 1,592 / 1,479 秒。墙钟来自相邻原子候选归档时间，文件系统时间精度有限；按每次 first16 / refill8 调用取最慢 rank 分解，不能解释成精确逐 kernel trace。CPU worker 时间之和不等于墙钟。CPU 评分时间中约 70% 用于车身道路/对象几何余量，约 24% 用于 canonical PDM。

旧收集器在一个 rank 上执行“GPU 生成一组 → CPU 评分并等待 → reference → 下一组”，尽管每个 rank 已有两个评分 worker，仍没有同时提交第二个场景。完整采样还反复将同一冻结 BF16 Linear 权重转为 FP32；GPU profiler 显示这部分转换有明显开销。

## 三项执行改动及证据

1. **GPU 生成与 CPU 评分重叠。** 生成第一组后提交原官方场景评分任务，再运行 reference 和下一组；按输入顺序收回结果。first16/refill8、组内全部 16 条轨迹及 dual 的首批人口不变。所有评分 future 在返回 collector 前完成，不能跨 behavior 更新或 checkpoint。基础设施失败仍中止更新，不变成零 reward。
2. **相同 GEOS 几何操作向量化。** 仍使用实际模拟轨迹的 41 个车身、同时间对象、可行驶区域并集、孔洞和相同边界；不减少物理时间点，也不改成中心点近似。静态几何上下文仅作有界缓存。
3. **缓存冻结 behavior 的 FP32 Linear 权重转换。** 保留原 BF16 参数及 dtype 语义，只缓存其精确 FP32 副本，每次 `load_state_dict` 后刷新稳定地址。当前可训练网络不走该缓存；不缓存 hidden、qwen_proj 输出、sigma 或跨版本输出。额外显存约为该冻结动作头 Linear 参数的一份 FP32 副本。

各项单独测量，不能相乘当成整体加速：

| 测量 | 原实现 | 优化后 | 范围 |
| --- | ---: | ---: | --- |
| 512 条真实轨迹几何余量 | 50.034 秒 | 3.809 秒 | 几何环节 13.14×；全部结果逐项一致 |
| C_SDE 两场景采集，仅重叠 | 12.299 秒 | 10.357 秒 | 采集约 1.19× |
| N_FLOW_NOISE 两场景采集，仅重叠 | 14.585 秒 | 9.146 秒 | 采集约 1.59× |
| 冻结权重转换缓存 | — | 采样约 1.4–2.0× | 两模型、SFT/已训练权重、固定 TRAIN 场景；不是全训练加速 |

证据：[几何及 CPU profile](vector_geometry_qualification_20261014T1713Z.json)、[仅重叠配对采集](collection_overlap_qualification_20261014T1710Z.json)、[冻结权重转换与刷新](frozen_cast_qualification_v1.json)、[真实 GPU 采样 profile](rollout_kernel_profile_20261014T1715Z.json)。小样本计时有共享资源负载影响。所有 warmup 和重复官方评分计入资格测试成本；它们不进入正式训练人口或预算。

## 等价验收与执行身份

原 83 个训练/评价源文件保持正式 seal 不变。新入口位于独立 analysis 命名空间，复用原生 trainer、checkpoint 和评分 worker；单独保存执行源文件 hash、源码快照和 acceptance。执行身份进入原子 checkpoint，而不是事后修改已封存恢复点。

重叠 + 几何的第一套八卡验收已经全部通过：[聚合证据](overlap_geometry_full_acceptance.json)。C/N 两组实际完整动作头学习、原始 replay dropout、KL、dual、全部八分片 FP32 optimizer、每 rank RNG、数据与 inner 游标、最终完整 ODE 导出均与原实现逐值一致。N 另验证第一 inner 更新后中断并原生恢复；所有保存链、请求、模拟反馈、成本、优势、组选择一致。仅时间字段和明确的执行 provenance 不参加相等比较。

加入冻结权重转换后的第二套组合验收也已全部通过：[完整八卡学习与恢复证据](combined_full_acceptance.json)。执行身份为 `92e9154a1ae7dcfd44167d9f6575890333da94d0440d49837886c4a24e2f8e1b`。

[三项组合的固定场景 ABBA 采集测试](combined_execution_benchmark_v1.json)：C_SDE 从 10.965 秒降至 5.908 秒，约 **1.86×**；N_FLOW_NOISE 从 14.040 秒降至 4.734 秒，约 **2.97×**。原链、均值、std、logprob、请求、模拟结果、全部指标、成本、优势、reference 与 RNG 均逐值一致。范围仍是单 rank 的两个真实场景采集，不是全训练。

正式 N 已在 update2580 / raw32230 的原生完整 checkpoint 无损接续；中断时处于 collection1290 的收集阶段，已评分的首批候选和数据游标保留，补样没有重新生成旧链。[初期生产吞吐](N_production_throughput.json) 比较切换前 64 个 collection 与排除冷启动边界后的 32 个 collection：**1.063 → 2.316 raw scenes/s，约 2.18×**；完整 collection 间隔中位数 **23 → 10 秒**，包含更新和 IO。两个时间窗场景不同、主机共享负载可能变化，不能把该比值说成任意人口/主机的保证。后窗 776 raw / 512 accepted scenes，12,416 次真实候选评分；没有减少 G16/K10 或训练曝光。后续运行已出现新的耗时变化，见下文，2.18× 不能代表目前持续吞吐。

## 复用已有批量 PDMS

找到已有 `DriveVLA-M0/tools/navsim_candidate_relative_audit/score_candidates.py::score_pose_batch` 及其 500 场景 / 6,000 候选历史报告。该版本调用定制 EpisodeDrive scorer，依赖自定义 cache 的 `pdm_progress`。它的历史 batch-vs-single PASS 不能直接证明本轮官方 v1.1 协议等价。[原文件、commit 与 SHA](batch_scoring_reuse_lock.json) 已固定。

新适配复用其批量 trajectory transform / simulator / metric 流程，直接调用本轮锁定的官方 v1.1 `PDMSimulator` 与所有原始分项实现。聚合时，给每条候选建立 `[同一实际官方参考, 该候选]` 的两行视图，调用**未修改的官方 `_aggregate_scores`**。没有复制一套 scorer，也没有使用 G16 的最大进度归一化。参考自身违规造成的乘法 mask、NC=0.5、5 米阈值和阈值下零分处理都继承官方代码；不是简单除以未评分参考的原始进度。

原 Flow-GRPO 与本轮改动前的 RewardService 本来就逐候选独立评分；这里是防止批量加速引入新污染，不是用“旧 EP 归一化 bug”解释过往 RL 结果。

明确的反例：参考进度 10 米、目标候选 6 米、其他候选 12 米，全部通过乘法分项。直接把三条送入原聚合会把目标 EP 从 0.6 改为 0.5，也改变总分；独立 pair 聚合保持 EP=0.6。新增 7 项测试包含此负面对照、参考失败、NC=0.5、零进度、阈值两侧及随机置换。

[真实数据验收](batch_score_qualification_v1.json)：32 个连续 TRAIN 场景、512 条已存策略候选；batch/chunk 1/3/8/16，前四场景另作逆序、chunk5、不同内容干扰候选及重复候选。所有请求、模拟状态/车身、NC/DAC/TTC/EP/Comfort/DDC/PDMS、原始进度、事件时间、几何余量和成本**逐值一致**；共 2,656 次资格测试候选评分。诊断用干扰轨迹不进入 RL 采样或更新。该固定顺序测试的 CPU 评分耗时为 31.213 → 10.361 秒，约 3.01×，含冷缓存差异；另以 warm ABBA 测量排除首轮初始化影响。

[warm ABBA 评分测量](batch_score_warm_benchmark_v1.json) 在固定前 8 个真实 TRAIN 场景、相同已优化几何实现下，排除双方 warmup 后测得 **2.68×** 评分提速，全部反馈再次逐值一致；共 768 次候选评分，仍不是整个训练的加速倍数。G16 的逻辑候选评价仍为 16，实际仿真从 16 次 `[参考, 候选]` 的 32 条 proposal，变成一次 `[参考, 16候选]` 的 17 条 proposal；最后仍分别进行 16 次官方 pair 聚合，不共享 EP 分母。

批量评分的独立八卡学习/恢复验收已全部通过：[完整证据](batch_full_acceptance.json)、[7 项分项协议测试](batch_score_unit_acceptance.json)。C/N 两组完整动作头与 N 噪声网络、八个 FP32 optimizer 分片、每 rank RNG、dual、KL、游标、保存链、全部反馈及最终原 ODE 导出逐值一致。恢复测试显式从前一执行版的第一个 inner 更新 checkpoint 切入批量评分版，再与连续更新比较，不只是新版本自身重启。

执行身份为 `0b054e56de8277df5faead4d1aab484afd69aacb484d2594b0a6c0af4119f3c4`。正式 N 在 update2722 / raw33958 完整边界停止并原生接续该执行版。[部署与源文件核对](batch_execution_activation_20261014T1812Z.json) 记录八个 rank 已实际完成 update2774 / raw34606；原 83 个科学源文件和 7 个执行源文件 hash 全部匹配。旧执行 manifest、acceptance、源码和 checkpoint 全部保留；额外的执行 ancestry 记录升级，科学 run identity 不变。

[批量评分接入后的生产窗口](N_batch_production_throughput.json)：比较切换前的最后 16 个稳态 collection 与排除启动边界后的前 16 个 collection，两窗恰好均为 400 raw / 256 accepted scenes、6,400 次候选评分；耗时 **381 → 374 秒**，吞吐 **1.050 → 1.070 raw scenes/s，仅约 1.019×**。这是不同场景、共享负载下的短窗口观测，不能认定稳定的整段提速。评分微基准的 2.68× 没有等比例传递到已经重叠执行的整个训练；不得将各项倍数相乘。

另一个尚未解决的变化发生在批量适配启动之前：update2601–2640 的八 rank 临界更新时间中位数为 **1.954 秒**，update2701–2722 已为 **4.897 秒**；批量版 update2723–2758 为 **4.947 秒**。目前不能把原因归给批量评分。现场检查未见其他可见 GPU 计算进程、GPU 温度/功率降频或 CPU quota 限流；这些检查不足以确定根因。保留原始耗时证据，不修改数值平台或训练目标来掩盖变化。当前批量版窗口的更新累计约 155 秒，占 374 秒总墙钟约 42%；其余仍包含采样、未被隐藏的评分、同步与 IO，后续定位应使用实际临界路径。

[验收资源成本](qualification_cost.json)：三个执行修订的八卡短训分别占用约 1.33 / 1.28 / 1.23 GPU-hours，各自评分 1,536 条真实候选；这些是限定工程资格测试，不计作正式训练新曝光，也没有隐去其开销。CPU 对比、warmup、评分微基准单列在各自 JSON。三次全部原生短训时间包含进程持有 GPU 时的保存与导出时间，不等于 GPU kernel 活跃时长。

保留的失败：最初包装入口使用 argparse REMAINDER，torchrun 吞掉 `--` 后在解析参数阶段退出，没有执行任何训练。已改为只解析包装入口选项并保留原生参数的 `parse_known_args`；失败源码和日志保留在私有 `efficient_execution_qualification_v1`。这是启动接口问题，不是算法结果。

## 实际命令与恢复

从本分支 worktree、已授权 Tsinghua 主机和原 Python 环境执行。八卡验收及正式接续需要足够空闲显存，不占用未知进程。

```bash
PY=/mnt/project/flow-safe-improvement-20261013/venv/bin/python
RUN=/mnt/project/constrained-flow-noise-20261014

# 已执行：原训练时间分解；该命令拒绝将重叠后的时间当串行时间相加。
$PY -m scripts.analysis.constrained_flow_noise.profile_training_cost \
  --run-root "$RUN" --collections 64 --output "$RUN/training_cost_20261014T1705Z.json"

# 已执行：八卡真实学习与 inner 中断恢复；输出目录不允许覆盖。
$PY -m scripts.analysis.constrained_flow_noise.qualify_efficient_training \
  --run-root "$RUN" --output "$RUN/efficient_execution_qualification_v3" \
  --adapter-module scripts.analysis.constrained_flow_noise.train_efficient_cast

# 已执行：现有真实轨迹的 batch/分块/顺序/干扰候选验收。
$PY -m scripts.analysis.constrained_flow_noise.qualify_batch_scoring \
  --run-root "$RUN" --output "$RUN/batch_score_qualification_v1.json" --scenes 32
$PY -m scripts.analysis.constrained_flow_noise.benchmark_batch_score \
  --run-root "$RUN" --output "$RUN/batch_score_warm_benchmark_v1.json"
$PY -m pytest -q tests/constrained_flow_noise/test_batch_scoring_contract.py

# 已执行：batch 版八卡学习、原执行版中断到 batch 版恢复。
$PY -m scripts.analysis.constrained_flow_noise.qualify_efficient_training \
  --run-root "$RUN" --output "$RUN/efficient_execution_qualification_v4" \
  --adapter-module scripts.analysis.constrained_flow_noise.train_efficient_batch \
  --interrupted-adapter-module scripts.analysis.constrained_flow_noise.train_efficient_cast

# 已启动的正式原生接续；所有旧结果和原 checkpoint 保留。
$PY -m scripts.analysis.constrained_flow_noise.resume_efficient \
  --run "$RUN/formal_N_FLOW_NOISE_seed42_v1" \
  --checkpoint "$RUN/formal_N_FLOW_NOISE_seed42_v1/checkpoints/update002722_raw033958" \
  --acceptance "$RUN/efficient_execution_qualification_v4/acceptance.json" \
  --source-lock "$RUN/formal_sources.json" \
  --receipt "$RUN/execution_deploy_v4/N_efficient_resume.json" \
  --adapter-module scripts.analysis.constrained_flow_noise.train_efficient_batch \
  --stop-request "$RUN/execution_deploy_v4/N_native_stop_request.json"

# 已执行：切换前后各 16 个完整生产 collection；不把重叠计时相加。
$PY -m scripts.analysis.constrained_flow_noise.compare_execution_throughput \
  --run "$RUN/formal_N_FLOW_NOISE_seed42_v1" --first-new-collection 1361 \
  --before-count 16 --after-count 16 \
  --output "$RUN/execution_deploy_v4/N_production_throughput.json"
```

正式接续先向本任务八个 trainer rank 发 SIGUSR1，等全部保存同一原生完整恢复点并退出，随后 `resume_efficient` 校验源码、acceptance、checkpoint 全部 hash、八 rank 停止状态、运行锁和空闲显存。不会重置 Adam、lambda、KL controller、LR 阶段、数据/RNG 或已保存链。恢复命令与实际 checkpoint 必须取已生成的执行 receipt，不从权重文件单独续训。

优化后的 `scoring_seconds` 是提交到收回结果的延迟，包含与 GPU 的重叠；不能再与 rollout 相加。新 `compare_execution_throughput` 使用完整连续归档间隔比较 raw scene 吞吐，另报重启/冷启动成本。优化前后连续数据窗不是同场景配对，生产吞吐变化需要与固定场景 ABBA 测量分开解释。

## 仍存在的问题

- 零最终优势组和 refill 后超过 16 组的有效 surplus 仍有采样/评分成本。直接丢场景、减少 G/K、跳过零组 reference/dual、改变组选择都会改变本轮实验合同，未作为“无损优化”实施。
- C_SDE 在原实现 update1850 后触发 fresh-ratio 一致性检查。八 rank 已从 update1024 完成全部 826 步历史序列的精确重放并保存完整原生恢复点；故障两组重新生成的真实链及 old 概率也逐值一致。[恢复及故障诊断](C_recovered1850.json) 显示：原 B1 重算 ratio 误差为 0，flat 中一个转移为 0.0001001358，略超过原 0.0001 阈值；最大均值差 2.861e-6，std 相同。确认是该状态下的布局数值差异，尚未部署数值修正；C 仍暂停，未放宽检查或 clip。
- 当前 DEV 显示平均 PDMS 有收益，但 TTC 安全保持仍未合格；更快执行不能替代最终单候选 ODE 评价。两条正式预算终点仍未完成。
