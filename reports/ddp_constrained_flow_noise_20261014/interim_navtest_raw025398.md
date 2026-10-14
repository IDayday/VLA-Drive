# 用户要求的中期完整 Navtest：相同 25,398 次原始曝光

状态：2026-10-14 完成三组完整 Navtest；正式两遍训练尚未完成。

**结论：两组都有约 0.33–0.35 分的单候选 ODE 平均收益，但 NC/TTC 退化，
未满足安全保持。N 的测试均分没有超过 C，目前没有额外部署收益证据。两组均未达到 92。**

## 完整单候选 ODE 结果

分数均为 0–100，差值单位为分；置信区间为按场景加权的 log-cluster bootstrap。

| 模型 | PDMS | 相对 SFT | 95% 配对区间 |
| --- | ---: | ---: | --- |
| SFT | 88.9116 | — | — |
| C_SDE | 89.2592 | +0.3476 | [+0.1552, +0.5480] |
| N_FLOW_NOISE | 89.2429 | +0.3313 | [+0.0987, +0.5540] |

| 模型 | NC | DAC | TTC | EP | Comfort | DDC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 98.1640 | 96.9702 | 95.0519 | 83.0034 | 100.0000 | 98.2834 |
| C_SDE | 97.9870 | 97.2501 | 94.7308 | 83.8822 | 100.0000 | 98.2710 |
| N_FLOW_NOISE | 97.9252 | 97.2748 | 94.3274 | 84.2292 | 99.9918 | 98.2340 |

两个 PDMS 区间均为正，支持本次固定 seed 下的小幅平均收益；不包含训练 seed
或推理 seed 的不确定性。不能据此宣布稳定全面提升。

直接配对 **N−C = −0.0163 分，95% 区间 [−0.1402, +0.0969]**，没有额外均分收益证据。
N 相对 C 的 TTC 违规率高 **0.4034 个百分点**，区间 **[+0.2166,+0.5949]**；
NC 违规率高 **0.0741 个百分点**，区间 **[+0.0097,+0.1463]**。
这说明本次匹配检查点下 N 的 NC/TTC 表现更差；不据此提前宣告尚未完成的两条正式训练胜负。

[完整配对汇总](interim_navtest_raw025398_comparison.json)、
[SFT 原始报告](interim_navtest_raw025398_SFT.json)、
[C 原始报告](interim_navtest_raw025398_C_SDE.json)、
[N 原始报告](interim_navtest_raw025398_N_FLOW_NOISE.json)。

## 安全保持及失败转移

违规定义为官方分项小于 1，NC=0.5 也计入；以下是场景数，不把 NC 违规数直接称为事故次数。

| 事件 | SFT | C | N | C 新增 / 恢复 | N 新增 / 恢复 |
| --- | ---: | ---: | ---: | ---: | ---: |
| NC 违规 | 231 | 251 | 260 | 28 / 8 | 38 / 9 |
| DAC 违规 | 368 | 334 | 331 | 16 / 50 | 24 / 61 |
| TTC 违规 | 601 | 640 | 689 | 62 / 23 | 122 / 34 |
| NC/DAC/TTC 任一违规 | 933 | 944 | 987 | 78 / 67 | 144 / 90 |
| PDMS 零分 | 569 | 559 | 563 | 43 / 53 | 60 / 66 |

两组 DAC 及零分总数改善，但原安全人口出现新增违规。尤其 N 新增 122 个 TTC
违规，仅恢复 34 个；它比 C 多 49 个 TTC 违规，联合安全人口的新增违规计数为 144 对 78。
零分减少不等于安全全面改善，也不能把所有安全分项概括为下降。

## 行为及研究含义

实际模拟前进行程相对 SFT 增加 C **0.191 m**、N **0.317 m**；平均速度增加
**0.050 / 0.086 m/s**，终点速度增加 **0.071 / 0.291 m/s**。
请求轨迹二阶位置差均值增加 **0.0022 / 0.0222 m**。二阶差只是平滑性描述，
不是官方 Comfort，也不是碰撞概率。

这些结果支持“进度增加与 NC/TTC 退化同时出现”，不能仅凭相关性认定因果。
原 ODE 已吸收一定策略更新收益，但安全目标还没有转化为合格的部署安全保持。
可学习噪声对应的完整训练策略尚未在本检查点证明额外价值；本比较同时改变
转移均值、协方差和可学习性，不能把全部差异归因于 sigma 是否可训练。

同曝光完整 DEV 为 SFT **93.6192**、C **94.2246**、N **94.5460**。
DEV 上 N 比 C 高 **0.3214** 分，Navtest 上却低 **0.0163** 分。
两组相对 SFT 的平均收益都迁移到了测试集，但 DEV 上两方法的排序没有保持。
下一步继续完成已登记的两条正式训练，重点观察固定终点是否能同时保留均分收益
并恢复 NC/TTC；不根据本次 Navtest 改 sigma、lambda、LR、人口或检查点选择。

## 同曝光 TRAIN 随机策略监测（单独口径）

这是固定 128 个 TRAIN 场景、每场景 G16 的真实随机 rollout，不能与上面的
Navtest ODE 分数直接相减，也不能作为最终部署成绩。

| 监测项 | C | N |
| --- | ---: | ---: |
| 随机候选 PDMS 均值 | 93.1077 | 93.2412 |
| NC/DAC/TTC 联合安全候选比例 | 98.4375% | 98.2422% |
| PDMS 等分组 | 43 / 128 | 40 / 128 |
| 最终优势全零组 | 29 / 128 | 28 / 128 |
| 官方安全成本有差异的组 | 7 / 128 | 8 / 128 |
| progress-only 官方奖励组 | 78 / 128 | 80 / 128 |
| pairwise ADE | 0.1369 m | 0.1750 m |

这份固定监测中，N 的几何差异更大，但最终零优势只少 1 组，联合安全候选比例更低。
结合完整 ODE Navtest，不能把“轨迹更多样”直接当作有益探索改善，也不能宣称
已解决无效组和进度主导问题。这不是完整 TRAIN 人口的冻结模型统计。

## 协议与登记

用户明确要求“测一下测试集结果”。因此新增一次中期 Navtest 观察，
不将它称为正式预算终点，不用该结果调整 sigma、lambda、margin、训练预算或检查点选择。
两条 RL 线均取预先登记的首个 25,398 次原始场景曝光检查点，均为 update2030；
这是 203,184 次正式预算的 12.5%。SFT 同时使用匹配原始入口重新评价。

三组均为完整 12,146 场景 / 136 logs，训练 seed42、推理 seed2101，
原始观测、原 10 步 ODE、单候选、canonical v1.1。不使用 noise_head、候选筛选、
训练 condition cache、搜索或新 scorer。Navtest 已有历史使用，本次中期观察也须在最终结论中披露。

[权重与协议锁](interim_navtest_raw025398_lock.json) 在任何本次 Navtest 推理之前写入。
为原样复用冻结 sampler，锁中保留其历史 `LOCKED_BEFORE_FINAL_NAVTEST` 解析字段；
独立 schema、purpose、`formal_budget_complete=false` 和结果 context 明确这是中期观察。
不会进入 `final_confirmation_v1` 的正式终点/安全择优记录。

[执行登记](interim_navtest_raw025398_execution.json)：SFT 在 Tsinghua zt2，
C 在 Tsinghua zt4，N 在 Tsinghua zt；每模型八个进程，共 24 张不同物理 GPU。
保留已有训练和 census，未停止任何压力或未知进程。每组 16 个 CPU scorer，
在原子预测发布后立即评分。三组总候选评分预算为 36,438；保留失败分母。
已验证原推理单进程峰值为 7,950,648,320 bytes，本次调度仍采用更保守的
每卡空闲显存至少 35,000 MiB 门槛，并在启动时重新检查。

新增入口只连接原 locked sampler、canonical scorer 和配对分析器；完整基础 SFT
尚未完成时等待其完整 score bank，不将局部分数当基线。新增和既有选择测试共 12 项通过。
冻结的 83 个科学源码保持不变。

## 实际成本与读取提速

每组均完成 12,146 次真实 canonical 评分、12,146 条单候选 ODE 输出，失败为 0。
没有候选重试或补样改变评价分母。主推理合计 364,380 次 velocity 调用，
另有每个 GPU 首场景的原入口一致性检查，共 240 次 velocity 调用。

| 模型 | 评价进程总墙钟 | 累计评分 worker 测量秒 | 推理预留 GPU-hours 上界估计 |
| --- | ---: | ---: | ---: |
| SFT | 41.63 分钟 | 580.91 | 4.206 |
| C | 39.02 分钟 | 403.39 | 3.549 |
| N | 46.23 分钟 | 571.74 | 5.063 |

三组并行；不能将上表墙钟相加成实际等待时间。C 的总墙钟含约 4.96 分钟等待
完整 SFT 基线。GPU-hours 定义为八个预留设备从 torchrun 启动到最后完整分片写入，
包含 CPU 装载阶段；属于非独占窗口上界估计，不是 CUDA 活跃时间，不能与同期训练、
census 的 GPU-hours 相加当物理总占用。实测推理峰值均为每进程约 7.40 GiB。

评分始终基本追上推理，最终分析的重复小文件读取形成额外等待。新增只读字节预取，
在内存中执行**完全相同的原分析器、校验和 bootstrap**，未改评分或数学公式：
完整 DEV 报告逐项一致，用时 **7.40 秒**；完整 C Navtest 报告逐项一致，用时 **60.01 秒**；
完整 N 用时 **71.05 秒**，随后与原串行完整报告逐项核对也完全一致。
这些是实测时间，不将不同主机和缓存状态下的时间比当作严格配对加速倍数。

[DEV 一致性](analysis_byte_cache_DEV_parity.json)、
[C 全 Navtest 一致性](analysis_byte_cache_Navtest_C_parity.json)、
[N 全 Navtest 一致性](analysis_byte_cache_Navtest_N_parity.json)、
[三组汇总预取执行记录](analysis_byte_cache_comparison_execution.json)。

共享 C 推理期间，N 训练的 16-collection 窗口从约 **2.42** 降至 **1.83 raw scenes/s**，
更新中位墙钟从 **1.94** 增至 **2.95 秒**。C 训练稳定窗口约 **1.43/s**。
这是共享 GPU 的临时实测成本，不属于训练算法或数值更改。
[评测前](interim_navtest_pre_evaluation_speed.json)、[评测中](interim_navtest_during_evaluation_speed.json)。

20:03 UTC 的[评测后快照](interim_navtest_post_evaluation_speed.json)确认 N 恢复至约
**2.54/s**，C 保持约 **1.44/s**。两组分别继续到 N update4030 / raw50286
（24.75%）和 C update2336 / raw29190（14.37%）。这些运行进度不是本次测试所用权重。
[本次测试完成记录](interim_navtest_raw025398_COMPLETE.json)只标记这次完整评测，
不将两条正式训练误标为完成。

保留了两次评分前/只读分析准备失败：首次锁文件父目录缺失，以及一次分析命令的
基线路径拼写错误。后者被完整性检查拒绝，没有产生任何指标；均保留私有失败记录后
修正，不曾改变预测、评分、训练或评价分母。模型与评分器运行没有失败。

```bash
python -m scripts.analysis.constrained_flow_noise.interim_navtest lock \
  --run-root /mnt/project/constrained-flow-noise-20261014 \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --c-run /mnt/project/constrained-flow-noise-20261014/formal_C_SDE_seed42_v1_probability_repair \
  --output NEW_OBSERVATION_LOCK

python -m scripts.analysis.constrained_flow_noise.interim_navtest evaluate \
  --lock OBSERVATION_LOCK --model SFT_OR_C_SDE_OR_N_FLOW_NOISE \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --output MODEL_OUTPUT --baseline-scores SFT_OUTPUT/scores --workers 16
```

真实已执行参数位于执行登记。恢复同一 `evaluate` 命令，保持模型、锁、输出和源代码身份，
复用已验证的原子预测/评分记录；不覆盖旧锁或重选检查点。

已实际执行的完整一致性及汇总入口：

```bash
python -m scripts.analysis.constrained_flow_noise.analyze_cached_scores \
  --scores COMPLETE_POLICY_SCORES --baseline-scores COMPLETE_SFT_SCORES \
  --population POPULATION --output NEW_ANALYSIS --compare-existing ORIGINAL_CORE_REPORT --workers 16

python -m scripts.analysis.constrained_flow_noise.summarize_cached_navtest \
  --root /mnt/project/constrained-flow-noise-20261014/interim_Navtest_raw025398_20261014 \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --output NEW_COMPARISON \
  --acceptance /mnt/project/constrained-flow-noise-20261014/interim_Navtest_raw025398_20261014/cached_C_full_parity/cost.json
```

同时修复最终确认控制器的 C 路径：验证原生恢复身份后跟随明确的 continuation 指针，
历史与续训曲线合并，重复采集的旧尝试保留成本记录。此修改只影响只读汇总和未来评价路由，
不改变两组训练参数、采样、梯度或终点选择规则。
