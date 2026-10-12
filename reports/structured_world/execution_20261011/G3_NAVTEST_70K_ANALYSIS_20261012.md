**G3 Navtest 70k：q0 = 88.238284，q_final = 88.870126。** q_final 比 50k 高 0.350578 分，比 75k 低 0.046589 分；前者的 log 聚类 95% 区间仅略高于零，后者跨零。本次结果支持有限的检查点比较，不能据此宣称稳定的训练收益、单调改善或辅助监督的因果收益。

只读核对时间：2026-10-12 03:18:50 UTC。六份原有 score CSV 均有 12,146 个唯一场景、136 个 logs，场景集合完全一致；所有指标有限、记录状态为 `ok`，均值与各自 COMPLETE 一致。50k、70k、75k 的队列完成收据均为 `valid: true`。下表分数及组件均按原始均值 × 100 展示，保留六位小数；差值按未舍入值计算。

| G3 checkpoint | q0 PDMS | q_final PDMS（主结果） | q0 零分场景 | q_final 零分场景 | 评分失败 q0 / final |
|---|---:|---:|---:|---:|---:|
| `milestone_0050000` | 86.684229 | 88.519548 | 816 | 625 | 0 / 0 |
| `latest_0070000` | 88.238284 | 88.870126 | 660 | 581 | 0 / 0 |
| `milestone_0075000` | 88.471249 | 88.916714 | 620 | 572 | 0 / 0 |

70k q0 相对 50k / 75k 的 PDMS 差分别为 **+1.554055 / −0.232965** 分，零分场景分别少 156 / 多 40；70k 内 q_final 相对 q0 为 **+0.631842** 分。原始结果见 [50k COMPLETE](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0050000/scores/COMPLETE.json)、[70k COMPLETE](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0070000/scores/COMPLETE.json)、[75k COMPLETE](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0075000/scores/COMPLETE.json)。

组件差值均为 70k 减对应检查点，单位为百分点评分；它们不能相加解释 PDMS。

| 组件 | 70k q0 | 70k final | Δq0 vs 50k | Δfinal vs 50k | Δq0 vs 75k | Δfinal vs 75k |
|---|---:|---:|---:|---:|---:|---:|
| NC（无责任碰撞） | 97.879960 | 98.056973 | −0.102915 | −0.119381 | −0.185246 | −0.102915 |
| DAC（可行驶区域合规） | 96.575004 | 97.060761 | +1.416104 | +0.477523 | −0.131731 | +0.049399 |
| EP（自车进展） | 82.625619 | 83.048541 | +1.737200 | +0.269451 | −0.007114 | +0.007028 |
| TTC（碰撞时间达标） | 94.426149 | 94.813107 | +0.098798 | −0.082332 | −0.279928 | −0.156430 |
| Comfort（舒适性） | 100.000000 | 100.000000 | 0.000000 | +0.008233 | 0.000000 | 0.000000 |
| DDC（行驶方向合规） | 98.353367 | 98.423349 | +0.024699 | −0.016466 | +0.065865 | +0.028816 |

以下区间直接来自现有 [G3_70K_ANALYSIS.json](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/g3_navtest_recovery_20261012/G3_70K_ANALYSIS.json)，本次没有重新抽样。候选均为 70k q_final。

| 配对基线 | PDMS 差（分） | 按 log 聚类的 95% 区间 |
|---|---:|---:|
| 同一 70k 模型 q0 | +0.631842 | [+0.423819, +0.863965] |
| 50k q_final | +0.350578 | [+0.004161, +0.735957] |
| 75k q_final | −0.046589 | [−0.264710, +0.182534] |

方法为 136 个 logs 的配对、有放回 bootstrap，20,000 次、seed 42；每次用抽中 logs 的场景总分除以场景总数，保持场景加权，而非等权平均各 log。它考虑同 log 内场景相关性，但只是该训练种子和评测总体条件下的不确定性。对 50k，PDMS 下界仅 +0.004161 分；DAC 的区间为 [+0.145700, +0.859551]，NC、TTC、EP 均跨零。对 75k，PDMS 及 NC、DAC、TTC、EP 均跨零，说明当前证据未能分离两检查点，不等于证明等效。70k final 对自身 q0 的 NC、DAC、TTC、EP 区间均高于零；这是同模型修正前后诊断，q0 不是独立训练的原始 DDP 基线。现有报告没有跨检查点 q0 或 DDC 的聚类区间，不能套用 final 的区间；也没有多重比较校正或跨训练种子重复。中间检查点 Navtest 为用户要求的重复观测，不是未触碰的最终留出集。

“零分”是实际评测结果，不是推理/评分失败。配对零分转移如下，候选仍为 70k final：

| 基线 | 基线零分 → 候选非零 | 基线非零 → 候选零分 | 双方均零分 | 候选零分净变化 |
|---|---:|---:|---:|---:|
| 70k q0 | 100 | 21 | 560 | −79（660 → 581） |
| 50k final | 160 | 116 | 465 | −44（625 → 581） |
| 75k final | 71 | 80 | 501 | +9（572 → 581） |

70k final 的 581 个零分场景中，组件零值模式为：仅 DAC 344、DAC+DDC 3、仅 NC 223、NC+DAC 10、NC+DDC 1；涉及 DAC 共 357、涉及 NC 共 234，两者重叠 10，不能相加。这是组件模式统计，不是事故原因归因。相对 50k，逐场景 PDMS 改善/变差/不变为 3,892 / 4,131 / 4,123；相对 75k 为 4,698 / 3,235 / 4,213，场景数量的多数方向不能替代幅度加权的均值。

三检查点均属于 `G3_EVENT_LOCAL`，训练 identity 为 `6a0b41ad25b9652474c16bfe2dab7c215446960e8e705d8adb6f01e6d2fd6d1e`；训练与评测模型源码均为正式冻结 `0c54f1c49e7ed2a76b22826b654d6c72625f5b6d`（`/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009`）。共同输入 identity 为 `b14585d89aecd8f67f088c6cc7eb467dcc12c9a654d9fac0a952626ff5759781`；原 10-step FM、场景绑定 CPU Gaussian seed 42、原有 q0/q_final 路径，native ZeRO FP32 master 严格恢复、FP32 推理、TF32 关闭。70k 模型文件 SHA256 为 `f357e5bd821d49c34416a819792d6db07d2cefb86d58ce6c5624a0907654da49`；完整 checkpoint/source 绑定见 [70k identity](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0070000/identity.json)。

70k 曾因 8 个 SSH 返回 255 被 dispatcher 标失败；原 failure 保留，已有全量输出经独立 CPU 校验与收尾后有效完成，没有重新启动 GPU 推理。恢复验证记录 12,146 份预测及每种轨迹各 12,146 份评分，membership/identity/payload/proposal/metric-cache 哈希核对通过，评分失败 0，见 [恢复 VALIDATION](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/jobs/G3_EVENT_LOCAL_0070000/recoveries/completed_transport_20261012T025550Z/VALIDATION.json)及 [有效完成收据](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/completed/G3_EVENT_LOCAL_0070000.json)。恢复 helper/证据归档 commit 为 `7925579`，不改变模型或评分合同。

本次复核确认分析绑定的四份 CSV SHA256、70k score COMPLETE SHA256 与实际文件一致，另外核对 50k/75k q0 CSV；分析源码 SHA256 为 `ba288ea8fa8f897eab96ffb2f44174363cc31d02b07d12a8b249d22edf5c409e`，分析 JSON SHA256 为 `e50fdd9fed5d5605e7fb5dddb1759311136dfeb6248badcf9b656a694753cb52`。没有新增评测、重算 bootstrap 或导出场景标识、图像、权重。

**80k 尚待完成，本文不包含其有效分数。** 截至上述核对时间，[STATE](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/STATE.json) 为 `INFERENCE_RUNNING`，attempt `20261012T031255_b7be5fe8c52a`、19 shards；[CONTROLLER](/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navsim_navtest_dispatch_20261011T0225/CONTROLLER.json) 登记 PID `1803485`，使用新 cohost 冻结树 `/mnt/project/VLA-Drive-structured-world-navtest-cohost-frozen-20261012`（`d161a2e`），尚无 80k score COMPLETE。此状态仅作时间快照，不能预告最终分数或有效完成。
