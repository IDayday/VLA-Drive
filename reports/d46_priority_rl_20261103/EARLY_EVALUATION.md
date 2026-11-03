# D46 提前完整评测：已完成结果

归档于2026-11-03 12:46:38 UTC。原有限计划的四个不同策略均已完成32分片、12,146场景、44物理log、invalid0；没有用部分场景均值发布成绩。

| 策略 | 完整PDMS | 相对同IL46 | checkpoint更新 | TRAIN曝光 |
|---|---:|---:|---:|---:|
| D0 | 90.17112884 | +0.67560786 | 1400 | 44,800 |
| FPO | 90.12278094 | +0.62725996 | 900 | 28,800 |
| D5 | 89.97889876 | +0.48337778 | 1400 | 44,800 |
| AWM | 89.95115511 | +0.45563413 | 800 | 25,600 |

固定IL46基线89.49552098，统一单B1 clean-x ODE K10/shift5/seed2101。D0/FPO双优先继续；曝光与每更新算量不同，当前不是同预算算法因果排名或上限结论。

D0修复248、新增87零分；安全指标提高但EP−1.552pp，固定IL46高分≥.9队列净抵消1.369PDMS。FPO修复164、新增65，高分损失较小。D5修复288但新增143及较大回退抵消修复。AWM EP+.487pp、高分保留99.29%，但仅修复72，且NC略降。

[零分与抵消图](plots/D46_zero_repair_and_gain_loss.svg)、[安全与进度图](plots/D46_safety_and_progress_delta.svg)、[事实和精确来源](plots/D46_EARLY_FULL_FACTS.json)、[中文readout](plots/D46_EARLY_FULL_READOUT_zh.txt)。完整摘要已小体积保存于 `results/early_full/`，不复制逐scene或权重。

D0/D1/D2的u1400完整训练参数payload及部署签名相等，故D1/D2只引用D0的等参数参考，不构成独立评测。D1的2epoch机制、D2的.5epoch机制尚未在此早评点启用，不能宣判无效。

后续0.5epoch目录IO故障已和数值问题分开，原生1614保存及恢复验证PASS，丢失请求由Root回填；D2启动90s Gloo超时0新Adam、同CP重试证据保留。D1/D5目前是保留完整CP的资源hold。world16实际pending恢复资格已PASS，但不等于正式长训已开始。详见[README](README.md)与[固定运行快照](analysis/COMPLETED_RESULTS_RUN_STATE_SNAPSHOT.json)。

NAVTEST只用于已披露的研究选点与完整诊断，任何场景ID/标签不回流训练。此归档未操作任何训练、锁、CAMPAIGN或GPU。

运维补充至12:50:25 UTC：`operations_v6@46cdd79` owner217718仅CPU接管、未停止GPU；D0/D1/D2各自0.5E的精确等模型引用另有[绑定表](results/operations_v6/EVALUATION_EQUIVALENCE.json)，不扩展到未来分化。旧I已4877_inner1八rank合作保存EXIT0、资源hold让D0独占Tsh5，原10000预算仍保留，不能称旧I算法终止。见[回执](results/old_I_resource_hold/NATIVE_SAVE_EXIT0.json)。

12:52:56 UTC补充：[D2真实rescue激活](results/recovery/D2_RESUMED_RESCUE_ACTIVE.json)已证八rank从原1614恢复到1628、Adam未重置、辅助项非零；累计448 TRAIN B1查询与800额外adaptive candidates仅为训练实耗/机制激活证据，不是scene激活率或新部署分。
