# 最终Navtest：三个锁定camera-only控制的共同24epoch结果

独立核验时间：2026-10-15 05:22 UTC。三组均已COMPLETE，每组12,146场景、136个log，q0/q_final评分失败均为0。primary MIN_NONE基本保持原S0分数，并减少了primary LOG_NONE的退化；balanced MIN_NONE虽比q0高0.06170pp，但配对区间、自然dev结果和失败转移均不支持稳定净收益。三者都是NONE控制，这些差值不能解释为新增环境知识的收益。

主证据为 [SUMMARY.json](/mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_RESULTS_v15/SUMMARY.json)、[RESULTS.csv](/mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_RESULTS_v15/RESULTS.csv)、[最终注册](/mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json) 和 [FINAL_METHOD_LOCK.json](/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/FINAL_METHOD_LOCK.json)。本报告沿用全部锁定结果，未按Navtest分数挑选方法、checkpoint或执行输出。三组主执行均为q_final，q0仅作为共同输入对照。

以下PDMS为百分制，Δ及区间单位为百分点（pp）。零分修复指q0=0→q_final>0，新增零分指q0>0→q_final=0；它们不是评分处理失败，也不等同于局部标签的REPAIR成功或责任碰撞次数。共同q0的PDMS为89.52787，共500个零分场景。

| 锁定方法 | PDMS | Δq0（pp） | 配对95% CI（pp） | 零分修复 / 新增零分 | q_final零分数 |
|---|---:|---:|---|---:|---:|
| [primary LOG_NONE](/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/scores/primary_seed42_LOG_NONE_epoch24/COMPLETE.json) | 89.29093 | -0.23694 | [-0.39243, -0.10145] | 17 / 44 | 527 |
| [primary MIN_NONE](/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/scores/primary_seed42_MIN_NONE_epoch24/COMPLETE.json) | 89.53612 | +0.00826 | [-0.01666, +0.03646] | 4 / 2 | 498 |
| [balanced MIN_NONE](/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/scores/balanced_seed42_MIN_NONE_epoch24/COMPLETE.json) | 89.58957 | +0.06170 | [-0.19539, +0.34547] | 74 / 57 | 483 |

注册中另有两项头间配对比较，保持primary与balanced身份分开：

| 比较（前者减后者） | ΔPDMS（pp） | 配对95% CI（pp） |
|---|---:|---|
| primary MIN_NONE − primary LOG_NONE | +0.24520 | [+0.10818, +0.40002] |
| balanced MIN_NONE − primary MIN_NONE | +0.05344 | [-0.20186, +0.33919] |

区间使用既有算子对136个log作10,000次配对聚类bootstrap，固定seed=20260926；每次重采样保留log内场景，按抽到的场景总数归一化。独立只读复算上述五项比较的全部分项CI，与保存分析逐项一致。区间只描述这组固定模型的log抽样不确定性，不含跨底座/训练seed不确定性，未作多重比较校正；跨零不等于证明无差异，微小正点估计也不等于稳定收益。

全部评分分项如下，统一把原0–1指标乘100。分项是场景均值，不用这些均值重新组合PDMS；它们也不认证完整路线、交通规则或现实安全。

| 分项（×100） | 共同q0 | primary LOG_NONE | primary MIN_NONE | balanced MIN_NONE |
|---|---:|---:|---:|---:|
| NC：无责任碰撞 | 98.23399 | 98.22987 | 98.24222 | 98.27927 |
| DAC：可行驶区域合规 | 97.55475 | 97.32422 | 97.56298 | 97.64532 |
| EP：自车进度 | 83.67849 | 83.50679 | 83.69172 | 83.71358 |
| TTC：碰撞时间约束 | 95.08480 | 94.99424 | 95.06834 | 95.04364 |
| Comfort：舒适度 | 99.99177 | 99.99177 | 99.99177 | 99.99177 |
| DDC：行驶方向合规 | 98.41923 | 98.41923 | 98.41923 | 98.35337 |

为便于核对，全部分项相对q0的配对差值及95% CI也按pp列出：

| 分项 | primary LOG_NONE：Δ [CI] | primary MIN_NONE：Δ [CI] | balanced MIN_NONE：Δ [CI] |
|---|---|---|---|
| NC | -0.00412 [-0.05281, +0.04909] | +0.00823 [+0.00000, +0.02689] | +0.04528 [-0.04481, +0.13798] |
| DAC | -0.23053 [-0.38585, -0.10175] | +0.00823 [-0.01786, +0.03684] | +0.09056 [-0.16283, +0.36342] |
| EP | -0.17170 [-0.31567, -0.04723] | +0.01323 [-0.01548, +0.04304] | +0.03509 [-0.19165, +0.28628] |
| TTC | -0.09056 [-0.18896, +0.00000] | -0.01647 [-0.04332, +0.00000] | -0.04117 [-0.19223, +0.11060] |
| Comfort | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] | +0.00000 [+0.00000, +0.00000] |
| DDC | +0.00000 [-0.04928, +0.04684] | +0.00000 [+0.00000, +0.00000] | -0.06587 [-0.11558, -0.02085] |

primary LOG_NONE的PDMS、DAC和EP配对区间均为负；primary MIN_NONE相对LOG_NONE的+0.24520pp主要应解释为减少LOG目标回归造成的损失，相对q0仍只有+0.00826pp。balanced的NC/DAC/EP点估计上升，但TTC下降，DDC为−0.06587pp、区间[−0.11558, −0.02085]；舒适度不变。不能用整体+0.06170pp掩盖分项退化，也不能把指标均值变化直接改称事故数量变化。

高分组只沿用既有预登记阈值q0≥0.9（百分制≥90），见 [旧分析的固定阈值记录](/mnt/project/decision-local-consequence-artifacts/20261014/final_navtest_v1/final_analyses/seed42_epoch24/summary.json) 的 `group_thresholds_fixed_before_formal_training` 及 [原汇总算子](../../../tools/decision_local/summarize_round1.py:27)。本次只读三组已有评分CSV补充统计，机器可读证据见[高分组收据](evidence/FINAL_NAVTEST_HIGH_SCORE_v15.json)；没有调整阈值、生成新评分或据此选择方法。三组共同高分组为8,929场景、136个log，下表仅报描述性点估计，未新增分组CI。

| 方法 | 高分组ΔPDMS（pp） | 高分组新增零分 |
|---|---:|---:|
| primary LOG_NONE | -0.42869 | 32 / 8929 |
| primary MIN_NONE | -0.02058 | 1 / 8929 |
| balanced MIN_NONE | -0.52388 | 34 / 8929 |

balanced在全体场景中把零分总数从500降到483，但同时修复74例、新增57例，其中34个新增零分来自原高分组；其原高分组均值下降0.52388pp。它改变的成功/失败案例更多，不能只呈现净减少17例而省略损失。primary MIN_NONE的修复/新增为4/2，高分组新增1例，整体行为更接近保持输入；这只是固定终点的观察，不是安全等价或可部署保证。

自然dev对照沿用 [原Stage A共同终点汇总](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_RESULTS_v13/SUMMARY.json) 和 [balanced共同终点汇总](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_RESULTS_v14/SUMMARY.json)：每项1,696场景、16个log，q0=91.38836，均为相同三个seed42、24epoch checkpoint。此处比较NATURAL，不混入合成RECOVERY。

| 方法 | 自然dev PDMS | Δdev q0（pp） | dev配对95% CI（pp） | dev零分修复 / 新增 |
|---|---:|---:|---|---:|
| primary LOG_NONE | 91.21920 | -0.16917 | [-0.46022, +0.26260] | 2 / 4 |
| primary MIN_NONE | 91.38751 | -0.00085 | [-0.00160, +0.00025] | 0 / 0 |
| balanced MIN_NONE | 91.05623 | -0.33213 | [-1.19179, +0.17465] | 4 / 9 |

primary MIN_NONE在自然dev与Navtest都接近q0；LOG_NONE在两套自然输入上的点估计均下降。balanced在自然dev为−0.33213pp、在Navtest为+0.06170pp，相对primary MIN_NONE的Navtest差值+0.05344pp也不确定；结合高分退化与分项变化，不能称为跨分布稳定收益。两个人口的绝对均分不能直接配对相减；本报告不把dev与Navtest合并成一个估计，也不根据Navtest反向选择loss或权重。限定loss修复的训练fit、六dev/四机制/六行为细节见 [bounded报告](BOUNDED_REPAIR_RESULTS_ZH.md)。

核验结果：未发现使本轮三项锁定评分失效的人口、身份、输入或数值汇总异常。检查覆盖三个完成收据、六份评分CSV及其摘要，具体如下：

- 三组q0/q_final均为12,146个唯一token、136个log，token→log映射与预测/特征索引一致，全部status=ok、分项有限、failed=0、valid=true；未知局部标签场景没有从canonical分母删去。
- 三份q0 CSV逐字节相同，包含相同的proposal哈希；共同SHA256为 `ac79bbfec664294959fb49034b06325e6f9196a09af367bb9e54f2c43ba174d1`。六份CSV实际哈希均与各自COMPLETE一致，逐项均值和零分转移与SUMMARY/RESULTS一致。
- score COMPLETE→score identity→prediction identity→锁定checkpoint SHA收据关联一致。三组具有相同feature/anchor、采样seed、canonical index、adapter、NAVSIM/nuPlan源码、runtime和failure policy；锁定轨迹合同为8×0.5s输入、40×0.1s模拟。
- 最终注册SHA256与SUMMARY一致；method-lock SHA256与注册一致；五个保存配对的均值、胜/负/平数和全部分项CI复核一致。核验仅使用现有收据、索引和CSV，没有重新推理或评分，也未逐场景扫描原始影像/缓存块。

元数据中需区分训练来源与执行来源：primary两头训练源码为 `0471af96e4f769eb2b31b787a68c57be24bebee0`，balanced训练源码为 `d2996601429044e3ef3b71055a4ca546a3ca928e`；三个导出器的 `source_sha` 均为前者，因为bounded改动仅在训练loss，forward合同未变。balanced导出记录出现0471并非用旧未平衡权重：其checkpoint SHA与最终锁定的balanced权重收据一致。三组训练均为B折48,881场景/588 logs、24epoch/9168有效optimizer更新；`full_training_population=false`表示限定B折，不是少跑了本轮终点。

| 方法 | checkpoint SHA256（锁定与导出收据一致） |
|---|---|
| primary LOG_NONE | `ad8fd68ae83c9755b214a719b75cc10952b07f1dcb1e8dd74db913f2d69c1ba9` |
| primary MIN_NONE | `9b111049ae8b399dfb0a6d74da8acf9780914e4f99016d08964d323c7326ac5e` |
| balanced MIN_NONE | `6f9adff1768dadea9ae1c80c0f2ba27bdd698626009b864dcabd1a516adc478f` |

分析来源：[按run ID汇总脚本](../../../tools/decision_local/summarize_locked_repair_evaluation.py)，SHA256 `5443f8fc701923c9f3758df8d14360863d37fc9f417c3f62ab8f3f3201fd3d6f`；[配对算子](../../../tools/local_interaction_mask_v2/compare_pdms.py)，SHA256 `0197912c0aba48e22f4e291ae66359c9268b0fb0ede410a5956f49cb5dc0a8b2`。本次复核在内存中完成，原SUMMARY、RESULTS、评分与训练产物均未改写；仅新增本报告。

解释边界继续成立：旧Navtest已经反复观察，本轮共同终点锁定不能把它变成新的盲测集；三个头共享同一个已训练S0底座，头seed均42，proposal seed也固定42，不能视为三个独立VLA训练复跑。balanced是看到原结果后的一次有限KEEP/REPAIR loss对照，不能追认为原六组预登记主效应，也不能与LOG差异直接作单因素目标语义归因。

NONE表示未输入显式CV/FUT真值关系或预测环境关系，执行身份为camera_only，oracle_labels与predictor_sha256均为空；底座视觉表示当然可能包含已有场景知识，但本轮控制没有识别“新增环境知识”的效果。所用轨迹和当前场景表示来自固定cache，导出收据明确 `cache_based=true`；GPU导出耗时、并行分片吞吐和CPU评分耗时均不是原图端到端在线时延，也不构成实时部署性能证据。

正式Stage B保持主线程既有 [NOT_RUN处置](/mnt/project/minimal-repair-artifacts/20261015/STAGE_B_DISPOSITION_v15.json)，该处置在最终Navtest前形成且未使用其结果；两种P及三条R_MIN路径的小集24epoch真实A/B隔离smoke已完整，正式P/R及DIFF净收益本轮未检验。本报告记录这三个锁定控制的证据，不据Navtest追加方法选择或改变处置。
