# corrected v5 Stage A：共同24epoch结果与独立状态核对

主结果核对快照：2026-10-15 04:28:19 UTC；behavior完成复核：2026-10-15 04:33:27 UTC。原始Stage A的9个训练终点、72个canonical开发评估、8个mechanism及18个behavior均已完成。NATURAL下各头相对输入的PDMS点估计均未提高；RECOVERY下LOG的改善明显大于MIN。FUT−CV的配对区间均跨零，当前证据不足以确认稳定的未来关系增益，也不足以证明未来信息无价值。

本报告只纳入训练源码 `0471af96e4f769eb2b31b787a68c57be24bebee0`、corrected v5标签和独立的 `stageA_v5_seed42/43` run。旧v4仍为正确性hold，不进入这些结果。核对时工作树HEAD为 `d2996601429044e3ef3b71055a4ca546a3ca928e`；工作树提交与主线程后续loss修改均不替代原始训练身份。

本次只读取既有注册表、run身份、COMPLETE及聚合收据，并运行既有CPU汇总脚本读取已评分的开发集CSV；未加载训练权重、扫描fullB逐场景文件、连接远程主机或启动GPU任务。

共同完成状态及证据如下。

| 范围 | 已完成 / 登记 | 核对范围 |
|---|---:|---|
| Stage A formal训练 | 9 / 9 | seed42六组；seed43 LOG_FUT、MIN_CV、MIN_FUT；均24epoch |
| NATURAL canonical | 36 / 36 | 9个头 × 6/12/18/24epoch；每项1696场景 |
| RECOVERY canonical | 36 / 36 | 同一组头和观察点；每项1696场景 |
| mechanism | 8 / 8 | MIN_FUT两seed × 两分布 × zero/reverse_time |
| behavior | 18 / 18 | NATURAL 9/9、RECOVERY 9/9；全部固定24epoch终点 |

训练注册见 [STAGE_A_V5_TASKS.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json)。逐run独立读取终点收据，确认每组9168个数据及有效optimizer更新边界、1,173,144次原场景曝光、完整B折48,881场景；终点checkpoint文件均存在。同seed初始权重哈希、科学配方和最终曝光计数一致。各组均为KEEP 190,062、REPAIR 2,969、有效目标曝光193,031；原生586,572、横向293,286、时间293,286。已有 [共同终点账本审计](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_COMMON_ENDPOINT_AUDIT_v13.json) 还记录了各seed全9168步的逐步匹配；本次独立复核范围为身份和终点聚合收据。

评分注册为 [NATURAL](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json) 与 [RECOVERY](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json)。72项canonical及8项mechanism的导出均COMPLETE，q0/q_final评分均为1696场景、failed=0、valid=true；80项共用相同metric index、NAVSIM/nuPlan源码、runtime和protocol合同，prediction身份与score身份关联一致。这里完成的是开发评价，未产生本轮最终Navtest结论。

现有汇总脚本已用 `--require-complete` 执行成功，返回 `observations=72, pending=0, common_endpoint_models=18`；18指9个模型在两种输入分布的终点评估。新目录创建时要求不存在，未覆盖旧 `STAGE_A_RESULTS_PARTIAL_v12`。产物为 [SUMMARY.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_RESULTS_v13/SUMMARY.json)、[learning_curves.csv](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_RESULTS_v13/learning_curves.csv) 和 [自动共同终点报告](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_RESULTS_v13/COMMON_ENDPOINTS_ZH.md)。

主线程归档的 [聚合JSON](stage_a_results/SUMMARY.json)、[学习曲线CSV](stage_a_results/learning_curves.csv) 和 [共同终点表](stage_a_results/COMMON_ENDPOINTS_ZH.md) 与上述v13产物逐字节一致；另见已有 [共同学习曲线图](figures/stage_a_common_curves.png)。

执行命令（工作目录 `/mnt/project/VLA-Drive-minimal-repair-20261015`）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -B \
  tools/decision_local/summarize_minimal_repair_results.py \
  --registries /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json \
               /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json \
  --output /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_RESULTS_v13 --require-complete
```

汇总脚本SHA256：`252139a73bd48b7046ad52cceb5632842c00732761624115740a74276212b4b0`。下列分数和区间均来自该新汇总；不按6/12/18/24epoch结果挑选最佳权重。

共同24epoch终点以PDMS百分制表示，Δ为相对本分布实际输入计划的百分点差。NATURAL输入为原S0 q0，基线91.38836；RECOVERY为固定合成扰动后的plan_in，基线82.10776。两个基线不同，RECOVERY增益不能直接写成自然部署增益。NONE使用camera-only执行；CV/FUT均为特权探针。

95% CI使用1696场景所属16个log的配对聚类bootstrap（10,000次，固定seed=20260926）。这些区间描述固定训练模型在开发log上的不确定性，不含修复头跨训练seed的不确定性，也未作多重比较校正；底座S0固定，seed43仅复跑关键三组。零分修复和新增零分分别指canonical分数0→非0、非0→0，不等同后文局部约束REPAIR成功。

NATURAL的共同终点：

| 头seed | 组 | PDMS | Δ输入（pp） | 配对95% CI（pp） | 零分修复 / 新增零分 |
|---:|---|---:|---:|---|---:|
| 42 | LOG_NONE | 91.21920 | -0.16917 | [-0.46022, +0.26260] | 2 / 4 |
| 42 | LOG_CV | 91.18649 | -0.20187 | [-0.35523, +0.06620] | 1 / 2 |
| 42 | LOG_FUT | 91.11493 | -0.27343 | [-0.46585, -0.13994] | 0 / 3 |
| 42 | MIN_NONE | 91.38751 | -0.00085 | [-0.00160, +0.00025] | 0 / 0 |
| 42 | MIN_CV | 91.27994 | -0.10843 | [-0.29873, -0.00554] | 0 / 1 |
| 42 | MIN_FUT | 91.24982 | -0.13855 | [-0.29657, -0.03372] | 0 / 2 |
| 43 | LOG_FUT | 91.20104 | -0.18733 | [-0.31904, -0.02626] | 0 / 2 |
| 43 | MIN_CV | 91.26894 | -0.11943 | [-0.26907, -0.00823] | 0 / 2 |
| 43 | MIN_FUT | 91.36074 | -0.02762 | [-0.09412, -0.00240] | 0 / 0 |

RECOVERY的共同终点：

| 头seed | 组 | PDMS | Δ输入（pp） | 配对95% CI（pp） | 零分修复 / 新增零分 |
|---:|---|---:|---:|---|---:|
| 42 | LOG_NONE | 90.95513 | +8.84736 | [+7.42032, +11.21769] | 144 / 12 |
| 42 | LOG_CV | 90.96466 | +8.85689 | [+7.39696, +11.21596] | 142 / 8 |
| 42 | LOG_FUT | 90.88206 | +8.77429 | [+7.26760, +11.20247] | 141 / 9 |
| 42 | MIN_NONE | 82.29533 | +0.18756 | [+0.03417, +0.39174] | 3 / 0 |
| 42 | MIN_CV | 82.47144 | +0.36368 | [+0.08590, +0.79698] | 6 / 0 |
| 42 | MIN_FUT | 82.35663 | +0.24886 | [+0.06367, +0.53506] | 4 / 0 |
| 43 | LOG_FUT | 90.79257 | +8.68481 | [+7.34953, +10.78063] | 141 / 11 |
| 43 | MIN_CV | 82.58051 | +0.47275 | [+0.24689, +0.79034] | 9 / 0 |
| 43 | MIN_FUT | 82.55185 | +0.44408 | [+0.12959, +1.04270] | 8 / 0 |

相同seed的主要配对对照如下，括号中为95% CI，单位均为pp。DID为 `(MIN_FUT−MIN_CV)−(LOG_FUT−LOG_CV)`；它是有符号差分，不是可执行策略的得分。

| seed | 对照（前者减后者） | NATURAL：Δ [CI] | RECOVERY：Δ [CI] |
|---:|---|---|---|
| 42 | MIN_FUT−MIN_CV | -0.03012 [-0.06594, +0.00523] | -0.11482 [-0.41604, +0.00418] |
| 43 | MIN_FUT−MIN_CV | +0.09181 [-0.03522, +0.24588] | -0.02867 [-0.25413, +0.35109] |
| 42 | LOG_FUT−LOG_CV | -0.07156 [-0.36225, +0.07526] | -0.08260 [-0.26738, +0.16241] |
| 42 | MIN_NONE−LOG_NONE | +0.16832 [-0.26285, +0.45911] | -8.65980 [-11.07930, -7.21541] |
| 42 | MIN_FUT−LOG_FUT | +0.13488 [-0.08752, +0.34871] | -8.52543 [-10.85928, -7.04018] |
| 43 | MIN_FUT−LOG_FUT | +0.15971 [-0.00207, +0.28985] | -8.24072 [-10.02850, -7.06818] |
| 42 | MIN_FUT−MIN_NONE | -0.13770 [-0.29647, -0.03286] | +0.06130 [-0.05321, +0.26822] |
| 42 | DID | +0.04144 [-0.12395, +0.36549] | -0.03222 [-0.46408, +0.24421] |

自然输入下，MIN_NONE基本保持输入分数（−0.00085pp），MIN_FUT两seed分别为−0.13855/−0.02762pp，LOG_FUT为−0.27343/−0.18733pp。恢复输入下，LOG四个已登记实例提升8.68481至8.85689pp，MIN五个实例提升0.18756至0.47275pp；相同信息源的MIN_FUT−LOG_FUT分别为−8.52543/−8.24072pp。可以报告这种分布和目标间的差异；不能据此把局部最小修复目标与日志GT拟合当成同一个任务。

FUT相对CV未显示跨seed、跨分布一致的优势：MIN的四个FUT−CV区间，以及seed42的LOG FUT−CV和DID区间均跨零。自然seed42的MIN_FUT−MIN_NONE为−0.13770pp，区间全负，但该比较同时改变了关系值与特权support，不能只归因于未来关系本身。

mechanism的8项均为已训练MIN_FUT终点上的固定干预，注册见 [NATURAL mechanism](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_MECHANISM_NATURAL.json) 和 [RECOVERY mechanism](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_MECHANISM_RECOVERY.json)。下表仅报告相对同一头正常执行的PDMS点差，不用干预分数选择执行策略。

| 输入 | seed | 正常MIN_FUT PDMS | 全关系置零Δ（pp） | 全关系时间反转Δ（pp） |
|---|---:|---:|---:|---:|
| NATURAL | 42 | 91.24982 | -0.05369 | -0.00005 |
| NATURAL | 43 | 91.36074 | -0.00185 | -0.00001 |
| RECOVERY | 42 | 82.35663 | +0.20779 | -0.00005 |
| RECOVERY | 43 | 82.55185 | +0.04715 | -0.05073 |

`zero`将全部关系数值置零，`reverse_time`反转全部五个关系通道的时间顺序，二者均保留原support；这不是仅干预未来信息的实验。时间反转的影响在这些点估计中较小，关系置零在RECOVERY反而提高分数；目前没有“更准确的未来关系稳定带来更高得分”的机制证据，但这些干预也不能证明模型完全不使用关系或未来信息没有价值。

behavior按 [18项固定终点注册](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_BEHAVIOR_REGISTRY.json) 核对，现已18/18 COMPLETE，全部退出码为0、未超时；每项覆盖1696场景且paired_planning.valid=true。下表“评分失败”指输入/输出评分处理失败数，不是PDMS零分或分数下降；“REPAIR未成功”为REPAIR总数减成功数。KEEP误改沿用聚合字段 `KEEP_unnecessary_modifications`，仅指超过局部KEEP改动阈值，不能据此断定路线或交通行为错误。

| 输入 | seed | 组 | 完成收据 | 评分失败（输入/输出） | KEEP误改 / 总数 | REPAIR成功 / 总数 | REPAIR未成功 / 总数 |
|---|---:|---|---|---:|---:|---:|---:|
| NATURAL | 42 | LOG_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_LOG_NONE_epoch24/COMPLETE.json) | 0 / 0 | 11 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 42 | LOG_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_LOG_CV_epoch24/COMPLETE.json) | 0 / 0 | 11 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 42 | LOG_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_LOG_FUT_epoch24/COMPLETE.json) | 0 / 0 | 16 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 42 | MIN_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_MIN_NONE_epoch24/COMPLETE.json) | 0 / 0 | 0 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 42 | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 0 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 42 | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed42_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 0 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 43 | LOG_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed43_LOG_FUT_epoch24/COMPLETE.json) | 0 / 0 | 15 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 43 | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed43_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 0 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | 43 | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/natural/stageA_v5_seed43_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 0 / 287 | 0 / 10 | 10 / 10 |
| RECOVERY | 42 | LOG_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_LOG_NONE_epoch24/COMPLETE.json) | 0 / 0 | 237 / 264 | 4 / 15 | 11 / 15 |
| RECOVERY | 42 | LOG_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_LOG_CV_epoch24/COMPLETE.json) | 0 / 0 | 242 / 264 | 4 / 15 | 11 / 15 |
| RECOVERY | 42 | LOG_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_LOG_FUT_epoch24/COMPLETE.json) | 0 / 0 | 240 / 264 | 4 / 15 | 11 / 15 |
| RECOVERY | 42 | MIN_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_MIN_NONE_epoch24/COMPLETE.json) | 0 / 0 | 0 / 264 | 1 / 15 | 14 / 15 |
| RECOVERY | 42 | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 0 / 264 | 0 / 15 | 15 / 15 |
| RECOVERY | 42 | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed42_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 0 / 264 | 0 / 15 | 15 / 15 |
| RECOVERY | 43 | LOG_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed43_LOG_FUT_epoch24/COMPLETE.json) | 0 / 0 | 237 / 264 | 4 / 15 | 11 / 15 |
| RECOVERY | 43 | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed43_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 0 / 264 | 2 / 15 | 13 / 15 |
| RECOVERY | 43 | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_v5/recovery/stageA_v5_seed43_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 0 / 264 | 2 / 15 | 13 / 15 |

KEEP改动阈值为最大位置变化>0.25m或最大heading变化>0.1rad；REPAIR成功要求输出通过密集局部几何、运动学和相对输入改动范围检查。自然分布9项的REPAIR成功均为0/10；恢复分布seed42的LOG三组均为4/15，MIN_NONE为1/15，MIN_CV和MIN_FUT均为0/15；seed43的LOG_FUT为4/15，MIN_CV和MIN_FUT均为2/15。canonical零分恢复可能发生在不同标签组，也不要求满足全部局部约束，因此不能替代此处的REPAIR成功率。

自然输入中，MIN各头均未触发KEEP改动阈值，LOG各头为11至16/287；恢复输入中，seed42的LOG_NONE/LOG_CV/LOG_FUT分别改动237/264、242/264、240/264个KEEP案例，seed43 LOG_FUT为237/264；两个seed的全部MIN头均为0/264。LOG在合成恢复上回归日志GT的行为与MIN保持局部可接受输入的目标存在差异。KEEP只认证所选标注下的局部约束；这些改动不能一律称为对正确路线的破坏，也不能从PDMS提高反推所有路线或交通规则均正确。

原聚合字段 `new_stop` 只统计输入计划终点位移≥1m、输出计划终点位移<1m的转换，是终点位移代理，不是基于速度的停车检测，也不称为真实停车。自然输入全部头为0/1696；恢复输入seed42 LOG_NONE/LOG_CV/LOG_FUT分别为5/1696、5/1696、4/1696，seed43 LOG_FUT为4/1696，其余MIN头均为0/1696。

训练侧fit与一轮KEEP/REPAIR平衡修复对照作为独立后续工作推进。已有 [TRAINING_CONDITIONAL_FIT_v13.json](/mnt/project/minimal-repair-artifacts/20261015/TRAINING_CONDITIONAL_FIT_v13.json) 完成了冻结seed42六个24epoch头的只读拟合诊断：固定4096诊断集与B折的交集，共1975个原场景，观察前8个登记计划选择epoch；有效曝光为KEEP 2515、REPAIR 47。重复计划不当作独立场景，模型状态没有更新；这既不是完整B折拟合统计，也不是独立泛化测试。

| 组 | KEEP目标ADE（m） | REPAIR输入→目标ADE（m） | REPAIR预测→目标ADE（m） | REPAIR平均改动（m） |
|---|---:|---:|---:|---:|
| LOG_NONE | 0.08883 | 1.09944 | 0.10521 | 1.07661 |
| LOG_CV | 0.08956 | 1.09944 | 0.10936 | 1.07697 |
| LOG_FUT | 0.08968 | 1.09944 | 0.11220 | 1.07311 |
| MIN_NONE | 0.00413 | 0.25361 | 0.23971 | 0.01788 |
| MIN_CV | 0.00581 | 0.25361 | 0.20573 | 0.06532 |
| MIN_FUT | 0.00585 | 0.25361 | 0.20220 | 0.06634 |

LOG的目标是日志GT，MIN的目标是KEEP/REPAIR teacher，不能直接跨目标比较ADE大小。MIN_FUT在REPAIR曝光上的误差从输入0.25361m降至0.20220m，但平均改动只有0.06634m；MIN_NONE改动更小。结合KEEP拟合接近恒等和训练曝光的严重不均衡，这些数据支持“修复监督没有被充分拟合”作为待检验解释，不足以单独证明loss不平衡就是原因。

主线程的一轮限定KEEP/REPAIR平衡对照已启动，状态为 **in progress**：[stageA_balanced_v14_seed42三组登记](/mnt/project/minimal-repair-artifacts/20261015/STAGE_A_BALANCED_V14_TASKS.json) 仅包含MIN_NONE、MIN_CV、MIN_FUT，源码为 `d2996601429044e3ef3b71055a4ca546a3ca928e`；三组均有启动收据，尚无COMPLETE。按 [固定训练曝光权重](/mnt/project/minimal-repair-artifacts/20261015/REPAIR_CLASS_BALANCE_v14.json)，KEEP=0.5078106091696394、REPAIR=32.507746716066016，仅调整训练loss加权，不改变采样或部署；权重不使用开发集统计。原始六组及关键三组复跑保持独立，本文所有主比较仍只取原9个正式24epoch终点。

本次未修改训练器或启动该对照；平衡结果应以独立源码、run身份和收据报告，不并入原登记的72项观察。后续对照属于观察本轮结果后推进的研究，不能回写成原预登记比较，也不能由当前fit诊断宣称平衡修复已经奏效。

共享A折标签构建已COMPLETE：52,711/52,711场景，errors=[]，见 [FOLDA_CLOSURE_RESULT_v5.json](/mnt/project/minimal-repair-artifacts/20261015/FOLDA_CLOSURE_RESULT_v5.json)。这是已有闭合聚合收据，本次未复扫A折；构建完整也不表示每条计划都有可靠监督。A标签准备与Stage B正式训练分开记录。

主线程另行执行的真实图像在线/缓存复核已有 [FORMAL_ONLINE_ACCEPTANCE_v14.json](/mnt/project/minimal-repair-artifacts/20261015/FORMAL_ONLINE_ACCEPTANCE_v14.json) 通过收据：原seed42正式LOG_NONE、MIN_NONE终点在同4个开发样本上的q0/q_final均exact=true、max_abs=0，实际visual/language/proposal前向检查有记录。这仅验证这两个头在有限样本中的执行一致性，不构成全开发集在线等价、延迟或路线正确性的证明；未做Navtest。

标签覆盖使用现有 [TARGET_QUALITY_FOLDB_v5.json](/mnt/project/minimal-repair-artifacts/20261015/TARGET_QUALITY_FOLDB_v5.json) 聚合证据：488,810个预构造计划中KEEP 76,836、REPAIR 1,930，有效目标比例16.1138%；只有8,456/48,881个原场景至少有一条有效计划。REPAIR中194项涉及已标注障碍交叠，原生计划仅21项REPAIR。实际训练曝光另外计数：2,969/193,031个有效曝光为REPAIR（1.5381%），KEEP:REPAIR约64:1；计划池计数与训练曝光不能混用。

目标资格仅控制loss监督；CV/FUT输入的`common_valid`则取CURRENT、CV、FUTURE有效性mask交集，包含未来标注支持信息。FUT−CV识别的是同一特权support条件下关系数值的增量；FUT−NONE还含support信息差异。低覆盖、动态修复稀疏、固定且已见A+B的同一S0底座，以及仅16个开发log，都限制了“未来信息价值”的可识别范围；A/B隔离只针对新增P/R。全部开发场景仍留在canonical分母中；未知不被伪造为KEEP。

局部KEEP、局部REPAIR成功和官方PDMS各有不同定义；本文不宣称完整导航路线、信号、交通规则或连续时间安全正确性。A折标签准备、训练fit和独立平衡对照均不代表Stage B正式P/R已启动。本文只记录已核验结果与尚缺证据，Stage B是否放行及最终方法选择留待主线程决策。
