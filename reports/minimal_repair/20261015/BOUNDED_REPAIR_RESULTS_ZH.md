# 限定KEEP/REPAIR平衡修复：共同24epoch结果

主结果核对：2026-10-15 04:52:43 UTC；诊断完成复核：2026-10-15 04:57:58 UTC。训练3/3、canonical主评分6/6、mechanism 4/4及behavior 6/6均已COMPLETE。三组恢复输入均有正向配对增益，自然输入均未超过输入基线；FUT−CV在两种分布的区间均跨零。四种关系干预的点估计方向一致，但区间均跨零，尚无明确的正向未来机制证据。

本报告仅分析 `stageA_balanced_v14_seed42` 的MIN_NONE、MIN_CV、MIN_FUT三组冻结24epoch终点。这是观察原Stage A结果后的一轮限定loss修复，训练源码 `d2996601429044e3ef3b71055a4ca546a3ca928e`；原始9个正式头及其72项观察保持独立，见 [原Stage A报告](ROUND1_STAGE_A_RESULTS_ZH.md)。本轮仅一个修复头seed，不把后续结果回写成原预登记结论。主线程已登记正式Stage B为 `NOT_RUN`，处置依据及smoke边界见文末。

共同24epoch分数均为PDMS百分制。NATURAL的实际输入为原S0 q0，基线91.38836；RECOVERY为固定合成扰动plan_in，基线82.10776。NONE为camera-only，CV/FUT为特权探针。六项导出和评分均1696场景，failed=0、valid=true；checkpoint SHA与各自训练fit收据一致，评分身份与预测身份匹配。canonical合同与原Stage A相同，同一分布三个输入评分CSV逐字节一致。

| 输入 | 组 | PDMS | Δ输入（pp） | 配对95% CI（pp） | 零分修复 / 新增零分 |
|---|---|---:|---:|---|---:|
| NATURAL | MIN_NONE | 91.05623 | -0.33213 | [-1.19179, +0.17465] | 4 / 9 |
| NATURAL | MIN_CV | 91.33142 | -0.05694 | [-0.55650, +0.26295] | 5 / 5 |
| NATURAL | MIN_FUT | 91.33713 | -0.05123 | [-0.89499, +0.35279] | 5 / 5 |
| RECOVERY | MIN_NONE | 84.21945 | +2.11168 | [+1.18867, +2.83950] | 44 / 5 |
| RECOVERY | MIN_CV | 84.44177 | +2.33401 | [+1.85813, +2.85749] | 44 / 2 |
| RECOVERY | MIN_FUT | 83.95654 | +1.84877 | [+1.45218, +2.50483] | 39 / 4 |

主要配对区间使用同一1696场景、16个开发log的聚类bootstrap，10,000次、固定seed=20260926。区间描述固定seed42模型的开发log不确定性，不包含跨训练seed不确定性，也未作多重比较校正。零分修复/新增零分指canonical 0→非0/非0→0，不等同局部标签的REPAIR成功。

| 输入 | 对照（前者减后者） | Δ（pp） | 配对95% CI（pp） |
|---|---|---:|---|
| NATURAL | MIN_FUT−MIN_CV | +0.00571 | [-0.38203, +0.21150] |
| NATURAL | MIN_FUT−MIN_NONE | +0.28090 | [+0.04062, +0.43490] |
| RECOVERY | MIN_FUT−MIN_CV | -0.48523 | [-0.85852, +0.07159] |
| RECOVERY | MIN_FUT−MIN_NONE | -0.26291 | [-0.80990, +0.74013] |

自然输入FUT−NONE为+0.28090pp，区间全正，但FUT−CV仅+0.00571pp且区间跨零；FUT本身相对输入仍为−0.05123pp。这里的正差说明FUT比本轮NONE少损失，不能写成自然输入性能提高，也不能归因于纯未来关系：FUT−NONE同时改变关系值和特权support。

恢复输入三组相对输入分别提升+2.11168、+2.33401、+1.84877pp，区间均为正；CV点估计最高，FUT−CV为−0.48523pp、FUT−NONE为−0.26291pp，二者区间均跨零。平衡后的三组比原同组恢复分数分别提高+1.92412、+1.97033、+1.59991pp；自然输入对应变化为−0.33128、+0.05149、+0.08731pp。后一组新旧差值仅作点估计描述，未另报配对区间。合成恢复收益不能直接写成自然部署收益。

mechanism按 [NATURAL v15注册](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_V15_MECHANISM_NATURAL.json) 与 [RECOVERY v15注册](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_V15_MECHANISM_RECOVERY.json) 执行，4/4已COMPLETE，均1696场景、零评分失败。所有干预保持同一MIN_FUT checkpoint和输入计划；下表Δ为“干预−同头正常执行”，负值表示正常关系输入的点估计较好。区间用既有 `compare_pdms.compare` 对已生成的开发评分CSV配对计算，方法与主比较相同。

| 输入 | 干预 | PDMS | 干预−正常（pp） | 配对95% CI（pp） |
|---|---|---:|---:|---|
| NATURAL | [zero](/mnt/project/minimal-repair-artifacts/20261015/mechanism_balanced_v14/natural/scores/stageA_balanced_v14_seed42_MIN_FUT_epoch24_zero/COMPLETE.json) | 91.18767 | -0.14945 | [-0.31603, +0.10325] |
| NATURAL | [reverse_time](/mnt/project/minimal-repair-artifacts/20261015/mechanism_balanced_v14/natural/scores/stageA_balanced_v14_seed42_MIN_FUT_epoch24_reverse_time/COMPLETE.json) | 91.27786 | -0.05927 | [-0.23773, +0.00066] |
| RECOVERY | [zero](/mnt/project/minimal-repair-artifacts/20261015/mechanism_balanced_v14/recovery/scores/stageA_balanced_v14_seed42_MIN_FUT_epoch24_zero/COMPLETE.json) | 83.81465 | -0.14189 | [-0.92288, +0.29659] |
| RECOVERY | [reverse_time](/mnt/project/minimal-repair-artifacts/20261015/mechanism_balanced_v14/recovery/scores/stageA_balanced_v14_seed42_MIN_FUT_epoch24_reverse_time/COMPLETE.json) | 83.88533 | -0.07121 | [-0.27451, +0.03823] |

`zero`置零全部五个关系通道，`reverse_time`反转全部五个通道的时间顺序，两者保留原support，均不是纯未来信息干预。四种干预均降低分数，支持方向一致的关系依赖迹象；但四个区间都跨零，且FUT−CV没有显示优势，因此尚不能据此确立稳定的未来关系增益，也不能宣布未来信息无价值。没有按干预结果选择执行策略。

机制配对计算只读取既有评分CSV，没有新推理。算子 [compare_pdms.py](../../../tools/local_interaction_mask_v2/compare_pdms.py) 的SHA256为 `0197912c0aba48e22f4e291ae66359c9268b0fb0ede410a5956f49cb5dc0a8b2`；每个干预及其正常执行的canonical合同相同，q0评分CSV逐字节一致。

behavior按 [六项固定终点注册](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_V14_BEHAVIOR_REGISTRY.json) 执行，6/6已COMPLETE，全部退出码0、未超时，每项1696场景。评分失败指处理失败数；REPAIR未成功为该组总数减成功数，二者含义不同。

| 输入 | 组 | 完成收据 | 评分失败（输入/输出） | KEEP改动 / 总数 | REPAIR成功 / 总数 | REPAIR未成功 / 总数 |
|---|---|---|---:|---:|---:|---:|
| NATURAL | MIN_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/natural/stageA_balanced_v14_seed42_MIN_NONE_epoch24/COMPLETE.json) | 0 / 0 | 8 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/natural/stageA_balanced_v14_seed42_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 5 / 287 | 0 / 10 | 10 / 10 |
| NATURAL | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/natural/stageA_balanced_v14_seed42_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 4 / 287 | 0 / 10 | 10 / 10 |
| RECOVERY | MIN_NONE | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/recovery/stageA_balanced_v14_seed42_MIN_NONE_epoch24/COMPLETE.json) | 0 / 0 | 15 / 264 | 2 / 15 | 13 / 15 |
| RECOVERY | MIN_CV | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/recovery/stageA_balanced_v14_seed42_MIN_CV_epoch24/COMPLETE.json) | 0 / 0 | 8 / 264 | 0 / 15 | 15 / 15 |
| RECOVERY | MIN_FUT | [COMPLETE](/mnt/project/minimal-repair-artifacts/20261015/behavior_balanced_v14/recovery/stageA_balanced_v14_seed42_MIN_FUT_epoch24/COMPLETE.json) | 0 / 0 | 4 / 264 | 1 / 15 | 14 / 15 |

KEEP改动阈值为最大位置变化>0.25m或最大heading变化>0.1rad；聚合字段 `KEEP_unnecessary_modifications` 仅是局部KEEP标签下的改动，不能一律称为路线或交通错误。原seed42三组在两种分布的KEEP改动均为0，本轮增加到自然8/5/4、恢复15/8/4。自然局部REPAIR仍全部0/10；恢复从原1/0/0变成2/0/1（按NONE/CV/FUT顺序），没有与整体恢复PDMS改善等比例的局部修复成功提升。标签REPAIR仅10或15例，且成功要求密集局部几何、运动学和相对改动范围均通过，不能用全1696场景的canonical零分修复数代替。

原字段 `new_stop` 仅为输入终点位移≥1m到输出终点位移<1m的代理，不是基于速度的停车检测。本轮六项该代理均为0/1696，不能由此推出没有真实停车或进度问题。

三组均完成完整B折48,881场景、588个log、24epoch、9168次数据及有效optimizer更新、1,173,144次原场景曝光。各组KEEP 190,062、REPAIR 2,969、有效目标193,031；原生586,572、横向293,286、时间293,286。独立核对新旧对应组的identity及COMPLETE，初始化、数据、曝光和配方一致；identity差异仅为权重收据、源码及其文件哈希、验收收据。既有 [共同终点审计](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_COMMON_ENDPOINT_AUDIT_v14.json) 与 [新旧匹配审计](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_MATCHING_AUDIT_v14.json) 均通过；本次没有复扫逐步日志或fullB场景。

固定训练曝光权重为KEEP=0.5078106091696394、REPAIR=32.507746716066016，见 [权重收据](/mnt/project/minimal-repair-artifacts/20261015/REPAIR_CLASS_BALANCE_v14.json)。仅改变训练loss加权，分母保留原optimizer batch有效场景数；不按microbatch类别数归一化，不改变采样、标签或部署输入，权重不使用开发集统计。训练中的两类有效曝光仍约64:1，不能把加权写成新增了REPAIR数据。

训练fit使用 [原v13](/mnt/project/minimal-repair-artifacts/20261015/TRAINING_CONDITIONAL_FIT_v13.json) 与 [balanced v14](/mnt/project/minimal-repair-artifacts/20261015/TRAINING_CONDITIONAL_FIT_balanced_v14.json) 的既有聚合结果。两者index SHA、标签identity、目标和分组曝光完全一致：固定4096诊断集与B折交集的1975个原场景，观察前8个登记计划选择epoch，KEEP 2515、REPAIR 47次有效计划曝光。这里的8epoch描述诊断计划选择，不是模型训练长度；重复计划不是独立场景，诊断不更新模型，也不是完整B折拟合或泛化检验。fit文件中的 `source_sha=0471af96...` 为诊断执行代码来源；平衡头的训练identity仍为 `d2996601...`。

| 组 | KEEP平均改动：旧→新（m） | REPAIR目标ADE：旧→新（m） | REPAIR平均改动：旧→新（m） | REPAIR较输入更近比例：旧→新 |
|---|---:|---:|---:|---:|
| MIN_NONE | 0.00413 → 0.05886 | 0.23971 → 0.18704 | 0.01788 → 0.12416 | 95.74% → 87.23% |
| MIN_CV | 0.00581 → 0.04741 | 0.20573 → 0.16542 | 0.06532 → 0.14420 | 91.49% → 87.23% |
| MIN_FUT | 0.00585 → 0.04323 | 0.20220 → 0.16937 | 0.06634 → 0.13603 | 91.49% → 91.49% |

三组REPAIR平均目标ADE相对旧头下降21.97%、19.59%、16.24%，共同输入→REPAIR目标ADE为0.25361m；与此同时KEEP平均改动增加。NONE/CV的“比输入更接近目标”曝光比例还下降，因此平均误差改善不是所有案例都改善。MIN_FUT的REPAIR目标ADE仍略高于MIN_CV（0.16937 vs 0.16542m）；这组fit本身没有证明未来关系更有价值。

实际评分routing为v15。原v14计时包装依赖容器中缺失的 `/usr/bin/time`，失败发生在canonical启动前；已有预测及旧失败记录保留，见 [ABSENT_TIME_RECOVERY_v15.json](/mnt/project/minimal-repair-artifacts/20261015/ABSENT_TIME_RECOVERY_v15.json)。恢复使用固定源码 `3431960c` 的Python stdlib `RUSAGE_CHILDREN` 包装；已逐项核对6个主评估和4个mechanism的新旧canonical子命令、导出命令及模型/输出路径均相同。RECOVERY MIN_CV增加等待旧导出COMPLETE的依赖，避免双写。此项是评分基础设施恢复，不是模型或评分数值变更。

现有汇总脚本已以v15主评估注册运行 `--require-complete`，输出 `observations=6, pending=0, common_endpoint_models=6`。新目录 `BALANCED_RESULTS_v14` 创建时要求不存在，未覆盖原Stage A或其他结果。产物为 [SUMMARY.json](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_RESULTS_v14/SUMMARY.json)、[六项终点CSV](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_RESULTS_v14/learning_curves.csv)、[自动共同终点报告](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_RESULTS_v14/COMMON_ENDPOINTS_ZH.md)。本轮CSV只有固定24epoch终点，不构成多epoch学习曲线。

运行命令（工作目录 `/mnt/project/VLA-Drive-minimal-repair-20261015`）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -B \
  tools/decision_local/summarize_minimal_repair_results.py \
  --registries /mnt/project/minimal-repair-artifacts/20261015/BALANCED_V15_EVAL_NATURAL.json \
               /mnt/project/minimal-repair-artifacts/20261015/BALANCED_V15_EVAL_RECOVERY.json \
  --output /mnt/project/minimal-repair-artifacts/20261015/BALANCED_RESULTS_v14 --require-complete
```

汇总脚本SHA256为 `252139a73bd48b7046ad52cceb5632842c00732761624115740a74276212b4b0`。本次只运行既有CPU汇总，读取已生成的收据、聚合统计和开发评分CSV；没有新建模型推理、训练或GPU分析，没有SSH连接或fullB逐场景扫描。

正式balanced MIN_NONE已有 [在线/缓存验收](/mnt/project/minimal-repair-artifacts/20261015/BALANCED_ONLINE_ACCEPTANCE_v14.json)：同4个开发样本q0/q_final均逐位一致，checkpoint SHA为 `6f9adff1768dadea9ae1c80c0f2ba27bdd698626009b864dcabd1a516adc478f`。这只是有限样本的执行一致性证据，不是全量在线性能、路线正确性或Navtest结果。

CV与FUT共用CURRENT/CV/FUTURE有效性mask交集，support本身含未来标注信息；FUT−CV只识别同一特权support下关系数值的增量。标签覆盖和动态修复稀疏的限制继续存在，KEEP只表示所选标注下的局部可行性，不认证完整路线、交通规则或安全。固定S0已见A+B，新增P/R折间隔离不等于底座完全OOF。

主线程已在 [STAGE_B_DISPOSITION_v15.json](/mnt/project/minimal-repair-artifacts/20261015/STAGE_B_DISPOSITION_v15.json) 登记正式Stage B为 `NOT_RUN`。该处置综合原配方与平衡修复两套匹配结果、原关键seed复核、训练fit及标签覆盖限制：一次限定学习修复改善了稀有REPAIR拟合，但仍未形成稳定的future-over-CV自然净收益；不是根据CI跨零机械停止，也不证明未来信息无价值。该记录确认Stage B两种预测器、三条R_MIN路径均完成真实A/B隔离的小集24epoch smoke，代码smoke已完整；正式P/R及DIFF净收益本轮未检验。处置未使用Navtest，本报告的比较与结论均限于上述开发结果。
