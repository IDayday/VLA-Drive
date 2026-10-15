# 已执行的正式终点、部署与结果复核命令
所有下列入口有真实执行或真实数据验证记录；GPU绑定由当前授权资源与登记runner决定，不将本次24卡切分固定成通用配置。所有COMPLETE run已结束，不应重复启动或覆盖结果。原RAM路径已释放，共享源文件保留。旧命令是执行历史，任何新复跑须使用独立输出和相同身份。
原六组与seed43训练源码0471af96；限定平衡配方d2996601；评估模型路径仍0471af96；CPU计时包装与按run ID汇总3431960c。实际完整SHA见SOURCE_VERSIONS_FINAL.json。
## 正式训练与原生恢复
原正式B折训练的一个实际启动命令；完整九组见evidence/STAGE_A_V5_TASKS.json。B=48881，24epoch，不是全A+B训练。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/LOG_NONE --kind LOG_NONE --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v5.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

限定平衡配方的三组命令和唯一额外loss权重参数保存在evidence/STAGE_A_BALANCED_V14_TASKS.json；它们当时使用字节相同RAM镜像。当前共享原件映射如下，逻辑身份不变：

| 历史RAM尾路径 | 共享原件 |
|---|---|
| anchor_train_full_v1 | /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 |
| anchor_dev_full_v1 | /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 |
| targets_foldB_v5 | /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 |
| targets_dev1696_v5 | /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5 |

实际恢复带`--resume`命令及各段更新边界见EXECUTION_AND_RESUME_ZH.md。以下真实64场景48updates验证已完成，包含第7步中断及完整模型/optimizer/scheduler/RNG/数据/曝光对照；脚本的既有输出也不能覆盖：

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python /mnt/project/minimal-repair-artifacts/20261015/verify_balanced_resume_v14.py
```

## 正式权重的真实当前图像部署与缓存对齐

每个模型各4图，实际日志及逐场景q0/q_final对齐保存在对应验收JSON；这些不是全Navtest在线时延测量。

LOG_NONE_FORMAL_ONLINE_ACCEPTANCE_v14.json，checkpoint SHA256 `ad8fd68ae83c9755b214a719b75cc10952b07f1dcb1e8dd74db913f2d69c1ba9`。
日志：`/mnt/project/minimal-repair-artifacts/20261015/LOG_NONE_predict_current.py_formal_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/predict_current.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --refiner /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/LOG_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/online_formal_v14/LOG_NONE --split dev --max-seconds 900
```
日志：`/mnt/project/minimal-repair-artifacts/20261015/LOG_NONE_evaluate_minimal_repair_planning.py_formal_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_minimal_repair_planning.py --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --checkpoint /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/LOG_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/cached_formal4_v14/LOG_NONE --distribution NATURAL --mode camera_only --max-seconds 300
```

MIN_NONE_FORMAL_ONLINE_ACCEPTANCE_v14.json，checkpoint SHA256 `9b111049ae8b399dfb0a6d74da8acf9780914e4f99016d08964d323c7326ac5e`。
日志：`/mnt/project/minimal-repair-artifacts/20261015/MIN_NONE_predict_current.py_formal_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/predict_current.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --refiner /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/MIN_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/online_formal_v14/MIN_NONE --split dev --max-seconds 900
```
日志：`/mnt/project/minimal-repair-artifacts/20261015/MIN_NONE_evaluate_minimal_repair_planning.py_formal_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_minimal_repair_planning.py --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --checkpoint /mnt/project/minimal-repair-artifacts/20261015/stageA_v5_seed42/MIN_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/cached_formal4_v14/MIN_NONE --distribution NATURAL --mode camera_only --max-seconds 300
```

MIN_NONE_BALANCED_ONLINE_ACCEPTANCE_v14.json，checkpoint SHA256 `6f9adff1768dadea9ae1c80c0f2ba27bdd698626009b864dcabd1a516adc478f`。
日志：`/mnt/project/minimal-repair-artifacts/20261015/MIN_NONE_predict_current.py_balanced_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/predict_current.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --refiner /mnt/project/minimal-repair-artifacts/20261015/stageA_balanced_v14_seed42/MIN_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/online_balanced_v14/MIN_NONE --split dev --max-seconds 900
```
日志：`/mnt/project/minimal-repair-artifacts/20261015/MIN_NONE_evaluate_minimal_repair_planning.py_balanced_v14.log`，实际exit=0。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_minimal_repair_planning.py --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --checkpoint /mnt/project/minimal-repair-artifacts/20261015/stageA_balanced_v14_seed42/MIN_NONE/epoch_24.pt --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --output /mnt/project/minimal-repair-artifacts/20261015/cached_balanced4_v14/MIN_NONE --distribution NATURAL --mode camera_only --max-seconds 300
```

## 锁定后的最终Navtest

实际登记：evidence/FINAL_NAVTEST_V15_REGISTRY.json。执行前endpoint readiness已accepted；全部78项开发观察和12个共同24epoch终点核验，未按分数选择终点。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python /mnt/project/VLA-Drive-minimal-repair-20261015/tools/decision_local/verify_minimal_repair_endpoint.py --method-lock /mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/FINAL_METHOD_LOCK.json --registry /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json --output /mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/FINAL_READINESS.json
```

实际主机：`training-vlawm-tsinghua-zt3`；日志：`/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/training-vlawm-tsinghua-zt3.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 1 --max-seconds 14400
```

实际主机：`training-vlawm-tsinghua-zt4`；日志：`/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/training-vlawm-tsinghua-zt4.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 1 --max-seconds 14400
```

实际主机：`training-vla-zt`；日志：`/mnt/project/minimal-repair-artifacts/20261015/final_navtest_v15/training-vla-zt.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 1 --max-seconds 14400
```

各模型实际export与canonical score子命令、16 CPU workers及child-inclusive计时包装均在上述registry，不需要再拼装另一套评分器。最终三组均12146场景，failed=0。

## 已运行的结果与成本汇总

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python /mnt/project/VLA-Drive-minimal-repair-scoring-v15-20261015/tools/decision_local/summarize_locked_repair_evaluation.py --registry /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_V15_REGISTRY.json --output /mnt/project/minimal-repair-artifacts/20261015/FINAL_NAVTEST_RESULTS_v15
```

```bash
python /mnt/project/VLA-Drive-minimal-repair-20261015/tools/decision_local/summarize_minimal_repair_costs.py --root /mnt/project/minimal-repair-artifacts/20261015 --output /mnt/project/minimal-repair-artifacts/20261015/FINAL_COSTS_v15.json
```

CPU总计仍为记录下界，不能由`exceeds_recorded_cap=false`推出真实CPU总量未超160小时。新正式P/R和折交换没有运行；旧小集P/R与全链路命令保留在CURRENT_VECTOR_COMMANDS_ZH.md，并明确是smoke。

既有行为输出的只读汇总也已执行完成：

```bash
python /mnt/project/VLA-Drive-minimal-repair-20261015/tools/decision_local/summarize_repair_behavior.py --registries /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_BEHAVIOR_REGISTRY.json /mnt/project/minimal-repair-artifacts/20261015/BALANCED_V14_BEHAVIOR_REGISTRY.json --output /mnt/project/minimal-repair-artifacts/20261015/BEHAVIOR_RESULTS_v16
```
