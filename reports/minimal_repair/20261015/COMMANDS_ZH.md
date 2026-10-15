# 实际执行、评价与恢复命令

**当前状态：旧v4九组已因初始速度向量边界错误安全暂停。下文v4命令是历史执行收据，
不要重新运行或恢复这些旧正式任务。** 新版源码0471af96与独立v5标签正在验证，
修正后已执行命令见 [CURRENT_VECTOR_COMMANDS_ZH.md](CURRENT_VECTOR_COMMANDS_ZH.md)。不可热修改冻结worktree，训练身份不由报告分支替代。

## 标签生成和核验

主机：`training-vla-zt`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-execution-v4-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --workers 24 --max-seconds 5400 --shards 4 --rank 0
```

主机：`training-rl-zt2`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-execution-v4-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --workers 24 --max-seconds 5400 --shards 4 --rank 1
```

主机：`training-vlawm-tsinghua-zt5`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-execution-v4-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --workers 24 --max-seconds 5400 --shards 4 --rank 2
```

主机：`training-vlawm-tsinghua-zt4`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-execution-v4-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --workers 24 --max-seconds 5400 --shards 4 --rank 3
```

四分片完成后的核验入口（已实际启动）：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python /mnt/project/minimal-repair-artifacts/20261015/close_foldB_v4_v2.py
```
原失败等待脚本和日志保留，不重跑生产分片。

## 阶段A固定训练

六组seed42和三组seed43均在同一完整B折、同一目标资格上训练24epoch。命令中的输出路径已登记；复现必须用独立输出，不覆盖原run。

`stageA_seed42_LOG_NONE`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/LOG_NONE --kind LOG_NONE --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed42_LOG_CV`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/LOG_CV --kind LOG_CV --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed42_LOG_FUT`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/LOG_FUT --kind LOG_FUT --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed42_MIN_NONE`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/MIN_NONE --kind MIN_NONE --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed42_MIN_CV`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/MIN_CV --kind MIN_CV --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed42_MIN_FUT`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/MIN_FUT --kind MIN_FUT --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed43_LOG_FUT`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed43/LOG_FUT --kind LOG_FUT --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 43 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed43_MIN_CV`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed43/MIN_CV --kind MIN_CV --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 43 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

`stageA_seed43_MIN_FUT`：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed43/MIN_FUT --kind MIN_FUT --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 43 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200
```

## 安全暂停与恢复

在某个run输出目录创建 `STOP_REQUESTED`，训练器在完成当前optimizer边界后保存latest并退出。只对本任务run操作；不要停止G2或旧任务。检查PAUSED、latest及进程退出后，把STOP_REQUESTED改名留档，再以完全相同命令追加 `--resume`。恢复器核对源码、目标版本、fold、plan pool、模型、optimizer、scheduler、全部RNG和数据位置，不重置epoch或warmup。

下面保留旧LOG_NONE的原生恢复语法作为历史记录（目前有CORRECTNESS_HOLD，禁止据此恢复旧run）：
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-training-v5-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v4 --dev-features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --dev-labels /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v4 --output /mnt/project/minimal-repair-artifacts/20261015/stageA_seed42/LOG_NONE --kind LOG_NONE --scope repair_formal --fold-partition /mnt/project/decision-local-consequence-artifacts/20261014/conditional_oof_v1/PARTITION.json --training-fold 1 --target-acceptance /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_ACCEPTANCE_v1.json --seed 42 --epochs 24 --batch 128 --microbatch 16 --lr 0.0001 --data-reader mmap_tensors --max-seconds 7200 --resume
```

## 固定开发观察

以下两个有限注册表各含36项，只含6/12/18/24epoch，不选择事后最好权重。GPU数由当前授权资源决定，分片不改变batch1算术。相同分布的q0评分只有在逐场景轨迹和metric-cache哈希相同时复用。

主机：`training-vla-zt`；分布：`NATURAL`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-evaluation-v6-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_EVAL_NATURAL.json --gpus 3 4 5 6 7 --cpu-parallel 2 --max-seconds 21600
```

主机：`training-rl-zt2`；分布：`RECOVERY`
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-evaluation-v6-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_EVAL_RECOVERY.json --gpus 5 6 7 --cpu-parallel 2 --max-seconds 21600
```

tsinghua-zt4另有两个共享GPU observer，使用相同注册表和原子任务claim，不新增实验。收据：`ADDITIONAL_EVALUATION_RUNNERS_TSG4.json`。

## 已执行真实图像部署验证

下面4场景部署已经完成，原输出须保留；再次验证需另设输出。P/R为链路smoke头，不是正式终点结果。
```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-execution-v4-20261015/tools/decision_local/predict_current.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --refiner /mnt/project/minimal-repair-artifacts/20261015/smoke_v3/R_MIN_DIFF/epoch_24.pt --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v3/P_DIFF/epoch_24.pt --output /mnt/project/minimal-repair-artifacts/20261015/online_repair_smoke4_v4 --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --split dev --max-seconds 900
```

## 汇总与阶段B边界

`summarize_minimal_repair_results.py` 已在真实小集分数上执行。正式结果全部闭合后，用两个阶段A注册表和 `--require-complete` 汇总，保留全部观察点及24epoch共同终点。阶段B正式训练目前尚未放行，不能把小集P_A→R_B命令或本文件当作已完成完整预测关系实验。阶段B须保持原A/B、相同新标签、对应P的真实留出预测和同一24epoch配方。

有限资源/状态根目录：`/mnt/project/minimal-repair-artifacts/20261015`。只有本轮分支推送到远端，不自动合并；数据、权重和私有图像不随Git上传。
