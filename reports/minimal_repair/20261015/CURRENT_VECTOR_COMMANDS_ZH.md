# 修正当前速度边界后的实际命令

执行源码：`0471af96e4f769eb2b31b787a68c57be24bebee0`，冻结目录 `/mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015`。下列命令均已实际启动，有对应日志；smoke不是正式成绩。输出已存在，不能盲目重复执行。旧v4训练保持暂停。

## V5_VALIDATION_JOBS.json

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_train256_v5.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/TRAIN_DIAGNOSTIC_256.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_train256_v5 --workers 16 --max-seconds 600
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_smoke_A_v5.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --index /mnt/project/minimal-repair-artifacts/20261015/TRAIN_SMOKE_A.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_A_v5 --workers 4 --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --index /mnt/project/minimal-repair-artifacts/20261015/TRAIN_SMOKE_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --workers 4 --max-seconds 300
```

主机 `training-vlawm-tsinghua-zt4`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/DEV1696.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_dev1696_v5 --workers 16 --max-seconds 1800
```

## TARGETS4096_V5_JOBS.json

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_train4096_v5_rank0.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --index /mnt/project/minimal-repair-artifacts/20261015/TRAIN_DIAGNOSTIC_4096.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_train4096_v5 --workers 16 --max-seconds 1800 --shards 2 --rank 0
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_train4096_v5_rank1.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --index /mnt/project/minimal-repair-artifacts/20261015/TRAIN_DIAGNOSTIC_4096.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_train4096_v5 --workers 16 --max-seconds 1800 --shards 2 --rank 1
```

## SMOKE_V5_JOBS.json

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_LOG_NONE.log`。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/LOG_NONE --kind LOG_NONE --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_LOG_CV.log`。

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/LOG_CV --kind LOG_CV --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_LOG_FUT.log`。

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/LOG_FUT --kind LOG_FUT --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_MIN_NONE.log`。

```bash
CUDA_VISIBLE_DEVICES=7 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/MIN_NONE --kind MIN_NONE --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_MIN_CV.log`。

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/MIN_CV --kind MIN_CV --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_MIN_FUT.log`。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/MIN_FUT --kind MIN_FUT --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_P_ABS.log`。

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_A_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_ABS --kind P_ABS --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_P_DIFF.log`。

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_A_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_DIFF --kind P_DIFF --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

## SMOKE_V5_PREDICTED_JOBS.json

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_R_MIN_CURRENT.log`。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/R_MIN_CURRENT --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_ABS/epoch_24.pt --kind R_MIN_CURRENT --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_R_MIN_ABS.log`。

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/R_MIN_ABS --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_ABS/epoch_24.pt --kind R_MIN_ABS --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/smoke_v5_R_MIN_DIFF.log`。

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/train_small_model.py --protocol minimal_repair_v1 --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_diagnostic4096_v1 --labels /mnt/project/minimal-repair-artifacts/20261015/targets_smoke_B_v5 --output /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/R_MIN_DIFF --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_DIFF/epoch_24.pt --kind R_MIN_DIFF --scope smoke --epochs 24 --batch 32 --microbatch 16 --data-reader mmap_tensors --max-seconds 300
```

## ONLINE_CACHE_JOBS_v10.json

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/online_repair_smoke4_v10.log`。

```bash
CUDA_VISIBLE_DEVICES=6 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/predict_current.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --refiner /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/R_MIN_DIFF/epoch_24.pt --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_DIFF/epoch_24.pt --output /mnt/project/minimal-repair-artifacts/20261015/online_repair_smoke4_v10 --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --split dev --max-seconds 900
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/cached_repair_smoke4_v10.log`。

```bash
CUDA_VISIBLE_DEVICES=4 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_minimal_repair_planning.py --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_dev_full_v1 --checkpoint /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/R_MIN_DIFF/epoch_24.pt --predictor /mnt/project/minimal-repair-artifacts/20261015/smoke_v5/P_DIFF/epoch_24.pt --output /mnt/project/minimal-repair-artifacts/20261015/cached_repair_smoke4_v10 --index /mnt/project/minimal-repair-artifacts/20261015/ONLINE_DEV4.json --distribution NATURAL --mode camera_only --max-seconds 300
```

## 中断恢复对照

真实B折小集，连续48updates与第7步中断后恢复；模型、optimizer、scheduler、RNG、数据顺序和曝光逐位一致。

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python /mnt/project/minimal-repair-artifacts/20261015/verify_resume_v10.py
```

该脚本已运行完成，原产物位于`resume_v10`。重做需独立输出，并保持原输入、源码和seed。正式头原生恢复命令待新run实际启动后按对应身份给出。

## 正式有限队列（已部署，等待完整B标签核验）

九个共同24epoch小头的逐run命令在 [STAGE_A_V5_TASKS.json](evidence/STAGE_A_V5_TASKS.json)；登记不等于训练完成。

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5_rank0.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --workers 24 --max-seconds 5400 --shards 4 --rank 0
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5_rank1.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --workers 24 --max-seconds 5400 --shards 4 --rank 1
```

主机 `training-vlawm-tsinghua-zt5`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5_rank2.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --workers 24 --max-seconds 5400 --shards 4 --rank 2
```

主机 `training-vlawm-tsinghua-zt4`；日志 `/mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5_rank3.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/build_minimal_repair_data.py --anchor /mnt/project/decision-local-consequence-artifacts/20261014/ANCHOR.json --features /mnt/project/decision-local-consequence-artifacts/20261014/anchor_train_full_v1 --index /mnt/project/minimal-repair-artifacts/20261015/FOLD_B.json --output /mnt/project/minimal-repair-artifacts/20261015/targets_foldB_v5 --workers 24 --max-seconds 5400 --shards 4 --rank 3
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_runner_training-rl-zt2.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/run_registered_heads.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json --gpus 2 4 --max-seconds 21600
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_runner_training-vla-zt.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/run_registered_heads.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json --gpus 2 4 6 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt4`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_runner_training-vlawm-tsinghua-zt4.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/run_registered_heads.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json --gpus 0 5 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt5`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_runner_training-vlawm-tsinghua-zt5.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/run_registered_heads.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_TASKS.json --gpus 2 4 --max-seconds 21600
```

主机 `training-vla-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-vla-zt_NATURAL.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json --gpus 0 1 3 5 7 --cpu-parallel 2 --max-seconds 21600
```

主机 `training-rl-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-rl-zt2_RECOVERY.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json --gpus 0 1 3 5 7 --cpu-parallel 2 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt4`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-vlawm-tsinghua-zt4_NATURAL.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json --gpus 1 2 3 4 6 7 --cpu-parallel 2 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-vlawm-tsinghua-zt_RECOVERY.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 2 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt2`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-vlawm-tsinghua-zt2_NATURAL.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_NATURAL.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 2 --max-seconds 21600
```

主机 `training-vlawm-tsinghua-zt3`；日志 `/mnt/project/minimal-repair-artifacts/20261015/stageA_v5_eval_training-vlawm-tsinghua-zt3_RECOVERY.log`。

```bash
/mnt/project/decision-local-consequence-artifacts/20261014/envs/qwen/bin/python -u /mnt/project/VLA-Drive-minimal-repair-current-vector-v10-20261015/tools/decision_local/evaluate_milestones.py --registry /mnt/project/minimal-repair-artifacts/20261015/STAGE_A_V5_EVAL_RECOVERY.json --gpus 0 1 2 3 4 5 6 7 --cpu-parallel 1 --max-seconds 21600
```

完整B分片收尾：`/mnt/project/minimal-repair-artifacts/20261015/close_foldB_v5.py` 已启动，最长等待6000秒，随后调用固定版本并发SHA核验。它不恢复旧任务、不缩减人口，也不跳过失败。
