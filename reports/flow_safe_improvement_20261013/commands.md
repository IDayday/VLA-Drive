# 实际命令与恢复入口

以下是已经执行的命令记录，不是新增实验授权。正式训练已完成，勿再次启动已完成run。当前仅锁定的六组Navtest继续运行。命令参数及原始receipt见[完整记录](evidence/actual_stage_commands.json)。

所有模型命令工作目录为`/mnt/project/VLA-Drive-flow-safe-improvement-20261013`；解释器为本轮`/mnt/project/flow-safe-improvement-20261013/venv/bin/python`。每次派发重新读取`/mnt/project/server_dispatch_policy.json`，只允许四台Tsinghua；本地盘恢复点必须在所属主机访问，不能把同名`/var/tmp`误认为共享文件。

实际CUDA环境：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 FLASH_ATTENTION_DETERMINISTIC=1
cd /mnt/project/VLA-Drive-flow-safe-improvement-20261013
```

## P0固定库实际单seed接续

主机：`training-vlawm-tsinghua-zt`。原receipt：`/mnt/project/flow-safe-improvement-20261013/training/P0_REWARD_FM_seed42/single_seed_launch.json`。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.flow_safe_improvement.run_learning_window_single_seed \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --cohorts /mnt/project/flow-safe-improvement-20261013/cohorts/manifest.json \
  --targets /mnt/project/flow-safe-improvement-20261013/targets/P0_REWARD_FM \
  --reference-scores /mnt/project/flow-safe-improvement-20261013/reference_scores/SFT_4seed_v1 \
  --eta-registration /mnt/project/flow-safe-improvement-20261013/calibration/common_eta/registration.json \
  --retention-registration /mnt/project/flow-safe-improvement-20261013/calibration/common_retention/registration.json \
  --baseline-dev-scores /mnt/project/flow-safe-improvement-20261013/evaluations/original_export/SFT_DEV5_v1 \
  --reference-dev-scores /mnt/project/flow-safe-improvement-20261013/evaluations/SFT_DEV_reference4_v1 \
  --dev-cache /mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache \
  --output /mnt/project/flow-safe-improvement-20261013/training/P0_REWARD_FM_seed42 \
  --checkpoint-root /var/tmp/flow-safe-improvement-20261013/P0_REWARD_FM_seed42 \
  --epochs 8 \
  --evaluation-epochs 1 2 4 8 \
  --training-seed 42 \
  --evaluation-override /mnt/project/flow-safe-improvement-20261013/resources/single_seed_override/REGISTRATION.json \
  --adopt-existing
```

## 安全ODE三轮实际接续

主机：`training-vlawm-tsinghua-zt`。原receipt：`/mnt/project/flow-safe-improvement-20261013/online/ODE_SAFE_FM_seed42/single_seed_launch.json`。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.flow_safe_improvement.run_online_iteration_single_seed \
  --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json \
  --cohorts /mnt/project/flow-safe-improvement-20261013/cohorts/manifest.json \
  --reference-scores /mnt/project/flow-safe-improvement-20261013/reference_scores/SFT_4seed_v1 \
  --eta-registration /mnt/project/flow-safe-improvement-20261013/calibration/common_eta/registration.json \
  --retention-registration /mnt/project/flow-safe-improvement-20261013/calibration/common_retention/registration.json \
  --baseline-dev-scores /mnt/project/flow-safe-improvement-20261013/evaluations/original_export/SFT_DEV5_v1 \
  --reference-dev-scores /mnt/project/flow-safe-improvement-20261013/evaluations/SFT_DEV_reference4_v1 \
  --dev-cache /mnt/project/DriveDreamer-Policy-paired/runs/paired_dev_pdms_v1/metric_cache \
  --train-cache /mnt/project/flow-safe-improvement-20261013/metric_cache_v1 \
  --registration /mnt/project/flow-safe-improvement-20261013/online_registration.json \
  --batch-qualification /mnt/project/flow-safe-improvement-20261013/validation/ODE_batch_qualification_attempt2/report.json \
  --physical-fallback-registration /mnt/project/flow-safe-improvement-20261013/calibration/behavior_bank_policy/registration.json \
  --variant ODE_SAFE_FM \
  --training-seed 42 \
  --output /mnt/project/flow-safe-improvement-20261013/online/ODE_SAFE_FM_seed42 \
  --checkpoint-root /var/tmp/flow-safe-improvement-20261013/online_ODE_SAFE_FM_seed42 \
  --evaluation-override /mnt/project/flow-safe-improvement-20261013/resources/single_seed_override/REGISTRATION.json \
  --adopt-existing
```

## 锁定Navtest实际SFT运行

主机：`training-vlawm-tsinghua-zt2`。原receipt：`/mnt/project/flow-safe-improvement-20261013/evaluations/final_navtest/SFT.launch.json`。

```bash
/mnt/project/flow-safe-improvement-20261013/venv/bin/python -m scripts.analysis.flow_safe_improvement.run_locked_navtest \
  --run-root /mnt/project/flow-safe-improvement-20261013 \
  --final-lock /mnt/project/flow-safe-improvement-20261013/evaluations/final_navtest/FINAL_SELECTION_LOCK.json \
  --model SFT \
  --gpu-count 8 \
  --cpu-workers 8
```

## 恢复语义

P/O控制器恢复使用相同命令参数并增加`--resume`，只从`latest_checkpoint.json`指向的完整封存点恢复；保留当轮behavior、bank、targets、optimizer、scheduler、RNG和数据游标。`--adopt-existing`是取消多seed时的明确接管模式，不可拿它绕过活动进程身份。启动前核对既有父/子进程PID与start ticks、所有者和源码快照；不允许两个控制器同时操作同一个run。

`run_endpoint_diagnostics --run-root ... --variant ... --resume`恢复同身份的TRAIN终点/v2诊断；已完成不会新增采样。Navtest使用上述`run_locked_navtest`的同模型、同host和相同GPU/CPU参数加`--resume`，它拒绝仍活跃的旧子进程，复用已封存scene和完整v1/v2分片，不新增噪声。当前不需要恢复；六个现有操作进程继续执行。

`train_fixed_bank --resume <完整checkpoint>`是底层实际恢复入口，正式父进程的`single_seed_phase_records.json`保存其完整torchrun命令。每个训练目录另有`source_snapshot`、`run_identity.json`、`storage_identity.json`、`latest_checkpoint.json`和分rank日志，具体存在文件以目录为准；滚动点与`milestones`分离，不能以未完成导出恢复优化器。

候选生成恢复继续同一个`build_candidate_banks`身份和逐场景hash，不重采behavior后沿用旧targets。各轮评分队列的归档receipt与逐字节核对已发布，原始请求、模拟状态及失败记录仍在私有运行目录。

## 只读汇总入口

以下入口都已实际执行。复跑分析时指定新的输出目录；不要覆盖封存结果。

```bash
SFI_RUN=/mnt/project/flow-safe-improvement-20261013
SFI_PY=$SFI_RUN/venv/bin/python
$SFI_PY -m scripts.analysis.flow_safe_improvement.summarize_direction \
  --run-root "$SFI_RUN" --output "$SFI_RUN/diagnostics/status_NEW"
$SFI_PY -m scripts.analysis.flow_safe_improvement.plot_research_results \
  --summary "$SFI_RUN/diagnostics/status_NEW/report.json" \
  --output "$SFI_RUN/diagnostics/curves_NEW"
$SFI_PY -m scripts.analysis.flow_safe_improvement.summarize_execution_costs \
  --run-root "$SFI_RUN" --output "$SFI_RUN/diagnostics/cost_NEW"
```

D1/D2/D3、目标求解、场景方向、配对转移和官方NFT对齐入口与早期实际命令见[执行历史](execution_history.md)、[验证映射](validation.md)；所有采样/评分操作必须在当前授权主机执行。最终选择锁已经创建，禁止再次按Navtest结果修改或重选。
