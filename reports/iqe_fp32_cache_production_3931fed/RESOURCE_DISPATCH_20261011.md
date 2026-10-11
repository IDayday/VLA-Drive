# Shared space and rl-zt4 dispatch, 2026-10-11 UTC

The latest user instruction explicitly authorizes `rl-zt4` for IQE. Its
task-scoped entry was added to `/mnt/project/server_dispatch_policy.json`;
other projects' authorization and running jobs were preserved. The machine is
`training-rl-zt4-worker-0`, with eight distinct A800 80 GB GPUs and the same
project Lustre mount as the Base server.

## Actual training and cache evidence

The 6,400-scene representative cache is complete: 267 immutable shards,
362,797,094,400 bytes. Full shard checksums passed before rl-zt4 training.

The bounded training job (launcher PID 2060916) completed all 200 updates from
the complete original step 33,000 state, finishing at step 33,200. It retained
global batch 32, repeat8, all four objectives, FP32 visual compute/storage,
FP32 optimizer state, and the original 100k learning-rate schedule.

| Same-runtime comparison | Mean s/update | p50 | p95 | Steady updates |
|---|---:|---:|---:|---:|
| Online FP32 | 2.450300 | 2.437893 | 2.506375 | 189 |
| Cached FP32, rl-zt4 | 1.202683 | 1.186394 | 1.360314 | 189 |

The cached result uses the slowest rank for each update. Throughput was
26.607 samples/s; peak allocated memory was 62,988,556,288 bytes and reserved
memory 66,630,713,344 bytes. Checkpoint writing took 45.9696 seconds, recorded
separately. This is bounded throughput evidence, not a long-run PDMS result.

Evidence:

- `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/CACHED200_LAUNCH.json`
- `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/REPRESENTATIVE_VERIFY.json`
- `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/cached200/RESULT.json`
- `/mnt/project/iqe-fp32-cache-production-runtime-3931fed/online200/RESULT.json`

The active original Base was not stopped or migrated. At 23:08 UTC it was at
40,231 steps, with finite recent losses and gradients and approximately
3.978 seconds of compute per update. The locked incremental Base remains
`step_022000.pt`, SHA256
`42320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93`.
The authoritative SHA is in the existing `LOCKED_S0.json` and launch receipts.

## Whole-cache space remains a separate constraint

The actual parent's fit manifest contains 90,892 observations. The measured
record size is 56,687,046 bytes: 5,152,398,985,032 bytes for the complete cache.
Reusing the representative shards avoids another 362,797,094,400-byte copy.
Remaining payload is therefore 4,789,601,890,632 bytes.

The measured 23:09 budget also preserves 60 future original Base checkpoints
(1,368,979,984,260 bytes), 22k preparation space, atomic-write temporary space,
and safety/growth reserve. Required free space was 6,841,403,586,447 bytes;
available space was 6,100,927,016,960 bytes, a 740,476,569,487-byte shortfall.
These are dated observations; subsequent checkpoint writes change free space.
The full cache has not been started or labelled complete. No checkpoint was
deleted. The detailed budget is `FULL_CAPACITY_REPORT.json` in the runtime root.

## Use freed training resources for the 22k experiment

`prepare_incremental_features.py` overlaps the existing official per-log
context preparation with the unchanged, frozen 22k `Pipeline.cache_scene_list`.
It validates complete per-log receipts and actual context checksums, preserves
scene roles and the final feature key, and never publishes `OFFICIAL_SCENES`.
It does not encode an uncommitted or failed context as a finished scientific
bank. The original locked source checkout remains at
`3855e06d7efac26ae4afb49116d0052e5dac253b`.

Ten behavioral tests passed (receipt interruption/reuse, corruption, removed
receipt, source/log/role substitution, final feature-key alignment, and late
GPU ownership changes). The real two-scene pilot strictly loaded the locked
22k Base, read actual images, wrote and read back valid frozen features of
shape `[1,16,256]` plus ego `[1,1,256]`. Each payload is 19,156 bytes. Its
`BUDGET_PAUSED` result means the explicit two-scene limit was reached, not that
the complete feature bank was produced.

At 23:22 UTC, rl-zt4 was assigned feature shards 8–15, PIDs 2063049–2063056.
The other initial shards on vlawm-zt3 exited safely when an unrelated
authorized navtest job arrived during CPU model loading. No external process
was stopped. A bounded rl-zt4 queue (PID 2074091) schedules missing shards
0–7 only after their corresponding current worker exits and the physical GPU
is free. As of 23:27 UTC all eight rl-zt4 feature workers had produced valid
records. This subsequent workload prepares expert-training inputs; it is
not another Base optimization run.

Progress and exact launch commands are under:
`/mnt/project/DriveVLA-M0-iqe-closeout-924b003/outputs/iqe/formal_round1_22k/launch_20261011/features_ready16/`.
The existing official context preparation continues independently; its
global manifest is committed only when all 101,592 requested train-domain
contexts are complete. Formal expert and Scorer training still depend on
the complete feature/candidate/score banks and the existing gap audit.

## Resume a feature shard

After verifying that its previous process has exited and its selected GPU is
free, this command resumes shard 8 without changing model or scene identities:

```bash
cd /mnt/project/DriveVLA-M0-iqe-closeout-924b003
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -u \
  /mnt/project/DriveVLA-M0-iqe-fp32-cache-3931fed/scripts/iqe/fp32_cache/prepare_incremental_features.py \
  --source-repo /mnt/project/DriveVLA-M0-iqe-closeout-924b003 \
  --source-commit 3855e06d7efac26ae4afb49116d0052e5dac253b \
  --config /mnt/project/DriveVLA-M0-iqe-closeout-924b003/configs/iqe/formal/round1_locked.yaml \
  --metric-version /mnt/project/DriveVLA-M0-iqe-closeout-924b003/outputs/iqe/formal_round1_22k/official_train_metric/8b67bbfc9e6924a533928d684f0c1628dd1282c176c06cb6371c7c2ee797ccc1 \
  --progress-root /mnt/project/DriveVLA-M0-iqe-closeout-924b003/outputs/iqe/formal_round1_22k/launch_20261011/features_ready16 \
  --shard-index 8 --num-shards 16 --max-samples 101592 \
  --max-new-scenes 6350 --reserve-gib 2000 --max-seconds 10800 --device cuda
```

Exit 0 means the official global commit and this feature shard agree. Exit 3
means bounded progress or pending official inputs; exit 2 denotes an error.
Record-level complete receipts, rather than process existence, control reuse.
