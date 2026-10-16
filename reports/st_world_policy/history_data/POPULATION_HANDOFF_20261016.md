# Full current teacher population handoff

Status: **NEEDS_REVIEW**. Full train metadata is running; full train teacher cache is not started by this agent. This document is not formal training approval. Read the final receipts before launching a cache or claiming complete coverage.

Latest Lead decision: no plan-waiting helper remains running. Lead will generate the plan after metadata COMPLETE and after implementing/testing deferred worker finalization plus one CPU finalizer. The 8e4e runtime command below documents the available pre-fix plan interface; it is **not the approved source for the upcoming full train cache**. Use the actual newly frozen runtime when Lead registers it. Existing dev artifacts from 8e4e remain immutable.

## Live CPU preparation

- Host: `training-vlawm-tsinghua-zt-worker-0` (SSH alias `training-vlawm-tsinghua-zt`).
- Remote PID: `1769420`; process start: `2026-10-16 22:09:06 UTC`.
- Foreground SSH execution session: `16189`; local SSH PID `266759`.
- Runtime/worktree: `/mnt/project/VLA-Drive-st-history-population-agent-20261016`.
- Registered writer SHA256: `347c9bee79c25e73902d069d4bdce3c88af468e4c7433a104098ab95af520ca8`; the running source is not modified.
- Log: `/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/TRAIN_BUILD.log`.
- Progress: `train/PROGRESS.json` under that directory. At the last recorded sample: 235 / 1176 logs, 676.55 seconds, 25,212,453,935 source bytes hashed. This is a partial progress snapshot, not completed population evidence.
- Budget: 2 CPU workers, BLAS 1, no visible CUDA device, global read limit 64 MiB/s, maximum 7200 seconds. No image decode, GT value decode, historical/future record reads, optimizer or teacher forward.
- Registration identity: `a04964f677155d3759fa8f3f737996a1c3f5ad294284e2cae41c56e3aa911dd7`.
- Population comes from the original train index: 101592 scenes / 1176 logs. Scene order and original train/dev roles are retained. Counts are not a substituted small subset.

Do not start a duplicate preparation while this PID is active. If it exits incomplete, the same command resumes immutable committed log shards and validates their pins:

```bash
cd /mnt/project/VLA-Drive-st-history-population-agent-20261016
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.prepare_current_teacher build \
  --student-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 \
  --partition /mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --split train \
  --output /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train
```

Final `COMPLETE.json` will bind the actual `TEACHER_REQUESTS.json` / `SELECTED_SCENES.json` hashes and unique image contents. Those hashes cannot be reported as complete before the process commits them. Per-log evidence remains under `train/log_shards/` on interruption.

## Completed dev metadata

Root: `/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev`.

- 1696 scenes / 16 logs / 5088 unique image contents; 25.617320634 seconds CPU preparation.
- Request identity: `81a46ee3ddd8c8f2602f53ca393e0c7d2ed91c1e4a43e801f98c7f374124a3da`.
- `TEACHER_REQUESTS.json` SHA256: `199f0018ee0120204a0f0bbef2930586a158f2445065e41812eb5868690ef25d`.
- `SELECTED_SCENES.json` SHA256: `4ce40893281f3cf246989cbbac6bbd80c40c06ec937a4405a5a7c353c6dd0573`.

Lead has already built the full dev native cache on 8 GPUs and is performing record verification, SP1 derivation and SP0 exact reuse. Do not launch another dev cache from this handoff. The teacher build and verification evidence belongs to Lead's actual cache namespace; metadata COMPLETE alone is not teacher COMPLETE.

## Train cache plan only

The plan generator uses immutable runtime `/mnt/project/VLA-Drive-st-cache-deployment-runtime-20261016`, commit `8e4e341a075b87a691fe664c26427008e39eb66f`. It waits for train metadata COMPLETE, checks final scene order and hashes, checks actual legacy C1 image SHA coverage, and writes exact per-variant budget/config/argv records. It never launches GPU or CPU cache workers.

```bash
cd /mnt/project/VLA-Drive-st-history-population-agent-20261016
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \
  /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/history_data/population_stage_plan.py \
  --metadata-root /mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1 \
  --output /mnt/project/st-world-policy-20261016/audits/history_data/full_current_teacher_train_stage_plan_v2 \
  --cache-runtime /mnt/project/VLA-Drive-st-cache-deployment-runtime-20261016 \
  --teacher-root /mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm \
  --legacy-c1-root /mnt/project/ddp-full-foresight-study-artifacts/20260929/dino_targets_v1/C1 \
  --pilot-complete /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384/COMPLETE.json \
  --pilot-log /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/CACHE_BUILD.log \
  --python /root/miniconda3/envs/ddp/bin/python --world-size 8 --roles train --wait-seconds 7200
```

Only after the helper completes, read `FULL_CACHE_STAGE_PLAN.json` and its `commands[].argv_prefix`, `append_required`, `preconditions` and exact budget files. GPU/rank assignment is Lead-owned. The helper leaves resource approval false and does not substitute a guessed identity if metadata is incomplete. SP1 requires the actual completed native cache identity. SP0 fails if original C1 image hashes are missing; do not silently run new DINO to fill the legacy control.

The old plan-only helper pointing to the superseded smoke runtime was stopped; its record is `STAGE_PLAN_HELPER_REPLACED.json`. The actual metadata process was not signalled. Its old log is retained.

## Measured pilot and preliminary capacity

Native current-only 256-scene pilot: 768 images, 1,359,294,528 file bytes / 24 shards; actual single-GPU build 42.637354804 seconds. All 768 record digests verified. Original PCA statistics were copied byte-identically into a new explicit scope namespace, never refitted; selected dimension remains 384.

The actual 256-unique-scene CPU one-pass measurements (2 loader threads, torch 1 thread, BLAS 1) are 43.252 scenes/s for verified teacher + PCA and 12.331 scenes/s for full real-image Dataset + PCA. They are not GPU training throughput and not physical cold-read evidence. Systems owns the disjoint 8-process shared-I/O measurement, using the same Dataset path and `loader_cost.py --torch-threads 2`.

Before final image-content deduplication, train has at most 304776 t0 view records. Native FP32 logical tensors are at most 539,424,261,504 bytes (502.378 GiB); SP1 pool3 is 55.820 GiB; exact SP0 C1 is 27.917 GiB. File/index overhead and checkpoint reserve are additional. Linear single-GPU pilot scaling gives about 4.70 GPU-hours for train; ideal 8-way wall is only an estimate, not a measurement. Full metadata completion replaces these bounds with actual unique counts. Current-only caching does not cover future target requests.

## Narrow finalization coordination issue

At immutable 8e4e341a, `TeacherCacheWriter.finalize()` obtains `FINALIZE.lock` and calls `committed()` for every shard, including a full tensor-file SHA. Every rank invokes finalize at the end. Once complete, several ranks can therefore serialize full scans of the same approximately 502 GiB train cache. Lead observed an approximately 18-second tail even on the smaller dev population.

Suggested narrow change, **not implemented or hot-applied here**:

1. Explicit worker `--defer-finalize` mode validates and commits only its assigned shards and emits a rank-completion receipt.
2. A separate CPU `cache_teacher finalize` operation runs once after all workers exit. It retains the existing full-shard SHA, coverage and immutable COMPLETE construction.
3. Missing/corrupt shards refuse COMPLETE. Resume preserves committed shards. Readers continue requiring existing COMPLETE and exact identities without modification.
4. Do not nominate rank 0 to finalize immediately without a barrier/coordinator: it may finish before another worker. Do not optimize by deleting checksum checks or trusting a COMPLETE path alone.

This changes orchestration only. Teacher outputs, FP32 computation, storage values, reader checks, PCA family and old artifacts remain unchanged. If adopted, register a new immutable runtime and regenerate the plan identity before the full train build. This report does not approve a new source hash implicitly.

## Tests and remaining handoff

Current data-side regression: 23 targeted teacher-cache/PCA tests passed in 3.87 seconds, plus 11 PCA namespace tests passed in 3.88 seconds. Logs: `train256_current_cost_v1/POPULATION_PREPARATION_TEST.log` and `PCA_NAMESPACE_TEST.log`. No additional GPU work was started by this agent.

The plan helper --help is verified. Its reporting-only role selection prevents duplicate dev commands after Lead has started dev work. Full population completion, teacher build, post-build verification, whole-population PCA scope registration and formal dataset approval remain Lead-controlled. The current completed PCA scope covers the original fitted population plus the real 256 pilot; it must not be relabelled as covering a newly built full cache.
