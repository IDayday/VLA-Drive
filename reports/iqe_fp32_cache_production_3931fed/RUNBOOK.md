# FP32 cache build, qualification and continuation

These commands use the actual checkout, complete checkpoints and manifests from
this task. They do not authorize stopping the original Base. No full build,
formal continuation, new navtest or ACTIVE update was executed by this task.
Read the current server dispatch policy and check physical GPU owners at each
job boundary. Existing scientific jobs must keep their resources.

## Common environment

```bash
cd /mnt/project/DriveVLA-M0-iqe-fp32-cache-3931fed
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
IQE_PY=/root/miniconda3/envs/ddp/bin/python
IQE_RT=/mnt/project/iqe-fp32-cache-production-runtime-3931fed
IQE_HOLD=/mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json
```

CLI help was executed; cache subcommand parsing is also covered by tests:

```bash
"$IQE_PY" -m iqe.vision_cache.cli --help
"$IQE_PY" -m iqe.cli freeze-base --help
"$IQE_PY" scripts/iqe/fp32_cache/build_full.py --help
"$IQE_PY" scripts/iqe/fp32_cache/continue_after_pause.py --help
```

## Inspect original Base, without changing it

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 training-vla-zt2 \
  'ps -eo pid,lstart,etime,args | grep "iqe.cli train-base" | grep -v grep'
cat /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/status.json
```

Never signal PID2124247 based solely on this historical document. Confirm live
host/PID/start/cwd/command again. If the user subsequently authorizes replacement,
request the original trainer's cooperative optimizer-boundary save, verify the
complete checkpoint receipt and `STOPPED_AT_BOUNDARY` result, and confirm all
old workers have exited. No automated stop command is provided or executed here.

## Full training-population plan and capacity gate

The40k checkpoint below was completely saved and verified. Re-running planning
is idempotent only for the same content. It does not change22k or the sampler.

```bash
"$IQE_PY" -m iqe.vision_cache.cli cache-plan \
  --config configs/iqe/fp32_cache/source.yaml \
  --parent-checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_040000.pt \
  --output "$IQE_RT/FULL_PLAN_40K.json" --scope full \
  --max-samples 90892 --records-per-shard 24

"$IQE_PY" scripts/iqe/fp32_cache/build_full.py --workers 8 \
  --rate-mib-s-per-worker 128
```

The second command is capacity-only by default and currently exits2 with
`BLOCKED_SHARED_CAPACITY`. It writes `full_fit_40k/LAUNCH_CAPACITY.json`;
it never launches a GPU job when the reserve does not fit. At the documented
snapshot the shortage is840.6GB, including the remaining original checkpoint
schedule,180GiB reserved for22k work,200GiB safety and atomic temporary writes.
No cache/old checkpoint cleanup is automatic.

After capacity is sufficient and an authorized host has8 idle GPUs, the already
authorized cache-only build is:

```bash
"$IQE_PY" scripts/iqe/fp32_cache/build_full.py --workers 8 \
  --rate-mib-s-per-worker 128 --execute
```

The same command resumes completed shards; no new rank-dependent cache identity
is generated. It reuses the verified6400-observation representative cache by
hardlink. A bounded subset of idle physical GPUs can be selected with
`CUDA_VISIBLE_DEVICES` and a matching smaller `--workers` value. The frozen
front still processes batches of4, with a legal short final batch.

To pause/reduce an owned cache build at a shard boundary, write a small atomic
`CONTROL.json`; this controls only the isolated cache output, not training:

```bash
"$IQE_PY" - <<'PY'
from iqe.io import atomic_json
atomic_json('/mnt/project/iqe-fp32-cache-production-runtime-3931fed/full_fit_40k/CONTROL.json',
            {'pause': True, 'rate_mib_s_per_worker': 64.0})
PY
```

Before resuming, set `pause` to false with the same atomic helper. Do not remove
completion receipts or edit immutable shard identities.

## Verify and measure a completed cache

Actual representative cache (already verified and consumed in200 updates):

```bash
"$IQE_PY" -m iqe.vision_cache.cli cache-verify \
  --root "$IQE_RT/representative200" --max-samples 64 --full-shards \
  --output "$IQE_RT/REPRESENTATIVE_VERIFY_RECHECK.json"
```

After the full build has a valid global COMPLETE, explicitly budget the full
shard integrity read (about5.15TB) before scheduling it:

```bash
"$IQE_PY" -m iqe.vision_cache.cli cache-verify \
  --root "$IQE_RT/full_fit_40k" --max-samples 128 --full-shards \
  --output "$IQE_RT/FULL_VERIFY.json"

"$IQE_PY" -m iqe.vision_cache.cli cache-benchmark \
  --root "$IQE_RT/full_fit_40k" --max-samples 128 --repetitions 1 \
  --output "$IQE_RT/FULL_READER_BENCHMARK.json"
```

`cache-benchmark` measures reader IO only. The1.202683s/update result comes from
the actual matched8-rank200-step training runs, not this reader command.

## Inspect the already completed checkpoint lifecycle

Actual cached continuation checkpoint is33200, not an arbitrary placeholder:

```bash
"$IQE_PY" -m iqe.vision_cache.cli checkpoint-roundtrip \
  --checkpoint "$IQE_RT/cached200/step_033200.pt" \
  --output "$IQE_RT/CHECKPOINT_SCHEMA_RECHECK.json"

"$IQE_PY" -m iqe.cli evaluate-base \
  --config configs/iqe/fp32_cache/lifecycle.yaml --mode smoke \
  --checkpoint "$IQE_RT/cached200/step_033200.pt" \
  --role stage_val --max-samples 32 --device cuda

"$IQE_PY" -m iqe.cli qualify-base \
  --config configs/iqe/fp32_cache/lifecycle.yaml --mode smoke \
  --checkpoint "$IQE_RT/cached200/step_033200.pt" \
  --evaluation "$IQE_RT/checkpoint_lifecycle/base_evaluation/4dab9b35c0b208a33f59393b6535f3e75c91686eeec490c8d816cca345796ce7/EVALUATION.json" \
  --role stage_val --max-samples 32 --device cpu

"$IQE_PY" -m iqe.cli freeze-base \
  --config configs/iqe/fp32_cache/lifecycle.yaml --mode smoke \
  --checkpoint "$IQE_RT/cached200/step_033200.pt" \
  --diagnostic-checkpoint --max-samples 32 --device cpu
```

The qualification command intentionally returns `BASE_QUALITY_UNTESTED` and a
nonzero status;32-scene engineering evidence is not formal baseline quality.
The diagnostic freeze succeeds in its separate namespace. For the full actual
online/cache/expert-clone/bundle chain, the command used was:

```bash
CUDA_VISIBLE_DEVICES=0 "$IQE_PY" scripts/iqe/fp32_cache/checkpoint_lifecycle.py \
  --config configs/iqe/fp32_cache/lifecycle.yaml \
  --checkpoint "$IQE_RT/cached200/step_033200.pt" --device cuda --max-samples 32
```

It already passed. These are reproduction commands, not requests to repeat
GPU qualification or steal the now-reassigned GPU0.

## Formal continuation, only after a separately authorized safe pause

Prerequisites: verified full cache, qualified checkpoint/visual/data identity,
adequate space and8 idleA800s on the certified runtime host
`training-rl-zt4-worker-0`. The certificate binds the actual host/backend/source;
moving it elsewhere requires a recorded qualification, not hash deletion.
Keep the original100k LR horizon, batch32 and all objectives.

On that host after the original job is confirmed paused:

```bash
"$IQE_PY" scripts/iqe/fp32_cache/continue_after_pause.py --updates 1000
```

This script rechecks the original host for an active Base, verifies the most
recent complete paused checkpoint and starts one stable continuation lineage.
It refuses partial newer checkpoints, duplicate output identities or a missing
full cache/certificate. It never stops a process, reverts to22k, starts from
generic weights or reinitializes optimizer/scheduler. The first window adds
1000 updates on the original100k timeline; it is not a new1000-step cosine run.
Output: `$IQE_RT/continuation`.

Normal interruption/next bounded window uses the newest completed checkpoint
inside that continuation run, with the same output and original schedule:

```bash
"$IQE_PY" scripts/iqe/fp32_cache/continue_after_pause.py --updates 1000 --resume
```

Choose an additional-update count no larger than the remaining100k budget.
No migration has been executed in this task, and currently the original job is
still active and the full cache is incomplete, so these commands correctly refuse.

## Actual exact-resume evidence and fallback

The executed8-rank resume selected
`$IQE_RT/cached_resume/step_033001.pt` (SHA
`dcfe25582f5655f4c1b6ee699298f83fcf956a025db9745004c3fbd3fb4c52e4`),
then saved the resumed diagnostic lineage at **33002**, SHA
`f60402b92e94f16902537cff544968dc2ca47c02b32573e7c03d748041e8718f`.
The exact executed arguments are in `RESUME_CONTINUE_LAUNCH.json`.
That64-record pilot is exhausted after2 updates. Do not rerun it blindly to
request a third missing batch; the full-cache continuation commands above are
the operational resume route.

If the future full cached run is safely paused and an online fallback is needed,
resolve its actual latest complete checkpoint rather than guessing a step:

```bash
IQE_CACHED_CHECKPOINT=$("$IQE_PY" - <<'PY'
from iqe.training.resume import resolve_resume
print(resolve_resume('/mnt/project/iqe-fp32-cache-production-runtime-3931fed/continuation', 'auto'))
PY
)
"$IQE_PY" scripts/iqe/fp32_cache/prepare_online_fallback.py \
  --parent-checkpoint "$IQE_CACHED_CHECKPOINT" \
  --production-config configs/iqe/fp32_cache/online_fallback.yaml \
  --qualification "$IQE_RT/CACHE_QUALIFICATION.json" \
  --output "$IQE_RT/online_fallback_state"
```

This performs an explicit metadata-only fork, under the paused source run lock;
it does not execute training or modify the parent. Full-state tensor fingerprints
are compared before and after serialization. It refuses an older source, changed
precision/backend/host/source, diagnostic-to-formal relabel or existing destination.
Six CPU tests pass; a new real cached->online GPU switch has not been performed.
Treat its first bounded continuation as a migration check, not already measured
long-run evidence. The original matched online200 path itself was measured.

On the same certified host, with no cached parent process left running, the
prepared full checkpoint can be loaded without resetting training state:

```bash
IQE_ONLINE_CHECKPOINT=$("$IQE_PY" - <<'PY'
from iqe.training.resume import resolve_resume
print(resolve_resume('/mnt/project/iqe-fp32-cache-production-runtime-3931fed/online_fallback_state', 'auto'))
PY
)
"$IQE_PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.vision_cache.cli train-base-cached \
  --config configs/iqe/fp32_cache/source.yaml \
  --production-config configs/iqe/fp32_cache/online_fallback.yaml \
  --runtime configs/iqe/performance_v2/combined_fp32.yaml \
  --parent-checkpoint "$IQE_ONLINE_CHECKPOINT" \
  --qualification "$IQE_RT/online_fallback_state/ONLINE_QUALIFICATION.json" \
  --output "$IQE_RT/online_fallback" --mode continuation --stop-after 32 \
  --warmup 10 --checkpoint-every 1000 --hold-file "$IQE_HOLD"
```

The shared command name is retained; `training_visual_source: online_fp32`
selects actual online vision, with no cache dependency. Subsequent recovery uses
`resume-base-cached` with the same arguments/output and omits `--parent-checkpoint`.
Do not restart the old cached lineage simultaneously.

## Tests and commit policy

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 "$IQE_PY" -m pytest -q tests/iqe
```

Delivery result:211 passed,2 skipped,4 warnings in39.66s. All cache/200-step/resume
proofs are already saved. No extra GPU retesting is needed for this delivery.
Local commits only; no push, PR, merge, checkpoint deletion or ACTIVE update.
