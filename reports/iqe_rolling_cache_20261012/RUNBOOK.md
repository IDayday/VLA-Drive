# Rolling FP32 Base runbook

Consumer worktree: `/mnt/project/DriveVLA-M0-iqe-rolling-cache-20261012`.
Branch: `perf/iqe-remote-rolling-cache-20261012`.
Producer worktree: `/mnt/project/DriveVLA-M0-iqe-rolling-producer-20261012`, branch
`perf/iqe-rolling-producer-index-8fe9f61`, commit
`319543e` (descends from accounting fix27f3b5e). It adds an explicitly versioned
producer storage budget and preserves the frozen frontend, codec, sampler and shard format. The active
consumer's protected source hashes stay at8fe9f61. Do not switch its worktree.
Runtime root: `/mnt/project/iqe-remote-rolling-cache-20261012`.
All commands below assume that worktree as cwd. No push was requested for this
rolling-cache change. Do not execute the initial train command again while the
continuation is running. The lineage/run locks also reject duplicate starts.

## Running identities

Original Base paused safely at46534. The parent for qualified continuation is
`/mnt/project/iqe-remote-rolling-cache-20261012/rolling200/step_046734.pt`, SHA256
`8e1e12ebf463c0e03c65bc9ec3ac6cd48cab97843cf4c4effb101684fe531dc2`.
This contains200 real updates proven exactly equal to the same combined-FP32
online path, with full optimizer/RNG/cursor and the unchanged100k scheduler.
The source46534 checkpoint and locked22k checkpoint are preserved.

Consumer: training-vla-zt2; launch identity is `CONTINUATION_LAUNCH.json` in the
runtime root. Producer hosts: training-vlawm-tsinghua-zt and
training-vlawm-tsinghua-zt2. Actual per-process IDs/start ticks/commands are in
`PRODUCERS_training-vlawm-tsinghua-zt_EXPAND_LAUNCH.json` and its zt2 counterpart.
Their `previous_receipt` fields preserve the prior INDEX/RATE96 processes and history.
Read current live state; historical RUNNING snapshots are not evidence.

## Inspect the bounded cache

```bash
PYTHONUTF8=1 /root/miniconda3/envs/ddp/bin/python -m iqe.rolling_cache.cli status \
  --root /mnt/project/iqe-remote-rolling-cache-20261012/pool46534
```

Read the live observer output without scanning the payloads:

```bash
python - <<'PY'
import json
p = '/mnt/project/iqe-remote-rolling-cache-20261012/monitor/STATUS.json'
d = json.load(open(p))
for k in ('timestamp_unix', 'alerts', 'consumer', 'buffer',
          'supply_measured_between_snapshots', 'demand', 'storage'):
    print(k, d[k])
PY
```

The existing observer samples every30s. Its current pointer is
`/mnt/project/iqe-remote-rolling-cache-20261012/MONITOR_CURRENT.json`, which binds
`MONITOR_EXPANDED_LAUNCH.json` by checksum; previous monitor receipts are preserved
history. After verifying that no observer owns the
same output lock, the actual read-only recipe is:

```bash
PYTHONUTF8=1 /root/miniconda3/envs/ddp/bin/python -u \
  scripts/iqe/rolling_cache/monitor.py \
  --root /mnt/project/iqe-remote-rolling-cache-20261012/pool46534 \
  --output /mnt/project/iqe-remote-rolling-cache-20261012/monitor \
  --consumer-log /mnt/project/iqe-remote-rolling-cache-20261012/continuation.log \
  --consumer-launch /mnt/project/iqe-remote-rolling-cache-20261012/continuation/LAUNCH.json \
  --consumer-receipt /mnt/project/iqe-remote-rolling-cache-20261012/CONTINUATION_LAUNCH.json \
  --producer-receipt /mnt/project/iqe-remote-rolling-cache-20261012/PRODUCERS_training-vlawm-tsinghua-zt_EXPAND_LAUNCH.json \
  --producer-receipt /mnt/project/iqe-remote-rolling-cache-20261012/PRODUCERS_training-vlawm-tsinghua-zt2_EXPAND_LAUNCH.json \
  --interval 30 --max-seconds 86400 --history-updates 100 --stall-seconds 180
```

Low-water/supply alerts are expected in the current measured implementation:
the fixed200-step window still spends8.086% waiting. They do not authorize sample
skipping, reduced precision or an automatic Base restart. The observer never
signals any process. Inspect the specific producer receipt and error first;
rerun only the same bounded worker after verifying its old owner has exited.
If shared reserve falls below the recorded1.825TB, producers refuse new writes.
Do not delete checkpoints/permanent caches to make the alert disappear.

## Producer command (one GPU; only after verifying no duplicate owner)

On training-vlawm-tsinghua-zt, workers0–7 use matching physical GPU0–7.
On training-vlawm-tsinghua-zt2, workers8–15 use physical GPU0–7.
The example is the actual worker0 recipe. Do not launch it alongside the current
worker0. Budget is at most24h from the original launch, shared IO<=1.5GiB/s.
The CLI checks placement, ownership/headroom, content identities and disk reserve.

```bash
cd /mnt/project/DriveVLA-M0-iqe-rolling-producer-20261012
CUDA_VISIBLE_DEVICES=0 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
  /root/miniconda3/envs/ddp/bin/python -u -m iqe.rolling_cache.cli produce \
  --root /mnt/project/iqe-remote-rolling-cache-20261012/pool46534 \
  --config configs/iqe/fp32_cache/source.yaml --device cuda:0 \
  --hold-file /mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json \
  --max-windows 10000 --max-seconds 86400 --max-gib-per-window 32 \
  --rate-mib-s 96 --worker 0 --workers 16 --minimum-free-mib 30000 \
  --resource-policy /mnt/project/iqe-remote-rolling-cache-20261012/pool46534/producer_resource_policies/f3fbeb72838ef80d0175eeb325917cc4734531faec843fb29415cfb3b025572c/POLICY.json
```

The launch forces UTF-8 independently of any process LC_CTYPE changes. It does
not alter torch precision, RNG, payload bytes or model execution.

The live expanded pool allows384GiB and8 windows of16 updates, replacing the
original160GiB/3 producer budget. The original `SESSION.json` remains byte-for-byte
unchanged for consumer exact resume. `PRODUCER_RESOURCE_POLICY.json` binds the
separate immutable budget version. The monitor reports both the original and
effective capacities. The consumer-side `status` command above still describes
its original3-window reading horizon; use the monitor for the8-window producer
buffer. Checkpoint/free-space reserve remains1,825,361,100,800 bytes.

For a replacement producer, retain its original absolute deadline from the
EXPAND receipt: set `--max-seconds` to the remaining seconds, not a fresh24h.
Do not run legacy producers concurrently: their old GC horizon would conflict
with extended lookahead. The two-host replacement checked full argv, cwd and
PID/start ticks, retired all old producers before activation, and sent no Base
signals. This changes storage scheduling only and does not require a new Base
checkpoint, data cursor, qualification, numerical identity or22k identity.

## Original formal-continuation launch (already running, do not duplicate)

```bash
cd /mnt/project/DriveVLA-M0-iqe-rolling-cache-20261012
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/iqe/rolling_cache/train.sh train \
  --config configs/iqe/fp32_cache/source.yaml \
  --rolling-config configs/iqe/rolling_cache/continuation.yaml \
  --runtime configs/iqe/performance_v2/combined_fp32.yaml \
  --parent-checkpoint /mnt/project/iqe-remote-rolling-cache-20261012/rolling200/step_046734.pt \
  --output /mnt/project/iqe-remote-rolling-cache-20261012/continuation \
  --mode continuation --stop-after 53266 --warmup 10 \
  --hold-file /mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json \
  --qualification /mnt/project/iqe-remote-rolling-cache-20261012/ROLLING_QUALIFICATION.json \
  --checkpoint-every 1000 --save
```

## Resume after a verified stop, without losing the current lineage

`resume` resolves the highest valid complete checkpoint from the continuation
output, not the original Base directory or the original46734 checkpoint. The
next command resumes at most1000 additional updates, within the original100k
schedule; near100k set the explicit budget to the actual remaining number.
Do not run it while another process holds this lineage.

```bash
cd /mnt/project/DriveVLA-M0-iqe-rolling-cache-20261012
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/iqe/rolling_cache/train.sh resume \
  --config configs/iqe/fp32_cache/source.yaml \
  --rolling-config configs/iqe/rolling_cache/continuation.yaml \
  --runtime configs/iqe/performance_v2/combined_fp32.yaml \
  --output /mnt/project/iqe-remote-rolling-cache-20261012/continuation \
  --mode continuation --stop-after 1000 --warmup 10 \
  --hold-file /mnt/project/iqe-fp32-cache-production-task-20261011/CACHE_BUILD_HOLD.json \
  --qualification /mnt/project/iqe-remote-rolling-cache-20261012/ROLLING_QUALIFICATION.json \
  --checkpoint-every 1000 --save
```

If the resumed cursor refers to evicted windows, remote builders recreate those
windows from the same original scene plan. Prepared/prefetched samples are never
used as a resume cursor. Exact resume is at optimizer-step boundaries; world
size, data, code and numeric identity mismatches are rejected.

## Re-run the bounded regression or state comparison

```bash
OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/iqe/test_rolling_cache.py tests/iqe/test_production_vision_cache.py \
  tests/iqe/test_fp32_cache_migration.py tests/iqe/test_fp32_cache_partition.py \
  tests/iqe/test_rolling_cache_monitor.py -q

PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python \
  scripts/iqe/rolling_compare.py \
  /mnt/project/iqe-remote-rolling-cache-20261012/online200 \
  /mnt/project/iqe-remote-rolling-cache-20261012/rolling200 \
  --output /mnt/project/iqe-remote-rolling-cache-20261012/FULL_200_STATE_COMPARISON_RECHECK.json
```

Do not run large state checksum comparisons concurrently with timing windows.
Deployment inference continues using online FP32 vision; caching is a training
storage dependency. No Base replacement, new navtest, locked22k mutation,
checkpoint cleanup or deployment ACTIVE update is part of these commands.

## Existing Navtest watch after checkpoint-directory migration

The existing20k/50k/100k plan keeps the same evaluation code, input population,
hosts, protocol, output directories and original absolute watch deadline.
Its checkpoint root has been reconnected to the new continuation directory via
`/mnt/project/iqe-review-runtime-1e876da/navtest/SCHEDULE_PLAN_CONTINUATION_46534.json`.
The original plan is preserved. The idle original observer cooperatively exited
through its existing STOP_SCHEDULING mechanism, then one replacement observer
started. No model, active evaluation or Base process was stopped, no evaluation
point was added, and no evaluation worker was started by this handover.

The49k complete receipt has the exact training run identity required by the
existing plan. The original native checker will still validate the full checksum
and state when50k/100k become available. Current status:
`/mnt/project/iqe-review-runtime-1e876da/navtest/SCHEDULE_STATUS.json`.
The actual observer command below is already running; do not duplicate it:

```bash
cd /mnt/project/DriveVLA-M0-iqe-review-1e876da
PYTHONUTF8=1 /root/miniconda3/envs/ddp/bin/python -u \
  scripts/iqe/navtest_schedule.py \
  --plan /mnt/project/iqe-review-runtime-1e876da/navtest/SCHEDULE_PLAN_CONTINUATION_46534.json
```

`NAVTEST_WATCH_HANDOVER_AUDIT.json` records the live PID/start ticks, successful
WAITING_FOR_CHECKPOINT state at50k and lack of duplicate observers. This preserves
the existing user-authorized monitoring plan; it does not create an additional
Navtest experiment or change the locked22k identity.
