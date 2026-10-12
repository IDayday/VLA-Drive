# Bounded FP32 cache continuation, 2026-10-12

The user requested approximately 5 TB first and explicitly authorized
`training-vlawm-zt3` and `training-vlawm-zt4`. This is cache construction only.
No Base process is stopped, no training is started or migrated, and the locked
22k Base, its shared representation and its incremental artifacts are unchanged.

## Population and storage

The immutable original full plan remains
`/mnt/project/iqe-fp32-cache-production-runtime-3931fed/FULL_PLAN_40K.json`.
Its 90,892 fit scenes, order, sampler, parent checkpoint, FP32 visual identity,
input processing and retention budget have not changed.

The separate storage phase requests **84,664 scenes**, including all 6,400
already verified representative observations. Measured maximum record size is
56,687,046 bytes. Estimated logical payload is **4,799,352,062,544 bytes**
(4.799 decimal TB); **362,797,094,400 existing bytes** are reused by hardlink.
This estimate is not a claim that those bytes have already been built.

| Host | Access scenes | Reused | New scenes |
|---|---:|---:|---:|
| training-vlawm-zt3 | 45,544 | 6,400 | 39,144 |
| training-vlawm-zt4 | 39,120 | 0 | 39,120 |

Assignments are deterministic and disjoint. The full population has **6,228
additional scenes** outside this phase. A phase completion receipt cannot
authorize full-population cached continuation. The existing production runtime
still requires complete fit-list coverage; its implementation was not edited.

The original **1,777,001,877,380-byte** checkpoint/other-work reserve remains
intact. With the original 5% payload allowance and 16 atomic shard writers, the
initial phase requires **6,487,884,187,020 free bytes**, including those reserves.
Builders are limited to 8 GPUs per host, 64 MiB/s per worker, 8 hours per
invocation. The combined configured average write ceiling is 1 GiB/s. Hashing
and source-image reads are not included in that write ceiling. Existing Base
throughput must be watched for IO contention and the per-phase `CONTROL.json`
can reduce the native builder rate at shard boundaries.

Both hosts expose the same Lustre project mount. A separate quota could not be
queried because no `lfs` client is installed; actual mount free space is checked
before launch and at each shard. No cache is redirected to local scratch.

## Live dispatch evidence

At 01:20 UTC both hosts were occupied by G3 Navtest workers launched at
01:09:42. They were preserved. The zt4 physical GPUs also showed substantially
more memory than the visible evaluation processes, so absence of a visible PID
alone is never treated as an idle GPU.

The cache-only queue was started at 01:20:42 UTC on the control host with
PID **1754900**, Linux process start ticks **1320665557**. This PID is historical
evidence, not a signal target. At the first queue check (01:20:49), shared free
space was 6.4806 TB, slightly below the 6.4879 TB admission requirement; both
GPUs and sufficient space were still pending. **No GPU cache builder had
started at this snapshot.** The queue checks every 60 seconds for up to 12
hours. Each host can start as soon as its GPUs are idle and the aggregate
capacity requirement passes; it does not wait for the other host's GPUs.

Runtime root:
`/mnt/project/iqe-fp32-cache-production-runtime-3931fed/phase_5tb_20261012/`

- `AGGREGATE_PLAN.json`: target phase, hash
  `6967e48e31c87148e69c1536ef3f9be2f45881372ad2ae12b2b42acfe643976e`.
- `PART_0_PLAN.json`, `PART_1_PLAN.json`: immutable host partitions.
- `COHORT.json`: exact config, paths, GPU/IO/time budgets and source hash.
- `COHORT_STATUS.json`: current resource/dispatch state, not a static report.
- `cache_part*/PHASE_STATUS.json`: actual native worker progress after launch.
- `cache_part*/PHASE_RESULT.json`: verified phase result after completion.
- Original `full_fit_40k/COMPLETE.json` is not published by this phase.

## Code and validation

`scripts/iqe/fp32_cache/build_phase.py` wraps the existing qualified builder.
`queue_phase.py` checks both GPU ownership and physical utilization/memory,
shared-mount identity, source identity, capacity and current dispatch policy.
It never terminates another task. Ambiguous or failed launches are retained for
inspection rather than automatically duplicated. A controller error signals only
its own exact child processes to preserve committed shards. An operator pause
must be explicitly released before resuming that phase.

Current tests:

```bash
cd /mnt/project/DriveVLA-M0-iqe-fp32-cache-3931fed
PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/iqe/test_fp32_cache_partition.py
```

**13 passed**, 0.13 s, one existing pynvml deprecation warning. Coverage includes
immutable-parent validation, deterministic partitions and reuse, role/input
rejection, unchanged reserves, incomplete/failed native worker semantics,
disjoint host coverage, duplicate rows, physical GPU occupancy including hidden
owners, aggregate storage accounting and own-child-only shutdown.

The 23 qualified `iqe/vision_cache` and `iqe/performance` source hashes were
checked against `CACHE_QUALIFICATION.json`: **all unchanged**. Existing real
FP32 payload/update/resume qualification remains the evidence for the numerical
path; the new CPU scheduling tests are not presented as new GPU qualification.
All 90,892 original records have unique ordered image-path/language identities.
Actual image-content hashes are revalidated by the native builder.

## Commands

Inspect the live queue and phase progress:

```bash
cd /mnt/project/DriveVLA-M0-iqe-fp32-cache-3931fed
export PYTHONPATH="$PWD"
IQE_RT=/mnt/project/iqe-fp32-cache-production-runtime-3931fed
cat "$IQE_RT/phase_5tb_20261012/COHORT_STATUS.json"
```

The existing queue is already running. Only after verifying that its exact
process has exited, the same manifest can resume observation/dispatch without
starting a duplicate remote controller:

```bash
/root/miniconda3/envs/ddp/bin/python -u scripts/iqe/fp32_cache/queue_phase.py \
  --manifest "$IQE_RT/phase_5tb_20261012/COHORT.json" \
  --poll-seconds 60 --max-seconds 43200
```

Worker failures preserve shards and launch evidence. Inspect their logs and
`PHASE_STATUS.json`; do not delete `REMOTE_CONTROLLER.json` or blindly relaunch
an old PID. `build_phase.py run --help` exposes the explicit same-plan resume
entry. If its `CONTROL.json` is paused, release it only after confirming the
previous exact controller and children have stopped and fresh resource admission
is available. No formal training continuation is launched from a partial phase.
