# T0 timing240 preparation

Status: **IMPLEMENTED, CPU_TESTED, NOT_DISPATCHED, NEEDS_INDEPENDENT_REVIEW**.
This package does not claim throughput qualification or formal launch permission.

Task owner: `/root/temporal_t0_timing_prepare`. Source worktree:
`/mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018`, branch
`agent/temporal-t0-timing240-20261018`, starting commit
`36104105` (Lead's requested integrated base).
Only this report directory is edited; native framework/trainer files are unchanged.

## Scope and actual evidence

The existing native runtime already implements the registered timing protocol:
`/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018`, commit
`38f86a15a92171e583bb5019cce89a63fb7a1d43`.
`temporal_formal.selected_recipe` and `consumption` explicitly admit a fresh
PROFILE with 240 updates; `train_world` rejects resume when profile mode is timing.
`profile_pipeline.TimingPlan` measures two windows of20 warmup+100 steady updates.
The main research contract is §17.3; this is execution of that existing design,
not a new model or training candidate.

The actual T0 capacity run completed4 updates/128 scenes with no retained owned
workers. Its monitor reports742.57 seconds allocation wall, and native checkpoint
cost48.43 seconds. The four diagnostic update times were8.81,8.16,7.92,7.92 seconds.
These are capacity evidence, **not** a sustained throughput estimate. At about8s
per additional update, the existing3600s native cap plausibly covers a240-update
window plus cold setup/save; it remains a bound, not a completion guarantee.

Capacity artifacts stay untouched:
`/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_capacity4/training_control/`.
Its checkpoint is **not loaded**. The new invocation starts from the same original
Base binding, fresh optimizer/RNG/cursor, and original20000-update/warmup1000 clock.

## Exact execution

- T0 current-only, SP2/PCA384/W48/96 sampled positions, all original loss definitions.
- Full101592-scene TRAIN manifest; same seed42 and native sample order.
- Four ranks, microbatch1, accumulation8, effective global batch32.
- Original BF16 execution / FP32 master and Adam / ZeRO2 / TF32-off policy.
- `profile_mode=timing`, `stop_after=240`, `save_every=240`, `max_updates=20000`.
- Two steady windows:21–120 and141–240, each preceded by20 warmup updates.
- 7680 scene exposures,1920 per rank; no resume and no automatic retry.
- Native3600s, child timeout3720s, outer wrapper3900s,180s cleanup.
- RSS192GiB, logical read cap1TiB, CPU2 threads/rank, eight admitted CPU cores.
- New output:
  `/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_timing240/training_control`.
- Output cap40GiB including partial checkpoint/log/metadata, at most one checkpoint
  directory. A periodic save at240 is reused by the native stop-after pause.
- Four exclusive *devices*0–3 on authorized Tsinghua-zt4; the host may still have
  independent GPU/CPU jobs. No exclusive-node performance claim.

The controller inherits the exact reviewed capacity supervisor, its pinned
descendant ownership/cleanup, GPU locks, source checks and per-rank telemetry.
Strict substitutions change only the timing invocation identity and accounting.
It additionally requires independent correctness and resource receipts tied to
the actual controller/launcher/inputs hashes. No `--skip-review`, full mode,
arbitrary topology, precision, resume or altered step budget is exposed.

## Metadata and identity

`prepare_inputs.py --write` only reads small pinned JSON documents and writes a
small immutable metadata clone under `prepared_inputs/`. It does not load weights,
open images, traverse full indexes, extract teachers, or fork a trainer.

The provider numerical identity, source index, PCA basis, Base and data manifest
remain identical. Only the provider admission's consumption/recipe binding,
descriptor/PCA-scope integrity hashes, config timing fields and output namespace
change. `native_preparation` in INPUTS identifies the reused **parent data factory**,
not a claim that the new timing workload has executed. METADATA_LINEAGE explicitly
records the derivation. A CPU test calls the original native recipe validator.

The absolute paths in prepared metadata intentionally bind the author's retained
worktree. Cherry-picking does not silently rewrite them. The worktree must remain
available, or a newly versioned metadata preparation and review are required.

## Storage obligations and unresolved admission

Independent Systems record:
`/mnt/project/VLA-Drive-st-spatial20k-efficiency-20261018/reports/st_world_policy/20261018/T0_capacity_settlement_and_timing_budget_v1/SETTLEMENT_AND_PROPOSAL.json`
SHA256 `b99b5ec90bf789c48971095385ac8ace8c37e661905b2d848bff76aac9a93f27`.
The completed capacity invocation's unused advisory allocation was settled; every
existing output is retained and no scientific checkpoint obligation is released.

At every sampled storage check, required free bytes are:

```
1TiB floor + actual scientific checkpoint remaining
+ 768GiB retained cache + 10GiB metadata + 4GiB C-evaluation reservation
+ 40GiB timing reservation - actual timing-owned bytes already written
```

Original gross scientific checkpoint commitment remains4143GiB. The current V3
remaining receipt is2266025903732 bytes; a fresh Systems admission must verify its
validity, device UUIDs, live CPU/memory/IO conditions and free space. The proposal
is tight and is not an allocation or quota. No job should start merely because
RESOURCE_ADMISSION_TEMPLATE exists. No files are deleted and no active job is
stopped by this package.

## Tests and limits

Actual command:

```
taskset -c 10,11 timeout 120s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python -B -m pytest -q -p no:cacheprovider reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1/test_controller.py
```

Result:46 passed in4.62s (`TESTS.log`). Tests include unchanged numerical metadata,
full native recipe validation, warmup/time-axis preservation, exact240 exposure,
negative full/precision/batch/loss mutations, missing/stale/self-authored reviews,
no resume/retry, one-partial-checkpoint cap, unchanged inherited monitor, and CLI
rejection of bypass flags. Fixtures are explicitly CPU unit evidence, not GPU
resource or real timing qualification.

Remaining prerequisites: independent exact-pin correctness review; fresh actual
Systems admission; Lead-only prepare/run. No GPU call or training was performed by
the author. No timing result, memory result or scientific gain is claimed.

## Lead commands after both reviews

`ST_T0_TIMING_ADMISSION` and `ST_T0_TIMING_ADMISSION_SHA` must be populated with the
actual independently produced receipt, not this package's template. Commands run
on the allowed actual host `training-vlawm-tsinghua-zt4-worker-0` after live checks.

```bash
test -n "$ST_T0_TIMING_ADMISSION"
test -n "$ST_T0_TIMING_ADMISSION_SHA"
bash /mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018/reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1/launch_once.sh --action prepare --resource-admission "$ST_T0_TIMING_ADMISSION" --resource-admission-sha256 "$ST_T0_TIMING_ADMISSION_SHA"
bash /mnt/project/VLA-Drive-st-temporal-t0-timing240-20261018/reports/st_world_policy/20261018/temporal_T0_timing240_preparation_v1/launch_once.sh --action run --resource-admission "$ST_T0_TIMING_ADMISSION" --resource-admission-sha256 "$ST_T0_TIMING_ADMISSION_SHA"
```

After a verified complete240-update exit, use the unchanged native summarizer:

```bash
cd /mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018
ST_T0_RUN=/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_timing240/training_control/runs/T0_fulltrain_seed42_profile240_world4_micro1_global32_v1
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -B -m tools.st_world_policy.profile_pipeline --output "$ST_T0_RUN/profile/TIMING_SUMMARY.json" summarize --ranks "$ST_T0_RUN/profile/rank0.json" "$ST_T0_RUN/profile/rank1.json" "$ST_T0_RUN/profile/rank2.json" "$ST_T0_RUN/profile/rank3.json"
```

The summary reports complete-window wall time including checkpoint/logging and
separate normal-update metrics. Also report cold setup/whole allocation wall from
the monitor, real checkpoint seconds and all-rank memory. Do not substitute the
older internal `slowest_rank_seconds` for complete pipeline timing. No `resume`
command exists: an incomplete timing run stays incomplete and requires a separately
reviewed new invocation rather than pretending resumed windows are continuous.
