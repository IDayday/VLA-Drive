# Exact32 causal-history preparation handoff

Status: **read-only preparation; builder not run, no admission or launch signed**.
Author working tree starts at `f8098c895411dc500bd6eaa56f6eaee3d778e72d`.
Consumers are the existing T3/T4 runtime `a8f8cb879a806ea674ed7be9b1ff7d82483e8b2d`
at `/mnt/project/VLA-Drive-st-temporal-combination-20261017`.

## Exact missing input

The selected manifest is
`/mnt/project/st-world-policy-20261016/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json`,
SHA256 `7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8`.
It contains **32 unique scenes from 2 logs**. Its overlap with the original
`smoke_train64_v1/SELECTED_SCENES.json` is **zero**. Do not change that list to
reuse history64. The proposed new immutable output
`/mnt/project/st-world-policy-20261016/audits/history_data/temporal_composition32_history_v1`
did not exist when checked.

This is causal input metadata: current record identity plus actual history at
-1.0 and -0.5 seconds, original camera order, timestamps, transforms and masks.
The separate teacher-request artifact is restricted to time0. No new future
selection, teacher tensor, cached W, image copy, GT rewrite, or training is
required. Missing causal frames retain the existing missing-view/frame masks;
this handoff does not assert all requested history exists.

The builder `prepare_history.run` validates the original complete student
population/partition, then resolves this exact ordered token/log subset. It
re-pins current/ego files from the real source. After building, compare the
output's token/log sequence and original current/ego pins to the fixed selected
manifest; its enriched serialization need not be byte-identical to the input.

## Existing implementation and monitor

No new builder is needed. Eight source files are byte-identical between the
completed history64 runtime `b847f5a99dd44ee5256362d380d9d96efb47d35b` and T3/T4's
`a8f8cb`: `prepare_causal_history`, `prepare_history`, `history_shards`,
`prepare_future_metadata`, `st_world_history`, `st_world_metadata_index`,
`st_world_future_provenance`, and `prepare_current_teacher`. Exact paths/SHA256s
are in `SOURCE_AND_SELECTION.json`. Existing sparse packed layout, source
hashing, timestamp geometry, before-write callbacks and immutable publication
therefore remain unchanged.

The actual route is:

```text
owned finite run_pilot_once controller
  -> existing run_build_once.core.monitor (4b2f0f source)
  -> same-process metadata_storage.run_registered_metadata
  -> prepare_causal_history.run
  -> original prepare_history.run + packed_log_v1 storage adapter
```

Reuse source:
`reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py`
and
`reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py`
in the frozen T3/T4 tree. The original controller is **not directly executable
for this new population**: it pins history64 selection/root, exact64 parser
arguments, CPU6/7, old admission status, and the historical2303GiB checkpoint
commitment. It intentionally rejects an arbitrary new argv. Do not weaken its
checks or call the builder bare to evade them.

The minimum controller work is a mechanical fixed32 binding: new output/control
roots, exact selected SHA, max_scenes32, current admitted CPU pair, fresh current
checkpoint commitment and exact32 admission identity. Retain the existing
absolute deadline, 4b2f monitor, nonblocking ledger acquisition, before-import
accounting baseline, child wait-boundary read accounting, owned-descendant
cleanup, single-shot marker, failure preservation and no automatic retry.
This report does not implement or pretend such a binding already exists.
`metadata_storage.py` has evolved since b847; use its actual a8f8cb pin and the
current policy/ledger rather than reinstalling an old storage implementation.

## Prior measured cost and proposed upper limits

The prior actual independent receipt is
`reports/st_world_policy/20261017/history_packed64_actual_independent_v1/README.md`
in `/mnt/project/VLA-Drive-st-world-policy-20261016`.
That **different 64-scene/64-log** build completed once:

| Quantity | Prior observed value |
|---|---:|
| Wrapper wall |81.6231s|
| Process-tree RSS peak |1,653,936,128B|
| Wrapper charged logical reads |1,991,446,716B|
| Builder source-byte meter |1,668,979,101B|
| Actual artifact+control content |1,156,349B|
| Allocated blocks |1,298,432B|

These are not exact32 measurements. Fixed full student metadata reads,
per-log pickle sizes, repeated image hashing and filesystem cache state prevent
halving them by scene count. The new subset has only2 logs, but that does not
authorize a fabricated cost estimate.

`STORAGE_REGISTRATION_DRAFT.json` carries conservative **proposed stop limits**
from the already executed history64 path, pending independent Systems approval:

| Resource | Proposal / required live binding |
|---|---|
| CPUs |One or two explicitly available CPUs; `cpu_affinity` remains null in the draft|
| Workers |1; no GPU, OMP/MKL/OpenBLAS2|
| Work / cleanup wall |900s +40s; outer GNU timeout covers bootstrap|
| Aggregate RSS |4GiB; no reason to raise it from prior observed1.65GB|
| Total logical read |16GiB, whole invocation including imports/controls/wait boundary|
| Builder source-read upper bound |16GiB; additionally throttled to32MiB/s|
| Metadata artifacts |31MiB:23MiB output+4MiB staging+4MiB metadata|
| Separate terminal/control evidence |1MiB **inside** combined32MiB, not32+1|
| Files/depth/history shards |256 entries, depth1, max32 packed shards|
| Invocation |Exactly1, no automatic retry|

Actual CPU availability, current host/lease times, same-ledger32MiB reservation,
checkpoint/free-floor commitments, control adoption and current policy hashes
must be provided by Lead/Systems. No old lease/CP timestamp can be reused as live
authorization. No quota or current claim capacity was inspected or modified here.
The teacher cache and large metadata indexes must not be rebuilt for this input.

## Exact existing CLI argv

The following is the **registered builder argv**, not a permitted bare launch.
`HISTORY32_BUILDER_BUDGET` must name a real small budget file with
`phase=data_preparation`, `split=train`, `max_scenes=32`, `workers=1`,
`max_seconds=900`, `target_times_s=[0.0]`; it is distinct from the typed storage
registration. All flags were checked in the actual original parser/source.

```bash
RT=/mnt/project/VLA-Drive-st-temporal-combination-20261017
PY=/root/miniconda3/envs/ddp/bin/python
ROOT=/mnt/project/st-world-policy-20261016
SELECTED=$ROOT/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json
HISTORY=$ROOT/audits/history_data/temporal_composition32_history_v1
: "${HISTORY32_BUILDER_BUDGET:?Lead-issued exact32 budget file required}"
cd "$RT"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.prepare_causal_history build \
  --student-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 \
  --raw-log-root /mnt/navsim/trainval_navsim_logs/trainval \
  --sensor-root /mnt/navsim/trainval_all/trainval_sensor_blobs/trainval \
  --fallback-sensor-root /mnt/navsim/trainval_sensor_blobs/trainval \
  --partition /mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --scene-list "$SELECTED" --split train --mode registered --max-scenes 32 \
  --seed 20261017 --tolerance 0.05 --target-times 0 --record-layout packed_log_v1 \
  --registration "$HISTORY32_BUILDER_BUDGET" --output "$HISTORY"
```

The existing admitted child interface is:

```bash
: "${HISTORY32_POLICY:?Fresh pinned current policy required}"
: "${HISTORY32_POLICY_SHA:?Current policy SHA256 required}"
: "${HISTORY32_STORAGE_REG:?Real storage registration required}"
: "${HISTORY32_STORAGE_REG_SHA:?Storage registration SHA256 required}"
: "${HISTORY32_CP:?Current checkpoint commitment required}"
: "${HISTORY32_CP_SHA:?Checkpoint commitment SHA256 required}"
CUDA_VISIBLE_DEVICES='' "$PY" -m tools.st_world_policy.metadata_storage run \
  --policy-path "$HISTORY32_POLICY" --policy-sha256 "$HISTORY32_POLICY_SHA" \
  --registration-path "$HISTORY32_STORAGE_REG" --registration-sha256 "$HISTORY32_STORAGE_REG_SHA" \
  --checkpoint-receipt "$HISTORY32_CP" --checkpoint-sha256 "$HISTORY32_CP_SHA"
```

This child must remain inside the existing owned supervisor. No exact32 parent
path is fabricated here: first produce the mechanical binding identified above,
then invoke its existing `--admission PATH --admission-sha256 SHA` interface with
fresh reviewed pins under the900+40s bootstrap-inclusive outer bound. The old
64 controller cannot serve as that command unchanged.

## Completion and downstream

Require both actual `HISTORY/COMPLETE.json` and supervisor terminal COMPLETE,
exit0/owned cleanup0/read+RSS+disk bounds. Preserve any failed attempt. Check the
exact32 ordered population, current/ego source pins, causal timestamps, original
view order and missing masks. `CONTROL_HISTORY_RESULT` may first say
`BUILD_RETURNED_PENDING_TERMINAL_ACCOUNTING`; that intermediate receipt is not
the terminal success criterion.

Then feed `$HISTORY/COMPLETE.json` into the **already implemented**
`tools.st_world_policy.temporal_composition --history-complete` command in
`temporal_composition_v1/RUNBOOK.md`. No future-scene reselection, tensor rebuild
or new training controller is required. Separate T3/T4 training/profile
authorization remains outside this metadata handoff.

This preparation's pinned source/metadata snapshot read423,734B and wrote only
small reports. Preliminary source inspection was also small, with no tensor,
raw log, image, source identity/index scan, GPU/SSH or ledger action. One report
generation shell snippet failed Python syntax compilation before executing any
statement; corrected report generation completed. No experiment or fixture
builder was executed.
