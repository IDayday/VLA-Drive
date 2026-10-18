# Explicit C1 TRAIN failure-cohort road diagnostic adapter

Implementation starts at exact mature road runtime
`0d0d19dd5ce2d2fd583d2cc265289e47649ca594` in isolated branch
`research/st-road-failure-direction-20261018`. The running rollout-normalizer
training/profile worktrees are untouched. This report is an implementation
handoff, not independent execution approval or evidence of road repair.

The prior actual v3 diagnostic used the original 32 TRAIN scenes at boundary4095,
candidate0; every original DAC was1. Seven directions were defined and25 remained
undefined. Its repair denominator was therefore0. Existing report:
`/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/road_direction_actual_v3/REPORT.md`.
The new adapter permits one separately frozen TRAIN failure frame to answer the
previously unidentifiable road-repair question. It does not select a favorable
proxy result or authorize proxy-driven policy training.

## Kept exactly

`road_margin_proxy.py`, the official consequence backend and the original
`diagnose_road_direction.py` are byte-identical to the base commit. Source hashes
and comparisons are in `SOURCE_BINDING.json`.

The actual existing direction math is the complete `[8,4]` normalized raw action
domain, including sin/cos channels, with L2 offset0.005. It is **not XY-only or
yaw-fixed**, and0.005 is not a meter offset. Physical yaw can change within the
original0.025rad guard. Negative, positive and seeded random directions, the
anchored FP32-to-FP64 decode gate,1e-9 official geometry parity, original road
cost, finite-difference rules and all original undefined regimes remain intact.
Random seeding still uses `scene_id + NUL + selection_hash + NUL +
road_direction_v1`; the new fixed selection has its own hash by construction.

The original default remains `--prepare` with `fixed32_v1`, its same evidence
selection and128-call cap. Changing implementation hashes naturally invalidates
old execution approvals; no prior approval is silently transplanted.

## Narrow additions

- `road_failure_selection.py` supplies a typed, immutable selection and joins
  original C1 receipts. It has no model, proxy, simulator or optimizer path.
- `--bind-failure-selection` is an explicit CPU preparation. It reads the
  content-hashed fixed frame in its original order, loads only its original
  native ledger/rank containers, and examines the selected slot's four actual
  candidates. The container format requires reading its complete serialized
  file; no unrelated model or metric-cache payload is opened.
- The failed candidate is the seeded-hash first actual DAC0 candidate; the safe
  sibling is the same-rule first actual DAC1 candidate in that group. PDMS0 is
  never treated as proof of DAC0. These choices are made before any new proxy
  or directional result.
- `--selection-mode failure_cohort_v1` is required for new registration. The
  registration pins the final selection file bytes, all selected-row content,
  source hashes, exact commit and original/failure execution schema. A metadata
  frame alone is refused. Replacing a file at the same path invalidates approval.
- The new mode has the original supervisor, wall, memory, output and IO limits,
  and a **96-new-call** cap with three durable variant intents per scene. It
  refuses original re-scoring, duplicate variant calls and automatic retry.
- Summary denominators are calculated from actual frozen original labels and
  actual scene results. Requested, eligible, t0-irreparable, complete-valid and
  unresolved counts are separate. DAC repair has a requested-failure denominator
  and a complete-valid-subset denominator. Known repairs/requested is explicitly
  a lower bound when some directions remain unresolved. Per-component safety
  denominators remain visible; original DAC0 usually means original PDMS0, so
  high-to-zero alone is not a useful safety test for this cohort.

Every original/sibling joins exact `scene_id/source_log/boundary/rank/slot`,
policy snapshot origin, saved FP32 raw hash, saved physical hash, canonical
metric masks/units, context hash, reference-input hash, complete/admitted receipt
and request identity. The original TRAIN index is checked. `RolloutBatch` verifies
the actual complete raw/noise content identity. A changed context, raw tensor,
source log, candidate mapping, selection order or receipt is rejected.

The native road flag at t0 is recorded and checked. An unexpected original t0
failure is undefined before geometry/scoring, retained in the fixed requested
population, and never replaced. Invalid provenance fails closed; the population
is still represented by the frozen frame and failed binding receipt.

## Binding limits and preservation

Preparation is bounded separately:512MiB logical reads,16MiB output,600s wall,
120CPU seconds,2GiB measured RSS, affinity at most2CPUs, one numerical thread,
32MiB/s logical read pacing. There is no2GiB address-space limit around PyTorch
imports. CPU soft limit, deadline alarm, explicit RSS/CPU checks and the parent's
external limits bound the work. Actual small control checksum discovery reads
are accounted as well as the subsequent pinned read.

The builder only stats selected metric caches to plan their compressed byte
budget; it never opens their content. Before writing a usable selection it
checks that the new execution's exact planned logical reads fit the original
256MiB cap. If they do not fit, it reports the concrete limit and preserves the
attempt; it does not expand execution limits. Originals, caches and previous
reports remain unmodified. A repeated binding output path is rejected.

The worker still preserves failed, zero-gradient, nearest-tie, decoder/parity,
native scoring and interrupted states. No case is replaced. Native calls have
durable intents before invocation; an uncertain interruption is never retried.

## Evidence and boundaries

The independent Evidence agent froze FRAME32 in commit
`758ecc85ccb44e3b6364aa0c0c9d72421daf56e1`, file SHA256
`5ea6524267bc4e02fde7bfd79389ba774bea3f77821898b2471b369f316b1841`.
Its frame is32unique scenes from31logs and31original feedback ledgers. It is
metadata-only and not itself an executable selection. No original candidate
payload, metric cache, image, model checkpoint, GPU or real scorer was read or
run by this implementation task.

CPU fixtures verify the old default, actual arbitrary boundary/rank/slot joins,
nonzero failing candidate indices, stored raw/physical identity, seeded choice,
missing safe sibling/provenance, wrong context/source, duplicate scene, changed
same-path content, explicit schema,96/per-scene call caps, t0 handling,
requested-denominator preservation, binding CLI success/failure/idempotent refusal,
and the original supervisor faults/cleanup and geometry tests. Fixtures are
explicitly synthetic; they do not stand in for actual assets or canonical scores.

`TEST_OUTPUT.txt` and `TEST_EXECUTION.json` record the final **85 passed** run
(24.97s pytest;26.86s total wall,27.05s child-inclusive CPU, sampled aggregate
peak RSS1,484,316,672B). No resource cap was triggered.
Earlier failed command attempts and intermediate passing runs are retained.
The first attempted test wrapper used unavailable `/usr/bin/time`; the second
wrapper initially resolved its repository one parent too high and exited before
tests. Those setup errors were corrected and are not presented as test passes.

Implementation status: **IMPLEMENTED, CPU_VERIFIED, NEEDS_INDEPENDENT_REVIEW**.
Real candidate binding, admitted real proxy execution, actual resource audit and
scientific repair results: **NOT_RUN** by this agent. The Lead performs those
stages after reviewing this exact source; this agent signs no correctness or
resource PASS for its own implementation.
