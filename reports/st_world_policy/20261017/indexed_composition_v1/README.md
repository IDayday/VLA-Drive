# Indexed cached/on-demand teacher composition: implementation handoff

Status: **IMPLEMENTED, CPU_FIXTURES_VERIFIED; REAL_QUALIFICATION_NOT_RUN**.
This is an execution qualification adapter for the existing fixed-8 FP32 DINO
teacher. It does not add a teacher, student, target recipe, trainer, cache builder
or formal admission. Source start: `cf5805016b5e23bdab9541c77b562ed5fda83dc3`.
Owned worktree: `/mnt/project/VLA-Drive-st-indexed-composition-20261017`.

## Why a new result is necessary

The completed fixed-8 qualification used a different request population and a
small current source. It establishes real fixed-8 implementation evidence; it
does not qualify a newly signed set of request/source paths. The planned new
population is the independently prepared 32-scene future metadata. Its 294
unique image keys have 189 hits and 105 misses in the full-current source index,
according to Lead's metadata audit. These counts are planning evidence, not a
new real teacher numerical result. The new CLI derives them from exact indexes.
No old qualifier, result, source cache, PCA basis or native checkpoint is edited.

The new receipt is `st_indexed_composition_qualification_v1`, with plan schema
`st_indexed_composition_plan_v1`. It binds:

- The unchanged prior fixed-8 result and resolved plan, all exercised cases,
  actual execution/source/backend identity, frozen-state and RNG checks.
- Exact request-only metadata index and original request bytes; source-cache
  index and original cache identity; all touched source receipt hashes.
- The unchanged native FP32 teacher contract, config and execution identity.
- The active implementation files, exact scene/key sequences, provider identity,
  bounded LRU policy and budgets.

The factory recognizes this schema only for the exact new indexed population,
maximum 32 train scenes. Existing `configure_smoke` still requires at most four
updates. Its PCA scope remains the independently reviewed indexed scope. No
formal/profile/full-population expansion or source/precision substitution exists.
The original legacy and fixed-8 qualifier admission branches remain unchanged.

## What the real run will measure

1. Generate an independent direct fixed-8 online reference for every new key,
   bypassing both cache and provider; retain finite FP32 features/masks in RAM.
2. Recompute every key in reversed order with request groups of four, while the
   executor still computes exactly eight slots with repeat-last padding.
3. Read every existing-cache hit with the original first-use shard checksum and
   per-record digest validation. Compare full tensors/masks with online reference.
4. Use the real indexed provider for all scenes under cold, partial-LRU,
   cold-restored and reversed-scene layouts. Preserve missing-source masks and
   compare every valid time/view slot, not only sampled patches or digests.
5. Preserve Python/NumPy/CPU/CUDA RNG through construction and every fixed-8
   call. Verify sampled patch/future-pair IDs under restored streams and retain
   actual cache/RAM/miss/eviction counts. Both a mixed cache+compute read and a
   partial-RAM-hit+compute read must actually occur.

`fixed_policy_repeatability` compares regrouped online outputs and provider
outputs against their composed reference (original cached values plus fresh
misses). `legacy_reference_compatibility` separately compares *all* cached hits
against the fresh fixed-8 reference. Both must be bitwise true for the new
receipt to admit a provider. A numerical difference is retained for review;
there is no tolerance argument. The receipt also retains online differences for
every slot, so a stable cached difference cannot disappear behind repeatability.

## Resource accounting and boundaries

`plan` opens bounded metadata readers, hashes metadata and stats model/shard
files. It never opens an image, model tensor or teacher target payload and does
not discover CUDA. The request population is capped at 32 scenes / 480 keys;
it never enumerates the full-current record population in Python.

The new run is capped at 900 seconds, two CPU threads, one explicitly admitted
GPU, 8 GiB host RSS, 16 GiB CUDA allocated peak, 64 GiB logical inputs and
32 MiB metadata output. It writes **zero teacher tensor bytes**. Fixed-8 padded
compute is counted, not just returned records. The plan calculates an input
upper bound including metadata validation, first-use shard SHA, mmap slices,
weight loading and image reads before accepting its budget.

The native contract uses 1,769,904 bytes per feature+mask record. For 294 online
records plus 189 composed cached copies, the calculated resident references are
854,863,632 bytes (815.261 MiB), below the explicit 2 GiB reference ceiling.
This excludes the separately bounded teacher, scene buffers, comparison temps
and metadata; whole-process peak RSS is checked at runtime. These are calculated
byte counts, **not measured GPU/RAM peaks or throughput**.

Runtime accounting retains `/proc/self/io` rchar and explicitly accounted
mmap-record/teacher-weight bytes. rchar and mmap accounting describe logical
reads, not physical storage traffic. Guard checks are cooperative between bounded
calls; the Lead's owning supervisor must enforce the outer deadline. A kernel
or checksum call is not preempted by an in-process check. Missing/incomplete
results are not qualifications; all attempts use fresh output namespaces.

## Validation and remaining inputs

CPU tests use explicitly synthetic temporary images, metadata, cache tensors,
prior-result fixtures and a fake executor. They exercise the actual indexed
reader/provider, first-use checksums and admission code. They do **not** stand in
for real DINO/image/FP32 numerical evidence.

Covered: metadata-only planning; bounded population and padded budget; valid new
factory admission; no borrowing old population qualification; all four layouts,
mask/keys/RNG/restore; wrong source/execution/source-code/pins; truncated or
re-signed false evidence; actual partial/mixed counters; sampling continuity;
separate legacy numerical failure; immutable-shard first-use SHA retained.
85 existing/new focused tests passed in 42.75 seconds; the subsequent additional
two-scene/shared-key/distinct-mask test passed in 5.90 seconds (86 distinct tests,
30 new tests total). Full output and exact commands are in `tests.log`,
`multiscene-test.log` and `RESULT.json`.
Actual CLI `--help`, `plan --help` and `run --help` are callable.

Still required before real dispatch:

- Lead-created request-only metadata index for the exact future32 manifest,
  using the shared metadata ledger (no ad hoc root/claim here).
- For later <=4-update student smoke: original PCA fitted-cache metadata index
  and a separately pinned 32-scene dataset; the qualifier itself does not need
  PCA or a student.
- A final clean integrated source snapshot; regenerate plan against its source
  hashes, then independent correctness/resource review and an authorized GPU.
- Real qualification result. Only that result can be used by `prepare_indexed`.

Full-future training, long-window throughput and formal temporal training remain
unqualified. No running training, evaluation, cache claim, resource policy or
locked artifact was changed by this implementation.
