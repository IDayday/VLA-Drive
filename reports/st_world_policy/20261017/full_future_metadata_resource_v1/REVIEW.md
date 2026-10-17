# Full future metadata: independent resource review

Review date: 2026-10-17. Reviewer: `/root/st_metadata_resource_review`.
This review reads source, small control JSON and file metadata only. It does not
launch a builder, read raw log payloads/images/weights, migrate a ledger, create
a cache root, or authorize training. The existing PROFILE64 result is separate
evidence; this review does not rerun it.

## Decision and concrete blocker

The next useful operation is metadata preparation and an SQLite index, with no
new teacher tensors on disk and no GPU. It does not require a multi-TB cache.
Full train/dev population preparation is feasible to attempt with the bounded
plan below, but has not been executed or admitted by this report.

The exact original population is train **101,592 scenes / 1,176 logs**, dev
**1,696 scenes / 16 logs**. Reading the small `SOURCE_POPULATION.json` manifests
and issuing `stat` for those exact logs found three train logs larger than the
existing 128 MiB ceiling: **147,814,233; 137,629,626; 171,848,564 bytes**. Existing
`plan(full_population)` rejects them before publishing PLAN. This is a real
blocker, not a reason to omit those scenes. Retain the default for old callers;
the proposed train invocation explicitly needs a reviewed 192 MiB ceiling,
one worker, and the unchanged 4 GiB process-tree RSS bound. A file-size bound
does not prove deserialized memory usage. Dev's maximum is 86,509,381 bytes and
does not need that source change.

`POPULATION_STAT_EVIDENCE.json` records the measured totals, original completion
receipt hashes and source-manifest identities. No raw pickle was opened. The
train raw files total 12,935,693,807 bytes; dev totals 235,171,260 bytes.

## Authoritative limits and additive migration

The live ledger read in this review is
`/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json`,
SHA `a4c1fb15f4799187b03035343b8582484d8feb02fd23c7b632b5bc968662789d`.
Its eight existing rows include the typed SP0 dev-W claim; the metadata schema
is still `st_metadata_storage_package_v1` with a 4 GiB envelope. The common cache
cap is **768 GiB**, the free-space floor is **1 TiB**, and the ledger overhead is
16 MiB. These are live recorded limits, unlike a proposed policy not migrated.
The user did not impose a 4 GiB metadata limit. Increasing that implementation
envelope is appropriate; creating multi-TB new teacher tensor banks is not.

Use the reviewed, explicit v2 **8 GiB combined metadata/temporary envelope**
introduced by `fc477e60`, after its separate correctness review. Preserve the
original v1 package digest, all original rows and legacy reservations, the
typed-W package and its original-parent pin, the 768 GiB cap, checkpoint reserve,
and the same ledger/lock. Do not edit the historical v1 file or open a second
uncoordinated ledger. A fresh Lead writer-quiescence observation and migration
receipt are still needed; read-only training is not a cache writer.

| Allocation | Inclusive MiB |
|---|---:|
| Existing ledger metadata, including adopted pilots | 1,136 |
| Existing administrative metadata reservations | 128 |
| Full train future preparation | 2,048 |
| Full train request index, including journal | 2,048 |
| Full dev future preparation | 128 |
| Full dev request index, including journal | 128 |
| Total after adoption | **5,616** |
| Uncommitted part of 8 GiB envelope | **2,576** |

The administrative 128 MiB is real and must be retained: 64 MiB for the earlier
indexed-composition plan/run reports, 32 MiB for SP0 controls, and 32 MiB for the
provider factory. Its source is the immutable
`ADMINISTRATIVE_METADATA_CARRYFORWARD.json` under
`audits/spatial_probe_actual_SP0_5000_v1/storage_preparation_dev_v1`, SHA
`1b3a22906bfebaa99bd425e855d9ec73550a4b34013490bb815814d1847aa530`.
These administrative reservations are not falsely called ledger claims.

Before migration, the two new PLAN roots receive a separate **64 MiB combined
administrative allowance** (32 MiB each), giving 1,328 MiB temporarily. The old
64 MiB report reservation cannot silently be repurposed: it names two different
roots. Upon adding the two preparation claims, their existing PLAN/controls
are absorbed into those claims, not charged again, leaving the stated 5,616 MiB
total. No old files are deleted. All partials, logs, directory allocation,
controls, indexes, SQLite journals and atomic-write overlap remain in the
relevant inclusive root allowance.

`RESOURCE_PLAN.json` contains exact byte accounting derived from the current
ledger. Shared space had 7,293,923,561,472 available bytes at the recorded stat.
Quota is **NOT VERIFIED**; an available-space snapshot is not a quota guarantee.

## Practical sequence and limits

1. Execute one separately bounded metadata/stat-only PLAN per role. Plan reads
   source control JSON and checks selected-manifest hashes/order. It stats the
   registered raw logs without reading them. `selected_rows` is `null` for
   `full_population`, so it does not duplicate the 30 MB train selection in PLAN.
   The known reserved control reads are 127,915,555 bytes train and 20,427,411
   bytes dev, plus bounded imports/source pins. The proposed outer budget is
   CPU 0/1, 2 GiB process-tree RSS, 180 s plus 30 s cleanup, 4 GiB logical reads,
   and 32 MiB inclusive output per role. The CLI limits in PLAN describe the
   later build; they are not the outer PLAN budget.
2. Bind the actual immutable PLAN identities in new future-build registrations
   and same-ledger additive claims. Refresh checkpoint commitments through the
   entire stage and cleanup. The existing 2,183 GiB receipt expires at
   1792232752.2251627 and cannot admit a new four-hour build. Retain all 1,983 GiB
   original run obligations, 120 GiB previous smoke, 80 GiB PROFILE, plus the
   independent 1 TiB free floor. Do not count saved checkpoints as deletions or
   silently subtract them from these conservative obligations.
3. Build serially, one worker on two CPUs, aggregate read/write pacing 64 MiB/s,
   4 GiB RSS. Proposed ceilings: train 14,400 s / 512 GiB logical source reads /
   2 GiB root; dev 1,800 s / 16 GiB reads / 128 MiB root. The exact root paths are
   in `RESOURCE_PLAN.json`. No image decoding into teacher tensors occurs, but
   this build **does** hash image files, read headers, deserialize raw logs and
   hash original opaque ego records. Do not call it zero-image I/O. The costs
   have not been measured for this full population.
4. Index only completed requests. Proposed train index: 3,600 s / 64 GiB reads /
   4 GiB RSS / 2 GiB root. Dev: 900 s / 4 GiB reads / 4 GiB RSS / 128 MiB root.
   `max_page_count` limits the DB to half the disk cap, reserving journal and
   publication headroom. Indexing does not read images or teacher tensors.
   Preserve the original teacher contract and original PCA; no refit.
5. Verify complete scene/log coverage and input identities, then index metadata.
   Keep failed and incomplete artifacts. There is no automatic retry, no
   automatic cap increase, no new teacher-bank build, and no training launch.

The existing build callbacks enforce root space, RSS, time and the shared
checkpoint/free-floor guard. The global ledger lock serializes writers. An
outer owned-child monitor is still required for non-cooperative imports or
long calls. Record logical bytes and observed wall/RSS, not wall multiplied by
workers. Existing completed request finalization streams an expected digest
instead of blindly writing a second full copy. Failed partials nevertheless
remain charged. Do not release their budget merely because a process stopped.

## Capacity estimate and stopping conditions

The actual 32-scene preparation plus index occupies about 879,160 logical bytes.
Simple extrapolation yields approximately 2.643 GiB train+dev, **not a measured
full-population result**. The 2 GiB train-preparation cap is only about 9% above
its extrapolated 1.835 GiB. The train index's 1 GiB database ceiling is about
31% above its extrapolated 0.764 GiB. Shapes, missing frames, distinct images,
per-log receipts and fragmentation can change both. The unused 2,576 MiB is
not silently assigned to a root. A cap hit produces preserved partial state
and a concrete replan; it does not justify trimming the data or repeated
unbounded rebuilds. Raw-log file size alone cannot guarantee the RSS bound.

## Pins and admission boundary

`SOURCE_PINS.json` binds clean metadata runtime
`/mnt/project/VLA-Drive-st-metadata-runtime-v2-20261017` at
`b805b00e80e1cc4e28f8e4a8f6ba7b79894fb9ab`. This is sufficient for dev PLAN but
not the pending train single-log ceiling change. Every new train runtime needs
its actual builder/dependency hashes in PLAN and registrations. No old runtime
that only understands v1 should write a migrated v2 policy. The plan's historical
`aggregate_preparation_plus_index_budget_bytes=4GiB` is explicitly
`PROPOSED_NOT_RESERVED`; the new actual storage admission must bind the reviewed
8 GiB package rather than misrepresenting that field as an active cap.

The initial `DEV_PLAN_ADMISSION.json` approves only a bounded dev PLAN. The exact
thin wrapper is being independently reviewed before dispatch; its final pin
supersedes any preliminary interpreter/launcher choice. Existing model,
provider/PCA, math and GPU qualifications are outside this narrow resource
review and are not re-approved here. No full build or formal temporal training
is authorized by this report.

### Narrow follow-up: full train ceiling

Lead supplied a second clean immutable metadata runtime at
`/mnt/project/VLA-Drive-st-metadata-runtime-v3-20261017`, commit
`ad1318328f81a0023490a99e1fece4a132b5a7c2`. It contains the explicit 192 MiB
single-log ceiling patch `fc6c6c70`; independent correctness commit `28830077`
reports eight targeted CPU tests passing. `TRAIN_SOURCE_PINS_V3.json` fixes its
actual source hashes. The train argv explicitly uses 192 MiB and one worker;
old defaults stay 128 MiB. This resolves the stat-only PLAN ceiling rejection,
and does not establish the memory cost of deserializing those logs.

The final plan wrapper must use the existing `ddp/bin/python` environment.
`DEV_PLAN_ARGV_V2.json` corrects the preliminary shell-interpreter choice
(`navsim/bin/python`); the preliminary argv/admission are retained as evidence
and must not be used to dispatch the final wrapper. Train uses
`TRAIN_PLAN_ARGV.json`. Neither command has been executed by this reviewer.
The final wrapper-bound admission is issued separately after the owner hands
off its source. Helper pins remain the separately reviewed original helpers,
not the unrelated new W-reconciliation controller.

The exact aggregate reservation uses full claim sizes, not merely current
precharged/observed bytes: **694,122,466,550 bytes existing**, rising to
**698,820,087,030 bytes** including retained administrative reservations and
the four proposed roots. This leaves **125,813,633,802 bytes** under the common
768 GiB cap. `RESOURCE_PLAN.json` records charged and reserved amounts
separately. Checkpoint obligations and the free floor are additional disk
requirements, not credits against that cache cap.
