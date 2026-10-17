# Full train/dev future metadata: smallest bounded admission

Reviewed source: **`6faa9f8360c0a17e25f96399df03bdc162206ead`** in a separate worktree. Read current AGENTS and ST contracts, `provider_profile_upgrade_v1/NEXT_STEPS.md`, the affected existing builders, small receipts, and file stats. No PLAN, build, index, teacher, GPU, image/raw-log/tensor read, source mutation or ledger mutation was performed. This is a narrow source/data precheck, not a resource approval or completed full-population qualification.

**Conclusion:** keep the existing builder and indexing mathematics. The preparation path already supports the exact full train and dev populations. The missing step is a small, explicitly versioned metadata-only budget migration and four exact registrations, not another data framework or a TB tensor cache. A combined **8 GiB hard envelope**, with **5,616 MiB initially reserved including existing administrative and ledger reservations**, is a practical conservative proposal for independent resource review. No tensor cache, precision, checkpoint reservation or existing receipt needs to change.

## What already works and what must be bound

`prepare_future_metadata.py::source_with_pins/plan/build` checks the original source, index, partition, current-population registration/COMPLETE and per-record hashes; full scope rejects truncation (`scope=full_population`, max-scenes equals actual source count). It preserves the complete original scene order and log grouping. `TargetLogLookup.scene` retains times `[0,.5,1,2,4]`, three original views, 0.05 s tolerance, strict future timestamp, ambiguous-nearest rejection and missing-view semantics. Existing independent exact32 selector evidence remains relevant to this unchanged function; it does not prove unvisited full data.

Actual source bindings, verified from small COMPLETE/REGISTRATION files:

| Role | Scenes / logs | Existing student root | Existing current population |
|---|---:|---|---|
| train | 101592 / 1176 | `/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1` | `/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train` |
| dev | 1696 / 16 | `/mnt/project/ddp-foresight-artifacts/20260928/student_dev_v1` | `/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/dev` |

Both use the pinned `/mnt/project/VLA-Drive-st-history-population-agent-20261016/reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json`. Existing qualification locates raw logs at `/mnt/navsim/trainval_navsim_logs/trainval`, and sensor roots at `/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval` and `/mnt/navsim/trainval_sensor_blobs/trainval`. These are discovered paths, not newly verified raw/image availability. A fresh stat-only full PLAN must confirm every full log exists and respects 128 MiB per-log bound before build. Do not reuse the train source identity for dev.

The builder writes immutable per-log JSON shards, a streamed complete `TEACHER_REQUESTS.json`, provenance ledger, shard receipts and COMPLETE. Partial files are retained and counted; log-shard resume does not resample or drop scenes. Source reads include raw-log hashing/deserialization, image hashing/header inspection and opaque ego-file hash verification; it creates **no teacher tensor** and does not decode GT/ego tensors. Thus "metadata output only" does not mean "zero image/raw input IO". Both output and IO must be bounded.

`index_teacher_metadata build --requests ...` then creates a streamed SQLite request index without opening images/tensors. Bind the new request semantic identity, complete request/provenance SHA dependencies, the existing pinned teacher contract receipt, exact scene-count upper bound, and output namespace. Do not rebuild the already complete full-current train index. A dev *current source* index is a separate future provider need, not a prerequisite to produce dev future request metadata; do not borrow the train source or silently add it to this four-root budget.

## The 4 GiB limit is ours, not the user's

The 4 GiB value comes from `metadata_storage.py:24 PACKAGE_MAX`, `validate_package`, the old registered `metadata_package.maximum_bytes`, and `prepare_future_metadata.py:538`'s proposed aggregate field. It is **not** a user-imposed maximum. The user forbids few-TB cache expansion. A source-reviewed 8 GiB metadata-and-staging envelope is consistent with that constraint, subject to the resource owner's fresh storage/checkpoint commitments.

The live ledger snapshot SHA and small administrative receipt pins are in `PRECHECK_EVIDENCE.json`. The apparently conflicting numbers reconcile as follows:

| Existing reservation | MiB |
|---|---:|
| Four typed metadata claims: current index 1024; future32 preparation/request-index/fit-index 32 each | 1120 |
| Seven adopted metadata-pilot legacy reservations | 16 |
| Actual ledger metadata package commitment | **1136** |
| Qualification report administrative reservation | 64 |
| SP0 stage controls administrative reservation | 32 |
| Completed factory parent administrative reservation, including its PROFILE child | 32 |
| Effective existing commitment to preserve | **1264** |

The last 128 MiB are explicitly **not ledger claims**. They are bound by `resources/QUALIFICATION_REPORT_BUDGET_v1.json` and `ADMINISTRATIVE_METADATA_CARRYFORWARD.json`; carry them forward once, without omission or duplicate charging. My initial 1120 MiB count counted typed claims only and was incomplete; this report uses the reconciled 1264 MiB.

## Practical combined cap, including staging

Actual existing32 metadata sizes remain preparation 620,617 logical / 651,264 allocated bytes; request index 258,543 logical / 266,240 allocated bytes, including a 249,856-byte SQLite DB. Naive full train/dev final-file extrapolation is approximately 2.643 GiB. It is an estimate from two logs, not a capacity guarantee.

Do not use that final-size estimate as a peak reservation. `st_world_metadata_index.py:309` sets `max_page_count = max_disk_bytes // (2*4096)` to reserve rollback-journal/publication room. A 0.764 GiB projected train DB requires approximately 1.528 GiB of index allowance under the current implementation. Preparation also retains shards and partials. This explains why current 4 GiB minus retained 1264 MiB is a poor full-pipeline fit; it is not evidence that metadata needs TB.

Proposed inclusive reservations, awaiting exact independent resource admission:

| Reservation | MiB | Includes |
|---|---:|---|
| Existing effective baseline | 1264 | Existing typed claims, adopted pilots and administrative carryforward |
| Full train future preparation | 2048 | PLAN/registration controls, log shards, final request/provenance, all retained partials |
| Full dev future preparation | 128 | Same complete categories |
| Full train future request index | 2048 | SQLite DB, journal/staging, receipts and retained partials; DB itself capped at 1024 MiB |
| Full dev future request index | 128 | Same; DB itself capped at 64 MiB |
| **Initially committed total** | **5616** | All four roots plus baseline |
| **Combined hard maximum** | **8192** | One versioned package, not 8 GiB per role/stage |
| **Unallocated margin** | **2576** | No implicit permission to consume it |

Retain current per-builder 2 GiB output cap. The train preparation estimate is relatively close to that cap; an overflow must remain a recoverable PARTIAL, not trigger automatic quota growth, shard deletion or scene truncation. A further claim change would require exact evidence and approval, but there is no reason to engineer multiple rounds of artificial claim shrinking before a practical first admission. Keep checkpoint/free-space reserves independent; do not fund metadata by reducing them.

## Minimal code/control change required

1. **Explicit metadata envelope version/migration only.** Current `migrate_policy` rejects *any* change to the old package (`metadata_storage.py:215–217`) and rejects changes/removals of old claims. Merely raising `PACKAGE_MAX` would neither make the migration legal nor adequately bind it. Permit only an explicitly registered 4→8 GiB maximum change for this new metadata package version, pinning exact old policy/ledger and new policy, preserving baseline, every existing claim/reservation, tensor cap, free reserve, checkpoint commitments and allowed kinds. Keep old 4 GiB policies/receipts valid and immutable. No generic unbounded override or formal-training flag.
2. New full train/dev PLANs must report the **actual new aggregate admission**, not imply their legacy `PROPOSED_NOT_RESERVED` 4 GiB field is authoritative. Preserve old PLANs; make any new plan field/option explicit, source-pinned and checked against the registered envelope. Source selection and target serialization remain unchanged.
3. Issue exact metadata-only preparation registrations and typed storage registrations for the four roots. Pin source modules and callback dependencies, current population COMPLETE/registration/selection, ordered scene IDs, input roots, plan SHA and actual count. Index registrations are completed only after their real request identity/SHA is available. They need no new teacher extraction or training permission.
4. Execute only through existing `metadata_storage run` guarded builder dispatch. Single worker, bounded CPU, aggregate IO throttling, ≤4 GiB RSS, finite per-attempt wall/read budgets and exact output patterns. Preserve no-follow/hardlink, before-create/publication hooks, census checks, immutable artifacts and source pins. Resource review must set full-population read/wall budgets; the old 8 GiB/900 s exact32 budget is not a full-population registration. Read-only32 costs include substantial fixed source-control reads, so do not blindly multiply total32 reads by 101592/32.

Lead's proposed implementation is appropriately narrow: keep schema v1 capped at 4 GiB, introduce v2 with **exactly 8 GiB** and `previous_package_sha256`, and verify that digest by reconstructing the original v1 object (maximum 4 GiB, all baseline fields identical). Do not trust a caller-supplied previous digest by itself. Allow only this exact envelope transition; reject subsequent v2 envelope mutations while preserving ordinary already-supported additive registrations under its fixed cap. `spatial_w_storage.py:313` currently hashes the live metadata package, so its compatibility check must use the **verified original v1 digest** under v2. Preserve the frozen-W package's existing digest, limit, claims and signature checks; do not rewrite its baseline or treat metadata v2 as a general exemption. This preserves its historical source commitment without changing any W artifact.

This is design-level acceptance only, pending the actual fixed patch. Relevant negative tests: wrong previous digest, altered baseline, malformed/extra v2 fields, v2 limit or provenance mutation, wrong old policy/ledger, old claim edits, overflow including staging, duplicate administrative reservation, and interrupted/repeated migration. Keep v1 behavior and frozen-W checks intact. No raw/image/model tests are needed merely to validate that control-only patch.

This is a narrow budget/admission change; no change to selector, model, teacher, PCA, dataset, loss, trainer or FORMAL provider gate is needed to build metadata. Full dev loss-provider support and full training admission remain separate downstream work and must not block metadata preparation or existing trajectory/PDMS inference.

## Required acceptance, without redundant model work

After source/control review and resource admission, build train/dev under the one envelope (sequential is simplest; do not run two processes each assuming the entire shared IO allowance). Verify exact full scene/log/order coverage, five times, missing masks, original source/selection/ledger hashes, all metadata index rows and COMPLETE dependencies. Use a predeclared small cross-log selector comparison covering new logs/missing frames; do not claim full-population tensor equality from exact32 evidence. Do not recompute all teacher tensors or run model/GPU work as a metadata prerequisite.

A fresh stat-only PLAN can expose oversized/missing logs or per-log metadata bounds before expensive reads. An actual malformed log, changed source or missing required current view must fail with evidence, not be silently filtered. Source parser still holds the original ~101k-row control list; the existing 4 GiB RSS guard remains mandatory. Full-scale RSS, IO duration and final size are unmeasured in this review.

**Status:** existing full-population builder path usable; exact full registration and versioned budget migration **WAITING_IMPLEMENTATION/RESOURCE_ADMISSION**. No newly discovered target-semantic defect in this narrow read. No current approval to dispatch. No CPU test suite was rerun because this precheck changes no implementation; earlier fixed-source correctness evidence is retained, and the forthcoming migration patch needs only its affected negative tests (wrong old/new pin, old claim mutation, cap/staging overflow, duplicate administrative charging, old 4 GiB behavior).
