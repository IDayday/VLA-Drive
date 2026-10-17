# Independent index-adapter resource review

The adapter's fixed resource and additive-storage boundary passes this narrow code/CPU-fixture review. **Current combined dispatch remains BLOCKED_SHARED_MONITOR_EXIT_RACE.** This review does not authorize an index run or issue a resource token.

Frozen owner commit: `15967d1de0615ed104a2c326a09b4ac7dd63d2ea`. Adapter SHA256: `e20b3f9e0317bd7b122b0fa9d74f99f8b6bc11cc12a598db70524bcb83bdbc0f`. Reused core: `a775a87532dab70a376cb64a7c0c19403e21e2f3a3b8040152fbefb8cebf10b9`. Source files were read without importing the real indexer/model or opening data payloads.

## Resource boundary

| Role | Logical read cap | Parent+child RSS | Wall + cleanup | SQLite/output+staging CLI limit | Inclusive claim |
|---|---:|---:|---:|---:|---:|
| dev |4GiB|4GiB|900s +40s|112MiB|128MiB|
| train |64GiB|4GiB|3600s +40s|2016MiB|2048MiB|

Both roles fix CPUs0–1, empty CUDA, one attempt and stdout/stderr at1MiB each. The SQLite page ceiling remains half the CLI root limit. Metadata reserves16/32MiB cover admission/START/RESULT/temporary publication; all files and retained partials are also bounded by the inclusive guard. Escaped pipe output and atomic staging are precharged with `json_write_bound`; excessive diagnostic output fails closed rather than enlarging the claim. Initial and final control writes use the same guard. There are no background counter-file writes during index inventory.

The imported core provides bounded pipe capture, owned-process observation/cleanup and wait-boundary accounting. The known leader-exit race can falsely report descendants after a leader that just exited. Its preserved combined evidence is113PASS/1FAIL with zero remaining owned children. No rerun is used here to erase that timing-dependent failure, and this report does not approve the core fix or active TRAIN terminal accounting.

The admission must span the full role wall+40s cleanup and be rechecked under the shared loader lock. A pinned CURRENT checkpoint receipt must retain2183GiB and a separate1TiB floor throughout its lease. The original metadata guard repeatedly checks the exact receipt, available disk, whole root and aggregate RSS. Shared quota has not been verified in this review; filesystem free space alone is not a quota claim.

## Same ledger, unchanged old reservations

`validate_additive_policy` requires every non-claim field and every old claim to survive exactly, allowing precisely the two registered index claims. It rejects another state directory, roots, cache cap, free floor, legacy reservations, changed old claim, extra claim or mismatched registration. The existing `StorageBudget._locked` then checks live ledger core and all admitted claims under the same `CLAIMS.lock`; `metadata_writing` durably precharges the full new claim before root creation or SQLite writes. Because this is a same-core declaration extension, no fabricated new core-migration receipt is needed. It is not a second ledger or permission to rewrite historical rows.

The new claims total2176MiB. The previously planned metadata total remains5620MiB/8192MiB including the existing separate4MiB build-control addendum. The768GiB overall cache cap and typed-W package stay unchanged. A new `audits/full_future_index_preparation_v1/...` control root is outside the historical4MiB `audits/full_future_metadata_preparation_v1` tree: it still needs an explicit administrative reservation. The reviewed preparation helper's1MiB inclusive allowance would make the conservative planned total5621MiB if granted; this review does not grant or pretend that extra allocation already exists. Repo source/tests are not cache claims.

## Actual completion schema

The wrapper correctly handles the indexer's real CLI projection, which has no `artifact.status`. It requires a newly emitted `st_teacher_metadata_index_v1` COMPLETE, exact projection fields, semantic identity, original source/contract/request pins, full scene count, null tensor-cache identity, legal hash/count/resource scalars and zero image/tensor opens/bytes. The actual SQLite source and content checking remain the pinned indexer's responsibility; this adapter does not rehash the large database. Request-only argv retains all1696dev or101592train scenes.

Both actual future populations must be bound before preparation. This review does not assert that current TRAIN is complete. If its primary terminal status is FAILED solely because of old accounting, the present adapter/preparation cannot silently reinterpret it as success; an explicit separately reviewed reconciliation is needed while preserving the original result and historical spent amounts.

## Tests and next dependencies

**71 small CPU tests passed**:53owner adapter tests plus18independent resource tests. The additional tests exercise every role's read/RSS/root/wall/cleanup limit, unchanged same-ledger core fields, atomic publication headroom, immutable START, and hardlink/atomic-replace rejection. Wall1.30s, peak RSS35,860,480bytes, process rchar3,109,729bytes, CPUs0–1 and2GiB address-space cap. No torch, GPU, real index, real ledger write or dispatch. Fixtures stayed below128MiB and are not committed.

Before any actual index admission:

1. Freeze the independently accepted shared-core exit-race fix and update the adapter's exact core pin; review that limited integration.
2. Obtain genuine complete train and dev inputs, plus the actual thin preparation/registrations/proposed same-core policy; preserve old failed receipts.
3. Explicitly account for the isolated index-control administrative namespace and include it in the metadata total.
4. Bind fresh checkpoint/floor commitments, actual allowed host/resources, exact code/helper/dispatch pins and per-role lease. Recheck live ledger under the original guard.
5. Only Lead starts a separately admitted one-shot role. Afterwards examine the real small COMPLETE/RESULT, final root size and terminal accounting; no success is signed in advance.

No need to repeat model/numerical audits or dispatch a GPU experiment for these metadata tasks. The active TRAIN controller and budget remain untouched.
