# Independent metadata storage implementation review

Reviewer `/root/st_metadata_resource_review`, 2026-10-17. Reviewed implementation
`6e20c8b6d9b883a21653b1f79dd39c132ca9bf2f`, isolated worktree
`/mnt/project/VLA-Drive-st-metadata-guard-review-20261017`.

**CHANGES_REQUESTED. No real migration, metadata build or training admission.**
This review follows the earlier resource proposal review; it now evaluates actual
code and CPU fault cases. It does not supersede the live receipt requirements or
claim the future preparation/index hooks have been integrated.

The original storage and owner tests were independently rerun: **75 passed,
5.79 s**, one existing pynvml deprecation warning. Three additional behavior
tests exposed three failures; full logs and the test are committed. All files,
ledgers, policies and simulated COMPLETE artifacts in these tests are temporary
fixtures. No real campaign ledger/cache was written, no image/tensor opened,
no source population scanned, and no GPU/model/teacher job launched.

## MG-F01: expired quiescence is accepted after an interrupted migration

Location: `metadata_storage.migrate_policy`, pending-path branch around lines266–280.
The no-pending path verifies quiescence time; the pending path loads its saved
old state without rechecking that time. Reproduction interrupts after pending
publication, advances time beyond the quiescence receipt while keeping checkpoint
commitments valid, then retries. It modifies the real fixture ledger to v5.

This is a substantive migration authorization failure: an old writer could have
become active during the delay. Pilot stat equality is not live writer evidence.
Recheck fresh quiescence before any old-to-new state transition, even when pending
exists. Preserve a separately justified idempotent completion-only path if state
is already exactly migrated; do not mutate/reapply using stale permission.
Refreshing expired receipts must be explicit and cannot bypass snapshot/hash
checks. The independent test requires rejection and byte-identical old ledger.

## MG-F02: pre-write checks ignore uncreated parent directories

Location: `MetadataWriteContext.check`, path validation around lines414–432.
A registered file pattern can name `log_shards/<hash>.json`, while no directory
pattern permits `log_shards`. If the directory does not yet exist, tree_inventory
cannot see it and the pre-write destination check succeeds. The later census
would fail only after the directory/output has been created.

Validate each planned ancestor relative to the registered root, including
nonexistent parents, against directory patterns and depth before IO. Existing
link/mount checks remain. Do not weaken tree_inventory's rules. The independent
test performs no such write: it requires the pre-write check itself to reject.

## MG-F03: missing callbacks are detected after builder publication

Location: `run_registered_metadata`, module execution around lines607–616.
The runner invokes the builder first and tests hook counters after return.
An unwired builder can publish COMPLETE and only then receive BLOCKED_STORAGE.
The committed actual index CLI/core at this reviewed commit contain no guard
hooks. Owner documentation correctly says integration is pending, but the public
runner must fail before invoking an old module, not after its writes.

The independent fixture simulates the same boundary: its unwired builder writes
COMPLETE; runner raises, but COMPLETE remains. Add preflight of an explicit
callback protocol capability and exact registered implementation source pins
before parser/build execution. Retain actual callback/commit counts afterward
as defense against missing runtime calls. Coordinate the capability name/source
manifest with index/future owners; this reviewer changes no production source.
Capability alone is not proof of correct hook placement: integrated owners still
need bounded read/hash/SQL callbacks and pre-write/publication calls reviewed.

## Accepted portions and remaining limits

The inspected implementation uses the existing flock and ledger, preserves old
claims/charges/devices and immutable v4 policy, retains snapshots/partials, counts
declared metadata plus adopted pilots in one four-GiB package, rejects a second
uninitialized ledger, binds actual registration bytes, and precharges a full
attempt before creating output. Existing tests cover race/idempotence windows,
package overflow, bad pins, checkpoint reserve, RSS, deadline and sticky errors.
These passing behaviors do not cancel the three concrete failures above.

Real Lead quiescence/process/controller evidence, quota/free space, checkpoint
commitments, metadata roots and runtime receipts remain NOT_VERIFIED by this
review. Callback checks are cooperative; external finite process-tree monitoring
is still necessary for blocked native calls. No GPU experiment is required to
fix or regress these three issues. Retest the independent cases against the
owner's narrowly revised commit before any actual migration/build.

Commands actually executed, sequentially, from the review worktree:

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_cache_storage_budget.py tests/st_world/test_metadata_storage_guard.py

env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 90s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_review_metadata_guard.py
```

The initial counterexample run also had three failures. The directory test was
then adjusted to expect a sticky resource failure at the correct enclosing
context boundary; its intent and production code were unchanged. Both private
logs are retained; the committed log is the final independent test version.
