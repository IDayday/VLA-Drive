# Minimal v5 single-ledger migration and writer admission

This is an implementation/dispatch plan, **not an implemented interface or
permission to run the full builder**. Lead remains the only dispatcher.

## Why direct v5 append is insufficient

`StorageBudget._locked` (lines201–223) requires the ledger core to match the
loaded policy and every existing ledger claim to appear unchanged in that
policy. Adding a claim through a new v5 means a still-running v4 writer will
fail the next lock acquisition because v4 lacks it. The current allowed roots
also exclude the six retained metadata pilot roots. Extending those roots or
adopting pilots changes core; ordinary `initialize()` rejects the old core.
The guard has **no safe explicit migration interface today**. Do not bypass
these checks or edit v4 in place. Do not start a separate uncoordinated ledger.

## Small implementation surface

1. Add an explicit migration function/CLI beside `cache_storage_budget.py`.
   Inputs are exact old-policy SHA, new-policy SHA, expected old-ledger SHA,
   a frozen pilot stat-inventory hash, new build registration hashes and a
   Lead-owned writer-quiescence receipt. It uses the **existing
   `cache_storage_ledger_v1/CLAIMS.lock`**, not a second global lock/ledger.
2. Lead verifies all v4 teacher/PCA/cache writers and pilot builders are terminal
   by actual process identity and complete/exit evidence, and prevents new v4
   dispatch. Merely no GPU activity is insufficient for CPU writers. Pure
   training cache readers continue unchanged; do not stop/restart them or edit
   their pinned policy/runtime. Current source usages call StorageBudget from
   `cache_teacher` and `spatial_probe`, not ordinary training target reads.
   Specifically inspect the launch/exit receipts and fresh PID/start-time/
   cmdline/cwd for `cache_teacher`, `pool_teacher`, `fit_teacher_pca`,
   `index_teacher_metadata`, and any wrapper registered for the v4 roots.
   Include the `storage_budget_real256_v1/cache`,
   `pca_fixed384_full_current_v1`, `pca_fixed384_full_current_v2` claims, all
   legacy declared writer roots, and six metadata pilot/review roots. A failed
   or interrupted but exited writer is quiescent; its files/reservation remain.
   A COMPLETE file does not override a live writer; an old PID alone is not
   evidence either way. Lead must record who owns future admission and check
   no queued automatic controller can restart an old writer. This reviewer
   did not perform that live process inventory and grants no quiescence PASS.
3. Under the same flock, verify old policy/core/claims exactly. Freeze the old
   ledger bytes into a new immutable checksum-bearing snapshot. Only allow an
   additive v5 migration: preserve campaign, state_dir, cap768 GiB, free floor,
   ledger reserve, every existing reservation and every existing claim,
   charged_bytes, observed_bytes and device. Put all new output roots under the
   existing allowed `/mnt/project/st-world-policy-20261016/caches` path; no
   allowed-root extension is needed for new output. The six already-existing
   pilots are outside allowed roots: append only those exact six roots in the
   explicitly approved core migration, not arbitrary `/audits` or
   `/mnt/project`. Do not silently bypass the allowed-root check. Append pilot
   legacy reservations based on the
   fixed inventory (15 MiB rounded total); source hashes/paths and retained
   versions stay unchanged. Reject paths overlapping existing cache roots.
4. Admit each new index/preparation claim only after its actual registration
   exists. Bind the metadata build-plan identity explicitly; do not call it the
   already-known final cache/index identity. Compute total package increment
   relative to the exact v4 snapshot under flock and require <=4 GiB, including
   adopted pilots, outputs, scratch/journals, receipts, logs and old partials.
   New claim amounts are parts of that package, not another package per build.
5. Validate the original campaign cap and disk floor plus current checkpoint
   reservations, then atomically replace the same CLAIMS.json with the new core
   and preserved state. Write an immutable migration receipt binding old/new
   raw hashes and the policy used by subsequent writers. This migration must
   be idempotent across snapshot/write/receipt failure windows: a saved pending
   record permits only exact completion of the same transition. Conflicting
   old/new hashes fail. Old v4 write admission must fail safely thereafter;
   no fallback to an empty ledger. No active source/policy file is rewritten.
6. Add a thin index-budget wrapper/hook, not a new training framework. Before
   opening any SQLite writer, acquire the actual v5 claim and precharge the
   complete bounded attempt including partial publication. The existing guard
   supports finite `reservation.writing(maximum_new_bytes)`. If holding this
   lock for a whole <=1,200-second metadata build, explicitly schedule it
   alone; other new cooperative cache writers wait, existing readers do not.
   Alternatively add tested bounded checkpoints without weakening precharge.
   The wrapper must honor current unknown-output/restart identity handling;
   never manufacture `identity.json` simply to adopt foreign files.
   For the first current-only attempt, a concrete one-GiB declaration is
   `output_bytes=0`, `staging_bytes=1056964608` (1,008 MiB),
   `metadata_bytes=16777216` (16 MiB). This deliberately reserves the entire
   bounded SQLite attempt as staging plus control metadata, rather than
   predicting tensor payload bytes. Bind its `cache_identity` field to the
   explicitly typed metadata build-registration identity; it is not the final
   SQLite index identity. Precharge the 1,008-MiB attempt before entry. On
   restart, the existing `reserve()` stat reconciliation works only for the
   known same claim; preserve leftover partials and reject a conflicting root.
   Root metadata/log writes also need precharge within the remaining16 MiB.
   The current-only root is flat and fits the existing `_inventory` interface.
   Do not use `_inventory` for future preparation's nested log shards: it
   deliberately rejects subdirectories. A future bounded-tree admission/check
   is a separate small implementation with explicit allowed suffixes, no links,
   bounded depth/file count and allocated-byte accounting. Future preparation
   remains blocked until that behavior is tested; do not weaken the current
   flat teacher-cache contract globally to make it work.
7. During build and verify, check process-tree RSS<=4 GiB, owned root
   **allocated** bytes plus all retained partials, time, and filesystem free
   floor including checkpoint reserves. The internal builder checks logical
   bytes, so the external physical allocation check remains required. Any
   failure stops only this owned finite job and leaves artifacts incomplete.
   A precharged one-GiB `writing()` context is **not** permission to ignore free
   space for twenty minutes. Add a resource check callback before each bounded
   commit/publication and at most every five seconds through long hash/parse
   phases, including free floor, remaining checkpoint reservations, root bytes,
   RSS and deadline. If the entire build holds the global flock, this callback
   must use its already-held state or read-only admission snapshot; it must not
   recursively acquire the same flock. An owned monitor may fail the finite
   builder when the callback cannot run. Do not merely rely on the current
   sampled `ru_maxrss`/logical-root checks.
   A failed attempt keeps its reservation and cannot be retried into new roots
   to escape the package cap. New complete metadata is independently verified
   before a reader identity is handed to future training qualification.

No new implementation PASS is implied by this plan. Once these small changes
pass independent tests, the current-only CPU task can be registered without
another GPU experiment. Future preparation needs its own actual implementation
and <=256-scene qualification before full-source admission.

## Required negative and fault tests

- Two migrations race: exactly one changes state; exact second invocation is
  idempotent, different transition rejects. Stale old-ledger/policy SHA rejects.
- Active/unknown writer, absent quiescence, or an unreceipted pilot builder
  rejects migration; no source process is signaled by the migration function.
- Core cap/free floor/state_dir change, old reservation removal, old claim
  alteration, lost charged bytes and overlapping/symlink paths reject.
- Pilot files outside prior roots are adopted only with exact bounded inventory;
  additional/changed files before admission reject. Pilots are charged once.
- Crash before/after snapshot, after atomic state write and before completion
  receipt: resume preserves the same new state, old snapshot and all claims.
- Existing v4 policy bytes stay unchanged; v4 writers fail after migration,
  new v5 writer succeeds, active read-only cache access is unaffected.
- Existing645.318GiB + **one** <=4GiB package is correct; per-root duplicate
  4GiB, another index, leftover partial and future request staging exceedance
  reject before a write. No failed-attempt reservation is silently released.
- Disk falls below floor + remaining cache and checkpoint commitments between
  admission and publication: no COMPLETE. RSS/deadline failures preserve partial
  evidence; a one-GiB root is not allowed to write two-GiB SQLite+scratch.
- Writer failure on metadata, teacher/image/tensor access attempts, changed
  original source metadata, wrong request/cache identity and non-current source
  cannot silently fall back to an eager/full build.

These are targeted admission tests. They do not require full models, teacher
forward, GPU/DDP or retesting the accepted metadata mathematical path.

## Actual CLI available now; invocation is not yet admitted

The following uses the real existing `build` interface and real source pins.
The destination is a **proposed new output**, not a claimed existing artifact.
Run only inside the implemented/reviewed v5 reservation wrapper; the raw command
alone is presently insufficient and was not executed by this reviewer.

```bash
cd /mnt/project/VLA-Drive-st-metadata-resource-review-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=30s 1200s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.index_teacher_metadata build \
  --cache-root /mnt/project/st-world-policy-20261016/audits/history_data/full_current_teacher_train_stage_plan_v3/cache/train/native384 \
  --expected-cache-identity d92dcced1f92106e0e10a361bc65ea67348be299f07b78e38b75ba973079aa56 \
  --expected-request-identity 2c107e58146482d3f3985d7803d52ecf64d2e3ce9939435d8fce86e24fcf6d6f \
  --contract-receipt /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json \
  --contract-receipt-sha256 a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32 \
  --output /mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1 \
  --max-scenes 101592 --max-disk-bytes 1056964608 \
  --max-builder-rss-bytes 4294967296
```

CPU affinity0,1 is a proposed example, not reservation of a busy host's cores.
Lead chooses two actually available allowed-host cores at dispatch. The reviewed
source must be an immutable integrated commit. At success `verify` takes the
actual new COMPLETE identity; no guessed future identity or imaginary command
is provided. Verify is CPU-only and sequential, under the same metadata IO/RSS
and time accounting. No training or teacher command is included.
