# Independent finite build-controller review

**PASS for the exact one-shot dev metadata build.** The train branch has the same
reviewed control implementation but requires its own time-valid resource token
and serial Lead dispatch. No build was dispatched by this reviewer.

Owner commit: `26320138b9a4f880f0b7bd6a98a9796a9f067553`, clean worktree.
Controller path:
`/mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py`.
Controller SHA:
`866c35bce29b2215def6f1fea215dd75d7a62dc650a39d15c6e861f170ec7216`.
The executable must use that exact owner path, not a relocated copy.

## Narrow correction and tests

FBC-01 was corrected with a finished flag after successful session cleanup.
A later receipt error no longer attempts to reap/signal the same finished
session again. Poll failures now retain exception type/message; final malformed
JSON remains a failure with its original exception type. The original owned
session implementation, builder/runtime, data and limits did not change.
Initial 29-pass/1-failure evidence is retained in `TEST_OUTPUT.txt` and commit
`9b60ad61`; it was not replaced by the successful log.

Current-source validation: **31 passed in 2.62 seconds**, no skips:
21 owner cases plus 10 independently written cases. Seven independent cases
spawned only their own tiny CPU process and exercised success, sampled RSS,
read budget, elapsed wall, stale counter, malformed counter and stdout overflow.
Cleanup returned no live owned processes; the malformed-counter case propagated
the original JSON exception after cleanup. Additional independent cases covered
symlink/content rejection, atime-only stability and JSON escaping/atomic-headroom
precharge. No real metadata builder, cache, ledger write, GPU, model, image or raw
log payload was used by these fixtures.

## Boundaries actually enforced

The controller verifies exact PREPARATION, role registration, new policy,
completed native migration, source/runtime commit, original monitor helper
hashes and current host policy. It uses existing `run_registered_metadata` and
original callback guards. Parent/child inherit CPUs 0/1; child CUDA visibility is
empty. No optimizer or tensor extraction path is introduced.

Dev: one worker, 1,696 scenes/16 logs; 1,800 seconds plus at most 40 seconds owned
cleanup, process-tree RSS 4 GiB, 16 GiB parent+child proc `rchar`, 128 MiB inclusive
cache root. Train is fixed independently at 14,400 seconds, 512 GiB reads and
2 GiB root. Builder source I/O is paced at 64 MiB/s. Sampled RSS/read checks are
not a cgroup and may detect a limit crossing after a bounded polling interval.
The child uses a 128 MiB lower `rchar` cap to retain parent overhead margin;
parent additionally checks the actual combined count. `rchar` is logical syscall
I/O, not measured physical disk bandwidth.

START and RESULT are immutable under the same ledger lock, with no auto retry.
The child holds the registered ledger-writing context and invokes its original
storage callbacks. Its mutable COUNTERS write is guarded and read under a
separate common counter lock; parent does not deadlock waiting for the ledger
lock held by the builder. Parent pipe buffers are bounded to 1 MiB each. Complete
JSON escaping and temporary-file overlap are precharged using the existing
`json_write_bound`, not the raw pipe byte length. Old PLAN/control evidence and
failed partial files remain counted. No cache replacement/deletion is allowed.

The outer loop only terminates the new verified child session via the existing
pidfd/unreaped-anchor implementation. GPU exports and unrelated training are not
members of this owned session. All actual build failures remain failures; the
supervisor does not manufacture a COMPLETE receipt from a partial artifact.

## Actual migration and accounting

Actual native migration receipt:
`/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/MIGRATION_01fc719a84318e75e7482a20f0b041ccc2a9954623b2b9e715e5081a3eb44053.complete.json`,
SHA `d53193e57b383e0299387da61f6262c69fc23352616638d0280cdcec7d4c18cb`.
It reports MIGRATED and the exact new policy/CP identities. Live ledger SHA before
build is `caebf94cd9b5d303feea6cfd18aebc8f5915eeebeac81d7230ed90cad38123ba`.
All eight previous admitted claim rows equal the immutable old snapshot. Two new
prep declarations are in the policy; their admitted ledger rows are created by
the real guarded write path, not fabricated by this review.

The separate 4 MiB administrative addendum has SHA
`2946083d7ed65aa931bcdf12bee333b340d9e0c9ec742509314ebfed9de61ddf`.
It preserves the old 128 MiB administrative reservations and gives final planned
metadata 5,620 MiB/8 GiB including the future indexes. Index claims remain pending
real request identities. Current prep+administrative allowance is 3,444 MiB.
The original common 768 GiB cap, typed-W package, original metadata parent and
same ledger/lock remain unchanged.

Fresh dev admission observes free shared space and retains the full 2,183 GiB
checkpoint obligation plus independent 1 TiB floor. Quota is not verified.
No already-written checkpoint space was subtracted. The receipt is valid through
15:20:52 UTC; the shorter build token below controls this invocation.

## Actual token and command

`DEV_BUILD_RESOURCE_ADMISSION_V1.json` SHA:
`35ff4275bea3cbd8ac579e8a7ce00241580d5eaf7d39c8d18eb990b65ea5018a`.
Latest start: 2026-10-17 09:51:46 UTC. Expiry: 10:36:46 UTC.

```bash
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py \
  --admission /mnt/project/VLA-Drive-st-sp0-dev-plan-resource-review-20261017/reports/st_world_policy/20261017/full_build_admission_v1/DEV_BUILD_RESOURCE_ADMISSION_V1.json \
  --admission-sha256 35ff4275bea3cbd8ac579e8a7ce00241580d5eaf7d39c8d18eb990b65ea5018a
```

This approves no train auto-chain, SQLite index before its actual request ID,
new teacher tensors, GPU extraction, model training or scientific/formal result.
Metadata build reads real raw logs, image headers/bytes and ego-file checksums;
its no-tensor scope must not be misreported as zero source/image I/O.
