# Independent correctness review: immutable future finalizer

**PASS_CODE_SYNTHETIC_ONLY** for exact commit
`1ce3a44134bee0aa52fd1f6b659103e989a0bbc2`. Reviewer:
`/root/st_calibration_independent_correctness_v2`, not the implementation author.
No implementation files were changed. No real finalization, actual ledger
mutation, GPU call, raw/image/teacher/weight read or dispatch occurred.
This is not an independent resource approval or permission to start a build.

## Findings and evidence

| Requirement | Actual call-chain review and current behavior evidence |
|---|---|
| Failed original build remains failed | `_terminal` accepts the actual `st_full_future_build_invocation_v1` / `FAILED_NO_AUTO_RETRY` lineage only with nonzero child return, cleanup0, no automatic retry/teacher/training/science authorization. Historical admission/preparation/controller/storage command are pinned and cross-bound to original plan/root/run. Five independent coherent-repin negative cases reject changed role/success/retry/cleanup/teacher scope. Old failure bytes are never rewritten. |
| Original source and scene order | `_lineage`, `source_with_pins`, exact ordered IDs/selected rows, `_validate_shard`, raw provenance and final parser coverage bind the population. Per-scene RAM records replay in original source order, even when log grouping differs. Same-header original serialization is byte-identical; all current/future row values are preserved. The new ledger path intentionally changes top-level manifest identity. |
| One source-shard read | `_read` obtains a bounded no-follow version snapshot, hashes and parses the same bytes; counter test observes each shard once. Later passes decompress per-scene RAM records and perform no file reads. Initial/final stat census and per-read version detect source changes. |
| Compression correctness | Exact stored length and SHA; bounded decompression, EOF, no trailing/unused/unconsumed bytes; truncated data, expansion bomb, digest change, concatenated zlib members, duplicate JSON keys and nonfinite constant all reject. |
| No false completion | Original writer's lock is acquired nonblocking/read-only; a held lock rejects. Bad source/frame time, postwork census change and late terminal mutation reject before FINALIZATION_COMPLETE. Existing atomic writer retains partials; injected request/final-marker failures resume deterministically and preserve original evidence. |
| New receipt consumed strictly | `_validate_full_population` explicitly branches to `validate_completion`, preserving original current/frame/population checks. New receipt filenames/schema, hashes, lineage, count and input header are checked. Independent wrong request SHA/identity/count/header/filename cases all reject. Legacy route remains accepted by the unchanged legacy checks. |
| Actual registered runner route | The existing storage runner calls `future_finalization` with `finalize-existing`, flat output and callback checks. The actual synthetic dispatch test passed on stable local fixture storage. Production guard checks were not bypassed. |

`test_independent.py` adds17 negative cases. The author's23 tests were actually
rerun, not inherited from a report. Initial execution: **39 passed,1 failed**.
Only the failing test was rerun, with identical source and a local `/tmp`
fixture: **1 passed**. Therefore40 unique cases have current passing behavior
evidence; it would be inaccurate to report a single clean40-pass first run.
Both complete logs are retained.

The initial failure was `BLOCKED_STORAGE: pilot inventory changed before
adoption`. The project shared filesystem first reported the24-byte synthetic
`audit/pilot/old.sqlite` as512 allocated bytes, then4096 bytes with unchanged
content/inode/mtime/ctime. `SHARED_FS_FINDING.json` pins both observations.
Strict migration correctly refused the changed inventory. Moving only that
synthetic test to local `/tmp` reproduced a valid dispatch without weakening
any production code. Future real migrations must still use a fresh stable
inventory. This fixture issue is not hidden as an automatic pass.

The first runner attempt failed before collecting tests because the script
path lacked the repo in Python's import path. Its traceback is retained;
only the independent runner's `sys.path` was corrected, not implementation.

## Commands and measured costs

From this isolated worktree:

```bash
taskset -c 6,7 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python reports/st_world_policy/20261017/future_finalization_correctness_independent_v1/run_bounded.py
taskset -c 6,7 env REVIEW_SINGLE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python reports/st_world_policy/20261017/future_finalization_correctness_independent_v1/run_bounded.py
```

These are administrative synthetic tests, not real-data launch commands.
The runner uses CUDA_VISIBLE_DEVICES empty and a180-second alarm. Environment:
Python3.9.25, pytest8.4.2, `/root/miniconda3/envs/navsim/bin/python`.

- First process18.9545s; import startup3.1537s /132,211,701 logical bytes;
  test window37,174,312 logical read bytes /2,303,098 written bytes;
  process total169,386,013 logical bytes; peakRSS615,706,624 bytes.
- Single-test process4.0786s; startup3.0976s /132,212,853 logical bytes;
  test window3,753,508 read bytes /140,259 written bytes;
  total135,966,361 logical bytes; peakRSS605,622,272 bytes.
- Combined measured wall23.0331s; test-window reads40,927,820 bytes and
  writes2,443,357 bytes. Each process remains below the separately allowed
  256MiB startup-inclusive read bound. Test windows remain below64MiB read/write
  allowance. Reviewer cost recording does not substitute for resource approval.

## Boundaries before actual use

Real1176-shard replay was not run. Real PLAN/index/shards were not opened by
this reviewer. Validating a synthetic schema does not assert that every actual
shard is legal or fits the compressed RAM budget. Lead still needs an exact
new signed registration/output claim and separate resource review/finite
controller. Old failure evidence and old source claims remain retained.

This review covers the new finalizer, its existing metadata-storage route and
new provider receipt branch. A later index-preparation routing change requires
its own narrow review. No scientific identity, loss, model, optimizer, teacher
function or training approval is changed by this code. No formal launch is
approved here. Source/file hashes and detailed boundaries are in REVIEW.json.
