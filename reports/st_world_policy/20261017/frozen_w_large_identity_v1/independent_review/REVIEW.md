# Independent review: large immutable native run identity

**PASS_LARGE_RUN_IDENTITY_CODE_AND_BOUNDED_CONTROL** for `a84bfe40860bf6cc657c7527fcc73c5d7aa08545` (parent `cf5805016b5e23bdab9541c77b562ed5fda83dc3`). This resolves the 19,105,073-byte SP0 control-file rejection. It is not an actual W extraction, migration, GPU/resource lease, numerical validation or formal-training approval.

Reviewer: `/root/st_metadata_resource_review`. Isolated worktree `/mnt/project/VLA-Drive-st-frozen-w-large-identity-review-20261017`, branch `agent/st-frozen-w-large-identity-review-20261017`. Only owned reports and temporary CPU tests were written. No implementation file or live policy/ledger was modified. Exact code and evidence hashes are in `SOURCE_AND_REVIEW_EVIDENCE.json`.

## Correctness boundary

- The shared `metadata_storage.CONTROL_LIMIT` remains **16 MiB**. An independent test supplies the same >16 MiB valid document to the generic pinned reader and the dedicated run reader: the former rejects while the latter accepts. No signature, manifest, registration or unrelated control limit is enlarged.
- Only native `run/identity.json` receives the **32 MiB** limit. The exact 32 MiB valid-JSON boundary passes; one extra byte is rejected before parsing. Private regular-file, canonical-path, hardlink, size and version checks remain mandatory.
- Cold lookup reads and hashes the entire file. The original full semantic identity is checked before projecting fields, so omitted population data is still covered. A population edit with unchanged projected config/source fields and an old semantic digest is rejected. Duplicate keys and nonfinite JSON are rejected without entering the cache.
- The retained projection contains only immutable primitive values/tuples: full identity, canonical config digest, exact source/registration/Base lineage, formal-origin flag, raw SHA256, byte size and file version. `validate_signature` still compares these fields with its existing pinned config, registration and native receipt. Config-digest comparison preserves canonical config equality; it does not replace the full original run-identity hash.
- The locked LRU contains **at most eight** projections, with no population/config dictionary or JSON byte array retained. An independent 32-call/8-thread test performs one cold open and returns one shared immutable projection. Eviction forces full revalidation; an invalid evicted replacement is rejected.
- Hot lookup rechecks canonical path/private-file metadata and `(device, inode, size, mtime_ns, ctime_ns)`. A version change is rejected without rereading, including identical-content replacement. Cold opening uses `O_NOFOLLOW` and `O_NONBLOCK`; leaf-link replacement at open and parent-link replacement during parse both fail closed. These are file-version safeguards for the immutable shared-store contract, not a claim that a writable file can never be changed immediately after any completed check.

No storage-package cap, claim membership, migration, checkpoint reserve, model, inference path or W tensor code is changed by this patch. The affected typed-W and per-write checkpoint-floor tests pass. The prior independent storage review remains applicable; it is not reissued as a new live admission.

## Executed tests

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 timeout 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_spatial_w_large_identity.py \
  tests/st_world/test_spatial_w_storage.py \
  tests/st_world/test_spatial_probe_checkpoint_reserve.py -q
```

**80 passed in 7.67 s**, one existing pynvml deprecation warning. See `RELATED_TESTS.log`.

Using the same CPU environment, `pytest reports/st_world_policy/20261017/frozen_w_large_identity_v1/independent_review/test_independent_identity.py -q` produced **10 passed in 1.11 s**, no warnings. See `INDEPENDENT_TESTS.log`. These are additional negative/boundary cases, not repetitions of the whole previous migration audit.

## One actual SP0 control read

After source review, the existing bounded checker was executed once in this independent worktree. It has a **2 GiB RLIMIT_AS**, **180-second CPU/outer limit**, CPU affinity **0–1**, and no CUDA visibility. It reads exactly the run identity and four small control documents; it does not traverse the run or read native shards. The separate `source_start_commit` in its output identifies the implementation's starting point; `SOURCE_AND_REVIEW_EVIDENCE.json` binds the actual reviewed execution source `a84bfe40` and checker hash.

| Observed item | Result |
|---|---:|
| SP0 run identity size | 19,105,073 B |
| All five control documents read | 19,132,153 B |
| Run identity cold opens | 1 |
| Full cold read/hash/parse/semantic check | 0.259761 s |
| 100 hot stat/version calls | 0.006752 s total, no additional opens |
| Whole diagnostic wall | 0.297493 s |
| Peak RSS | 127,488,000 B (121.582 MiB) |
| Retained projection JSON | 636 B |
| Native model/image/tensor opens | 0 |
| Live policy/ledger reads or mutations | 0 |

The complete run semantic identity is `703ac3d75f4651a29688ed226de7ebf0593c12df359cc13086aa47106b442789`; raw SHA256 is `33b15316a564379f811bcef430e96842e1a13b366fe53212ab7344e5cdc12444`. Pinned CONFIG/REGISTRATION/Base binding and the actual native FORMAL5000 COMPLETE agree on lineage, config, source, formal origin and exact step. The raw digest is computed and recorded, while the semantic identity supplies the existing external binding. No native tensor hash was revalidated here.

Actual command (output retained in `REAL_METADATA_ONEPASS.json`):

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/frozen_w_large_identity_v1/check_real_metadata.py
```

The dedicated metadata reader can now validate this actual SP0 run within the registered CPU/memory scope. Its cost is a control-file lookup, not W extraction or training throughput. Full signature tests use synthetic inputs; actual cache-plan/native/image verification and exact fresh migration/extraction receipts are still required by the existing process. This patch supplies no new permission to use expired quiescence or admission evidence, change the 768 GiB campaign envelope, or launch training.
