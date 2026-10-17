# Frozen-W native run identity admission correction

Source: `cf5805016b5e23bdab9541c77b562ed5fda83dc3`. Isolated branch `agent/st-frozen-w-large-identity-20261017`. This is an engineering control-file correction, not model/extraction qualification or formal admission.

## Finding and minimal repair

`spatial_w_storage.validate_signature` read native `run/identity.json` with the shared `metadata_storage.CONTROL_LIMIT`. In this exact source that shared limit is **16 MiB**, not 1 MiB. SP0's actual file is **19,105,073 B** because the run identity includes training population information. Its valid native FORMAL5000 endpoint was therefore rejected before extraction.

Only native run identity now has a **32 MiB** ceiling. All other control limits, source/PCA/model contracts, checkpoint inventory and existing tensor verification remain unchanged. The first lookup checks canonical location, a private regular file, exact byte bound, before/opened/after/final versions, full raw SHA256, original full semantic identity, schema, config, source, registration and Base lineage. The JSON is parsed once; config equality is checked through its canonical SHA256.

A process-local, locked LRU keeps at most **8 immutable small projections** containing hashes and `(device,inode,size,mtime_ns,ctime_ns)`. It does not retain population/config dictionaries or JSON bytes. Each later call rechecks the path and file version. A changed version fails closed without rereading; a fresh process or an evicted entry must perform full validation again. `O_NOFOLLOW` and `O_NONBLOCK` protect the open boundary. Same-byte inode replacement is also rejected while cached.

This reuses the repository's existing canonical semantic hash and duplicate/nonfinite JSON rejection. It introduces no new training/API/backend semantics and no disk cache. The complete frozen-W package, single-ledger quotas, native shard hashes and payload checks remain the responsibility of their unchanged existing paths.

## Current tests

`CPU_TESTS.log`: **73 passed**, one existing torch `pynvml` deprecation warning, **7.54 s**. Tests use synthetic temporary controls only. They cover >16 MiB identity success, exactly one cold open, bounded small immutable projection, hot stat-only checks, max-eight LRU and eviction, semantic/schema/source/registration/Base/config failures, oversized/empty/nonregular/link files, read/parse replacement or growth races, version changes after caching, and all existing W storage package/migration tests.

The old test fixture used to rewrite an identical shared train/dev run identity. It now checks and preserves the existing immutable file. No production gate was relaxed to accommodate a mutable fixture.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_spatial_w_large_identity.py \
  tests/st_world/test_spatial_w_storage.py -q
```

## Actual SP0 control evidence

`REAL_METADATA.json` and reproducible `check_real_metadata.py` record one allowed read-only check of the actual run identity plus four small control documents. The check inherited a hard 2 GiB address-space ceiling, CPU affinity 0–1 and 180-second timeout. No native payload was opened.

- Actual run identity: `703ac3d75f4651a29688ed226de7ebf0593c12df359cc13086aa47106b442789`.
- Raw identity-file SHA256: `33b15316a564379f811bcef430e96842e1a13b366fe53212ab7344e5cdc12444`.
- Full run identity, pinned CONFIG/REGISTRATION/Base binding and actual native5000 COMPLETE all agree.
- Run identity cold opens: **1**, bytes **19,105,073**.
- All control bytes read: **19,132,153**, below the 32 MiB diagnostic budget.
- Cold complete hash/parse/semantic validation: **0.257125 s**.
- 100 hot file-version checks: **0.006272 s total**, no extra identity opens.
- Retained projection serialized size: **636 B**.
- Peak RSS: **127,508,480 B**. Total diagnostic wall: **0.287677 s**.
- Model/image/tensor opens: **0**. Ledger, policy and existing asset mutations: **0**.

These are control lookup costs, not W extraction or training throughput. No real W signature existed for this endpoint yet; the actual check therefore exercises the dedicated helper and all relevant original lineage controls, while synthetic tests exercise complete signature admission. It does not claim native payload revalidation or grant launch approval.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/frozen_w_large_identity_v1/check_real_metadata.py
```
