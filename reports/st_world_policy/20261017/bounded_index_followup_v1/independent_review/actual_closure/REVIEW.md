# Actual bounded metadata index closure

**PASS — the two approved metadata-only index invocations completed within their exact resource and identity limits.** This closes these two one-shot stages; it does not approve frozen-W extraction, teacher execution, new training, throughput claims or another index invocation.

The independent check ran from review commit `81dad039e1482cef0c28dda59a6654113e18ffe7`, using only Python standard-library reads. It passed **39 checks in 0.1528 seconds**, with 29,327,360 B peak reviewer RSS. `EVIDENCE.json` retains exact file pins, actual counters, reservations and checks; `CHECK.log` preserves stdout. No live policy, ledger or runtime was modified.

| Observed property | Future32 request index | Original PCA-fit64 metadata index |
|---|---:|---:|
| Original scenes | 32 | 64 |
| Metadata records | 294 | 957 |
| Complete database bytes | 249,856 | 1,155,072 |
| Final regular-file logical bytes, including controls | 258,543 | 1,183,507 |
| Final regular-file allocated/charged bytes | 266,240 | 1,196,032 |
| Root directory logical and allocated bytes | 4,096 | 4,096 |
| Final root-inclusive charged bytes | 270,336 | 1,200,128 |
| Inner guard observed peak, including temporary writes, before outer RESULT | 266,240 | 1,656,304 |
| Invocation seconds | 5.748660784 | 5.833587660 |
| Peak aggregate controller/child RSS bytes | 650,620,928 | 652,521,472 |
| Guard checks | 12 | 20 |
| Writer callbacks / precommit callbacks | 10 / 7 | 18 / 15 |

Each stage had a 32 MiB reservation, 28 MiB inner builder limit, 2 GiB aggregate RSS limit, 180-second invocation limit and CPU affinity `[0, 1]`. Both exited with return code 0, `automatic_retry=false`, CUDA visibility disabled, and `status=PASS`. Their timestamps prove serial execution. Both completed inside the independent approval and checkpoint-receipt validity windows. Final roots contain exactly the SQLite metadata DB, COMPLETE, admission, START, RESULT and empty lock file; no partial database or tensor output is present.

The final two-root total is **1,470,464 B**, calculated as `sum(max(st_size, st_blocks*512))` over each root and its files. The first review pass omitted root-directory allocation; `CHECK_FILES_ONLY_INITIAL.log` preserves that superseded measurement. The table and final evidence correct it. The ledger's `observed_bytes` values (266,240 / 1,196,032 B) are publication-time snapshots and are not labeled final directory usage. Final RESULT publication adds one allocated block after that snapshot; each claim remains fully precharged at 32 MiB. No reservation or cap is relaxed by this accounting correction.

## Identity and contents

The observed result hashes match Lead's supplied receipts:

- Request result: `54cf472cabc61234063ab9750d4bd73d3e420560a6cccafdb4c18278ce2963e0`.
- Fit result: `5508a350b2777d909dc83297e997f43c02b63108c23b766f02ead5428abb5373`.

Only these two small databases were hashed and opened with SQLite `mode=ro&immutable=1`. Every compressed JSON row was checked for bounded decompression, canonical serialization and row SHA. All scene rows exactly match the original request manifests in their original order; all content-addressed records match the original image-reference metadata. No referenced image was opened. SQLite `quick_check` passed on both.

The request index preserves request identity `ab5021926ee49d70d086869f7c65a0f888a240c55fda6e11b6d29987fb67325f`; it correctly has no teacher cache identity or shard receipts.

The fit index preserves original cache identity `fae6ed77041cdd318e647a9f44d907ffe63c62487805b0d683e9b247b6d7714d` and request identity `eda2256c3a4a63a9cf3d5a1db3a582ed68de6cacf5e6014a1e9e0f310a86cfe9`. All 957 records and all 30 original shard-receipt hashes agree with the old cache controls. The command is the pinned `index_teacher_metadata build` path; it indexes the existing native384 fit population and has no PCA fit or tensor transformation operation. Thus this operation did **not refit PCA**. This is a metadata/source-contract conclusion, not a new numerical validation of PCA tensors.

Both COMPLETE and returned artifacts report `image_files_opened=0`, `tensor_files_opened=0`, `tensor_disk_bytes=0`, and `training_admission=false`. The reviewer confirmed metadata-only command/source pins and output types. These are the executed builder's reported counters, not a new external syscall trace. The closure itself opened no images, raw logs, tensors or model weights and imported no model/runtime code. It did not read or hash the large full-current database. An initial overly broad AGENTS filename lookup was canceled after approximately 10 seconds without reading file contents; all subsequent checks used exact paths and the two small output directories only.

## Ledger and next migration handoff

The current ledger is:

`/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json`

Observed SHA256: **`7fcfc0c18fa69fc65a125ee36acc1204f9976e62926bede016269136f79cebbe`**.

Removing only the two newly admitted claim rows and serializing with the original ledger format reconstructs the exact pre-invocation SHA256 **`ddeff34b7c9dc09d44dc32d14f6ba876a6f6343bb3b279d8f14fd00d78bb85dd`**. Every previous claim, reservation, core field and migration-history entry is therefore unchanged, rather than merely having a similar total. The reviewer verified the ledger remained unchanged during closure.

The effective declaration policy is:

`/mnt/project/st-world-policy-20261016/resources/CACHE_STORAGE_BUDGET_POLICY_v7_index_followup.json`

SHA256 **`b7062c4da878519c435db2e88b899bd476e5cd6a0e9f16dca191fe61522bb831`**.

It retains the v6 core and the same ledger/lock. Two 32 MiB claims were admitted: `bounded-request-index-v1` and `bounded-fit-index-v1`. **Both must remain unchanged in the subsequent typed-W additive migration.** Do not use the earlier v6 ledger snapshot as the new migration baseline or release their unused reservation. Their exact rows and registration hashes are retained in `EVIDENCE.json`.

Metadata reservations, including retained pilots and these controls, remain **1,191,182,336 B (1,136 MiB) within the existing 4 GiB package**. All declarations plus legacy and ledger reservations total **694,096,380,774 B**, within the original 768 GiB cap. The 1 TiB free floor is unchanged. The executed commands used the exact pinned checkpoint receipt `1bf12212b29202cd3f5cf80e73425ab5d663c778e7ba35df43a83bb2c27289ad`, retaining 1,983 GiB of checkpoint obligations. The closure's observed free space exceeded even full 768 GiB cache capacity plus these checkpoint obligations plus the 1 TiB floor.

This handoff is evidence of completed stages, **not a reusable admission token**. Any typed-W migration needs its own exact policy, current ledger pin, fresh writer/quiescence facts and a checkpoint commitment valid through its planned window. It must retain the same global lock and both new metadata claims.

## Reproduce this limited closure

```bash
cd /mnt/project/VLA-Drive-st-index-followup-review-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 60s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/bounded_index_followup_v1/independent_review/actual_closure/check_closure.py
```

The checker writes only its sibling review evidence. It intentionally rejects an altered baseline ledger or changed inputs; it is not a live migration/build command.
