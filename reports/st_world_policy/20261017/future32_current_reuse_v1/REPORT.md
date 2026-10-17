# Future32 → full-current cache key reuse (read-only diagnostic)

**Result: 336 / 480 present view/time slots match the existing current-cache index (70%).** Excluding time zero, 240 / 384 future slots match (62.5%). This is the fixed qualification population of **32 scenes from two logs**, including consecutive windows; it is not a representative estimate for the full training population.

| Time | Present | Truly missing | Current-cache key hits | Misses | Hit fraction |
|---|---:|---:|---:|---:|---:|
| 0 s | 96 | 0 | 96 | 0 | 100% |
| 0.5 s | 96 | 0 | 81 | 15 | 84.375% |
| 1 s | 96 | 0 | 75 | 21 | 78.125% |
| 2 s | 96 | 0 | 57 | 39 | 59.375% |
| 4 s | 96 | 0 | 27 | 69 | 28.125% |

There are **294 distinct image SHA + complete teacher-contract keys**: 189 are present in the original current index and 105 are absent. The 480 slots contain 186 repeated present keys. These repetitions arise from the supplied observation/time windows; no sample was added, removed, reordered or resampled. The diagnostic does not change missing-frame masks. There are no genuinely missing slots in this particular 32-scene population; the earlier independent 64-scene qualification had different coverage and is not substituted here.

## Exact inputs and integrity

- Future metadata root: `/mnt/project/st-world-policy-20261016/caches/future_metadata_qualification32_v2`.
- Future request identity: `ab5021926ee49d70d086869f7c65a0f888a240c55fda6e11b6d29987fb67325f`.
- Full-current index: `/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1`.
- Index identity: `326a3f37344b4526b8595954cb7d0b1dbb6eabdfdc836ea9582d2dc239dc725b`.
- Indexed current cache population: **101,592 scenes / 304,776 records**. This is the lookup source population; the diagnostic query population remains 32 scenes / 2 logs.
- Original cache identity: `d92dcced1f92106e0e10a361bc65ea67348be299f07b78e38b75ba973079aa56`.
- Teacher contract remains native384, FP32 compute/storage, 18×24 patches, 1,024 channels and original three-view order.

The script used the original `MetadataIndex` constructor: full SQLite SHA, original cache COMPLETE/identity/requests SHAs, immutable file versions and receipt identity all remained enabled. Each distinct hit was decoded through `MetadataIndex.row`, which checks the original indexed record checksum; `payload_key` used the actual image SHA and full teacher contract. Future COMPLETE file pins, the whole request semantic identity and raw-log provenance ledger coverage were verified. Raw-log **files** themselves were not opened. The read allowlist rejected image/tensor/raw-log paths and write modes during lookup. It allowed only the existing index and pinned control dependencies; SQLite was opened read-only/immutable by the original reader.

## Measured metadata costs and bounds

The PLAN was persisted before execution. Source base commit is `7cb76144cd4cc0bfa64fbc46284b0470f9a31dd5`; exact diagnostic/module hashes are in PLAN. The script was a new isolated file, without edits to the index/model/data implementation.

- Original startup hashes: **699,243,037 bytes**, **1.023647 s**.
- Full MetadataIndex startup: **1.024988 s**.
- **294** unique checksummed SQLite lookups: **0.050569 s** total.
- Diagnostic internal wall (after Python imports): **1.147940 s**. This excludes interpreter/import startup and is not teacher or training throughput.
- Process peak RSS: **629,989,376 bytes** (~600.80 MiB), below the 2 GiB limit.
- `/proc/self/io` delta: logical `rchar` **702,587,851 bytes**; physical `read_bytes` **710,639,616 bytes**. Both were below 1 GiB. Logical accounting includes small control/proc reads; physical accounting may include filesystem readahead.
- Two CPU affinity slots, no GPU, 180-second external timeout and 175-second internal stop, 16 MiB SQLite metadata budget, cooperative checks at every original hash chunk and each query.
- Images/tensors/raw logs opened: **0 / 0 / 0**. Teacher extractions and tensor disk writes: **0**. No policy, ledger, claims, model, source dataset or cache was mutated.

The PLAN's conservative read envelope was 1,024,564,736 bytes, including one MiB per distinct query and control slack. It is a pre-run allowance, not the measured read cost. No page cache was cleared or cold-storage claim made. Hash and lookup timings are one bounded invocation on this host, not a sustained storage benchmark.

## Interpretation for on-demand FP32 work

This sample supports reusing the existing current cache by exact image+contract key: a future observation often appears elsewhere as a current observation. It does **not** justify building another full tensor cache. Among this fixed sample's 294 unique images, 105 still need on-demand frozen-teacher computation. Within-sample repeats could additionally benefit from a bounded RAM cache, but residency, read amplification, eviction, target repeatability and teacher-compute throughput were not measured here.

A metadata hit does **not** prove the corresponding safetensors shard is currently readable/correct or that mixed cached/generated targets pass training qualification. Actual `TeacherCacheReader` first-access shard/record verification remains required when consuming tensors. No tensor checksum or numerical check is waived by this result, and no provider/temporal training scope is opened.

## Reproduction

```bash
cd /mnt/project/VLA-Drive-st-frozen-w-storage-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.diagnose_future_current_reuse run \
  --plan reports/st_world_policy/20261017/future32_current_reuse_v1/PLAN.json \
  --plan-sha256 0b8f52930879cf890952244e1f46f6aa98e803f66e0592109d024460da1787f5 \
  --output /absolute/new-report-root/RESULT.json
```

Use a new report output for a newly authorized invocation; immutable timing evidence is not overwritten. Source hashes must match PLAN. This command is a reproduction record, not authorization to spend another read budget automatically.

Six bounded synthetic CPU tests passed (3.68 s): repeat/missing counts, exact key matching, wrong lineage, corrupt record rejection and three resource limits. Their fixtures are not real-cache evidence; the actual result above comes from `RESULT.json` and `RUN.log`. No scientific or training PASS is inferred.
