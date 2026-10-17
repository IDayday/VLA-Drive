# Bounded future metadata implementation and PLAN-only handoff

2026-10-17 UTC. Based on `400a5607` in
`agent/st-bounded-future-provider-20261017`; follows proposal `eaabb607`.
The live `8c1e3f86` teacher/student runtime and controllers are unchanged.

**Implementation/CPU behavior: PASS. Real future preparation: NOT_RUN. Full
population: WAITING_INPUTS. Teacher/student/formal admission: NOT_GRANTED.**
No real raw log, image, GT tensor or teacher tensor was opened by this task.
The tests below use synthetic files. Only source-control JSON, known file stats
and a new private 32-scene PLAN were inspected/generated.

## Implemented path

`tools/st_world_policy/prepare_future_metadata.py` exposes actual `plan` and
`build` commands. It reuses original source population validation, the original
trusted raw-log unpickler, camera path resolution and exact original target
selection algorithm. It does not create a trainer/service or instantiate a teacher.

- `plan`: binds the original current population/selection and its raw hashes.
  Qualification uses leading contiguous windows from the first two source logs,
  restored to original source order, at most 256 unique scenes. Full scope requires
  the explicitly declared entire original scene count; it cannot subset a full run.
  File stats estimate raw-log comparison reads before any raw/image open.
- `build`: requires a separately issued immutable registration, pinned by SHA,
  matching PLAN identity/SHA, scope and metadata-only permission. It checks code
  hashes, original source, partition, current/ego pins and selected row order.
- `TargetLogLookup`: one token map/timestamp vector per loaded log, same five
  times `[0,.5,1,2,4]`, 50 ms tolerance, nearest/tie/current-frame rules as
  `prepare_history.build_target_scene`. Missing physical frames/views retain the
  original `None` semantics; missing computation is not fabricated as absence.
- `ImageMemo`: per-log, bounded 64 MiB accounting; repeated unchanged resolved
  images reuse their original SHA/header. Changed file versions fail explicitly.
  Actual image content is hashed on first access; no decoded image/feature cache.
- Per-log shards: immutable, complete selected log subset, raw-log SHA, original
  selected/current/ego pins, all frame records, image versions and content identity.
  A restart rechecks raw bytes/source records/image versions before reusing a shard.
- Finalizer: streams original-order scene rows with one log in memory. Canonical
  identity and sorted indent=2 JSON match the existing metadata representation.
  Interrupted temporary files and completed shards remain; COMPLETE is published
  only after exact full *declared scope* and hash checks. A qualification COMPLETE
  is not full temporal coverage or teacher-cache COMPLETE.

`st_world_future_provenance.py` implements the new small immutable raw-log ledger.
The request binds path, raw SHA, semantic identity and explicit schema. Admission
opens and verifies the ledger, original source/partition/registration/selection
identities, every raw-log SHA field, exact registered log order/counts and per-log
ordered token hashes. It never infers future coverage from the current-only cache.

The narrow optional `st_world_metadata_index.py` change adds the ledger dependency
and rejects mismatched coverage or mutated ledger bytes at build/open/access.
Legacy manifests have no new log-coverage requirement and keep their previous
validation path. The 64 KiB header guard was not raised. A synthetic 1,176-log
ledger exceeding 64 KiB retains a request header below 4 KiB and verifies correctly.
New ledger requests are intended for this verified indexed path; no legacy
teacher writer, shared provider factory, PCA loader or trainer was modified.

## Resource and interruption behavior

Hard accepted configuration maxima are 2 workers, 64 MiB/s aggregate metered IO,
4 GiB process RSS, 2 GiB new preparation metadata and 128 MiB per raw-log file.
The read-byte/deadline limits are explicit; the prepared qualification uses 8 GiB
logical reads and 900 s. Metadata writes share the read pacer's schedule, in at
most 1 MiB write chunks. Existing control-metadata validation reserves its known
logical byte reads before calling the reused verifier. Raw/image/current/ego
reads are individually metered. Reader/writer rates measure logical file bytes,
not guaranteed physical shared-storage traffic or page-cache misses.

Disk admission includes existing partial/orphan files. Over-budget/time execution
returns PARTIAL (CLI exit 3), retains completed shards and does not publish COMPLETE.
Data/identity failures return nonzero (CLI exit 2), without implicit resampling,
mask changes, file cleanup or tensor generation. The RSS check is performed at
read/write/record boundaries; real raw-log deserialization peak must still be
qualified. An individual trusted pickle allocation is not a claim of an OS-level
RSS sandbox. The proposed real run remains externally limited to two CPU cores.

The independent index allowance is another <=2 GiB. **Combined new preparation
plus index <=4 GiB is a proposed registration, not a reservation/permission.**
A storage-ledger integration/review remains required. No existing tensor caches,
checkpoints or original selection files are changed. No full temporal build is
launched by preparing a PLAN.

## Focused tests

Actual final command used CPU affinity `0,1`, empty CUDA visibility and
OMP/MKL/OpenBLAS threads 2:

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_prepare_future_metadata.py \
  tests/st_world/test_teacher_metadata_index.py \
  tests/st_world/test_metadata_contract_pin.py -q
```

**53 passed, one existing pynvml warning, 5.08 s.** This is 24 new future tests,
24 existing index tests and five contract-pin tests, not a sum of old executions.
Complete log: `evidence/future_metadata_preparation_v1/focused_tests.log`.

Coverage: original versus optimized exact frame/reference/missing rows; current
required, ties/duplicate/nonmonotonic timestamps rejected; repeated-image memo;
streamed bytes/canonical identity; interruption after first log and between final
request/COMPLETE; reused source mutation; byte/disk/rate/worker bounds; denied
unregistered builds; ledger SHA/source/log/token coverage and post-index mutation;
1,176-log bounded header; legacy index/provider and contract-pin regressions.
The first local test collection had an incorrect fixture import, fixed before
execution; it produced no source-data work. These tests do not substitute for
real image/raw-log equivalence or throughput qualification.

## Actual fixed PLAN, no data scan

Final private path:
`/mnt/project/st-world-policy-20261016/audits/bounded_future_metadata_plan32_v3/PLAN.json`

- SHA256: `1f1493ccd6996a8eb56a124a52f25d581b6ec13105c581c2e8e9b241177aa722`.
- Identity: `3965a384e4b4daf5c1fede62a3007aedda53e3a24994dda132a7e0b52f999254`.
- 32 scenes, two logs, original index order; original population still 101,592
  scenes / 1,176 logs with the exact hashes in the preceding proposal.
- Raw-log stat sizes: 1,412,161 + 822,124 bytes. Reference plus new implementation,
  each hash + deserialize once: **8,937,140 bytes estimated raw-log reads**.
- 480 scene/time/view references maximum, not unique images or computed tensors.
  Image read costs have not been measured; the 8 GiB hard budget still applies.
- Audit hook forbade `.pkl`, image, `.pt` and `.safetensors` opens: 0 attempts.
- PLAN preparation only: 4.929316 s, peak RSS 715,608,064 bytes. This is not future
  processing speed. No registration/build/teacher dispatch occurred.

Earlier PLAN v1/v2 are preserved and are superseded by v3's exact code pins;
those revisions changed implementation guards, not scene selection or results.
The actual argv, source hashes, resource limits and code SHAs are in committed
`evidence/future_metadata_preparation_v1/PLAN_PREPARATION_RESULT.json`.

## Lead follow-up commands and gates

Verified help:

```bash
python -m tools.st_world_policy.prepare_future_metadata --help
python -m tools.st_world_policy.prepare_future_metadata plan --help
python -m tools.st_world_policy.prepare_future_metadata build --help
```

The final PLAN already exists. Run its exact recorded `argv` only to verify
idempotent plan reuse. Build is intentionally unavailable until Lead issues a
registration with schema `st_future_metadata_registration_v1`, self identity,
`plan_sha256`, `plan_identity`, phase `data_preparation`, matching
`approved_scope=qualification`, and all three GPU/teacher/training authorization
flags false. No such registration has been created in this task.

After independent review and shared-storage accounting, Lead can execute:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  -m tools.st_world_policy.prepare_future_metadata build \
  --plan /mnt/project/st-world-policy-20261016/audits/bounded_future_metadata_plan32_v3/PLAN.json \
  --registration "${REGISTERED_METADATA_RECEIPT:?Lead registration required}" \
  --registration-sha256 "${REGISTERED_METADATA_SHA256:?Exact receipt SHA required}"
```

The same command resumes verified shards; it does not issue approval or delete
partials. The existing raw reference helper must be run on the same locked 32
scenes under the **same combined 8 GiB / 900 s accounting**, then every scene row
must match. The new request has an intentionally different provenance identity.
Do not equate whole-file hashes across distinct provenance headers.

Only after that qualification may Lead register a complete-population plan/build.
Factory integration needs new indexed request/provider identities and an explicit
new PCA channel scope to the unchanged basis; no fake old cache identity. Formal
scope, exact-resume qualification and resource approval remain separate gates.

## Source pins

| Artifact | SHA256 |
|---|---|
| `prepare_future_metadata.py` | `0ccb79700eb3adfca315e2ae7822837a4c67bfe86031f472712e84e173362321` |
| `st_world_future_provenance.py` | `387371e4d75434cbcaee11d50af20a441b78b26b93f4a340b3f140302c3a5df1` |
| Updated metadata index | `bf31af21e40278de2714f987ffc2c1830e1b5dc0f59ac25ad604a6147ad7cbe2` |
| New future tests | `c158616242299bbde8c95509e5946a0496bc7b68e15623174f5163f975a92e7d` |
| Final complete test log | `7505cc623f97f390f79dc345eb45998b3f9c20ca01711692da0fc4af5e71f6d9` |
