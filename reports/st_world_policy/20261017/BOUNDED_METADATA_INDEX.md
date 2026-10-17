# Bounded teacher metadata index: implementation and narrow qualification

Date: 2026-10-17 UTC. Isolated branch: `agent/st-bounded-future-provider-20261017`.
Work started at `2e84a5f3bb4eb5ebeafa23cfbde5e8486dd53ffa`. The active native
teacher/student qualification runtime remains frozen at `8c1e3f86`; this work
does not alter that runtime, either controller, teacher execution, trainer,
loss, sampler, PCA basis, manifests, or any training scope gate.

**Metadata-only parity passed. Full temporal population remains WAITING_INPUTS.**
No GPU, teacher extraction, image read, tensor read/build, training, or formal
dispatch was performed. No current-only cache was relabelled as temporal coverage.

## What changed

Only new owned modules, tests and this evidence were added:

| File / interface | Actual behavior |
|---|---|
| `starVLA/dataloader/st_world_metadata_index.py::build_index` | Stream the exact original sorted `atomic_json` files; recompute original canonical identities and raw SHA256; validate every scene using the existing validator; validate exact source records/order, COMPLETE coverage, teacher contract, role and three lineage hashes; publish an immutable metadata SQLite sidecar. |
| `MetadataIndex` / `Rows` / `Lookup` | Read-only, process-owned indexed lookup; streaming scene/record iteration preserves original order. No all-population Python dict/list. Source SHA is checked at open; source/index file-version stamps are checked on access; row hashes and bounded decompression are checked on each row. |
| `starVLA/dataloader/st_world_indexed_provider.py::IndexedTeacherCacheReader` | Subclass of the existing reader. Original cache identity and ordinal-to-shard/row mapping are unchanged. Inherits the actual shard receipt SHA, first-access tensor SHA, tensor header, per-record digest, finite/dtype/grid checks. Corrupt stored data never falls back to compute. |
| `IndexedBoundedTeacherProvider` | Subclass of the existing provider. Reuses its source/executor checks, all-times assembly, true missing-source masks, RNG isolation, tensor LRU, and identity-only cold rebuild. Only metadata containers change. New provider execution identity binds the exact metadata sidecars; old PCA/provider identities are not reused. |
| `tools/st_world_policy/index_teacher_metadata.py` | Real `build` / `verify` CLI; strict pinned small contract receipt, explicit population and disk/RSS bounds. No trainer or executor construction. |

The source cache path requires `COMPLETE`; request-only indices have their own
`st_teacher_metadata_index_v1` receipt, **not a teacher cache COMPLETE receipt**.
The latter can describe valid missing teacher records, but cannot claim those
records were computed. A missing valid target still blocks/raises through the
unchanged provider; only genuinely absent physical source frames retain false
masks. Dataset consumption still loads all four future times before the existing
time-pair/patch request sampler.

The index supports only the repository's sorted, indent=2 `atomic_json` layout.
It rejects external compact/reformatted JSON rather than silently `json.load`ing
a huge file. Every row is parsed with stdlib JSON; duplicate keys, bad semantic
identity, truncated arrays, row/header oversize, nonfinite JSON, wrong source,
coverage reordering and incomplete receipts are rejected. Source files are not
rewritten to satisfy the format.

Canonical JSON **metadata rows only** use lossless zlib level 1 to keep the
sidecar small. Original metadata content/identity and teacher tensors are not
compressed, quantized, or converted. Decompression output is bounded to 64 KiB,
and rejects incomplete streams, trailing compressed data, and oversized output.
The request-only record table is kept directly in canonical payload-key order,
avoiding a second copy of the full unique-record table. Source cache records keep
their original integer ordinal for exact shard offsets.

## Resource boundaries and measured pilot

Builder: explicit maximum 2 GiB per sidecar directory (including orphan evidence),
4 GiB process RSS; SQLite page count reserves half the directory budget for the
journal/publication window, 8 MiB suggested page cache, no SQLite mmap. Header/row
size is bounded; the largest full metadata object is never decoded. Partial DBs
remain diagnostic evidence. A published index is reused only with identical
dependencies/contract and sufficient requested budgets. No cleanup is automatic.

Reader: default 16 MiB metadata allowance per connection, half allocated to the
SQLite suggested page cache; no decoded-row LRU and no mmap. Decoded rows are
bounded to 1 MiB and slices to 2 MiB. The provider rejects combined metadata
allowances above 256 MiB per rank. These are metadata allowances, not a claim
that Torch, caller-retained batches, teacher weights or the entire process fit
in that amount. SQLite `cache_size` is its documented suggested cache limit.
The original source reader's small per-shard version-stamp map and bounded
receipt LRU remain in place; their size is bounded by the registered shard
population, rather than forcing multi-GiB shard rehashing after every LRU miss.

Actual final pilot: private
`audits/teacher_metadata_index_pilot_v5/{run_metadata_pilot.py,RESULT.json,run.log}`
under `/mnt/project/st-world-policy-20261016`.

| Actual source | Scenes | Unique image records | SQLite bytes |
|---|---:|---:|---:|
| `history_data/train256_current_cost_v1/native384` | 256 | 768 | 983,040 |
| `history_data/smoke_train64_v1/TEACHER_REQUESTS.json` | 64 | 957 | 638,976 |

- All current256 scene rows, ordered image records, shard locations and receipt
  hashes match the original `TeacherCacheReader` exactly, not sampled rows.
- All future64 scene rows and repeated/reversed ordinal queries match original
  metadata. The request has 192 current image hits, 765 future misses against this
  current256 source, and three truly absent future-view records.
- Python audit hook forbade image/tensor opens: **0 opens / 0 blocked attempts**.
  Teacher calls: 0. Tensor bytes written: 0.
- CPU affinity `[0,1]`; OMP/MKL/OpenBLAS threads 2; GPU visibility empty.
- Elapsed 1.804741 s; process peak RSS 624,246,784 B (includes Torch and the small
  original-reader parity reference); combined metadata cache allowance 33,554,432 B.
  This is metadata preparation/verification, **not teacher or training speed**.

For the already registered 101,592-current-scene population, simple linear
extrapolation from these small indices is 390,113,280 B current + 1,014,294,528 B
future metadata, or about 1.31 GiB combined. **This is an estimate, not a full
build result or capacity qualification.** Real full future deduplication, paths,
page fill and build journal headroom require the actual manifest. Aggregate
current+future+temporary/orphan use must remain within the independently reviewed
2 GiB campaign budget; per-index acceptance alone is not aggregate admission.

Earlier v1–v4 pilot evidence is preserved. One v4 metadata check overlapped an
earlier CPU suite briefly (each configured for two threads); the final v5 pilot
and final test suite were run serially with a shared two-core affinity. Only v5
is cited for the bounded final evidence; no GPU/runtime changes occurred.

## Tests and corruption behavior

Final command (CPU affinity restricted to the first two allowed cores):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_teacher_metadata_index.py \
  tests/st_world/test_provider_scope_boundaries.py \
  tests/st_world/test_ondemand_teacher.py \
  tests/st_world/test_teacher_cache.py -q
```

**73 passed, one existing pynvml deprecation warning, 9.88 s.** This includes 24
new metadata tests and 49 relevant existing provider/cache/scope cases; previous
test totals are not added. Full log and the real metadata result are copied to
`evidence/teacher_metadata_index_v1/` beside this report.

Covered: strict format/hash rejection, duplicate/oversized/truncated metadata,
lineage required with zero reuse sources, exact source row/order/shard mapping,
post-open original/index mutation, wrong/corrupt/incomplete publication, source
tensor corruption without recompute, preserved true missing masks, missing valid
labels raise, source role/contract mismatch, cold identity resume, indexed-only
bounded source requirement, summed per-rank budgets, no partial-population
publication, repeated build reuse, receipt-window fault/orphan preservation,
real CLI, bounded metadata decompression, and existing scope rejection.

Fixture tensor comparisons are behavior tests only. Real teacher numerical
qualification and student exact resume of **this indexed adapter** are NOT_RUN.
No previous raw fixed8 PASS is silently promoted to an indexed trainer PASS.

## Actual full-population prerequisites

Read-only stat and bounded tail inspection found:

- Full current cache:
  `audits/history_data/full_current_teacher_train_stage_plan_v3/cache/train/native384`:
  `identity.json` 145,979,873 B, `requests.json` 170,157,016 B. The cache identity
  is `d92dcced1f92106e0e10a361bc65ea67348be299f07b78e38b75ba973079aa56`.
- `audits/history_data/full_current_population_v1/train/TEACHER_REQUESTS.json`
  exists, but explicitly has `times_s=[0.0]` and
  `source=original_current_json_exact_t0`. It is **not** a full temporal request.
- The corresponding train/dev `HISTORY_MANIFEST.json` files are absent in those
  checked roots. Full future/history coverage was not found there. This is a
  statement about inspected paths, not a claim to have searched every filesystem.
- Actual current source `student_train_v1/index.json` is 9,549,650 B; its current
  dataset validation remains unchanged. Existing small future/history assets are
  train64/dev16. No large raw metadata file was deserialized in this task.

Required next input: legal full ordered future requests for times `[0,.5,1,2,4]`,
bound to the unchanged source/partition, plus original selected-scene current/ego
pins. History cells additionally need genuine complete causal history metadata.
Generating these requires the existing real preparation process; this metadata
index does not fabricate them or convert current-only requests into futures.

## Exact minimal Lead follow-up points (not implemented or enabled here)

1. `teacher_provider.py::{validate_descriptor,prepare,create_provider}`: add an
   explicit indexed-metadata descriptor branch. Pin each metadata index identity
   and receipt SHA, original request/cache identity and raw SHA. Build
   `IndexedTeacherCacheReader` and `IndexedBoundedTeacherProvider`, not eager
   `TeacherCacheReader` / `deepcopy(full_manifest)`. Preserve all fixed8 actual
   qualification, backend, executor/source, budget and source role checks. An
   approved full-current source is a different source from the 256-scene qualified
   diagnostic source: record that change explicitly; do not rewrite the old plan.
2. `provider_pca.py`: explicitly register a new provider channel scope pointing
   to this provider identity and the **same frozen PCA basis**, original fit
   cache/COMPLETE and raw channel family. Do not use an old target-cache identity
   or refit statistics. The actual fit cache metadata is small and can retain
   the existing verified fit-reader path.
3. `STWorldTrainingDataset`: existing explicit provider injection already accepts
   this subclass and required `manifest`/`scenes` lookups; original selected rows,
   source/current/ego checks, targets and sampler remain. Do not convert lazy
   `manifest['scenes']` back to `list`/JSON or hash it by full serialization in the
   factory. HistoryReader still uses its original metadata path; assess a real
   full history manifest before claiming its memory cost is qualified.
4. Retain `loader_workers=0` and one process-owned index per rank. Existing
   collective data-error handling must wrap provider reads; worker pickle/fork is
   explicitly rejected by the indexed source reader. Resume preserves the exact
   index/provider identity and cold-rebuilds the tensor LRU, not a sampler cursor.
5. Scope/sample-budget changes remain independent review work. This commit does
   not allow PROFILE/FORMAL, pick a temporal/W winner, or dispatch a teacher.
   Actual indexed real-teacher/student checks and resource review precede such
   admission; existing immutable-cache paths/output identities remain untouched.

## Evidence card and source pins

Local chain:
`st_world_targets.py::{validate_target_manifest,unique_image_records,TeacherCacheReader}`
defines original semantic identity, sorted image record order, strict COMPLETE,
and per-record tensor checks. `st_world_ondemand.py::BoundedTeacherProvider`
defines all-future assembly, strict miss behavior and RNG isolation.
`st_world_dataset.py::STWorldTrainingDataset` owns source/current/ego/PCA checks;
`st_world_history.py::HistoryReader` is the existing causal history interface.
Original contract sections 5.3/6 require source/time/grid/dtype/role identity and
unchanged time-pair/patch semantics. None of these files changed.

Primary API evidence:
[Python SQLite](https://docs.python.org/3/library/sqlite3.html) documents the
process-owned connection, parameter binding and streaming cursor interface.
[SQLite cache/mmap pragmas](https://www.sqlite.org/pragma.html) describe suggested
page-cache limits and disabling mapped database regions.
[SQLite immutable URI](https://www.sqlite.org/uri.html) requires truly unchanged
files; this adapter separately pins SHA and rejects file-version changes.
[Python zlib](https://docs.python.org/3.10/library/zlib.html) documents bounded
decompression and EOF/trailing-stream checks. No package installation/upgrade.

| Artifact | SHA256 |
|---|---|
| `st_world_metadata_index.py` | `a404d79f657abffefe9635374d8a5714f10878da286f54e588be6b32613a7dc4` |
| `st_world_indexed_provider.py` | `95a011b022542a38009e572d149bb4eb40263c1cd283877ff27f580a9c126f85` |
| `index_teacher_metadata.py` | `b3f9516b3cd5b936d06e2b589d0766519d153dcb85969d5209e8009bfb226b47` |
| New focused test | `0fbc1028fe1096256a740753c6027a80e04a957d74091dc59f9ac0e1e353251c` |
| v5 `RESULT.json` | `d8be67861ca7e5330ca669793be1f928b4094520d90b36f6e46729d97199a154` |
| v5 complete test log | `cd41d894e07cefd05cda46300c26b57581794a64edb26f9f7bb8dc8b04fb67ed` |

## Executable metadata-only commands

Run from this isolated source or the subsequently reviewed integrated commit;
these commands do not launch training. Existing v5 output is reusable only with
its exact inputs. Change output identity for different source files.

```bash
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/root/miniconda3/envs/ddp/bin/python
ROOT=/mnt/project/st-world-policy-20261016
$PY -m tools.st_world_policy.index_teacher_metadata --help

$PY -m tools.st_world_policy.index_teacher_metadata build \
  --cache-root "$ROOT/audits/history_data/train256_current_cost_v1/native384" \
  --expected-cache-identity 38691667383d9968956260a9cdb8895ed4392373d0d60bfb91395d1fbf96c9e3 \
  --expected-request-identity 23ab544bd08e6c278d1f74ba97da74780d766ebee06ad2a77db35cf408a792b3 \
  --contract-receipt "$ROOT/audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json" \
  --contract-receipt-sha256 a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32 \
  --output "$ROOT/audits/teacher_metadata_index_pilot_v5/current256" --max-scenes 256

$PY -m tools.st_world_policy.index_teacher_metadata build \
  --requests "$ROOT/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json" \
  --expected-request-identity eda2256c3a4a63a9cf3d5a1db3a582ed68de6cacf5e6014a1e9e0f310a86cfe9 \
  --contract-receipt "$ROOT/audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json" \
  --contract-receipt-sha256 a1de94f3e51b3e8ed050fa7f63b55d819cfa6453af7c1268b04d20c51f5e1d32 \
  --output "$ROOT/audits/teacher_metadata_index_pilot_v5/future64" --max-scenes 64

$PY -m tools.st_world_policy.index_teacher_metadata verify \
  --index "$ROOT/audits/teacher_metadata_index_pilot_v5/current256" \
  --expected-request-identity 23ab544bd08e6c278d1f74ba97da74780d766ebee06ad2a77db35cf408a792b3 \
  --expected-index-identity e5e347107b1a008238df57f4c86a9f25231159ea908a44d7e29f3642cc5085d1

$PY -m tools.st_world_policy.index_teacher_metadata verify \
  --index "$ROOT/audits/teacher_metadata_index_pilot_v5/future64" \
  --expected-request-identity eda2256c3a4a63a9cf3d5a1db3a582ed68de6cacf5e6014a1e9e0f310a86cfe9 \
  --expected-index-identity 022be4dbf8349189a3810bc56b5cc75e6e6190b835905498d89e9e7b154f4fb2
```

No full-future command uses a guessed path. The next legal full manifest must be
prepared/pinned first. No active runtime/source/controller was modified.

## Next preparation proposal only: bounded full future metadata

No source scan/build or code change for this proposal was executed. The actual
existing paths are `prepare_history.py::{build_target_scene,run}` and
`prepare_current_teacher.py::{source_population,plan,build_log,build}`.

Observed issue: `prepare_history.build_target_scene` rebuilds the raw-log token
map, timestamp array and monotonicity check for each selected scene and hashes
shared image files again through `image_reference`. `run` subsequently retains
all `history_rows`, all `targets`, a full final teacher list and history list.
The current-only preparation already has suitable registration, source/partition
checks, `ReadBudget`, at-most-two bounded workers, per-log immutable shards,
progress and resume. However, its finalizer also rebuilds whole-population
`rows`/`scenes` dictionaries; simply copying that finalizer would retain the
large-memory problem.

Minimal proposed future-only preparation:

1. Reuse `source_population` and the complete original ordered selection/pins,
   without `stable_selection` or a new training subset. Register exact source,
   partition, all four future times, physical timestamp tolerance and budgets.
   Current and ego files remain byte-hash-pinned; GT values are not decoded.
2. Reuse per-log shard ownership/resume. Read and SHA-pin each raw log once, build
   its token/timestamp lookup once, then call the same target-selection semantics
   for each scene: exact t0, future-only nearest frame within original tolerance,
   ambiguous tie rejection, all three views and genuine missing-source `None`.
   Add a per-log bounded image-reference memo using resolved path plus file
   version, but actually calculate the original image SHA before first reuse.
   A changed file invalidates the memo/registration rather than becoming missing.
   This is an explicit source scan and needs its own dispatch authorization.
3. Emit one immutable future metadata shard per log, preserving source-log SHA,
   per-image SHA, ordered source tokens and the original current-image mapping.
   Do not also produce history inputs unless the separately chosen cell needs
   them. A single log exceeding registered RSS must fail/replan, not silently
   drop scenes or truncate frames.
4. Finalize the original `st_teacher_requests_v1` file by streaming rows in the
   **original source-index order** from verified shards. Keep only a small
   token→shard/ordinal directory and bounded log-shard LRU, not global target
   objects. Compute canonical semantic identity and raw file SHA incrementally;
   publish atomically only after exact ordered coverage. Preserve the same
   `atomic_json` layout accepted by this index. Unfinished shards/temporaries
   remain resumable evidence; no automatic deletion.
5. Only after this genuine full future artifact exists should the metadata index
   build against it. An initial <=256-scene old/new preparation parity check must
   compare complete target metadata, source hashes, times/masks and order. This
   is not permission for full image scanning or new teacher execution now.

Measured metadata sizes (stat only, no history image/record payload opened):
train64 requests 502,318 B; 64 history-record JSON files total 936,677 B;
history manifest 38,221 B. Linear projections for 101,592 scenes are about
797,367,035 B for future requests and about 1.49 GB for history-record JSON,
before per-log receipts and filesystem overhead. These are estimates, not full
population measurements. Writing future requests plus retained log shards is
about 1.6 GB; propose a separate **2 GiB future-preparation metadata cap** plus
the **2 GiB aggregate index/temporary cap** (4 GiB reservation total, no teacher
tensor bytes). If history inputs are chosen later, register a separate metadata
budget based on the actual causal record population; do not hide those bytes
inside the future-only allowance. No such reservation or full build was made.

Initial future-preparation resource proposal: two CPU workers total, shared
read cap <=64 MiB/s, existing four-hour hard preparation window, <=4 GiB builder
RSS, no GPU. These are proposed limits for independent resource review, not
measured throughput or a claim the full scan will finish in that window.
