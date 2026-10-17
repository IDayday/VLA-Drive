# Indexed provider / PCA wiring handoff (read-only audit)

Audited source: `1162b955b3bf9abb6728823a84a50c38e1a19bb8`, read using
`git show`; no real metadata, images, tensors, PLAN or GPU were opened. This is
a proposed integration list, **not** PROFILE/formal admission. Existing SMOKE
guards and the running immutable runtime remain unchanged.

## Reusable path and smallest integration

| Location at audited source | Current behavior | Proposed narrow change / checks to retain |
|---|---|---|
| `teacher_provider.py::validate_descriptor`, lines 149–226 | Eager `pinned(requests)`; exactly the qualified current cache is opened through `TeacherCacheReader`; creates `BoundedTeacherProvider`. | Add an explicit **new indexed descriptor schema**, retaining the old branch byte-for-byte. Bind request-index COMPLETE pin, request semantic identity, original raw file SHA, source index pins, original COMPLETE cache identities and per-rank metadata budgets. Instantiate `MetadataIndex` → `IndexedTeacherCacheReader` → `IndexedBoundedTeacherProvider`. Never substitute an index receipt for the old cache identity file. |
| `teacher_provider.py::prepare`, 256–280 | Opens each source and all requests before validating/publication. | New indexed preparation branch only reads verified bounded headers through those readers; validate the entire descriptor before immutable publication. Do not call `read_small()` on full requests/identity and do not materialize `Rows` via `list`, `dict`, JSON serialization or `deepcopy`. |
| `teacher_provider.py::create_provider`, 229–245 | Chooses the already qualified executor and compares exact execution/contract identity. | Reuse this constructor sequence and `FixedBatchDINOExecutor`; only the metadata-provider factory changes. Retain fixed8 source/backend/RNG/partial-LRU qualification, FP32 compute/storage, limits and exact executor equality. No new executor/backend. |
| `provider_pca.py::register_provider_scope`, 10–30; `validate_provider_scope`, 34–55 | Reopens eager fit cache and reloads eager provider preview. | Version an **indexed provider application scope**. Use the verified fit index's compact original `cache_header` for `teacher_channel_family`, retaining original PCA fit cache identity and contract, original raw identity/COMPLETE SHA pins, indexed descriptor/provider identity and basis SHA. The header without `records` must not be rehashed as though it were the original complete cache identity. |
| `teacher_pca.py::TeacherPCA.load`, 237–270 | Already dispatches explicit provider scope separately from cache scope. | Reuse its exact frozen basis/receipt/tensor checksum, train-fit, standardize/non-whiten and projected-contract checks. Delegate only the new scope validation. **No PCA refit, basis copy, dimension choice or old `target_cache_identity` alias.** |
| `train_world.py`, 115–130, 174–180, 324–327 | Metadata preview before run ownership; creates actual provider after backend flags; replaces `data.teacher`; restores provider identity before RNG. | Keep the ordering and collective failure handling. Reuse/explicitly close preview indices so repeated factory/PCA validation does not retain extra SQLite caches. Do not share SQLite connections across ranks or loader threads. Keep synchronous `loader_workers=0`, native checkpoint and cold-RAM rebuild. |

The existing adapter is already a `BoundedTeacherProvider` subclass, so
`STWorldTrainingDataset`'s type check accepts it. `read_scene`, all declared
future times, missing-frame masks, PCA application and target assembly need no
new algorithm. The dataset's population-sized bookkeeping below is a separate
remaining integration issue; changing only the factory does not solve it.

## Checks the indexed modules already supply

- `st_world_metadata_index.py::build_index` (349–405): streamed original
  request semantic identity, source/partition hashes, exact ordered scenes,
  times/views, raw-log ledger and role checks; original cache COMPLETE and
  sorted record coverage/offsets. A current-only source is not future coverage.
- `MetadataIndex.__init__` (456–502): COMPLETE/SQLite/raw dependency SHA checks,
  bounded header, verified ledger, read-only SQLite, bounded page cache and
  lazy `Rows`/`Lookup`. `check` and `decode_row` (504–524) reject file-version or
  individual-record changes. Initial construction **streams full metadata SHA**
  (database and original JSON dependencies), but does not deserialize the full
  population or read target tensors. This is startup IO, not per-step hashing.
- `IndexedTeacherCacheReader` (17–44) preserves original cache identity, contract,
  immutable source receipt checks and inherited actual tensor read validation.
- `IndexedBoundedTeacherProvider` (59–113) binds a distinct adapter/index/provider
  identity, original provenance, a combined per-rank metadata-cache cap of
  256 MiB, and existing bounded tensor LRU. It does **not** grant training
  admission. SQLite cache budget is not total process RSS; teacher/model,
  caller batches and dataset metadata are additional allocations.

## Remaining eager allocations and IO (not fixed by indexing teachers)

1. `teacher_provider.py::read_small/pinned` (24–47), `validate_descriptor`
   (163, 219–223), and `prepare` (263–269) parse complete requests/identity
   files. The current 32 MiB guard rejects oversized files; it must not simply
   be raised. Within the guard, multiple eager providers can coexist during
   preview/PCA validation/real construction.
2. `TeacherCacheReader.__init__` (231–254) loads all request scenes and identity
   records, recomputes `unique_image_records`, and creates locations/scenes
   maps. Replacing **every** factory/PCA eager construction matters; one hidden
   fit-cache construction reintroduces this startup peak.
3. `STWorldTrainingDataset.__init__` (42–58) parses the full source `index.json`
   and selected manifest, builds a token lookup and duplicate set; source
   lookup is constructor-local but `self.rows` remains resident. Lines 83–95
   parse an optional whole `HISTORY_MANIFEST` and keep full current-pin maps;
   lines 103–109 keep full ego-pin maps. Lines 138–151 embed both maps in the
   dataset/run identity, creating population-sized serialization as well.
   These need an explicit bounded **selected-input metadata adapter** preserving
   the original ordered rows, split/log joins and raw/semantic hashes, rather
   than interpreting the teacher-request index as an unverified source index.
4. Dataset startup (96, 108) hashes every selected current JSON and ego `.pt`
   file, including all 101,592 ego files if the full population is selected.
   This is a full **ego-label byte scan**, not DINO tensor deserialization.
   `__getitem__` (158, 166, 172) repeats consumed current/image/ego SHA checks.
   Preserve per-consumption integrity; any change to eager startup verification
   needs an explicit immutable admission receipt, not silent removal.
5. If history is enabled, dataset (112) reads the full history manifest once,
   then `HistoryReader.__init__` (251–263) reads it again and builds its index.
   Teacher metadata indexing does not cover history. The no-history temporal
   cell does not incur this optional path; do not choose a cell to avoid it.
6. `TeacherCacheReader::_verify_locked` (289–313), inherited by indexed sources,
   hashes a whole **accessed** safetensors shard on first use/file-version change.
   `verified_versions` persists tiny stamps when receipt LRU entries are evicted,
   so ordinary LRU eviction does not rehash unchanged shards. A new process
   rechecks its accessed shards. `read_image` (315–328) always checks record
   digest/header/dtype/finite values. No factory constructor hashes all DINO
   tensor shards; a pass touching all shards eventually verifies them all.
7. `TeacherPCA.load` (240–265) hashes/loads its small fixed statistics tensor at
   startup; this is necessary basis verification, not a full target-cache scan.
   Avoid redundant descriptor/fit preview construction; do not weaken basis
   checks to save this bounded work.

## Scope / sequencing boundary

`validate_option` (50–64) still requires SMOKE and synchronous loading.
`validate_descriptor` (163–179) binds at most 64 train scenes to the exact
completed real qualification request path/SHA; `configure_smoke` (284–318)
requires at most four updates and exact sample count. A full-population index
cannot be admitted by reusing that 64-scene evidence or by toggling a flag.
Future indexed SMOKE should retain these gates with its own new identity/PCA
scope. Larger PROFILE/formal population authorization is a separate Lead
decision, including exact original order/coverage and resource/resume review.

The upcoming 32-scene metadata qualification can establish writer/reference
agreement only. It does not establish full future population availability,
indexed trainer numerical/exact-resume equivalence, or full-scale throughput.
The narrow next implementation would be factory + explicit PCA scope + focused
indexed-vs-eager metadata behavior tests; dataset/input/history boundedness must
be completed or explicitly budgeted before calling the full training path
bounded. No implementation, new tests or runtime work was performed in this audit.
