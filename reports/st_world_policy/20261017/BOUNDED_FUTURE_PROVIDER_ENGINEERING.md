# Bounded future teacher provider: engineering handoff

Task `st-bounded-future-provider-20261017`. Isolated source start:
`4ca99ec1427929b9b91e5c8db70a5314a301ec6e` on
`agent/st-bounded-future-provider-20261017`.

Status: **IMPLEMENTED; CPU behavior tests verified; real teacher integration,
GPU numerical equivalence, throughput and formal temporal launch NOT RUN.**
No active runtime, existing cache, trainer, data split, PCA basis, scientific
selection or resource approval was modified. No GPU, model weight load, real
teacher forward, new tensor cache, training or scoring was launched.

## Evidence card and adaptation

Question: supply the same immutable full-grid teacher targets without building
a multi-TB future cache or changing sampled scenes, future times and patches.
This is a storage/label execution adaptation, not a new scientific objective.

The original research contract is
`docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md` §§5.3,6:
retain native teacher coordinates and frozen PCA, allow exact image deduplication,
and preserve the adjacent future-pair and uniform patch streams independently
of label availability. The current feasibility evidence is
`audits/temporal_bounded_storage_feasibility_v1/{FEASIBILITY.json,METADATA_REUSE_COUNTS.json}`
under the private campaign root. It measures only smoke-population image-SHA
reuse, not full-population miss rates. The 768 GiB aggregate cache cap and 1 TiB
free floor remain unchanged. No new disk claim is made by this RAM-only module.

Inspected source at the fixed start:

| Source / symbol | SHA256 | Reused behavior |
|---|---|---|
| `starVLA/dataloader/st_world_targets.py`: `payload_key`, `TeacherCacheReader.read_image/read_scene` | `af56bf23d750e7465788669a466400aa852969d45ba2f3279f58aec9fd1d918e` | Complete immutable readers; image SHA plus full teacher contract; per-record checksum; channel-last raw targets |
| `starVLA/dataloader/st_world_dataset.py`: `STWorldTrainingDataset` | `edaa1d3619ccb6ea90a375c899dead0b44c53a94578fc3b478f58866be43cd8f` | Current-input boundary; full future fetch precedes pair sampling; PCA after raw fetch |
| `starVLA/model/modules/st_world/requests.py`: `sample_requests`, `gather_future` | `64c37c593e3602ab424ad5adb5c32513eddda487af860776bdf05337d6f48fb7` | Separate horizon/patch generators, no availability-dependent sampling |
| `tools/st_world_policy/cache_teacher.py`: `make_contract`, `preprocess_image`, native build loop | `2356f4c9314b379c590ff596d9ad1dfee56f0a451be600bf6a6d9fd79829c877` | Exact existing source validation, RGB/PIL resize/normalization, frozen encoder, pooling and storage cast |
| `starVLA/model/modules/foresight/dinov3_target_encoder.py`: `DINOv3TargetEncoder` | `709d5522e707eb97b1b576918816f2ef6ddb1a5e380e079abbe459f135f737a8` | Strict existing DINO weights, five excluded prefix tokens, eval/frozen FP32, autocast disabled |

The teacher is the already locked timm DINOv3 asset revision
`30c1109559f65dea34316b0d4842d35c5771fe11`, weight SHA
`45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`.
No teacher or architecture was chosen by this change.

Primary API sources checked before implementation:

- [PyTorch v2.5.1 `torch.random.fork_rng`](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/random.py): saves CPU and explicitly named device RNG and restores them in `finally`. Local installed source SHA256 `3cb3d532c854f65f29e3e5ea0dc1a878abfb44a740a1d0eab4e35bbb530227ab`. Explicit `devices=[]` on CPU avoids implicit enumeration; Python/NumPy need separate save/restore.
- [Python 3.10 `OrderedDict`](https://docs.python.org/3.10/library/collections.html#collections.OrderedDict): reorder and oldest-entry removal support the bounded RAM LRU. LRU is an execution choice, not a claim from a representation-learning paper.

New adaptation: an immutable manifest identifies requests while complete existing
caches and a separately bound synchronous executor supply individual raw records.
A new provider identity names this combination. It never claims that an old
current-only cache suddenly contains future records or has a new COMPLETE state.

## Implemented modules

`starVLA/dataloader/st_world_ondemand.py`:

- `BoundedTeacherProvider` validates complete source readers, exact teacher
  contract, source student/index/partition identity and split. A future image
  can reuse a different scene's current record only through identical source
  image SHA and complete contract; timestamp proximity is insufficient.
- `read_scene` has the existing current/future tensor layout. It fetches all
  declared future times, before any request sampling, and leaves physically
  missing sources exactly zero/false as the original reader does.
- A valid source missing from RAM/disk calls a **bounded batch** executor or
  fails. Failure is never turned into a false mask, skipped scene, random
  replacement or fake zero target. A corrupt existing record raises and is
  not quietly regenerated. Generated native records may not hide failure with
  false masks.
- Every requested unique record is delivered to its original time/view slots;
  repeated views or images cause one generation. No time/patch generator is
  passed to the provider. Global CPU/Python/NumPy RNG is restored on both
  success and failure of the synchronous callback.
- RAM retains independently owned raw record tensors, never slices retaining
  an entire returned batch. No tensor files are written. `state_dict` contains
  immutable identity only; cold resume clears recomputable RAM, not the
  trainer's consumed cursor or RNG. It does not checkpoint fake cache coverage.
- `max_cached_bytes`, `max_scene_bytes`, `max_batch_images` and
  `max_manifest_bytes` are explicit, positive and identity-bound. These are
  **tensor payload and encoded manifest limits**, not an RSS/teacher activation
  guarantee. The caller must separately budget source-reader metadata, decoded
  pixels, model/activations, returned batches and process overhead. At most one
  executor call runs in this provider; its output is released before the next.

`tools/st_world_policy/ondemand_teacher.py`:

- `FrozenDINOExecutor` is a concrete batch implementation using the existing
  verified model/preprocessor. No optimizer/backward/loss is constructed.
- Construction and forward preserve Python/NumPy/CPU plus the explicitly chosen
  CUDA device RNG. This includes temporary random model initialization before
  strict weight load, not merely deterministic eval forward.
- Parameters/buffers remain frozen/eval FP32 with version checks. TF32 must
  already be disabled; the adapter refuses incompatible flags instead of
  changing a live trainer's flags. Precision, SDP flags, deterministic settings,
  software/teacher/transform identity and implementation hashes are bound.
- Direct native SP2/SP3 and direct pool3 SP1 contracts are supported. Legacy C1
  and a source-cache-derived SP1 identity are explicitly rejected for generation.
  Reusing their exact immutable records remains legal. A derived-cache contract
  cannot silently be relabeled as direct teacher computation, even when shapes
  match. Supporting that selected path would need an explicit derivation adapter
  and a new audited identity.
- It returns raw teacher outputs, all true native masks and declared stored
  dtype. PCA/standardization stays in the existing downstream label path.
  No approximate values, interpolation, compressed targets or student features
  are introduced.

## Deliberate integration boundary

This implementation is **not yet a selectable trainer backend**. No change was
made to shared files owned by the Lead. Minimal integration is explicit:

1. Add a resolved provider-kind configuration and registration artifact for the
   fixed full future manifest, current-reader identities, raw contract, execution
   identity and RAM/resource limits. Source/precision/role checks stay mandatory.
2. Permit explicit provider injection in `STWorldTrainingDataset` in place of
   constructing `TeacherCacheReader`; validate the provider identity, full scene
   population, source pins and role. Do not change `__getitem__` target/PCA logic.
3. The initial synchronous path requires **`loader_workers=0`**. Both provider
   and executor reject non-owning thread, forked process or DataLoader worker.
   The current `ThreadPoolExecutor` prefetch path is therefore incompatible;
   no CUDA teacher may be silently created in that thread. This serialized
   execution change must be profiled and registered, not described as free.
4. Retain the existing `train_world.collective_call` around data/teacher fetch
   before requests and backward. It already distributes Python exceptions;
   a finite process-group timeout is still necessary for a killed/hung process.
   The new module creates no collectives or generic daemon. It does not make an
   indefinitely stuck external executor interruptible.
5. `TeacherPCA.validate_channel_scope` currently accepts immutable cache roots,
   matching `target_cache_identity`, source `channel_family` and identity file
   hashes. A new provider scope artifact/validator must admit the new provider
   and the unchanged raw channel family/basis explicitly. **Never pass an old
   current-cache identity as the provider, remove checks, or refit PCA.**
6. Checkpoint native training identity binds the new provider/state identity;
   do not store RAM tensors or producer lookahead as consumed cursor. Reload the
   same manifest/contract/execution and let the cold provider recompute requests.

A later request-aware optimization could fetch only the already-sampled pair,
but that would require changing the dataset/trainer/gather protocol. It is not
implemented or claimed here. This all-four adapter is the smaller, clearer
first equivalence reference and does not remove unused teacher work on misses.

## Tests and actual evidence

Executed in the installed ddp environment, CUDA hidden, CPU thread limits two:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout 120 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_ondemand_teacher.py \
  tests/st_world/test_teacher_cache.py \
  tests/st_world/test_requests.py
```

Initial focused run: **17 passed in 8.17 s**. After adding false-mask rejection
and concrete executor behavior tests, combined run: **46 passed in 8.89 s**.
The final source rerun is recorded in `BOUNDED_FUTURE_PROVIDER_TEST.txt`.
One existing pynvml deprecation warning, no skip.

Coverage includes exact fixture full-cache parity and current-cache SHA reuse,
role/contract/corruption rejection, true missing-source masks, temporary executor
failure/retry, no writes, duplicate delivery, finite/dtype/key/order validation,
RAM eviction and cold resume, request RNG and target equality, normal tensor
backward after an inference-mode executor, owning-thread enforcement, and a
real two-process CPU/Gloo test where one rank's miss fails both ranks before
backward. The DINO executor unit uses a small fake encoder plus the actual
preprocessor to exercise RNG/freeze/budget guards. It does **not** load DINO or
provide real teacher numerical/performance evidence.

## Required subsequent qualification (not run)

- Exact raw native and (if selected) pool3 targets against immutable real-image
  records on the registered teacher runtime, including original production batch
  size, regrouping, tail batches and cold vs warm requests. FP32 is not a claim
  of bitwise equality across different batch kernels; no BF16 tolerance is used.
- Same frozen PCA and masks, sampled pair/patch IDs, losses and actual student
  gradients/updates on real batches; preserve normal dropout and all RNG states.
- Check construction/miss/hit/exception CPU/CUDA RNG, cold native resume and next
  consumed scene IDs. Compare all label tensors, not selected scalar summaries.
- Actual memory including teacher activations and whole rank process; measure
  sustained teacher cost, sharing/interference and complete update throughput.
  No GPU is assigned and no number here is a predicted speed result.
- Independent Evidence/Correctness/Efficiency reviews on the final integrated
  SHA/config/provider/basis/resource identities before any temporal pilot.

No spatial winner or future experiment was selected, no multi-TB prebuild was
started, and no existing cache or checkpoint was removed.

## Independent Evidence follow-up: source provenance on pure compute

Evidence identified that the initial implementation validated
`source_student_identity`, `source_index_sha256` and `partition_sha256` only
inside the immutable-source loop. With `sources=()` an otherwise checksummed
manifest could omit them. This was a provider provenance bug; no integrated
training or cache was produced from that path.

The provider now unconditionally calls `require_hash` on all three fields
immediately after manifest validation, before examining sources or executor.
The existing train/dev split restriction remains unchanged. Nine negative cases
cover each field absent, `None` and malformed while using a nonempty executor
and zero immutable sources; all reject before the executor is called.

Focused source/reuse/current-only command (`-k 'source_lineage or content_reuse
or complete_current_reuse or current_only'`) passed **12 tests in 3.66 s**, with
16 unrelated tests deselected. Complete output is
`BOUNDED_FUTURE_PROVIDER_LINEAGE_FIX_TEST.txt`. This narrow fix uses no GPU,
teacher weights, trainer change or PCA change. Previous test evidence is retained
and does not claim coverage of this later revision.
