# Full-index bounded PROFILE admission

Implementation start: `d4fa6c3e0eb07f3216136207c860315efa122e01` (only independent-report additions after Lead's `04d90fd2`). Isolated branch `agent/st-full-index-profile-20261017`. This changes only `teacher_provider.py`, adjacent synthetic tests and this report. No live runtime, metadata ledger, real index, teacher payload, GPU, or training job was modified or launched.

The input/resource handoff is `da5fa3578b51159e58a630523b0c2c272cf4f83f`, `full_population_indexed_profile_preparation_v1/RUNBOOK.md`. Its pending full future identities are **not** assigned by this implementation. An existing PLAN cannot stand in for an actual COMPLETE or request index.

## Exact scope

The old `st_indexed_provider_profile_admission_v1` remains at **<=256 scenes / <=256 updates**. Old exact32 composition and <=64 SMOKE paths, source checks and PCA registration are unchanged. New `st_full_index_provider_profile_admission_v1` requires an explicit entire original train population above256, but still caps total optimizer updates at256.

Its `sample_limit` means **the whole indexed population**, not the number of samples consumed during this diagnostic. A two-rank / micro1 / global2 / stop64 example has accumulation1 and at most128 scene exposures across the initial invocation **plus all resumes combined**. It does not grant a fresh64 or256 updates per invocation, cannot introduce a continuation configuration, and cannot be presented as formal global32 throughput qualification. One-rank / micro1 / global2 remains available for the matched diagnostic; the new scope rejects other topologies. `profile_mode=none` avoids reinterpreting the separately registered240-step timing protocol.

The original20k learning-rate timeline, warmup1000, chosen temporal cell, losses, fixed PCA basis, teacher FP32 fixed8 execution, deterministic/official student execution, LRU budget, current source and original sampler remain bound to the original qualified recipe. The existing source-migration API still accepts only its three named control modules and requires the same15 mathematical/reader/PCA/patch modules byte-for-byte. All15 were freshly compared with the archived exact32 runtime; see `SOURCE_COMPATIBILITY.json`. This revision changes only one of those three control modules.

## Implementation and checks

- `full_profile_consumption`: explicitly derives cumulative updates/exposures, rank exposures and accumulation from the existing diagnostic topology; no new sampler or state machine. The strict descriptor identity includes this declaration. Boolean aliases and altered budgets fail.
- `validate_profile_recipe`: branches on the new schema; preserves the old schema's count cap; binds exact proposed recipe and original SMOKE reference. FORMAL remains rejected, and `formal_approved` / `new_population_tensor_qualified` remain false.
- `_qualified_profile_execution`: retains archived original PLAN/RESULT/descriptor verification, source migration, numerical/runtime/current-cache/PCA contract and source/partition/time/view checks. It does not relabel the exact32 numerical report as an all-population comparison.
- `_validate_full_population`: requires actual full-future COMPLETE and original full-current REGISTRATION/COMPLETE, all pinned. These must bind the same original selected manifest, full scene/log counts, original source, request semantic/raw hashes, raw-log provenance ledger, train role and all five times. Missing/partial/subset receipts fail.
- It reuses `MetadataIndex.Rows` and the existing bounded `_atomic_rows` parser to stream every selected/current/future row in order, checks current observations/timestamps, current/ego pins, ordered token hash and selected-row semantic hash. It never calls `list` or `json.load` on all requests/selection. One row per input and a bounded log-name set are retained; SQLite has the original per-rank cache bound. The selection retains the existing32MiB maximum (actual prior manifest stat30,071,766B); this patch does not raise it.
- `_validate_indexed_descriptor` and CLI route the new admission through the same provider and source readers. PCA application scope is newly bound by the existing factory; PCA statistics are not refitted or relabeled.

The existing `train_world` registered-manifest check, `check_formal_approval`, full-population `epoch_batches`/rank slicing, native complete checkpoint loader, cumulative `state.completed >= stop_after` guard and terminal handling are unchanged. Native resume therefore cannot restart the allowance. A new run identity is a separate registered job, not a way to extend this diagnostic automatically. No additional trainer patch is required.

## Commands after independent review and actual input completion

These are actual parser interfaces. Paths below come from the prior input handoff; the full-future roots still require actual successful COMPLETE/index checks. A clean frozen runtime and newly registered bounded metadata/config output are prerequisites. No command below ran in this implementation task.

```bash
EXACT32_DESCRIPTOR=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json
ARCHIVED_EXACT32_RUNTIME=/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
EXACT32_CONFIG=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/CONFIG.json
EXACT32_RESULT=/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json
CURRENT_POP=/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train
FULL_CURRENT_INDEX=/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1
FUTURE_PREPARATION=/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1
FUTURE_INDEX=/mnt/project/st-world-policy-20261016/caches/full_future_request_index_train_v1
TEACHER_CONFIG=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json
PCA_FIT_CACHE=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1
PCA_FIT_INDEX=/mnt/project/st-world-policy-20261016/caches/original_pca_fit64_metadata_index_v1

# NEW_METADATA_ROOT and PROFILE_RECIPE must be Lead's new registered paths.
python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor "$EXACT32_DESCRIPTOR" --parent-runtime "$ARCHIVED_EXACT32_RUNTIME" \
  --output "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json"
```

Create `PROFILE_RECIPE` from the actual original config, retaining all mathematical fields. Set `execution.scope=PROFILE`, `sample_limit` to the actual full request-index scene count (not merely the expected101592), `stop_after=64`, `world_size=2`, `micro_batch=1`, `global_batch=2`. Keep `max_updates=20000`, warmup1000, seed and the exact official policy. Set `runtime.profile_mode=none`, `loader_workers=0`, distinct output root and the independently registered wall/checkpoint limits. Recompute `official_execution.declared_recipe_overrides` using the existing `REFERENCE_RECIPE` and accurate adaptation text; do not change effective precision/optimizer fields. Remove old `data.teacher_provider`; the existing factory installs the new descriptor/PCA scope.

Before calling the CLI, write its explicit `CONSUMPTION.json` using `full_profile_consumption` on that exact proposed config (inside the registered metadata budget). The helper validates topology and finite bounds; it does not construct models, datasets or tensors:

```python
from tools.st_world_policy.teacher_provider import read_small, full_profile_consumption
from starVLA.dataloader.st_world_targets import atomic_json
atomic_json(consumption_path, full_profile_consumption(read_small(profile_recipe_path)), immutable=True)
```

```bash
python -m tools.st_world_policy.teacher_provider \
  --request-index "$FUTURE_INDEX" --source-index "$FULL_CURRENT_INDEX" \
  --teacher-config "$TEACHER_CONFIG" --qualification-result "$EXACT32_RESULT" \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 --max-metadata-cache-mib 16 \
  --profile-parent-descriptor "$EXACT32_DESCRIPTOR" \
  --profile-source-migration "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json" \
  --profile-data-manifest "$CURRENT_POP/SELECTED_SCENES.json" \
  --profile-reference-config "$EXACT32_CONFIG" \
  --profile-full-population \
  --profile-current-registration "$CURRENT_POP/REGISTRATION.json" \
  --profile-current-complete "$CURRENT_POP/COMPLETE.json" \
  --profile-future-complete "$FUTURE_PREPARATION/COMPLETE.json" \
  --profile-consumption-budget "$NEW_METADATA_ROOT/CONSUMPTION.json" \
  --training-config "$PROFILE_RECIPE" --output "$NEW_METADATA_ROOT/PROVIDER.json" \
  --pca-fit-cache-root "$PCA_FIT_CACHE" --pca-fit-cache-index "$PCA_FIT_INDEX" \
  --configured-output "$NEW_METADATA_ROOT/configured"

# Only after Lead independently registers actual config/artifacts/source/resources:
python -m tools.st_world_policy.check_formal_approval --registration "$PROFILE_REGISTRATION" --repo "$FROZEN_RUNTIME"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION"
# Same registered run stopped at an interior native boundary k<64; unchanged ceiling:
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION" --resume --acknowledge-stop
```

The existing approval/resource/lease and clean-source checks still execute. Actual index dependencies are fully hashed by the existing readers; setup may read gigabytes of existing metadata and must have a separately registered finite read/RSS/wall budget. The factory does not create a full future tensor cache or alter the768GiB ledger. Do not run these commands under the small budget of this code-only task.

## Boundaries still requiring real evidence

Synthetic tests exercise258 distinct scene IDs with tiny repeated fixture observations; they are not teacher or full-population training qualification. The old exact32 real numerical and student evidence is inherited only at its original scope. New data, actual two-rank finite training/failure behavior, native cold-provider resume and startup/shared-IO costs still require Lead's independently admitted bounded PROFILE. No speed, full-scene numerical PASS, formal training or global32 approval is asserted.

The unchanged student dataset eagerly loads its current source/selection and hashes original current/ego records at startup. This patch adds no full request lists, but does not claim the entire trainer has bounded256MiB metadata RSS or zero startup tensor reads. Existing index DB/dependency hashing, source-shard integrity and record validation are retained, including repeated validation at existing factory/PCA phases. No precision change, resampling, false mask, missing-label substitution or new target tensor disk write is introduced.

## Verification

`TESTS.log`: **118 passed, one existing pynvml deprecation warning,24.81s** under CPU affinity0–1 / two library threads / CUDA disabled /120s timeout. This includes successful execution of the actual CLI over synthetic258-scene indices, explicit new config/PCA writing, negative incomplete CLI flags, provenance/budget/recipe/order checks, old PROFILE/SMOKE and PCA/config regressions. `CLI_HELP.log` is the real parser output. No real GPU/model/image/teacher data was read. RSS was not separately measured; no real throughput or numerical result is inferred. The earlier117-pass log is retained and not added to the final count.
