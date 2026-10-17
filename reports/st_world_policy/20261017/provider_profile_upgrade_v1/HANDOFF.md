# Bounded indexed-provider PROFILE upgrade

Source start: `83862fce`. Isolated branch `agent/st-provider-profile-20261017`. This is a reviewable, opt-in metadata/configuration extension of the existing trainer. It creates no teacher tensors, launches no job, does not select a temporal cell or spatial winner, and does not admit FORMAL training.

## Existing evidence and preserved mathematics

The exact32 indexed composition already has real full-tensor, source-cache, cold/partial-LRU/restored-layout and RNG/sampler evidence. Lead separately reports the subsequent actual indexed student smoke completed two updates, finite losses/gradients and FP32-master updates, 48 immutable hits / 12 computed images, 14,159,232 B peak LRU, and zero new target-tensor disk. That student report is Lead-owned (`abd51fd0`); this work does not replace it with synthetic results or claim speed from two updates.

Local design evidence: `fixed_batch_teacher.FixedBatchDINOExecutor` retains fixed-eight FP32 compute/padding with actual requests only returned. `qualify_indexed_provider.validate_receipt` rechecks every stored full-tensor/mask/layout comparison. `IndexedBoundedTeacherProvider` preserves target assembly, bounded RAM, cold restore and source identity. `provider_pca` retains the same frozen basis and explicit indexed application scope. `training_config.official_runtime_policy` resolves the existing official DriveDreamer policy execution (`drivedreamer_f35fe783`), rather than merely accepting a precision label. This change introduces no mathematical model, loss, precision, augmentation or teacher API.

## Why a separate source migration is necessary

The old composition PLAN pins `teacher_provider.py` and its qualifier's exact bytes. Editing the control entry points must not silently reinterpret that old result. Old SMOKE validation remains strict and unchanged by default.

The new `prepare-profile-migration` command creates an explicit, immutable proposal binding the archived runtime commit, original PLAN/RESULT/descriptor and source inventory. It verifies the archived files against both their byte pins and `git show <original commit>:<path>`. Only these control modules may change:

- `tools/st_world_policy/teacher_provider.py`
- `tools/st_world_policy/qualify_indexed_provider.py`
- `tools/st_world_policy/training_config.py`

Fifteen explicit teacher/preprocess, immutable/indexed reader, target/schema, PCA and patch-sampler modules must remain byte-identical. The new control implementation requires its own independent review and source-bound run registration. There is no `skip_source_check` boolean. A migration is never a PASS review or launch approval.

## New explicit admission and unchanged old path

An indexed descriptor may carry `profile_admission` with schema `st_indexed_provider_profile_admission_v1`. It pins:

- the original exact32 descriptor and explicit source migration;
- an actual request index and unchanged COMPLETE current-source index;
- the fixed selected dataset path/SHA, scene/log order and current/ego record hashes;
- the original selected provider SMOKE recipe and the exact proposed PROFILE recipe;
- a hard maximum of **256 train scenes and 256 optimizer updates**;
- `new_population_tensor_qualified=false`, `formal_approved=false`.

The raw FP32 contract, execution/backend, fixed8, LRU/scene budgets and current source cannot change. Source student/index/partition, all five target times and view order must match the prior legal population. Every target scene must equal the selected dataset order. These checks occur before materializing more than 256 scene rows.

The existing selected model/space-time cell, original optimizer, frozen PCA, data roots, precision/backend, 20k schedule and 1000-step warmup remain fixed. Explicit diagnostic scope, sample count, stop count, GPU/batch topology and logging/runtime budget may change; the entire proposed recipe is pinned and validated by the existing official execution resolver. Only explanatory adaptation text and its correctly resolved override list may accompany topology changes. `loader_workers=0` remains mandatory. PROFILE cannot be relabeled SMOKE; FORMAL still fails.

The same provider factory, indexed readers, PCA loader, trainer, sampler, checkpoints and optimizer are reused. No second trainer or disk tensor cache exists.

## Required two-line trainer integration

`TRAINER_INTEGRATION.patch` adds `validate_training_manifest(...)` immediately after the existing registered-teacher dependency check. This is necessary: matching scene membership alone does not pin the actual registered dataset order/path/hash. Lead owns integration of this patch; **the extension must not be dispatched before that check is integrated**.

The current `check_formal_approval` path still validates source cleanliness/fingerprint, the exact execution/config/data/environment/resource registration and forbids diagnostic weights as formal initializers. FORMAL's independent-review gate is untouched. A source migration does not grant a resource lease or approve a PROFILE dispatch.

## Commands, after independent review and actual metadata/resource registration

The variables below are required paths supplied by Lead, not fabricated existing files. `EXACT32_DESCRIPTOR`, `ARCHIVED_RUNTIME`, and `EXACT32_CONFIG` must refer to the actual completed experiment. `PROFILE_REQUEST_INDEX` may initially be the same verified future32 index; a larger <=256 index must already exist and pass metadata admission. Never set it to the full-current index or fabricate future coverage.

```bash
python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor "$EXACT32_DESCRIPTOR" --parent-runtime "$ARCHIVED_RUNTIME" \
  --output "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json"

python -m tools.st_world_policy.teacher_provider \
  --request-index "$PROFILE_REQUEST_INDEX" --source-index "$FULL_CURRENT_INDEX" \
  --teacher-config "$QUALIFIED_TEACHER_CONFIG" --qualification-result "$EXACT32_RESULT" \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 \
  --profile-parent-descriptor "$EXACT32_DESCRIPTOR" \
  --profile-source-migration "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json" \
  --profile-data-manifest "$PINNED_PROFILE_SCENES" --profile-reference-config "$EXACT32_CONFIG" \
  --training-config "$PROFILE_RECIPE" --output "$NEW_METADATA_ROOT/PROVIDER.json" \
  --pca-fit-cache-root "$ORIGINAL_PCA_FIT_CACHE" --pca-fit-cache-index "$ORIGINAL_PCA_FIT_INDEX" \
  --configured-output "$NEW_METADATA_ROOT/configured"

# After Lead creates the actual source/config/data/environment/lease registration:
python -m tools.st_world_policy.check_formal_approval --registration "$PROFILE_REGISTRATION" --repo .
torchrun --nproc_per_node="$REGISTERED_WORLD_SIZE" -m tools.st_world_policy.train_world \
  --registration "$PROFILE_REGISTRATION"
# Existing interrupted diagnostic run only:
torchrun --nproc_per_node="$REGISTERED_WORLD_SIZE" -m tools.st_world_policy.train_world \
  --registration "$PROFILE_REGISTRATION" --resume --acknowledge-stop
```

Use `--help` to inspect the existing exact resume arguments before dispatch. The commands do not create registrations, choose resources, launch automatically, or claim any path above already exists. New metadata must stay in the existing registered ledger; no side directory bypass is authorized.

## Actual next PROFILE evidence required

1. Independent review of the new source migration, config binding and integrated trainer check.
2. Actual same-recipe bounded run on the intended multi-GPU topology, preserving source order and target masks; per-rank cache hits/misses, wait time, compute slots, RAM, GPU memory, metadata/shard IO and exposed data stalls.
3. Ordinary interruption/cold-RAM resume: next scene IDs, target bytes/patch pairs, RNG, loss and FP32-master/optimizer updates. Old numerical evidence is not claimed to be multi-rank trainer evidence.
4. Single-rank read/compute failure must terminate coherently through the existing collective path. No substitute zero mask, skip, resampling or fallback precision.
5. If new scenes produce a target discrepancy, retain it and block advancement; no retroactive tolerance relaxation.

No all-scene tensor equivalence, throughput or formal full-population readiness is asserted by this patch. The actual 32-scene report remains limited to its exact keys.

## Existing full-future metadata interface

`prepare_future_metadata.plan(scope='full_population')` already requires `max_scenes == source.scenes`, original index order, all original log groups, no selected prefix, explicit hard CPU/read/RSS/metadata budgets and registered atomic shard completion. `build` retains all five times, source/raw-log ledger and missing-frame semantics. Nothing here invokes it. Current-only COMPLETE is not temporal completeness.

Qualification preparation already supports <=256 scenes, selected before image/label reads using the first two source logs and leading contiguous windows. It still needs actual admitted preparation/index budgets. Full future preparation remains a separate controlled operation; full-population provider FORMAL admission remains blocked. Student current-index eager loading is not rewritten in this control-only patch; its per-rank RSS must be measured before broader scope.

## Final bounded CPU verification

`FINAL_TESTS.log`: **108 passed, 1 existing pynvml warning, 44.35 s**, CPU cores 0–1 / two library threads / no visible CUDA. The suite covers the new 65-scene synthetic indexed path, old SMOKE/indexed/PCA paths, source migration, exact manifest order, population/contract/recipe rejection and scope bounds. Synthetic tests are not real teacher or training qualification. Earlier intermediate fixture/source-edit failures are preserved in `INTERMEDIATE_TEST_FAILURES.log`; the final run used frozen source. Counts from separate runs are not added. `CLI_HELP.log` records the real parser interfaces.

## Actual existing inputs for the first bounded PROFILE

These existing paths were read as small metadata only. They are suitable input bindings, not permission to launch. The archived numerical runtime is `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`; the new control source must be independently reviewed and registered.

```bash
EXACT32_DESCRIPTOR=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json
ARCHIVED_RUNTIME=/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
EXACT32_CONFIG=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/CONFIG.json
EXACT32_RESULT=/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json
PROFILE_REQUEST_INDEX=/mnt/project/st-world-policy-20261016/caches/future32_request_metadata_index_v1
PINNED_PROFILE_SCENES=/mnt/project/st-world-policy-20261016/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json
FULL_CURRENT_INDEX=/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1
QUALIFIED_TEACHER_CONFIG=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json
ORIGINAL_PCA_FIT_CACHE=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1
ORIGINAL_PCA_FIT_INDEX=/mnt/project/st-world-policy-20261016/caches/original_pca_fit64_metadata_index_v1
```

The first PROFILE recipe can preserve this exact 32-scene manifest and the original single-GPU microbatch 1 / global batch 2 topology. Copy the existing CONFIG into a new registered metadata output; set `execution.scope=PROFILE`, `execution.stop_after<=256`, `runtime.loader_workers=0`, a distinct `runtime.output_root`, explicit `runtime.max_seconds`/`save_every`, and optionally the existing `runtime.profile_mode=timing`. For timing, use the existing timing-window requirements; 240 updates fit the new cap. Keep `max_updates=20000`, warmup=1000, deterministic=true, BF16 execution/FA2/FP32 master/ZeRO2, no language recomputation, all model/loss/patch settings and the original PCA basis. Before passing the unconfigured recipe into the provider CLI, remove its old `data.teacher_provider` and set `data.teacher_root` to the original COMPLETE current cache `/mnt/project/st-world-policy-20261016/audits/history_data/full_current_teacher_train_stage_plan_v3/cache/train/native384`; `configure_smoke` then replaces that dependency with the new exact provider and PCA application scope. A topology change must also keep the official resolver's declared overrides accurate. Repeating 32 scenes is not full-population throughput evidence.

For a larger <=256 qualification, use the existing `prepare_future_metadata plan --scope qualification --max-scenes 256` from the same student/partition/full-current population. It selects before labels and retains original ordered windows; build via the existing registered metadata-storage runner. Derive the selected scene JSON from PLAN.selected_rows without reordering (`st_fixed_scene_list_v1`, train, `source_index_hash=PLAN.source.source_index_identity`). Build only the request index using `index_teacher_metadata build --requests .../TEACHER_REQUESTS.json --expected-request-identity <actual COMPLETE request_identity> --contract-receipt /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1/RESOLVED_PLAN.json --contract-receipt-sha256 <actual SHA> --output <registered new index> --max-scenes 256 --max-disk-bytes <registered metadata allowance> --max-builder-rss-bytes 4294967296`. New metadata preparation/index accounting and source-read limits require actual registration; no unregistered build is authorized here. Substitute that actual index and its selected-manifest pin in the same provider command. No future tensor cache is built.
