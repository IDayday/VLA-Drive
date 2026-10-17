# Full-population PROFILE: existing CLI handoff

Source is the clean, frozen `/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017` at `a8b0036f33554fd2a00ea1af8c9b5f20fb6b965a`. This handoff changes no source/runtime, mathematical contract, sampler or storage policy. It does not launch anything. The current source already has the strictly validated finalizer-receipt branch; both independent static composition reviews are PASS (exact receipts and hashes in `BINDINGS_AND_COSTS.json`). Their scope remains static composition, not actual full-population training approval. The protected 15-file source comparison already passed; it need not be repeated as another model qualification.

Both actual indices are now COMPLETE. DEV: 1,696 scenes,9,948 records,8,585,216B SQLite, receipt SHA `339efa75e1cb41499c84b90783e09c375bd4ffd5b7a69a512c66c0207136f811`. TRAIN:101,592 scenes,603,750 records,515,739,648B SQLite; small receipt bytes freshly match SHA `07cdbbd00d7c9be219c23c75bf75aee4651cad6c93b136a0f2b19b4bdca4a148`. Its request identity/raw-request pin matches the finalizer receipt. Database/dependency verification remains the existing factory’s job. The actual full TRAIN finalizer is COMPLETE at101,592 scenes/1,176 logs; its request identity and raw-file SHA are recorded in the bindings. Only small receipts were read here; neither database nor request body was opened. DEV cannot substitute for TRAIN. This PROFILE uses TRAIN only; DEV is ready for its separate subsequent purpose.

Completed DEV/TRAIN indexes are followed by the already queued original CPU2/3 fit300 work → separately admitted metadata/provider preparation → separately admitted two-GPU PROFILE. There is no new all-population teacher tensor cache.

## Resource bounds to register before executing

The implementation preparation done here was local, CPU-only, with bounded small control/source reads (<64MiB); metered capture read134,850B small JSON and72,646B prior profile logs, plus earlier manual source/control inspection under2MiB. No model, image, teacher tensor, SQLite body or ledger was read or changed.

The *next actual* metadata/provider preparation is a separate CPU job, not this small-read handoff. Proposed finite envelope for resource review:2 CPU threads,4GiB aggregate RSS,1,500s wall,16GiB aggregate logical reads,32MiB new control outputs,zero new teacher tensors. These are stopping limits, not a measured completion/throughput prediction. Use the existing resource supervisor/counters; do not treat `timeout` or final `/usr/bin/time` output as an RSS/read limiter. No automatic retry or concurrency with the pending full-index/fit loader. If the actual TRAIN index size makes this envelope insufficient, report it before running rather than weakening hash checks or adding another validation pass. No new resource/controller implementation is supplied here.

Known cost accounting:

- One current `MetadataIndex` opening hashes381,902,848B DB +317,326,781B original dependencies (=699,229,629B, excluding its small receipt). The TRAIN request body is772,307,583B by stat only; its actual database size declared by the completed receipt is515,739,648B. One future-index DB/dependency hash pass is1,288,550,516B before small receipts. The original frozen PCA-fit index is1,155,072B plus967,162B dependencies.
- The single factory invocation below includes approximately four current/request index opening pairs and four complete streaming ordered comparisons through its existing descriptor/PCA admission calls. Hash work alone is approximately `4*(H_current+H_future)+H_fit`, plus original selected-list scans, decoded SQLite reads, original qualification receipts/source pins and small frozen PCA-basis loading. This is **not** four new checker commands. No teacher/image/model extraction occurs in this preparation; the small original PCA basis is loaded and never refitted.
- A training invocation repeats its original metadata admission per rank (roughly three ordered comparisons/rank); native dataset/input validation and model/teacher cold loading also remain. A resumed invocation repeats startup work. Do not report receipt-read cost as total startup cost or claim zero-cost full population validation. Budget actual `/proc` process-tree logical/physical reads and RSS in the existing supervisor, separately from steady updates.
- Prior exact32/world1 actual run:64 updates,128 exposures,712.03s allocation wall; endpoint checkpoint73.17s; observed rank RSS peak8,565,043,200B. The population repeated; these are neither full-population nor world2 speed results.

For the next PROFILE retain the already used `runtime.max_seconds=1500` and `save_every=64`. Retain the existing1,620s outer bound and180s cleanup allowance if the new resource review adopts them; lease must cover that whole interval. The full startup may consume this budget and stop early: no extension or speed promise is implicit. Target is local `training-vla-zt-worker-0`,GPU0/1,world2/micro1/global2/accum1;2 torch threads/rank, synchronous loader0. Whole-run limit is64 updates/128 total exposures across all resumes, with the original full101,592-scene sampler,20k LR timeline and warmup1000. Diagnostic weights cannot initialize formal runs and this is not global32 efficiency qualification.

Request a **new** checkpoint reservation, initially80GiB for a single endpoint-save invocation subject to review of the world2 writer. Do not spend the old profile's80GiB reservation or erase its checkpoint. An interior-stop/resume that retains additional checkpoints must fit an explicitly reviewed total reservation; no automatic extra invocation. Current commitments are2,303GiB (`b786cd…` receipt), plus a separate1TiB free floor; the receipt expires2026-10-17T20:22:33.193247Z and must be refreshed if stale. The current metadata policy SHA is `7a78476af60b191da42439dfdda0c63dae6334f7e911ee7bc3ed299e77486148`. This handoff does not create a new claim, lease or registration. Output bytes, config/PCA-scope control files, run identity, checkpoints, temp files and logs must all be assigned by the resource owner before writes.

## Metadata preparation commands (Lead only, after queued original CPU fit300 completes and separate resource admission)

These flags were checked against the actual source CLI, with no new wrapper or guessed subcommand. `PREP` is the resource-registered new control-output namespace, **not** a cache/index root. `PROFILE_RECIPE` is the registered copy of the included draft; set its output marker to that namespace before hashing. `configure_smoke` internally replaces the final training output with `$PREP/configured/runs`; it does not change the mathematical recipe. The actual new provider/PCA/registration hashes are intentionally absent until generated.

```bash
RT=/mnt/project/VLA-Drive-st-finalized-full-profile-runtime-v2-20261017
PY=/root/miniconda3/envs/ddp/bin/python
TRAIN_INDEX=/mnt/project/st-world-policy-20261016/caches/full_future_request_index_train_v1
CURRENT_INDEX=/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1
CURRENT_POP=/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train
EXACT32_DESCRIPTOR=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json
EXACT32_RUNTIME=/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
EXACT32_CONFIG=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/CONFIG.json
EXACT32_RESULT=/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json
TEACHER_CONFIG=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json
PCA_FIT=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1
PCA_INDEX=/mnt/project/st-world-policy-20261016/caches/original_pca_fit64_metadata_index_v1
FINAL_COMPLETE=/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_finalized_v1/FINALIZATION_COMPLETE.json
: "${PREP:?Lead must assign and resource-register the control output root}"
: "${PROFILE_RECIPE:?Actual registered draft recipe path required}"
: "${CONSUMPTION:?Actual immutable copy of supplied CONSUMPTION.json required}"
cd "$RT"
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

# Small explicit control-source migration; no teacher extraction or new model.
"$PY" -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor "$EXACT32_DESCRIPTOR" --parent-runtime "$EXACT32_RUNTIME" \
  --output "$PREP/SOURCE_MIGRATION.json"

# This is also the original-population/order validator. Do not prepend a second scanner.
"$PY" -m tools.st_world_policy.teacher_provider \
  --request-index "$TRAIN_INDEX" --source-index "$CURRENT_INDEX" \
  --teacher-config "$TEACHER_CONFIG" --qualification-result "$EXACT32_RESULT" \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 --max-metadata-cache-mib 16 \
  --profile-parent-descriptor "$EXACT32_DESCRIPTOR" \
  --profile-source-migration "$PREP/SOURCE_MIGRATION.json" \
  --profile-data-manifest "$CURRENT_POP/SELECTED_SCENES.json" --profile-reference-config "$EXACT32_CONFIG" \
  --profile-full-population --profile-current-registration "$CURRENT_POP/REGISTRATION.json" \
  --profile-current-complete "$CURRENT_POP/COMPLETE.json" --profile-future-complete "$FINAL_COMPLETE" \
  --profile-consumption-budget "$CONSUMPTION" \
  --training-config "$PROFILE_RECIPE" --output "$PREP/PROVIDER.json" \
  --pca-fit-cache-root "$PCA_FIT" --pca-fit-cache-index "$PCA_INDEX" \
  --configured-output "$PREP/configured"
```

Existing functions perform the necessary checks: `teacher_provider.prepare_indexed`(668), `_validate_indexed_descriptor`(548), `_validate_full_population`(273), finalizer `validate_completion`, and `provider_pca._register_indexed_scope`(103). Ordered scene/log IDs, timestamps, complete t0 frames, original current/ego pins, selected semantic/raw hashes, source/partition/teacher identities and all four future times stay enforced. `MetadataIndex` verifies full DB/dependency hashes and immutable versions. Missing actual COMPLETE is a hard failure, not a path placeholder. Source migration preserves the original15 mathematical file SHAs. The unchanged PCA family is bound through the indexed scope; no fake old cache identity or PCA refit is used.

## Native PROFILE command (separate registration/admission, not executed)

Lead creates a new `st_world_run_v1` registration using these actual seven artifacts: final configured training config, original full `CURRENT_POP/SELECTED_SCENES.json`, unchanged target contract, new `$PREP/PROVIDER.json` as `teacher_manifest`, existing `/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`, freshly matched environment and new world2 resource lease. Source is `a8b0036f…`, `weights_usable_for_formal_init=false`; no formal approval is manufactured. The BASE binding SHA inherited from the prior registration is `84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38`; normal startup verifies it and its actual source as before. Registration binds the complete manifest, not the first128 rows.

```bash
: "${PROFILE_REGISTRATION:?New exact world2 PROFILE registration required}"
: "${PROFILE_CPUSET:?CPU affinity from its actual resource admission required}"
cd "$RT"
"$PY" -m tools.st_world_policy.check_formal_approval --registration "$PROFILE_REGISTRATION" --repo "$RT"
export CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
# Run only under the existing admitted supervisor/lease with its locks and whole-job bounds.
taskset -c "$PROFILE_CPUSET" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION"
# If an interior native pause occurred and a fresh invocation is admitted, same identity:
taskset -c "$PROFILE_CPUSET" "$PY" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION" --resume --acknowledge-stop
```

The resume line is not an instruction to run a second64-update job. Native completed steps and consumed cursor retain the same64 total ceiling; a completed64 PROFILE adds zero updates. Keep the original invocation evidence. No eager reinitialization, fresh run ID or modified stop budget is allowed to extend exposure. `train_world` already owns the run lock, all-rank data errors, consumed-state checkpoint and prior-stop acknowledgement.

Only actual unresolved inputs remain: normal full index/dependency/order verification by the existing factory; resource-admitted prep namespace/window and fresh source-migration/provider/PCA outputs; new world2 registration/environment/lease/CP reservation; the resulting bounded full-sampler runtime measurements. Source composition, fixed8 body numerical tests and exact32 real composition evidence have already passed and are not rerun here. New-population per-tensor qualification and formal training remain unclaimed.
