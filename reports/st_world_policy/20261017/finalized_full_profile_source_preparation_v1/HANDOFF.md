# Finalized full-population PROFILE: source/config draft only

This handoff stays within the existing ST bounded-future task. No IQE, new teacher tensor cache, full training, model import, real index-body read, ledger change or dispatch was performed. The original frozen `bccebb00319512049e93df58d18cb5ead49575f0` tree remains clean/read-only. This report uses a separate sparse implementation worktree based on `3df052ea422c4ba042ef3be33dea19438c5702dc`; it is not a prepared runtime.

## Observed completion and missing inputs

The actual train finalizer receipt was bounded-read and its bytes matched the supplied SHA:

- Path: `/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_finalized_v1/FINALIZATION_COMPLETE.json`.
- SHA256: `16c47d6603144dcabe0c764f85c23c231bb0edfaf347d869f93c34d79bdc5438`.
- Receipt declares COMPLETE / full_population /101592 scenes /1176 logs; zero teacher extractions/tensor disk and no training admission.
- Request identity: `0c8132c3b3ac97aa8adf88736a46594427f845a03712e34f2e53ea91ce5be57b`.
- Raw request SHA declared by receipt: `5f43583a4a0f35a2fc9374d3b6c59442332b8f00721f63bd1ef1fdc9f5bb71ef`.

The large request body was **not** opened/rehashed here. The actual matching full-train request-index identity, future index COMPLETE, SQLite/dependency verification, new provider descriptor/PCA application scope, runtime registration and resources remain unassigned. `INPUT_BINDINGS.json` uses null for these; a PLAN or the old failed writer outcome is not substituted.

## Minimal source composition and ownership

Directly running the new3df source against the original31c3 qualification is not valid. The old control migration requires15 exact core files;14 match, but `starVLA/dataloader/st_world_history.py` now contains `packed_log_v1` handling. This is unrelated to the history-off selected PROFILE and must not be put on a broad migration whitelist just to pass.

`SOURCE_DELTA.json` records all actual SHA comparisons. The minimal proposal is:

1. Lead creates a new clean source identity from the existing original `bccebb00319512049e93df58d18cb5ead49575f0` baseline. Do not edit that frozen tree itself.
2. Apply `MINIMAL_RUNTIME.patch`. It adds the **exact reviewed** `tools/st_world_policy/finalize_future_metadata.py` bytes from3df and applies only the donor provider's12-line diff (10 additions/2 deletions). No hand-copied validator rewrite, dynamic unchecked import path, sys.path override, or source-check bypass is added.
3. Leave all15 old core SHAs and existing `train_world`, dataset, sampler, PCA and teacher execution files unchanged. Also retain the baseline `prepare_history.py` and `metadata_storage.py`; no new metadata-writer kinds or packed-history APIs are needed for read-only provider admission.
4. A narrow independent review must approve this particular two-file composition. Finalizer implementation review at3df is useful evidence, not automatic approval of a different composed runtime. Lead alone owns the resulting runtime creation, new commit, source migration and dispatch.

The provider calls `finalize_future_metadata.validate_completion` only for the explicit new completion schema and canonical `FINALIZATION_COMPLETE.json` name. That function strictly checks new identity/schema, actual source/failure/registration lineage, original selected population, SHA-pinned ledger/shard receipts and the caller's actual verified request-index header/count/hash. It does not reopen raw logs, images or target tensors. The old COMPLETE branch stays intact.

Static dependency trace: `validate_completion -> _registration/_lineage/_terminal/_pin/_read` within the exact new helper, plus existing `prepare_future_metadata` functions and metadata provenance. `prepare_future_metadata.py` is byte-identical betweenbccebb and3df, as are `prepare_current_teacher.py` and `st_world_metadata_index.py` (including `decoded_bytes`). The consumer calls `_lineage(check_code=False)`; only the separate finalizer writer calls `_lineage(check_code=True)` to verify the original preparation source pins. Original producer/finalizer paths stay in their actual immutable receipts; they are not relabeled as the new PROFILE runtime. The new helper's writer path is not invoked or admitted by this handoff, and baseline metadata storage continues rejecting the new writer kind. No new student/teacher mathematics is introduced.

## Draft recipe

`DRAFT_TRAINING_CONFIG.json` inherits the actual original indexed2-update CONFIG, not a reconstructed model recipe. It retains T2/S-P2, current+future targets, history/delta off,48 queries/view,96 sampled positions, frozen PCA384, deterministic true, FP32 fixed8 teacher targets and official BF16 execution / FP32 master+Adam / FA2 / ZeRO2 student policy. It keeps the20k scheduler, warmup1000, seed42, original optimizer/loss/normalizer settings and no activation recomputation.

The diagnostic overrides are full population101592, world2 / micro1 / global2 / accumulation1, stop64 total updates. `DRAFT_CONSUMPTION.json` fixes128 global exposures and64/rank across **all invocations combined**, including already completed updates before resume. It is not permission for64 new updates after every restart or for formal global32 throughput. The original full sampler/rank slicing stays in use; no selected-prefix dataset is created.

This is deliberately not runnable yet: `runtime.max_seconds` and `save_every` are null and the output root is an explicit unassigned marker. Lead must fill these from real resource/checkpoint admission before constructing/pinning a descriptor or invoking validation. Diagnostic weights are discarded as formal initializers (`weights_usable_for_formal_init=false` in the eventual registration). No new scientific cell or precision variant is selected.

## Commands after inputs and narrow source/resource review

Known existing paths are supplied below. `FULL_TRAIN_REQUEST_INDEX`, `NEW_METADATA_ROOT`, `PROFILE_RECIPE`, `PROFILE_REGISTRATION` and `FROZEN_RUNTIME` must be actual Lead-created and reviewed artifacts; they are not created here.

```bash
EXACT32_DESCRIPTOR=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/PROVIDER.json
ARCHIVED_EXACT32_RUNTIME=/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
EXACT32_CONFIG=/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/CONFIG.json
EXACT32_RESULT=/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json
CURRENT_POP=/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train
CURRENT_INDEX=/mnt/project/st-world-policy-20261016/caches/full_current_metadata_index_v1
FINAL_COMPLETE=/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_finalized_v1/FINALIZATION_COMPLETE.json
TEACHER_CONFIG=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json
PCA_FIT=/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1
PCA_INDEX=/mnt/project/st-world-policy-20261016/caches/original_pca_fit64_metadata_index_v1

# Run from the NEW two-file-composed frozen source, not either old source tree.
python -m tools.st_world_policy.qualify_indexed_provider prepare-profile-migration \
  --parent-descriptor "$EXACT32_DESCRIPTOR" --parent-runtime "$ARCHIVED_EXACT32_RUNTIME" \
  --output "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json"

python -m tools.st_world_policy.teacher_provider \
  --request-index "$FULL_TRAIN_REQUEST_INDEX" --source-index "$CURRENT_INDEX" \
  --teacher-config "$TEACHER_CONFIG" --qualification-result "$EXACT32_RESULT" \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 --max-metadata-cache-mib 16 \
  --profile-parent-descriptor "$EXACT32_DESCRIPTOR" \
  --profile-source-migration "$NEW_METADATA_ROOT/SOURCE_MIGRATION.json" \
  --profile-data-manifest "$CURRENT_POP/SELECTED_SCENES.json" --profile-reference-config "$EXACT32_CONFIG" \
  --profile-full-population --profile-current-registration "$CURRENT_POP/REGISTRATION.json" \
  --profile-current-complete "$CURRENT_POP/COMPLETE.json" --profile-future-complete "$FINAL_COMPLETE" \
  --profile-consumption-budget "$NEW_METADATA_ROOT/CONSUMPTION.json" \
  --training-config "$PROFILE_RECIPE" --output "$NEW_METADATA_ROOT/PROVIDER.json" \
  --pca-fit-cache-root "$PCA_FIT" --pca-fit-cache-index "$PCA_INDEX" \
  --configured-output "$NEW_METADATA_ROOT/configured"

# Only after actual index/provenance verification and Lead's complete registration/lease:
python -m tools.st_world_policy.check_formal_approval --registration "$PROFILE_REGISTRATION" --repo "$FROZEN_RUNTIME"
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION"
# Same run at a preserved interior native boundary; unchanged total64 ceiling:
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
  -m tools.st_world_policy.train_world --registration "$PROFILE_REGISTRATION" --resume --acknowledge-stop
```

The existing factory still performs original request/index/contract/PCA checks and a full streaming ordered-population comparison. It must receive the index built from the **finalized** request path and exact identity above. Old partial/pre-finalization request paths and their index plans cannot silently be reused. Finalizer receipt/source provenance checking is additive; current observation/ego and original population remain pinned. The same FP32 current bank plus fixed8 synchronous miss computation and bounded RAM LRU is retained, with zero new teacher-tensor disk writes.

## Cost and remaining review

The bounded stdlib-only capture used CPU10/11,1GiB address-space limit and180s timeout:1,197,661 source bytes,13,645 control-receipt bytes,21,995,520B peak RSS,1.865s. Earlier manual bounded source diff/header inspection was under1MiB and is explicitly not included in that metered capture. No torch/model/index/payload import occurred, and no new test matrix was run.

Still needed: narrow review of two-file source composition/legacy-core ownership; actual full request index/dependencies; fresh source migration and provider/PCA metadata; real two-rank resources/storage/checkpoint budgets; bounded original-sampler training and native next-batch recovery. Full-scene numerical equality, complete data exposure, full throughput and formal training are not asserted. The previous32-scene qualification remains its own exact-scope evidence.
