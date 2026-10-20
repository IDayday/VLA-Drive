# Independent fixed-DEV context and A prediction admission

Decision on 2026-10-20: **PASS for registration and a bounded two-GPU extraction of the fixed complete DEV1696/16-log context**, retaining all rejection conditions below. This does not certify extracted features, original-output replay or PDMS before those operations actually finish.

Sources inspected: collector/reader `87577e24e24783b5600d73397d5eaea5fc6beb9e`, A prediction `5cac262e6961e0ac3a267414162d611f7678fe33`, and integrated prediction/population handling `1bcd29639018bfdd9e4db3eac521c3894b7f5412`. Review worktree is based on `c8514e4f`. The reviewer has not modified collector, reader, inference modules or source fields.

## Findings

No fatal issue was found within the registration/extraction scope. Approval is limited to two explicitly allocated GPUs, disjoint shard ranks and a new campaign-owned DEV output namespace. Do not change the original Base, training feature caches or 68-file extraction contract. Existing native4x2 repeat evidence remains `INCONCLUSIVE` for cross-host executed-state bitwise equality as originally classified; its exact terminal metrics/events/reference inputs are a separate native-scoring result, not permission to relax DEV model replay tolerances.

Population and split:

- Registration requires the existing complete current-only DEV identity, all 1696 unique scenes and 16 logs, historical full-export identity, exact Base checkpoint, original metric-index pin and consistent scene/log joins.
- Every DEV log is checked against the full original training input index, not only the 1024-scene effect cohort. The DEV schema is independently typed `split=dev`, `role=dev_report`, `for_training=false`. TRAIN readers and effect/repair fit gates remain separate.
- The population is fixed before new DEV model results; it is not selected by repair success, GT quality, current expert performance or low scores. The registration reads historical protocol/receipts, not score CSV values.

Legal model inputs:

- `ForesightCurrentDataset` calls `current_observation` and does not instantiate `ForesightTrainingDataset`; it reads no ego-target, future auxiliary target or metric payload.
- Model encoding receives exactly image/state/lang/token observations, then `encode_current` rebuilds image/lang/state again before tokenization. Offline metric references remain registration metadata, absent from feature records and the actual model context.
- Feature records have a strict field whitelist. They contain the bound frozen W/A, normalized relative-history pose, planner condition, original raw/physical trajectory and fixed scene noise. GT, future actors, score, feedback and metric paths are rejected.

Numerics and source:

- Actual S0 FP32 masters are reconstructed and loaded strictly. Auxiliary prediction heads are stripped; W144 remains. The model is frozen/eval, TF32 off, deterministic, with the original action-only condition and Euler10 sampler.
- Each scene is checked by hooks for exactly one visual and one language invocation. Noise uses the original CPU SHA256 scene-seed generator, and the decoder does not run an extra VLM.
- Every generated physical trajectory must match its original FP32 historical export bitwise. A mismatch aborts the shard before completion; this review authorizes no tolerance change. The final reader also validates the same historical witness.
- The existing campaign's complete extraction-code/package closure is checked by `prepare_campaign`; the DEV registry additionally binds its collector and reader source files.

Atomicity and completeness:

- Default fixed shards contain 16 scenes, giving 106 shards for all 1696 observations. Rank modulo world size changes work ownership, not output population or order. Exclusive shard claims protect overlap during resume or worker reassignment.
- A receipt is published only after full payload creation and its checksum. Per-shard flags assert DEV-only scope, shared-forward counting and historical equality. Existing receipts are revalidated before reuse.
- Finalization requires all 106 expected shard receipts, exact ordered coverage, full payload checksum, finite/detached FP32 feature schema, original noise and every historical physical witness. It cannot seal a prefix as complete1696. Missing or corrupt shards fail closed. Reader access uses bounded LRU and file-version invalidation.
- Worker time bounds are observed at shard boundaries; an in-progress 16-scene shard may finish after the requested boundary. Do not use a wall timeout as permission to kill unrelated tasks.

A prediction:

- `--role dev_report` requires exactly all1696 scenes and obtains the population from `EvaluationContextReader`, not a TRAIN repair manifest. It does not require or open `dataset_path` in this branch.
- Refiner weights must belong to the same bound Base. Cached current context plus original Base trajectory are passed to deployment; GT/feedback/effect teacher are absent.
- The prediction registration distinguishes DEV context-only population from TRAIN repair data. `base_metrics`, `base_feedback` and target fields are unset for new DEV prediction. Later official scoring is a separate, explicitly budgeted CPU invocation.

## Independent actual-asset checks

Executed in the review environment with CUDA hidden and BLAS threads set to one. No repeated full weight/image/metric-file hash scan was performed.

- Read the actual existing DEV current identity/index/complete metadata, original training index, exact original metric index and original export identity.
- Actual DEV: 1696 unique scenes, 16 logs. Actual training input count read: 101592. Train/DEV source-log intersection: zero. Metric scene/log join: complete.
- Original Base SHA256: `78f870647015194798af067eefc8d8fedb6e59e6db2f3ae719c63b783871446f`. Original protocol explicitly records FP32, TF32 off, seed42, Euler10, three current views, no scorer, no history images and no future conditioning.
- Opened the first real current observation's three actual images and applied its original crop/resize path, producing three1024x576 RGB inputs and legal state shape1x4.
- Repeated this input transformation after adding poison GT trajectory, future actor, metric cache, score and teacher-cache fields to a copy of the current JSON. Image pixels, state, language and token remain exactly identical. These are actual image/input checks, not a model-forward or W/A equality claim.
- Separately hashed only the 68 small source files in the immutable extraction closure in both Lead and review worktrees: zero mismatches. Actual recorded Torch/Transformers/DeepSpeed package versions match the existing campaign contract.
- Metadata/image check wall: 0.37384809739887714s. GT files read: zero. Historical score values read: false. Model/GPU/simulator calls: zero.

Structured evidence: `DEV_CONTEXT_INDEPENDENT_REAL_METADATA.json`. Existing owner14 CPU results are inspected as owner evidence only; the reviewer did not rerun or relabel that matrix as a new independent test run.

## Dispatch boundary

The Lead may register the complete immutable DEV population and launch rank0/rank1 extraction on two available authorized GPUs, with explicit `--world-size 2` and per-worker `--max-seconds`. This is only shared current-feature preparation. A successful worker is not a completed full dataset; seal only after all shards and actual original-output comparisons pass.

If actual FP32 trajectories differ, retain the failure and diagnose software/device/numeric identity without broadening thresholds. No new NAVSIM/navtest evaluation, checkpoint selection or DEV-based recipe tuning is authorized by this extraction approval. Real extraction duration/bytes, full history replay, final evaluation and deployment claims remain NOT_RUN within this review.
