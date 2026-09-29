# DDP shared foresight + GT vehicle MAE

This is an explicit local Action-Only DDP variant. Generic Qwen3-VL plus random driving modules; three current front views, original ego FM head, optional64shared W queries before action queries. A/B/C/D have identical W/main branches, R omits W. Only labels use future images and GT vehicle trajectories. Deployed models retain W and remove both auxiliary heads. No joint-actor generator, detector, video/depth generation or scorer is instantiated.

Use the `ddp` environment (Torch2.5.1cu124, Transformers4.57, DeepSpeed0.16.9). Freeze a clean source commit/worktree before a run. Every path below is a CLI argument/environment variable, not a model-module constant. `ART` denotes the new campaign artifact directory; public generic files must match `reports/ddp_shared_foresight/GENERIC_SOURCES.json`. Prior driving checkpoints are forbidden.

```bash
export FORESIGHT_QWEN=/path/to/verified/generic/Qwen3-VL-2B-Instruct
export FORESIGHT_SOURCES="$ART/generic/sources.json"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Actual evidence is under `reports/ddp_shared_foresight/student_preflight` and `batching_fix`. CPU tests, real generic initialization pairing, main/auxiliary shared gradients, target poison invariance, auxiliary-head removal, two-rank global normalization, two-rank real optimizer resume and four-rank batch32 training have run. Teacher full training completed30epochs at frozen sourced3c05d1; the registered freeze and dev/train interaction exports also completed. Formal A/B/C/D/R and Navtest have not run. The exact registered FLUX VAE is now available from the Diffusers team's public standalone distribution; its config and weight hashes are identical to the pinned BFL artifacts.

## Reproducible public VAE download — tested

The BFL download entry requires authentication. The Diffusers team distributes the byte-identical standalone VAE publicly, so no token is needed for this command. Both the immutable public revision and the canonical BFL config/weight hashes are verified. Existing different files are rejected; partial downloads can be resumed.

```bash
python -m tools.foresight.download_flux_vae \
  --root "$ART/generic/flux_vae" \
  --identity-output "$ART/generic/flux_vae_identity.json"
```

The model is in `$ART/generic/flux_vae/vae`; use the parent as `--vae-root` for this campaign. `reports/ddp_shared_foresight/vae_download` records actual download and CPU load checks.

## Already tested commands

```bash
python -m pytest tests/foresight -q

# Actual public weights, random driving modules, no update; release needed GPU first.
python -m tools.foresight.check_batching --config configs/foresight/A.yaml \
  --current-root "$ART/student_train_v1" --output "$ART/new_batch_check.json" \
  --campaign-root "$ART" --run-id new_batch_check

# Real startup, not a complete experiment. Four-GPU batch32/micro8 tested at56bf61e.
torchrun --standalone --nproc_per_node=4 -m tools.foresight.train_student \
  --config configs/foresight/A.yaml --data "$ART/student_train_v1" \
  --campaign-root "$ART" --run-id new_real_startup --scope startup --limit 64 \
  --global-batch 32 --micro-batch 8 --updates 4 --schedule-updates 100 --warmup 1 \
  --save-every 100 --milestones '' --max-seconds 1800 --campaign-gpu-hours 120 --deterministic
```

For a2+2resume test, use `--stop-after 2` on the first allocation, then the same source/config/command with `--resume --acknowledge-stop` and no `--stop-after`. Do not resume a COMPLETE run or change its source/config/data/precision/device count. The demonstrated exactness boundary is the same deterministic setup; cross-device bitwise equality is not promised.

`tools.foresight.prepare_student_data` has prepared independent current JSON/ego label files for101592train and1696dev scenes. Teacher data uses the same ordered token/log lists. Never pass ego files, future caches or teacher labels to `encode_current` or `predict_action`.

Current-only checkpoint export was tested with two dev scenes from a new-campaign startup checkpoint. It restores FP32 masters, strips auxiliary heads and saves one ego sample per scene. Exporting startup weights is only plumbing evidence, never a formal score.

```bash
python -m tools.foresight.export_predictions --training-run "$RUN" --checkpoint-tag "$TAG" \
  --current-root "$ART/student_dev_v1" --output "$ART/new_export" \
  --campaign-root "$ART" --run-id new_export --sampling-seed 42 \
  --limit 2 --max-seconds 1800
```

## Target preparation and subsequent stages

Full teacher training and freeze selection were registered before launch. `freeze_teacher` rejects incomplete training; it selects only the registered milestones after all30epochs finish. The selected teacher is shared by C/D.

`python -m tools.foresight.summarize_teacher --run "$ART/teacher_full30_v1" --output "$ART/new_teacher_snapshot"` has been tested on the running full-data teacher. It snapshots complete training log records, fixed development milestones, role/fallback counts, static/moving and complete/partial groups, log-cluster intervals and standalone learning curves. Peer removal results remain GT-conditioned mechanism diagnostics.

```bash
python -m tools.foresight.freeze_teacher --teacher-run "$ART/teacher_full30_v1" \
  --registration "$ART/teacher_full_registration.json" --output "$ART/frozen_teacher.json"
python -m tools.foresight.export_interaction_targets --data "$ART/teacher_data_full_v1" \
  --teacher-run "$ART/teacher_full30_v1" --checkpoint "$SELECTED_TEACHER_CHECKPOINT" \
  --frozen-teacher "$ART/frozen_teacher.json" --split train --output "$ART/interaction_train_v1" \
  --campaign-root "$ART" --run-id interaction_train_v1
# Repeat split dev with distinct output/run ID. Export always re-encodes after ego-future masking.

python -m tools.foresight.cache_future_latents \
  --split-manifest reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
  --vae-root "$VERIFIED_FLUX_ROOT" --vae-identity "$VERIFIED_FLUX_IDENTITY" \
  --split train --shard 0 --shards 8 --output "$ART/future_train_v1" \
  --campaign-root "$ART" --run-id future_train_shard0_v1
# Complete all shards and dev separately, retaining missing-frame masks and every ego scene.
```

Before encoding, register the already authorized local VAE using `python -m tools.foresight.prepare_flux_identity --root "$VERIFIED_FLUX_ROOT" --output "$VERIFIED_FLUX_IDENTITY"`. The directory must contain the official `vae/config.json` and `vae/diffusion_pytorch_model.safetensors` at revision741f7c3ce8b383c54771c7003378a50191e9efe9. Both are checked against pinned public Git/LFS metadata. An empty file inventory, an arbitrary caller-declared weight hash or a different revision is rejected before loading. This command does not obtain access or download weights. Actual CPU encoding/decoding on three real current images has now passed; full target caching remains NOT_RUN.

The full timestamp/file audit has run independently of the VAE:101592train and1696dev scenes, zero failures, all with at least one valid future horizon.101344train/1695dev scenes contain all nine future views. Other cases have timestamp gaps, with no scene removed. Available train frames deviate at most16.415ms from their requested horizon, within the registered50ms tolerance. This proves file availability, not image decoding or VAE quality.

```bash
python -m tools.foresight.audit_future_frames \
  --split-manifest reports/ddpolicy_vehicle_from_scratch/NAVTRAIN_PARTITION.json \
  --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
  --fallback-sensor-root "$FALLBACK_SENSOR_ROOT" --workers 4 \
  --output "$ART/new_future_availability" --campaign-root "$ART" --run-id new_future_availability
```

Real-gradient calibration needs both completed, identity-checked target caches. Its single fixed training-only rule measures W and first/last Qwen q-projection gradients, setting each auxiliary's median norm to25%of the main norm; it records all missing/zero-label scenes. This rule is a starting gradient calibration, not a test-score search. Nonzero weights are required by the student trainer for enabled tasks.

```bash
FORESIGHT_LAMBDA_VIS=1 FORESIGHT_LAMBDA_INT=1 python -m tools.foresight.calibrate_auxiliaries \
  --config configs/foresight/D.yaml --data "$ART/student_train_v1" \
  --future-root "$ART/future_train_v1" --future-identity "$FUTURE_TRAIN_ID" \
  --interaction-root "$ART/interaction_train_v1" --interaction-identity "$INTERACTION_TRAIN_ID" \
  --output "$ART/gradient_calibration.json" --campaign-root "$ART" --run-id gradient_calibration
```

Apply the resulting integer VAE geometry and shared nonzero lambda values to the B/D and C/D configs, then run matched short training and freeze the common formal plan. The100000-update/global32plan remains provisional until that validation; do not mislabel a short run as the full experiment. Teacher, cache, loading, failures and evaluation consume the new campaign budget. Existing-resource ceiling6000GPUh, no external rental.

Formal training uses `--scope formal` without `--limit`, the registered common update/schedule lengths and seed, and only the relevant explicit cache roots/identities. Independent fresh starts are mandatory. `evaluate_auxiliary_tasks` evaluates every1/2/4shorizon/view and the same-cache copy-current reference; no latent MSE is a PDMS result.

Final Navtest export rejects diagnostic checkpoints, a truncated population and absent/changed model/source/data/sampling locks. A final lock must list the exact formally selected checkpoint hashes, evaluator sourceSHA, complete current-data identity/count, inference steps and five sampling seeds42–46. Final official scoring and paired planning results are still NOT_RUN. The exported NPZ contains only the executable ego trajectory; the official scorer must retain the full traffic environment.

The main checkpoint-selection rule is the common registered complete training endpoint for all arms. The endpoint length is frozen after the matched short fits; this avoids per-arm best-epoch selection. The development set still measures learning quality before final test execution. `lock_navtest` verifies the completed training identities/exposure and exact matching five-run development checkpoints. No lock has been generated for a real model yet.

```bash
# NOT_RUN: requires completed formal R/A/B/C/D and the complete development report.
# MODELS is a JSON list of {"training_run": ..., "checkpoint_tag": ...}.
python -m tools.foresight.lock_navtest --models "$MODELS" \
  --development-report "$PAIRED_REPORT/summary.json" --current-root "$NAVTEST_CURRENT_ROOT" \
  --metric-index "$NAVTEST_METRIC_INDEX" --output "$FINAL_LOCK"
# Execute export_predictions from this same clean, frozen source with --final-lock "$FINAL_LOCK".
```

## Official CPU scoring and paired analysis

The existing official NAVSIM v1.1 worker has been validated on two real development scenes from the startup export, with zero failed rows. This confirms the pipeline only. It is not a full development result. Use the `navsim` Python environment for CPU scoring; the wrapper loads no driving model and consumes only the exported executable ego trajectory. It can run concurrently with GPU export. Incomplete partitions retain rows and are finalized by the identical command with `--resume`.

```bash
CUDA_VISIBLE_DEVICES='' python -m tools.foresight.score_pdms \
  --devkit "$NAVSIM_V1_DEVKIT" --metric-index "$FULL_OFFICIAL_METRIC_INDEX" \
  --current-index "$ART/student_dev_v1/index.json" --predictions "$PREDICTIONS" \
  --output "$SCORES" --campaign-root "$ART" --run-id "$SCORE_RUN_ID" --workers 16

# Formal five-seed aggregation is NOT_RUN until all registered models finish.
python -m tools.foresight.summarize_experiments --registry "$SCORE_REGISTRY" \
  --output "$PAIRED_REPORT" --split dev
```

The registry is a JSON list with `arm`, `training_seed`, `sampling_seed`, and `score_dir` for each result. Formal groups require the same checkpoint for inference seeds42–46; training seeds remain separate. The analysis preserves failed samples at zero and marks the benchmark invalid, averages inference seeds within scenes, and resamples complete logs for paired intervals. It writes B−A, C−A, D−B, D−C and A−R scene/log CSVs; missing comparisons remain NOT_RUN. A one-log diagnostic produces no confidence interval. `--diagnostic --sampling-seeds 42` is allowed only for a development pipeline check, never for the main table or Navtest.

Offline ego ADE/FDE and yaw errors have also been checked on the same two development exports. This command reads separate GT labels after prediction; it loads no model and does not modify the exported trajectory. Failed rows invalidate their group means instead of disappearing from the denominator.

```bash
python -m tools.foresight.evaluate_ego --predictions "$PREDICTIONS" \
  --current-root "$ART/student_dev_v1" --output "$EGO_FIT_REPORT"
```

The read-only current Navtest cache has12146scenes/136logs, complete current JSON/three-image paths, and no token/log overlap with either train or development. Its token/log population exactly matches the official metric cache. Only current images, navigation and observed ego poses are reused; no old driving prediction or hidden cache is used. This data check did not run a Navtest model. For a locked future Navtest export, `evaluate_ego` accepts `--processed-root` solely on this offline label side.

GPU policy applies to local and vla-zt2: verify and stop only the pressure parents on allocated cards before work; restore pressure after the allocation exits. Current parent/worker identities are in `$ART/gpu_pressure` and job launch JSONs; re-check `/proc` before signalling because PIDs can be reused.
