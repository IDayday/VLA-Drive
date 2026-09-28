# DDP vehicle-only camera training

This is a new public-generic-to-driving campaign. Never initialize it from released DriveDreamer-Policy driving weights, Qwen3-VL-2B-WorldAction, previous foundation/Reader/graph/scorer weights, driving hidden caches, or a small-fit checkpoint. `GENERIC_SOURCES.json` pins public model/config/tokenizer bytes; loading checks precede model construction. Unknown generic pretraining data are not claimed fully auditable.

The official architecture reference is [DriveDreamer-Policy](https://github.com/youngzhou1999/DriveDreamer-Policy/tree/8cefcac46e5944add529e1be19cba78bc06cc2bd). The implementation retains Qwen3-VL-2B, original Wan and PPD auxiliary tasks, 3 current front cameras, original 1536-wide/24-block DiT, FM repeat8 and Euler10. A is the framework baseline; B adds a vehicle Reader, box/motion supervision and joint actor-time states; C has identical parameters and adds role completion. Main FM remains present on every update. B/C add weight0.1 auxiliary every4updates, with B all-hidden and C role-hidden; their final phase is all-hidden. The unique executed ego is joint slot0. Original video/depth ego auxiliary regressors are loss-only and never feed a second execution head.

Only `vehicle` annotations enter new labels, before capacity truncation. Parked cars remain eligible. Native images and official PDM environments keep every object category. Inference accepts current cameras/calibration/navigation/ego history only; no GT box/track/future is read by the prediction endpoint. Hungarian loss assignment has no class-correctness or2m gate. The fixed2m gate in offline detection metrics is separate from training assignment. Neighbor future yaw is unmodeled, canonical zero before encoding/after every Euler step; missing future xy labels do not freeze deployment integration.

## Verified environment and data

Commands below were run with `/root/miniconda3/envs/ddp/bin/python` for learning/export and `/root/miniconda3/envs/navsim/bin/python` for official CPU scoring. Runtime versions and source hashes should be kept with each run. No shared environment was upgraded. Optional DA3 export/pose-alignment dependencies were made lazy because this task only needs depth inference.

The complete official navtrain token population is103288/1192logs. Fixed dev is1696/16whole logs; train is101592/1176logs, with no token/log overlap. Source split SHA256 is `1eef553afcf0896674c109b40853e81c2c2acb04e4dee48973c57952e717b46d`. New vehicle label identity is `59a0f36f22f33cc0e10bc82328191d126a9acfdc89b8c59b3931873ec212a423`. Every source scene was retained. A first cache attempt had448 missing camera paths; the second declared sensor root repaired all448 in a NEW cache, preserving the failed attempt. All309864 current camera transforms are the expected uniform1920x1080→1024x576 resize, so the original generic depth labels align with the current views.

These environment variables name the existing local artifacts; change them together for another machine:

```bash
export DDP_PYTHON=/root/miniconda3/envs/ddp/bin/python
export DDP_ARTIFACTS=/mnt/project/ddpolicy-vehicle-joint-artifacts/20260928
export DDP_ASSETS=/mnt/project/DriveDreamer-Policy
export DDP_META=/mnt/navsim/navsim_dataset/meta/train
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
```

Fresh installations need the original repository requirements and the pinned generic weights. `verify_sources` checks the local files against public immutable Hugging Face revisions, rather than trusting directory names:

```bash
"$DDP_PYTHON" -m tools.ddpolicy_vehicle.verify_sources \
  --asset-root "$DDP_ASSETS" --output /new/path/generic_provenance
"$DDP_PYTHON" -m tools.ddpolicy_vehicle.configuration \
  --asset-root "$DDP_ASSETS" \
  --source-manifest "$DDP_ARTIFACTS/generic_provenance_verified/sources.json" \
  --output /new/path/formal_configs --seeds 42 43
```

Use new output directories. Data builders are `prepare_data`, `prepare_depth`, and `prepare_current`; each exposes paths through `--help`. Label and current-observation outputs are separate. No prediction cache or driving teacher is involved. The current Navtest cache contains only allowed observations, and creating it did not run a model or inspect PDM values.

## Executed checks and ongoing small fit

```bash
CUDA_VISIBLE_DEVICES='' "$DDP_PYTHON" -m pytest tests/vehicle_joint -q
CUDA_VISIBLE_DEVICES=7 DDPOLICY_TEST_CUDA=1 "$DDP_PYTHON" \
  -m pytest tests/vehicle_joint/test_cuda_state.py -q
```

Actual results:20CPU passed, the explicit CUDA test passed. Full-camera A/B backward gradients are finite; B's vehicle Reader and vehicle heads have nonzero real-scene gradients. A/B shared driving tensors and B/C complete driving tensors hash identically at initialization. Single-,two-,eight-GPU startup training completed16real optimizer updates/176scene presentations in total. Continuous4 and2+2 restore preserve all RNG/progress states; maximum BF16 model-parameter difference was2.98e-8, so this is not a bitwise-resume claim.

The independently registered small fits run in the immutable worktree `/mnt/project/VLA-Drive-ddpolicy-smallfit-20260928`, source `5076336f657980d46631657d2d2dfa8c5c007281`. Each A/B/C uses the same64training scenes, two GPUs, globalbatch16, microbatch2,512updates, lr1e-5, diagnostic warmup32/horizon512, and final64all-hidden. The temporary campaign diagnostic ceiling is20GPU-hours including earlier work. This diagnostic schedule is different from the official100000-step schedule and cannot establish full-training performance. Its checkpoints cannot enter the Navtest exporter or initialize a formal model.

The following reproduces one launched command. Replace the run ID for a NEW repeat; do not restart the active run without checking its state:

```bash
cd /mnt/project/VLA-Drive-ddpolicy-smallfit-20260928
CUDA_VISIBLE_DEVICES=2,3 "$DDP_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=2 -m tools.ddpolicy_vehicle.train \
  --config "$DDP_ARTIFACTS/small_fit_v1/B.yaml" \
  --processed-root "$DDP_META" \
  --vehicle-root "$DDP_ARTIFACTS/vehicle_targets_v1_complete" \
  --depth-root "$DDP_ARTIFACTS/depth_labels_v1" \
  --tokens "$DDP_ARTIFACTS/small_fit_v1/tokens.json" \
  --campaign-root "$DDP_ARTIFACTS" --run-id small_fit_B_seed42_001 \
  --global-batch 16 --micro-batch 2 --updates 512 --save-every 256 \
  --milestones 64,128,256,512 --small-fit --campaign-gpu-hours 20 --max-seconds 10800
```

For a safely PAUSED run, use the **same** code/config/data/world size/run ID and add `--resume --acknowledge-stop`; remove an elapsed `--stop-after` boundary when intentionally continuing. Completed runs cannot be silently extended. An incomplete write cannot become the `latest` pointer. `--resume-tag` can select an explicitly complete saved tag. Periodic saves retain only this run's newest complete periodic checkpoint; milestone, pause and endpoint checkpoints are immutable. Every resume attempt has its own charged GPU time. The old startup ledger's cumulative counters are not summed as if they were incremental updates.

A logging-only omission in the active small-fit source leaves Base's `coordinates.ego` field zero; its actual main-FM supervision is present. Derive Base's count from scene exposures×repeat8×8points×4coordinates. This logging field is corrected for subsequent runs; active training source was not modified.

## Formal training — NOT_RUN

The original recipe is100000updates, globalbatch32, AdamW1e-5, betas0.9/0.95, eps1e-8, weight decay0.001, warmup5000, cosine floor5e-7. At101592scenes, a pass is3175updates with a24scene final batch.100000updates expose3199752scenes, not exactly3200000. B/C use their final10000updates without the auxiliary task. Formal models restart independently from generic/random initialization, including after a successful small fit.

Eight-GPU startup measured6.6–7.0sec/update after the first update. Five full runs extrapolate to about7500GPU-hours of training; complete evaluation, cold data I/O, failures and reserves are additional. The registered full campaign cap is8000GPU-hours on existing authorized resources, including the diagnostic stage. This default follows the optional budget question receiving no reply; user changes override it. No rented or additional device is included. The command below is **NOT_RUN** while small-fit diagnostics are active; it must not inherit the old48GPU-hours:

```bash
export DDP_GPU_HOURS_CAP=8000  # Registered new campaign cap; includes all prior work.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$DDP_PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m tools.ddpolicy_vehicle.train \
  --config /new/path/formal_configs/A_seed42.yaml \
  --processed-root "$DDP_META" \
  --vehicle-root "$DDP_ARTIFACTS/vehicle_targets_v1_complete" \
  --depth-root "$DDP_ARTIFACTS/depth_labels_v1" \
  --tokens "$DDP_ARTIFACTS/vehicle_targets_v1_complete/train_tokens.json" \
  --campaign-root "$DDP_ARTIFACTS" --run-id formal_A_seed42 \
  --global-batch 32 --micro-batch 4 --updates 100000 --save-every 1000 \
  --milestones 0,1000,5000,10000,25000,50000,75000,90000,100000 \
  --campaign-gpu-hours "$DDP_GPU_HOURS_CAP" --max-seconds 86400
```

Run B/C seed42 with their own configurations/IDs, then the matched B/C seed43 pair. A seed43 is optional if resources permit. Stop/resume preserves the100000-step scheduler; a24-hour allocation boundary is a safe pause, not an experiment completion. Only allocated idle GPUs may be used; no other task is stopped.

## Current-camera prediction and evaluation

The FP32 exporter loads only this campaign's complete checkpoint identity and recovers trainable FP32 optimizer masters. `--local-checkpoint-cache` makes hash-checked exact copies on allocated local disk to avoid slow network mmap consolidation. Prediction reads `current_*` only, never the vehicle label root or metric cache. One joint sample is produced per scene/seed; final ego is slot0 plus the original single xy/yaw decoder.

The two-scene dev startup export and official scorer smoke have actually run, with0export/scoring failures andPDMS0. It is a near-random4-update model, **not** a baseline result or a development score. Full1696development and complete12146Navtest results remain NOT_RUN.

```bash
"$DDP_PYTHON" -m tools.ddpolicy_vehicle.export_predictions \
  --training-run "$DDP_ARTIFACTS/training/formal_B_seed42" \
  --checkpoint-tag milestone_100000 --current-root "$DDP_ARTIFACTS/current_dev_v1" \
  --output /new/path/dev_B_seed42_sample42 --sampling-seed 42 \
  --campaign-root "$DDP_ARTIFACTS" --run-id dev_B_seed42_sample42 \
  --max-seconds 7200 --local-checkpoint-cache /tmp/ddpolicy-vehicle-checkpoints

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/navsim/bin/python -m tools.local_interaction_mask_v2.score_async \
  --devkit /mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1 \
  --index /mnt/project/local-interaction-mask-v2-artifacts/20260927/official_dev_metric_v2/cache_index.json \
  --predictions /new/path/dev_B_seed42_sample42 --output /new/path/dev_B_scores42 \
  --workers 12 --chunk 8 --export-shards 1
```

Repeat the same sampling seeds42/43/44/45/46 for all arms. These are the existing paired-project inference seeds (`DriveDreamer-Policy-perf/starVLA/rl/flow_grpo/evaluation.py`); no RL implementation or weights are used here. Average metric scores, never trajectories or oracle selections. The checkpoint-selection rule must be frozen on complete dev results before Navtest. `lock_final` binds complete dev evidence, checkpoint hashes, evaluation source and seed list; Navtest export requires the lock and exact12146/136log current population. A missing second seed is explicitly NOT_RUN and a one-sided B/C second-seed comparison is rejected.

Offline vehicle evaluation: `tools.ddpolicy_vehicle.evaluate_vehicles` writes one row per GT target, including misses, missing endpoints and failed exports. `paired_results` retains all official scene rows and resamples whole logs for the paired interval. Its interval does not represent training-seed variability. Source raw vehicles, ROI/FOV supervised targets, detections and joint-selected targets have distinct denominators. Current geometric support is not an occlusion confidence. Neighbor CV is not reported without legal current velocity. Joint trajectories always come from the same sample.

Training-only auxiliary modules remain loaded in the current exporter for strict checkpoint verification; this is an implementation cost, not extra inference supervision. Further deployment memory optimizations would need parity checks. No v2EPDMS, scorer, oracle bank, Navtest-hard expansion or nonvehicle prediction branch is part of this campaign.

## Optimizer validity correction

The early two-GPU logs counted optimizer calls that did not change parameters. Do not use those runs as learning evidence. The installed DeepSpeed0.16.9 FusedAdam header stores tensor sizes as int32; a ZeRO partition above2^31 silently becomes a no-op. GPU reproduction and historical run corrections are in `reports/ddpolicy_vehicle_from_scratch/optimizer_stasis/`. The trainer now bounds same-hyperparameter groups to500million elements and checks actual FP32-master changes on EVERY step before incrementing progress. Earlier raw logs and costs remain preserved. Corrected experiments start afresh; no invalid diagnostic checkpoint is resumed. The provisional xy-scale correction is not selected because its evidence was confounded by this optimizer defect.
