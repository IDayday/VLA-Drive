# V1.1 signal rehabilitation (active)

Base `393c53b`, isolated branch `feature/structured-world-v1p1-signal-rehab-20260927`.
Old campaign is sealed. New caps: 24000 optimizer updates and 48 GPU-hours (stop at either).
No navtest tuning, no changes to camera/time/candidate/FM protocol.

## Executed P0

```bash
cd /mnt/project/VLA-Drive-structured-world-v1p1-20260927
export PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WORLD_PYTHON=/root/miniconda3/envs/ddp/bin/python
export OLD=/mnt/project/structured-world-v1-artifacts/20260926
export NEW=/mnt/project/structured-world-v1p1-artifacts/20260927
"$WORLD_PYTHON" -m pytest tests/structured_world_v1p1 -q
```

The following commands were executed. **Output paths must be fresh**; use new versioned output directories to reproduce, never overwrite completed artifacts.

```bash
"$WORLD_PYTHON" tools/structured_world_v1p1/reaudit_metrics.py \
 --old-root "$OLD" --targets "$OLD/targets_v6_dev_full" \
 --output "$NEW/metric_reaudit_863ab99"
"$WORLD_PYTHON" tools/structured_world_v1p1/audit_ego.py \
 --config /mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy/config.yaml \
 --manifest "$OLD/train_tokens.json" --data-root "$OLD/dataset_v1" \
 --old-root "$OLD" --output "$NEW/ego_audit_v1"
"$WORLD_PYTHON" tools/structured_world_v1p1/audit_raw_ego.py \
 --manifest "$OLD/train_tokens.json" --data-root "$OLD/dataset_v1" \
 --raw-log-root /mnt/project/onevl_navsim_data/navsim_logs/trainval \
 --output "$NEW/raw_ego_audit_v2"
```

Legacy `tools/structured_world/rediagnose.py` remains explicitly historical.
New live inference `evaluate.py` uses independent metrics, while old historical reports/CSVs remain unchanged.
New re-audit summarizes from its own schema; do not use the historical `summarize.py` on new-schema scene diagnostics.

## P1 real injection regression

```bash
CUDA_VISIBLE_DEVICES=0 DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts \
"$WORLD_PYTHON" tools/structured_world_v1p1/check_injection.py \
 --checkpoint /mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy \
 --vlm /mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction \
 --data-root "$OLD/dataset_v1" --manifest "$OLD/overfit_tokens.json" \
 --output "$NEW/injection_c2f0024"
```

No updates; same 64 scenes, same per-scene noise seed, original mixed precision, same existing .02 absolute/relative tolerance. Compares original, legacy insertion, append-tail, and append-tail + actual zero-gated adapter. Reports native-prefix inputs/positions/hidden, action conditions, and trajectories.

The initial P0/P1 snapshot is superseded by the executed P2/P3 commands below. P4 remains pending. Resume from `CODEX_GOAL_STATE.md` and live process records.

## P2 actual training (updated)

`train_world.py` is the **small-set** trainer. It freezes original Qwen/vision/DiT and trains Reader/reference heads. Classification eligibility uses fixed current-observation references, motion loss uses GT displacement without centre gradients. It caches only frozen current visual features in process, keyed by pixel/grid content; no trainable world hidden state is cached. Accumulation uses a no-grad count pass then one backward per scene, never eight full Qwen graphs.

Runs are pinned to implementation `d537401` in `/mnt/project/VLA-Drive-v1p1-runs-d537401`. This allows continued report/interface development without changing the running experiment implementation. Each branch's16+64 **combined initial isolation budget** is bounded by1000 updates. W_PRE16 consumed304 (checkpoint300 used for matched comparison,4 interrupted updates retained/charged), W_POST16 completed300, each fresh64 run is696 updates. Initial64 launches were interrupted before updates to reconcile this conservative cap; their GPU time is retained. Do not resume the finished16 phases.

Executed core64 command (substitute `W_PRE`/GPU2 or `W_POST`/GPU3):

```bash
cd /mnt/project/VLA-Drive-v1p1-runs-d537401
CUDA_VISIBLE_DEVICES=2 DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts \
"$WORLD_PYTHON" tools/structured_world_v1p1/train_world.py \
 --checkpoint /mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy \
 --vlm /mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction \
 --data-root "$OLD/dataset_v1" --manifest "$OLD/overfit_tokens.json" \
 --targets "$OLD/targets_v6_train8192" --config configs/structured_world_v1p1/W_PRE.json \
 --output "$NEW/W_PRE_64_bounded696_d537401" --ledger "$NEW/budget_ledger.json" \
 --run-id W_PRE_64_bounded696_d537401 --steps 696 --limit 64 --batch 8 --eval-every 100
```

Do not launch duplicates of currently running jobs. After verified termination and ledger reconciliation only, resume with:

```bash
cd /mnt/project/VLA-Drive-structured-world-v1p1-20260927
"$WORLD_PYTHON" tools/structured_world_v1p1/resume_world.py \
 --checkpoint "$NEW/W_PRE_64_bounded696_d537401/checkpoint.pt" \
 --worktree /mnt/project/VLA-Drive-v1p1-runs-d537401
```

The wrapper refuses live/unreconciled jobs and checkpoints behind charged updates; it respects approved phase end steps. Real W_PRE16 and W_POST16 resumes have executed successfully. Trained Reader/heads/reference buffers/new gate0 adapter strict roundtrip and target contamination checks pass on real GPU; see CHECKPOINT_REGRESSION.json. Default fast attention is not claimed to provide topology-independent bitwise training continuation.

To export full-GT geometry/motion and failure objects from a finished checkpoint:

```bash
"$WORLD_PYTHON" tools/structured_world_v1p1/evaluate_world.py \
 --trained "$NEW/W_POST_16_d537401/checkpoint.pt" --manifest "$OLD/overfit_tokens.json" \
 --targets "$NEW/targets_v6_full128" --limit 16 \
 --output "$NEW/W_POST_16_fullgt_eval300" --ledger "$NEW/budget_ledger.json" \
 --run-id W_POST_16_fullgt_eval300
"$WORLD_PYTHON" tools/structured_world/visualize.py \
 --predictions "$NEW/W_POST_16_fullgt_eval300/predictions" \
 --target-cache "$NEW/targets_v6_full128" --sensor-root / \
 --output "$NEW/W_POST_16_visualizations300" --matching geometry --limit 16
```

These16-scene commands completed for both branches:32 figures, plus scene/object CSV. Use fresh output paths for reproductions.

## P3 actual external prior

Depth Anything V2 Large official weights were verified against Hugging Face LFS SHA256. This is **pretrained visual backbone plus new calibrated BEV**, not pretrained BEV. Its Large weights license is CC-BY-NC-4.0, with source/license references recorded in PROVIDER_AUDIT.json. Only RGB current CAM_F0/L0/R0, calibration and fixed grid enter extraction.

```bash
export WORLD_PRETRAINED_VISUAL_WEIGHTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts/depth_anything_v2_vitl.pth
"$WORLD_PYTHON" tools/structured_world_v1p1/extract_pretrained_bev.py \
 --weights "$WORLD_PRETRAINED_VISUAL_WEIGHTS" \
 --weight-sha256 a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345 \
 --manifest "$NEW/provider_probe128_tokens.json" --observations "$OLD/targets_v6_train8192/observations" \
 --sensor-root / --output "$NEW/pretrained_bev_128_345d42c" \
 --ledger "$NEW/budget_ledger.json" --run-id pretrained_bev_128_345d42c
```

The first128 cache includes64 training scenes and64 candidate token-only-held-out scenes. The candidate split was **not** used for adaptation/validation. Before adaptation, holdout was fixed by complete log: `train_log_holdout64_tokens.json`,57 training logs vs59 held-out logs with zero overlap. A separate64 feature extraction used this manifest and output `pretrained_bev_log_holdout64_dd789c3`. Original three-camera/timing protocol unchanged.

The independent perception probe completed600 updates/batch8 =4800 presentations/75 epochs on64 scenes, then a single fixed holdout evaluation. No Qwen or backbone updates:

```bash
"$WORLD_PYTHON" tools/structured_world_v1p1/probe_provider.py \
 --train-manifest "$OLD/overfit_tokens.json" --holdout-manifest "$NEW/train_log_holdout64_tokens.json" \
 --train-features "$NEW/pretrained_bev_128_345d42c" \
 --holdout-features "$NEW/pretrained_bev_log_holdout64_dd789c3" \
 --targets "$NEW/targets_v6_full128" \
 --vlm-config /mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-WorldAction/config.json \
 --source-audit reports/structured_world_v1p1/PROVIDER_AUDIT.json \
 --output "$NEW/provider_probe64_v1" --ledger "$NEW/budget_ledger.json" \
 --run-id provider_probe64_v1 --steps 600
```

Source SHA/arguments/cache fingerprints for exact executed versions are in the run manifest/ledger. Config `PRETRAINED_BEV_INTERFACE.json` and `check_pretrained_injection.py` provide tested online Qwen/DiT connectivity after the independent probe. Provider is required at deployment; measured provider ~.24s and one-scene complete policy~.60s (not a paired isolated latency benchmark). Closed gate preserves native output exactly; nonzero gate changes the action. OOD blanking tests only connectivity.

P4 four-epoch8192-scene pretraining and planning commands remain NOT_RUN and not yet implemented; only proceed after64-scene learnability gate. The small-set1000-step script is not a substitute for that training budget. Both convergence repair rounds remain available, each must document a concrete fault hypothesis before launch.
