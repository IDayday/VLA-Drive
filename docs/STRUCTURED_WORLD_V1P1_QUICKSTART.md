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

P2/P3/P4 training commands are not yet implemented/validated. Do not reuse old 1000-step-limited pilot scripts for the required four 8192-scene pretraining epochs. Resume current development from `CODEX_GOAL_STATE.md` and the live process records.
