# Executed validation and explicit launch commands

Final source: `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64`. Commands below run in `/mnt/project/DriveVLA-M0-planreg-wm-v2-review-20260908`. Output directories are immutable: use new paths for a rerun, not overwrite an existing artifact.

## Environment

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUPLAN_MAPS_ROOT=/mnt/navsim/maps CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python
ROOT=/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908
CFG=navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml
export PLANREG_BASE_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-base-aligned
export PLANREG_VQA_VLM_PATH=/mnt/project/DriveVLA-M0-models/planreg-formal/InternVL3-2B-driving-vqa-dense
export PLANREG_V2_NORMALIZER=$ROOT/input_train48/normalizer.json
export PLANREG_V2_SHARED_INIT=$ROOT/shared_init_std02_seed0.pt
```

Actual environment: Python3.9.25, torch2.5.1+cu124, Transformers4.57.6, PEFT0.17.1, A80080GB. No environment package upgrades were performed.

## Final regression and production audits (executed)

```bash
$PY -m compileall navsim/agents/EpisodeDrive
$PY -m pytest -q tests/test_planreg_v2*.py \
  tests/test_drivor_scorer_parity.py tests/test_legacy_forward_parity.py \
  tests/test_student_checkpoint_export.py tests/test_read_only_register_attention.py \
  tests/test_register_patch_parity.py tests/test_internvl_planning_registers.py \
  tests/test_vision_qv_lora.py tests/test_future_image_paths.py \
  --junitxml=$ROOT/logs/final_regression.xml
$PY scripts/audit_drivor_scorer_parity.py
$PY scripts/audit_planreg_v2_review.py --config "$CFG" --output "$ROOT/final_audit"
$PY scripts/audit_planreg_v2_pair.py --base-config "$CFG" \
  --vqa-config navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml \
  --output "$ROOT/final_vlm_pair.json"
$PY scripts/audit_planreg_v2_shared_pair.py --base-config "$CFG" \
  --vqa-config navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml \
  --output "$ROOT/final_shared_pair.json"
git diff --check
```

Red evidence was obtained at `0fde322` with the same production-regression file over the unchanged base. See `logs/red.log`: 14 semantic failures. Green `logs/green_initial.log`: 14 passed. Final expanded suite: 121 passed, zero fail/error/skip. The exact passed test IDs and source hashes are included in the bundle; no full-repository test count is claimed.

## New data/init construction (executed, then final-source revalidated)

```bash
$PY scripts/build_planreg_v2_input_cache.py \
  --logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --sensors /mnt/project/DriveDreamer-Policy/navsim_raw/sensor_blobs/trainval \
  --metric-metadata /mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full/metadata/metric_cache_navtrain_full_metadata_node_0.csv \
  --source-vector-frame ego --smoke-scenes 48 --output "$ROOT/input_train48"
$PY scripts/build_planreg_v2_input_cache.py \
  --logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --sensors /mnt/project/DriveDreamer-Policy/navsim_raw/sensor_blobs/trainval \
  --metric-metadata /mnt/project/DriveVLA-M0-stage2/cache/metric_cache_navtrain_full/metadata/metric_cache_navtrain_full_metadata_node_0.csv \
  --source-vector-frame ego --smoke-scenes 8 \
  --exclude-recorded-drives "$ROOT/input_train48/recorded_drives.json" --output "$ROOT/input_probe8"
$PY scripts/create_planreg_v2_shared_init.py --config "$CFG" --seed 0 \
  --output "$ROOT/shared_init_std02_seed0.pt"
```

The final-source read-only regeneration of all 56 raw targets is in `CACHE_REVALIDATION.json`; original targets and statistics match bitwise. These are bounded diagnostic caches, not final-fit approval.

## Real smoke / DDP / restore / export (executed)

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/train_planreg_v2.py --config "$CFG" \
  --manifest "$ROOT/input_train48/manifest.json" --output "$ROOT/final_smoke32" \
  --run-step-limit 32 --microbatch 1 --accumulate 1 --workers 2 \
  --probe-manifest "$ROOT/input_probe8/manifest.json"
CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --standalone --nproc_per_node 2 \
  scripts/train_planreg_v2.py --config "$CFG" --manifest "$ROOT/input_train48/manifest.json" \
  --output "$ROOT/final_ddp4" --run-step-limit 4 --microbatch 1 --accumulate 2 --workers 2
CUDA_VISIBLE_DEVICES=0 $PY scripts/validate_planreg_v2_resume.py --config "$CFG" \
  --manifest "$ROOT/input_train48/manifest.json" --output "$ROOT/final_resume" --accumulate 2
CUDA_VISIBLE_DEVICES=0 $PY scripts/validate_planreg_v2_agentinput_export.py \
  --training "$ROOT/final_smoke32/last.ckpt" --student "$ROOT/final_student.ckpt" \
  --manifest "$ROOT/input_train48/manifest.json" \
  --logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --sensors /mnt/project/DriveDreamer-Policy/navsim_raw/sensor_blobs/trainval \
  --output "$ROOT/final_agentinput_export.json"
$PY scripts/audit_planreg_v2_precision.py --checkpoint "$ROOT/final_smoke32/last.ckpt" \
  --output "$ROOT/final_precision.json"
CUDA_VISIBLE_DEVICES=0 $PY scripts/migrate_planreg_v1_to_v2.py --config "$CFG" \
  --input /mnt/project/DriveVLA-M0-formal-runs/formal_dual_init_gb128_asyncpdm_20260903/formal_base_init_wm_seed0/checkpoints/epoch_27_final.ckpt \
  --output "$ROOT/final_migration.ckpt"
```

Real run logs are under `$ROOT/logs/`; restore subprocess logs under `$ROOT/final_resume/`. Fixed budget is 21,789, not 32 or eight. All final evidence is from the tested source snapshot. Development failures are retained separately and not counted as passing runs.

## Bounded GB128 profiles

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_planreg_v2.py --config "$CFG" --manifest "$ROOT/input_train48/manifest.json" \
  --output "$ROOT/gb128_8x4_acc4" --profile-only --microbatch 4 --accumulate 4 --workers 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_planreg_v2.py --config "$CFG" --manifest "$ROOT/input_train48/manifest.json" \
  --output "$ROOT/gb128_8x8_acc2" --profile-only --microbatch 8 --accumulate 2 --workers 2
```

Each completed profile has four warmup + eight measured optimizer steps. The earlier world1/micro1/accum128 attempt was explicitly aborted before its first optimizer update and is retained as ABORTED, not hidden or counted as passed. No fourth profile is authorized/run. Timing and selection are in `GB128_PROFILE.json`.

## Formal training / resume / deployment / evaluation (NOT RUN)

Do not reuse smoke weights or its statistics. First build full 103,288-scene **new-version** cache and matching original-GT statistics, audit Base/VQA/shared identity, and validate the full dataset's maximum tile envelope against an actual source-bound GB128 lock. Then set:

```bash
export PLANREG_V2_MANIFEST=/new/full_v2p2/manifest.json
export PLANREG_V2_NORMALIZER=/new/full_v2p2/normalizer.json
export PLANREG_V2_SHARED_INIT=/new/formal_v2p2_shared_seed0.pt
export PLANREG_V2_LAYOUT=/new/validated_v2p2_layout_lock.json
# One run only, explicit authorization flag:
LAUNCH_FORMAL=1 OUTPUT_DIR=/new/runs/v2p2_base bash local_planreg_wm_v2/train.sh
# Alternative paired initialization, not launched automatically:
LAUNCH_FORMAL=1 OUTPUT_DIR=/new/runs/v2p2_vqa \
  V2_CONFIG=navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml \
  bash local_planreg_wm_v2/train.sh
LAUNCH_FORMAL=1 OUTPUT_DIR=/new/runs/v2p2_base \
  RESUME_CHECKPOINT=/new/runs/v2p2_base/last.ckpt bash local_planreg_wm_v2/resume.sh
$PY scripts/export_planreg_v2_student.py --input /new/runs/v2p2_base/epoch_27_final.ckpt \
  --output /new/runs/v2p2_base/epoch27_student.ckpt
PLANREG_V2_STUDENT=/new/runs/v2p2_base/epoch27_student.ckpt \
  PLANREG_NAVTEST_LOGS=/data/logs/test PLANREG_NAVTEST_SENSORS=/data/sensors/test \
  PLANREG_NAVTEST_METRIC_CACHE=/cache/official_navtest EVAL_OUTPUT=/new/eval/v2p2_official \
  LAUNCH_NAVTEST=1 bash local_planreg_wm_v2/navtest.sh
```

These entries are not performance results. Full normalizer/cache and full tile-envelope verification remain prerequisites; code completion does not imply a trained model above 91.3 PDMS.
