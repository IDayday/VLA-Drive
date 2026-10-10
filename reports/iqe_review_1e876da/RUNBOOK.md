# Executable review and Base monitoring runbook

Repository `/mnt/project/DriveVLA-M0-iqe-review-1e876da`, branch `fix/iqe-formal-readiness-1e876da`, audit base `1e876da626b02af31459f44229248afa0cc55bed`. The Base job remains in the original worktree on `training-vla-zt2-worker-0`; do not restart it or substitute repair config into that process. Repair code commit `687eb2506f0d295ef155fa07ea6a0a1cd7cbcd72`; explicit navtest entry `3859f1c` and later scheduler commits are additive.

Latest priority is direct navtest PDMS on suitable complete Base checkpoints. No validation-set expansion is needed for that operation. All paths below refer to actually discovered data, configs and checkpoints. New output directories are deliberate output destinations, not fictitious input assets. Execute GPU jobs only after the current dispatch policy and available GPUs pass qualification.

```bash
cd /mnt/project/DriveVLA-M0-iqe-review-1e876da
IQE_PY=/root/miniconda3/envs/ddp/bin/python
IQE_METRIC_PY=/root/miniconda3/envs/navsim/bin/python
IQE_NAV=/mnt/project/iqe-review-runtime-1e876da/navtest
IQE_CKPT=/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_014000.pt
```

The full navtest dataset has already been locked and hashed. Re-running lock-data verifies its identity; it does not construct targets or enter training:

```bash
$IQE_PY -m iqe.navtest lock-data --config configs/iqe/review/navtest.yaml --current-root /mnt/project/ddpolicy-vehicle-joint-artifacts/20260928/current_navtest_v1 --metric-index /mnt/project/local-interaction-mask-v2-artifacts/20260927/navtest_cache_index.json --output "$IQE_NAV/dataset" --workers 8
$IQE_PY -m iqe.navtest lock-checkpoint --config configs/iqe/review/navtest.yaml --dataset "$IQE_NAV/dataset/DATASET.json" --checkpoint "$IQE_CKPT" --output "$IQE_NAV/step_014000"
```

The14k export/scoring jobs completed successfully: PDMS85.2969324/100,12146 scenes. The following commands reproduce/reuse their immutable receipts. **Inspect receipts/processes before recovery; do not duplicate any live job.** Export and scoring are independent, atomic per scene and resumable. Source hash must match the evaluation lock. No Scorer, GT, future actors or metric cache is supplied to inference.

```bash
ssh training-vlawm-zt 'cd /mnt/project/DriveVLA-M0-iqe-review-1e876da && PYTHONPATH=. /root/miniconda3/envs/ddp/bin/python scripts/iqe/navtest_export.py --config configs/iqe/review/navtest.yaml --dataset /mnt/project/iqe-review-runtime-1e876da/navtest/dataset/DATASET.json --output /mnt/project/iqe-review-runtime-1e876da/navtest/step_014000 --gpus 0,1,2,3,4,5,6,7 --timeout 14400'
bash scripts/iqe/navtest_score.sh --config configs/iqe/review/navtest.yaml --dataset "$IQE_NAV/dataset/DATASET.json" --output "$IQE_NAV/step_014000" --workers 8 --timeout 14400
cat "$IQE_NAV/step_014000/NAVTEST_RESULT.json"
cat "$IQE_NAV/SCHEDULE_STATUS.json"
```

The finite observer watches **20k/50k/100k** from the same training-run identity, fixed before14k navtest results. It requires a complete checksum-valid checkpoint and a currently idle authorized8-GPU host. It never starts training, stops another job or changes ACTIVE. `SCHEDULE_LAUNCH.json` and `SCHEDULE_RELAUNCH.json` record its PID/command history; SCHEDULE_STATUS contains the current PID. Create `STOP_SCHEDULING` in the navtest root to hold future dispatch without stopping an evaluation already running. If the observer exits, inspect its status/log and ownership before a single restart:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES='' $IQE_PY -u scripts/iqe/navtest_schedule.py --plan "$IQE_NAV/SCHEDULE_PLAN.json"
```

Independent stage_val evaluation/qualification remains implemented for reproducing the already-completed R1 audit. These commands are **not required to run the user-authorized navtest benchmark**. They will retain BASE_QUALITY_UNTESTED/BASE_UNQUALIFIED because the discovered rules are draft; they do not invent a formal acceptance gate from observed scores.

```bash
$IQE_PY -m iqe.cli evaluate-base --config configs/iqe/review/base_qualification.yaml --checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_013000.pt --mode smoke --max-samples 4 --device cpu
$IQE_PY -m iqe.cli qualify-base --config configs/iqe/review/base_qualification.yaml --checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_013000.pt --evaluation /mnt/project/iqe-review-runtime-1e876da/base_qualification/base_evaluation/2aed6d23ed4cb939b54626013571760bac680ad1f3c9d3dca04bd8df5e75e4ab/EVALUATION.json --mode smoke --max-samples 4 --device cpu
$IQE_PY -m iqe.cli freeze-base --config configs/iqe/review/base_qualification.yaml --checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_013000.pt --mode full --max-samples 4 --device cpu
```

The following are real, completed diagnostic runs (existing1-step Base, resampling_only), safe to verify through stage receipts. Their explicit2-step caps do not alter formal defaults:

```bash
$IQE_PY -m iqe.cli run-round --config configs/iqe/review/real_qualification.yaml --round 1 --mode smoke --max-samples 32 --max-steps 2 --device cpu --resume
$IQE_PY -m iqe.cli run-round --config configs/iqe/review/real_qualification.yaml --round 2 --selector-only --mode smoke --max-samples 32 --max-steps 2 --device cpu --resume
$IQE_PY -m iqe.cli run-round --config configs/iqe/review/real_qualification.yaml --round 3 --mode smoke --max-samples 32 --max-steps 2 --device cpu --resume
$IQE_PY -m iqe.cli train-expert --config configs/iqe/review/real_qualification.yaml --round 1 --mode smoke --max-samples 32 --max-steps 2 --device cpu --resume /mnt/project/iqe-review-runtime-1e876da/real_qualification/rounds/round_001/expert/step_000002.pt
$IQE_PY -m iqe.cli train-scorer --config configs/iqe/review/real_qualification.yaml --round 3 --mode smoke --max-samples 32 --max-steps 2 --device cpu --resume auto
$IQE_PY -m iqe.cli load-bundle --config configs/iqe/review/real_qualification.yaml --bundle /mnt/project/iqe-review-runtime-1e876da/real_qualification/rounds/round_003/bundle --mode smoke --max-samples 32 --device cpu
```

Exact-resume evidence is already at `/mnt/project/iqe-review-runtime-1e876da/real_resume/RESULT.json`. Its interrupted step1 folder now has step2: explicit step1 into that same output is intentionally rejected. `--resume auto` picks maximum compatible complete progress; a genuine older-point fork requires a fresh output identity and lineage, preserving original higher-step state. Corrupt-latest fallback requires `resume.allow_fallback: true`, with reasons in RESUME_RESOLUTION.json.

Formal incremental commands below are provided for completeness, **not run or authorized by the current Base-only navtest request**. `configs/iqe/review/formal_posttrain.yaml` points to the actual discovered checkpoint and full scene manifest but currently has an unqualified Base receipt, so formal freeze fails closed. A deliberate new qualified identity must be established before these commands can proceed; navtest monitoring does not silently relabel the old draft receipt.

```bash
IQE_FORMAL=configs/iqe/review/formal_posttrain.yaml
$IQE_PY -m iqe.cli freeze-base --config "$IQE_FORMAL" --checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_013000.pt --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli build-splits --config "$IQE_FORMAL" --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli prepare-metric-contexts --config "$IQE_FORMAL" --roles incremental_fit stage_val selector_cal --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli verify-reference --config "$IQE_FORMAL" --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli cache-features --config "$IQE_FORMAL" --roles incremental_fit stage_val selector_cal dev_report --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli export-candidates --config "$IQE_FORMAL" --round 0 --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli score-candidates --config "$IQE_FORMAL" --round 0 --backend reference --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli build-round --config "$IQE_FORMAL" --round 1 --max-steps 3000 --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli train-expert --config "$IQE_FORMAL" --round 1 --max-steps 3000 --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli audit-frozen --config "$IQE_FORMAL" --round 1 --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli export-candidates --config "$IQE_FORMAL" --round 1 --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli score-candidates --config "$IQE_FORMAL" --round 1 --backend reference --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli evaluate-oracle --config "$IQE_FORMAL" --round 1 --role stage_val --mode full --max-samples 101592 --device cpu
$IQE_PY -m iqe.cli train-scorer --config "$IQE_FORMAL" --round 1 --max-steps 3000 --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli calibrate --config "$IQE_FORMAL" --round 1 --role selector_cal --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli evaluate --config "$IQE_FORMAL" --round 1 --role dev_report --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli export-bundle --config "$IQE_FORMAL" --round 1 --require-gates --mode full --max-samples 101592 --device cuda
$IQE_PY -m iqe.cli run-round --config "$IQE_FORMAL" --round 2 --max-steps 3000 --mode full --max-samples 101592 --device cuda --resume
# A later interrupted formal trainer uses the same config/budget, --resume auto.
$IQE_PY -m iqe.cli train-scorer --config "$IQE_FORMAL" --round 1 --max-steps 3000 --mode full --max-samples 101592 --device cuda --resume auto
```

Do not use those full commands to bypass the current phase gates. No activation flag is present. Existing smoke bundles never update formal ACTIVE; rollback/load functionality retains the previous complete bundle. No push/PR/merge is authorized in this review.
