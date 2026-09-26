# Structured world V1 quickstart

Read `reports/structured_world_v1/FINAL_REPORT.md` and `CODEX_GOAL_STATE.md` before launching. The base uses **three current front cameras**, not single-front input. Cache v6 fixes monotonic distortion support; original v4/v5 evidence remains immutable. All paths are configurable.

## Tested environment

Python3.10, Torch2.5.1+cu124, transformers4.57.0, NumPy1.26.4. See `reports/structured_world_v1/ENVIRONMENT.json`. Use the existing baseline environment. For another host, install baseline project requirements and pin those versions in an isolated environment; clean-machine installation is **NOT_RUN**. Geometric BEV adds no MMCV dependency. Do not install old BEVDet dependencies into Qwen's environment; the provider here is trained from scratch, not imported BEVDet weights.

```bash
cd /mnt/project/VLA-Drive-structured-world-v1-20260926
source /mnt/project/structured-world-v1-artifacts/20260926/campaign.env
export DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts
export PYTHONPATH="$WORLD_ROOT:$WORLD_ROOT/navsim:$NUPLAN_DEVKIT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$WORLD_PYTHON" -m pytest tests/structured_world -q
```

On another machine supply `WORLD_ROOT`, `WORLD_ARTIFACTS`, `WORLD_PYTHON`, `BASE_CHECKPOINT`, `BASE_VLM`, `BASE_CHECKPOINT_SHA256`, `NUPLAN_DEVKIT`, `PDM_DEVKIT`, `PDM_CACHE_METADATA` and `DEPTH_MODEL_CKPTS`. The last directory contains legacy `ppd.pth` and `depth_anything_v2_vitl.pth`, needed to construct the original checkpoint. No new module depends on a developer-specific symlink. Large/private assets stay outside Git.

## Real end-to-end check

```bash
CUDA_VISIBLE_DEVICES=0 "$WORLD_PYTHON" tools/structured_world/check_bev_policy.py \
 --checkpoint "$BASE_CHECKPOINT" --vlm "$BASE_VLM" \
 --data-root "$WORLD_ARTIFACTS/dataset_v1" --manifest "$WORLD_ARTIFACTS/dev_tokens.json" \
 --target-cache "$WORLD_ARTIFACTS/targets_v6_dev_full" \
 --delta "$WORLD_ARTIFACTS/E_support_v6_seed42/checkpoint.pt" \
 --config configs/structured_world_v1/E.yaml --output "$WORLD_ARTIFACTS/bev_policy_regressions_v6.json"
```

This checks real BEV→Reader→Qwen→DiT online/cache/external equality, rejected future time and invalid cache identity, actual batch repeat ordering, and zero/open adapter gate gradients. Actual GPU evidence is separate from CPU unit tests.

## Independent data and target generation

Use trusted local NAVSIM logs. `prepare_dataset.py` copies processed metadata into a fresh directory and resolves only allowed current images. `build_targets.py` produces separate current-observation and target files; deployment never needs the target files.

```bash
"$WORLD_PYTHON" tools/structured_world/prepare_dataset.py \
 --manifests "$WORLD_ARTIFACTS/train_tokens.json" "$WORLD_ARTIFACTS/dev_tokens.json" \
 --source "$PROCESSED_META_SOURCE" --sensor-roots "$SENSOR_ROOT" --output "$NEW_DATASET_ROOT"
"$WORLD_PYTHON" tools/structured_world/build_targets.py \
 --manifest "$WORLD_ARTIFACTS/train_tokens.json" \
 --processed-root "$WORLD_ARTIFACTS/dataset_v1/meta/train" \
 --raw-log-root "$RAW_LOG_ROOT" --sensor-root "$SENSOR_ROOT" \
 --output "$NEW_TARGET_ROOT" --limit 8192 --capacity 64
```

For development use the fixed1696-token manifest and `--capacity 999`; retain all GT, not only64 slots. The existing split is by complete log; base checkpoint development exposure is unknown/potentially seen. Never create a new split using evaluation results. Scales and coverage are in `DATA_LABEL_AUDIT.json`.

## Real feature extraction and online interface

```bash
CUDA_VISIBLE_DEVICES=0 "$WORLD_PYTHON" tools/structured_world/provider_cli.py \
 --observations "$WORLD_ARTIFACTS/targets_v6_train8192/observations" \
 --sensor-root "$SENSOR_ROOT" --weights "$WORLD_ARTIFACTS/provider_dense_isolation/checkpoint.pt" \
 --output "$NEW_BEV_CACHE" --precision bf16 --token 2b0deb39ff8355ec
```

Omit `--token` for directory extraction. Online callers use `GeometricBEVProvider(ModelInputs)`; a separate process/service can provide the same tensor payload through `ExternalBEVFeatures` with a pinned producer signature. Freeze and `bind_cache_identity()` before cache use. Source, weights, images, calibration, augmentation, sensor contract and dtype are checked. Regenerate stale caches; do not reuse final Qwen hidden states while Reader/tokens train. No GT or future files are required. Frozen BEV remains part of inference cost.

## Training and complete development evaluation

Configs: A0 original path; A1 action continuation; A2 extra image queries without world loss; B/C bbox/bbox+motion; D/E corresponding BEV variants; E_adapter separate action adapter. Original Qwen/vision and old auxiliary modules are fixed, Qwen LoRA off in every group, original prompt preserved, historical auxiliary research losses explicitly off.

```bash
# Executed recipe; reruns need fresh run IDs and authorized remaining budget.
CUDA_VISIBLE_DEVICES=0 bash tools/structured_world/run_train.sh C NEW_C_RUN 1000 42
# Corrected-label confirmation used 400 steps for C, E and E_adapter:
CUDA_VISIBLE_DEVICES=0 bash tools/structured_world/run_train.sh E NEW_E_RUN 400 42
CUDA_VISIBLE_DEVICES=0 bash tools/structured_world/run_eval.sh E E_support_v6_seed42
bash tools/structured_world/run_score.sh E_support_v6_seed42
```

Training launchers accept `WORLD_TARGET_CACHE` and evaluation `WORLD_DEV_TARGET_CACHE`; defaults are corrected v6. Every variant starts from the same base. `run_score.sh` uses sixteen CPU workers with the validated NAVSIM v1 compact-cache adapter; failed rows are retained as zero and invalidate promotion. Score completed proposal banks while other GPU inference continues. Protocol: one candidate,10 FM steps,8×0.5s, original BF16 Qwen/action precision boundary, no scorer. No v2 EPDMS mixing.

The first1000-step matrix predates the final FOV label correction and is reported separately from corrected400-step confirmation. All failed/repair/test updates count toward12000. No automatic long training or hyperparameter search.

## Recovery

```bash
"$WORLD_PYTHON" tools/structured_world/resume_run.py \
 --checkpoint "$WORLD_ARTIFACTS/C_support_v6_seed42/checkpoint.pt"
```

Arguments are recovered from the checkpoint, not guessed. Completed runs do zero new steps. Incomplete runs restore delta/base/config/optimizer/scheduler/random/sampler state and retain the original budget reservation. Exact continuation is supported at optimizer boundaries with unchanged topology and `--deterministic` math attention. Fast-attention pilots are not claimed bitwise continuation-equivalent. Real tests: separate-process `resume_split` versus `resume_reference`; full two-GPU Qwen/world/DiT `check_real_ddp.py --joint-action --run-id NEW_TEST_ID`; independent loss/empty-rank/accumulation NCCL test `check_ddp_losses.py`.

## Figures and reports

```bash
"$WORLD_PYTHON" tools/structured_world/visualize.py \
 --predictions "$WORLD_ARTIFACTS/C_support_v6_seed42_dev/predictions" \
 --target-cache "$WORLD_ARTIFACTS/targets_v6_dev_full" --sensor-root "$SENSOR_ROOT" \
 --output "$NEW_FIGURE_ROOT" --limit 64
```

Figure categories are diagnostic selection after inference; they do not filter formal score rows or select model inputs. Complete PDMS sub-scores, matched perception/motion metrics, failed rows, paired scene differences and log-cluster intervals are linked from the final report. Navtest is never used for tuning; any unexecuted final test is explicitly `NOT_RUN`.

## Final navtest recipe — NOT_RUN

No candidate was promoted after this underfit pilot and label correction. Therefore no navtest score or model selection claim is made. After model selection is frozen in a future bounded experiment, the existing evaluator can use test metadata and current observation calibration without loading world labels:

```bash
CUDA_VISIBLE_DEVICES=0 "$WORLD_PYTHON" tools/structured_world/evaluate.py \
 --checkpoint "$BASE_CHECKPOINT" --vlm "$BASE_VLM" --delta "$FROZEN_DELTA" \
 --config "$FROZEN_CONFIG" --data-root "$NAVTEST_DATA_ROOT" --split test \
 --manifest "$NAVTEST_MANIFEST" --target-cache "$NAVTEST_CURRENT_OBSERVATIONS" \
 --skip-world-diagnostics --output "$NAVTEST_OUTPUT"
"$WORLD_PYTHON" tools/structured_world/score_pdms.py \
 --devkit "$PDM_DEVKIT" --cache-metadata "$NAVTEST_METRIC_METADATA" \
 --manifest "$NAVTEST_MANIFEST" --predictions "$NAVTEST_OUTPUT/predictions" \
 --output "$NAVTEST_OUTPUT/pdms" --workers 16
```

These test-split commands are supplied but **not empirically validated in this campaign**. They are not a substitute for the actual complete development evaluations.

Provider adaptation also saves strict provider/head/optimizer/random/sampling state and accepts `--resume` (same run ID/data/steps). Its fast CUDA grid-sample backward is not bitwise deterministic. Start with `--deterministic` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for tested exact continuation: only the differentiable geometry resampler moves to CPU during adaptation; GPU CNN training and frozen GPU deployment remain intact. Legacy provider checkpoints without the newly recorded resume contract cannot be silently resumed.


## Provider adaptation command record

The following600-step fast-mode adaptation was actually executed on the historical64-scene training cache. It is a command record, not an instruction to reuse the already-completed run ID or exceed the ledger. The final v6 perception audit uses `evaluate_provider.py` with the corrected cache and does no updates.

```bash
"$WORLD_PYTHON" tools/structured_world/adapt_provider.py  --manifest "$WORLD_ARTIFACTS/overfit_tokens.json"  --target-cache "$WORLD_ARTIFACTS/targets_v3_overfit64"  --sensor-root "$SENSOR_ROOT" --output "$WORLD_ARTIFACTS/provider_dense_isolation"  --ledger "$WORLD_ARTIFACTS/budget_ledger.json" --steps 600
"$WORLD_PYTHON" tools/structured_world/evaluate_provider.py  --weights "$WORLD_ARTIFACTS/provider_dense_isolation/checkpoint.pt"  --manifest "$WORLD_ARTIFACTS/overfit_tokens.json"  --target-cache "$WORLD_ARTIFACTS/targets_v6_train8192" --sensor-root "$SENSOR_ROOT"  --output "$WORLD_ARTIFACTS/provider_perception_v6.json" --device cpu
```

A fresh600-step adaptation on v6 was **NOT_RUN**. To run it in a future bounded campaign, use a fresh output/run ID (`--run-id`), the corrected cache, and `--deterministic` if exact continuation is required. Do not silently treat the historical provider as one trained on corrected targets.
