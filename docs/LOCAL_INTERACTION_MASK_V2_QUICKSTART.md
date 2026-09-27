# Local interaction mask V2

This campaign is active. Formal planning results are not available yet. Its binding protocol is `reports/local_interaction_mask_v2/OBJECTIVE.md` plus `USER_STEERING.md`; user corrections require an independent public-Qwen starting point, one training seed, both authorized hosts and complete Navtest acceptance. The old private-foundation results are historical engineering references only.

## Environment and inputs

The real GPU runs use Python3.10, PyTorch2.5.1/CUDA12.4 and the repository's Qwen3-VL dependencies in `/root/miniconda3/envs/ddp`. Official CPU NAVSIM scoring runs separately in `/root/miniconda3/envs/navsim` with NAVSIMv1.1 and the nuPlan devkit. No external service is needed. Keep native Qwen BF16 and original DiT FP32; BF16 noise draws are converted to FP32 at the action head boundary. Model loading verifies every public file against `PUBLIC_QWEN_PROVENANCE.json`.

`configs/local_interaction_mask_v2/requirements-gpu.txt` and `requirements-scorer.txt` record the core versions actually used, with Python3.10 and3.9 respectively. Install them in separate environments. The full runtime snapshots are `GPU_ENVIRONMENT.json` and `SCORER_ENVIRONMENT.json`. A fresh clean installation of these lists is NOT_RUN; they are not represented as a solved universal lockfile. This repository bundles the NAVSIMv1.1 source at `navsim_v1.1/navsim`; add it and the separately installed official nuPlan source to the scorer's `PYTHONPATH`. No private driving checkpoint is required.

Set paths for your own licensed NAVSIM installation; every data/model/output location is a CLI argument. `PYTHONPATH` must include this repository and nuPlan. Example shell setup:

```bash
export PYTHONPATH="$PWD:$NUPLAN_ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NO_ALBUMENTATIONS_UPDATE=1
```

Current data specifications are JSON objects containing `tokens` (path to a JSON token list), `observations` (current calibrated NPZ directory), and either `meta_root` (legacy per-token pickle) or `current_records` (strict current-only per-token JSON). Deployment reads only currentF0/L0/R0 images, their calibrated transforms, navigation and four allowed ego history states. `prepare_current_records` generates Navtest records from the exact metric-cache token/log index without reading annotations or future frames into outputs. WorldTargets and ego futures are opened separately on the loss side.

Exact scene/log manifests are published in `configs/local_interaction_mask_v2/splits/`, containing public dataset identifiers only. For researchers without this workspace's preprocessing files, rebuild from licensed raw NAVSIM logs and sensor images:

```bash
python -m tools.local_interaction_mask_v2.prepare_training_data \
  --index configs/local_interaction_mask_v2/splits/train_index.json \
  --raw-log-root "$NAVSIM_TRAIN_LOGS" --sensor-root "$NAVSIM_TRAIN_SENSORS" \
  --output "$NEW_TRAIN_DATA"
```

This new entry point is implemented; its raw-data parity check is pending at this documentation stage. It writes current-only `current_records/observations` separately from `labels/targets` and `labels/ego_meta`. Use the emitted `training_dataset.json` to set foundation/graph label paths, and `current_dataset.json` for deployment feature extraction. Repeat for holdout and dev with their fixed index files. For Navtest use `prepare_current_records` directly, with the Navtest index and test sensor/log roots; no model-input target cache is needed. Existing caches are never overwritten.

## Actually run checks

The following test command passed50 related tests at source815f7c2:

```bash
python -m pytest -q tests/local_interaction_mask_v2 tests/joint_world tests/structured_world_v1p1
```

Real runs, with exact commands and immutable source SHAs, are recorded in `RUN_LEDGER_SNAPSHOT.json` and the live external `budget_ledger.json`:

- Public source validation and actual full-Qwen2GPU training with an empty-annotation rank, checkpoint pause and resume.
- Frozen public current-vision extraction:7348train/holdout,1696development,12146Navtest; zero disk-roundtrip error in all shards.
-8GPU public-foundation common rank8Q/V LoRA/driving-token training, Reader/current heads and original fresh DiT. Source3972cab completed8epochs; the continuation is pinned0937532, preserving optimizer/RNG/scheduler from an immutable epoch8parent.
-64real-scene cached graph and current-memory bridge checks:128presentations/16updates, interrupted after3updates and resumed. Graph parameters/losses are bitwise identical to uninterrupted execution; planner cross-GPU differences are at most1.2e-7, not bitwise identical. `P1_REAL_RESUME.json` records this engineering test.
-64real-scene gate-zero comparison yields exactly unchanged original DiT outputs. Four real scenes across CURRENT/ALL/MASK architecture paths also test trained bridge online/cache agreement, frozen visual cache bypass, target poisoning and batch/singleton equivalence. These architecture checks share one diagnostic graph and are not scientific ALL-vs-MASK results.
- Historical7284scene graph audit and32private visualizations. A separate public epoch2 engineering audit has64scenes and32images. Low early perception/matching coverage is reported, not hidden by local-only metrics.
- Official CPU scorer smoke on four artificial stationary plans over real full Navtest metric caches; this is evaluator validation, not a learned-model Navtest score.

The first continuous graph loader check failed with file-descriptor exhaustion. The fix uses bounded tensor IPC/prefetch; failed artifacts and GPU charges remain. The independent cap is48GPU-hours, including failed jobs and extraction. Use the supervisor for every GPU job:

```bash
python -m tools.local_interaction_mask_v2.supervise \
  --ledger "$LEDGER" --run-id "$UNIQUE_RUN_ID" --output "$OUTPUT" \
  --gpus "$GPU_COUNT" --max-gpu-hours "$RUN_LIMIT" -- \
  python -m tools.local_interaction_mask_v2.TRAIN_OR_EXTRACT_MODULE ...
```

## Public foundation recipe (running, not a paper-equivalent reproduction)

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tools.local_interaction_mask_v2.train_foundation \
  --public-qwen "$PUBLIC_QWEN" \
  --config configs/local_interaction_mask_v2/public_baseline.yaml \
  --provenance reports/local_interaction_mask_v2/PUBLIC_QWEN_PROVENANCE.json \
  --tokens "$TRAIN_TOKENS" --meta-root "$TRAIN_META" \
  --observations "$TRAIN_OBSERVATIONS" --targets "$TRAIN_WORLD_TARGETS" \
  --visual-cache "$VISUAL_CACHE" --output "$FOUNDATION_RUN" \
  --epochs 16 --initial-epochs 8 --schedule-epochs 16 \
  --global-batch 32 --workers 2 --holdout "$FOUNDATION_HOLDOUT_SPEC"
```

Start only through the bounded supervisor, with17GPU-hours reserved for this run. Public pretrained tensors/vision are fixed; rank8 languageQ/V LoRA and driving tokens use1e-5, fresh history/DiT/Reader/current heads1e-4. The shared foundation learns egoFM + current classification/box losses. It is therefore not the paper's action-only ablation.7284scenes and a finite pilot are not equivalent to the published100k-scene/100k-update recipe. No earlier private VLA parameters are loaded.

The original8pass two-point holdout stop fired despite7–9% training-loss reductions over6→8passes. `EXPERIMENT_DECISIONS.md` records the correction: continue the immutable epoch8checkpoint to16, optionally24 using training and holdout curves, retaining the original16epoch scheduler floor. This is a documented stopping-rule repair, not a convergence claim. Use the same command with a new output, `--continue-from "$EPOCH8_CHECKPOINT" --epochs 24 --initial-epochs 16 --extension-use-training-loss`; the continuation supervisor cap is16GPU-hours. Actual cross-host recovery passed, but BF16 arithmetic is not guaranteed bitwise identical.

## Current perception repair (generalization pilot pending)

A500update64training-scene fixed-feature fitting check improved current detection recall25.3%→79.0% and precision10.0%→64.3%. This is not generalization or planning evidence. The declared next check uses all7284training scenes and the independent64scene/59log holdout,8passes, best holdout currentF1 including epoch0. It reuses the existing heads, preserving graph thresholds and fullslot matching. Only if it improves will the same recipe be applied for at most16passes after the final foundation freezes; all comparison arms share the chosen head.

```bash
python -m tools.local_interaction_mask_v2.train_current_heads \
  --cache "$TRAIN_CURRENT_CACHE" --targets "$TRAIN_WORLD_TARGETS" \
  --foundation "$FOUNDATION" --holdout-cache "$HOLDOUT_CURRENT_CACHE" \
  --holdout-targets "$HOLDOUT_WORLD_TARGETS" --output "$CURRENT_HEAD_RUN" \
  --epochs 8 --schedule-epochs 16 --batch 64 --lr 0.001 --workers 2
```

The `selected.pt` file is a separate, strictly parent-bound current head. Future labels are erased. The original post-Qwen BF16 input dtype is restored from lossless float32 cache storage before head training/inference. `extract_current` and `check_online` accept `--perception-checkpoint "$CURRENT_HEAD_RUN/selected.pt"`; online `load_foundation` must receive the same argument. A missing/different override is rejected by `PublicLocalPolicy`.

To reuse immutable current features without another Qwen pass, run `refresh_current_heads --cache "$TRAIN_CURRENT_CACHE" --perception-checkpoint "$CURRENT_HEAD_RUN/selected.pt" --dataset "$CURRENT_DATA_SPEC" --output "$NEW_CURRENT_CACHE"`. It verifies all source payload hashes/current observation identities, reproduces the head at native precision, and writes a fresh cache with the head hash. It opens no targets. Actual online/cache validation remains required before promotion. These new refinement commands are implemented but NOT_RUN at this documentation stage.

The7284scene8pass current-head pilot has now completed: independent holdoutF1 .164847→.213918, recall .239279→.297079, precision .125735→.167133. These are native-adapter epoch8features and remain diagnostic, not final pipeline/planning results. The64scene head pause/resume check restores parameters/loss/RNG/selection exactly. Formal refinement will be trained anew on the final foundation and repaired language numerics below.

The trained Qwen prefix check uncovered length-dependent BF16 LoRA matrix rounding: no information leakage from erased tail tokens, but a .08165m actual DiT trajectory difference between native and appended sequence shapes. Explicit FP32 computation ONLY for the small language LoRA repairs this check; original Qwen staysBF16. This changes inference numerics relative to the foundation's BF16-adapter training, and is recorded rather than called identical to that old arithmetic.64real epoch8training scenes now have exact native/append conditions under the repaired path. All formal arms use this same runtime. Add `--language-adapter-precision fp32` to `extract_current` and `check_online`; online `load_foundation` accepts the same named setting. It is included in cache/override identities. Native-precision current caches/heads cannot be reused across this boundary.

## Main graph and planner commands (formal runs NOT_RUN yet)

Freeze an immutable foundation checkpoint before formal current extraction. Never hash/load a changing rolling checkpoint. The extractor reconstructs all modules strictly using the saved public-origin identity and configuration from its recorded Git commit.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tools.local_interaction_mask_v2.extract_current \
  --checkpoint "$FOUNDATION" --public-qwen "$PUBLIC_QWEN" \
  --dataset "$CURRENT_DATA_SPEC" --visual-cache "$VISUAL_CACHE" \
  --language-adapter-precision fp32 \
  --selector configs/local_interaction_mask_v2/selector_v1.json \
  --graph-config configs/local_interaction_mask_v2/graph.json --output "$CURRENT_CACHE"

python -m tools.local_interaction_mask_v2.train_graph \
  --cache "$TRAIN_CURRENT_CACHE" --targets "$TRAIN_WORLD_TARGETS" \
  --meta-root "$TRAIN_META" --holdout "$GRAPH_HOLDOUT_SPEC" \
  --config configs/local_interaction_mask_v2/graph.json \
  --mode all --seed 42 --batch 32 --epochs 16 --schedule-epochs 32 \
  --workers 2 --output "$G_LOCAL_ALL"
```

Run the matching `--mode mask` from the same fresh initialization/data/config as `G_LOCAL_MASK`, on the other authorized host/GPU. The main comparison shares frozen current graphs and fullslot→GT→local matching. At16passes, `python -m tools.local_interaction_mask_v2.convergence --runs "$G_LOCAL_ALL" "$G_LOCAL_MASK" --output "$DECISION"` applies the registered training-holdout rule; any extension applies to both. Missing neighborhood supervision is flagged for review. `regraph` builds the nearest control from the same current predictions, changing only a registered selector JSON with `graph_variant: nearest`; labels are not available to this tool.

```bash
python -m tools.local_interaction_mask_v2.train_planner \
  --cache "$TRAIN_CURRENT_CACHE" --targets "$TRAIN_WORLD_TARGETS" \
  --meta-root "$TRAIN_META" --foundation "$FOUNDATION" \
  --holdout "$GRAPH_HOLDOUT_SPEC" \
  --graph-config configs/local_interaction_mask_v2/graph.json \
  --mode mask --graph-checkpoint "$G_LOCAL_MASK/checkpoint.pt" \
  --seed 42 --batch 32 --epochs 8 --schedule-epochs 16 \
  --workers 2 --output "$P_LOCAL_MASK"
```

The current-only arm uses `--mode current` without a graph checkpoint. ALL uses `--mode all` and its own independently trained ALL graph. Equal trainable current projection, graph projection and gated attention are used in every arm, with identical initial weights and full current scene/risk context. Qwen/LoRA/Reader/current heads/DiT and graphs stay frozen during this transfer phase. Do not train from GT-conditioned future hidden states.8→16extensions must apply to all three bridges based on6→8holdout DiTADE, never Navtest.

## Asynchronous complete Navtest (formal evaluation NOT_RUN yet)

Freeze models before starting. `variants.json` maps `A0` to null and the three learned arms to their checkpoint files. Start CPU scorers independently so they consume atomic per-scene NPZ exports as GPUs generate them; inference processes never load metric caches.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m tools.local_interaction_mask_v2.export_plans \
  --cache "$NAVTEST_CURRENT_CACHE" --foundation "$FOUNDATION" \
  --variants "$VARIANTS_JSON" --batch 16 --output "$EXPORT_ROOT"

python -m tools.local_interaction_mask_v2.score_async \
  --devkit "$NAVSIM_V1_ROOT" --index "$NAVTEST_CACHE_INDEX" \
  --predictions "$EXPORT_ROOT/A0" --output "$SCORE_ROOT/A0" \
  --workers 12 --chunk 8 --export-shards 8 --benchmark-navtest
```

Use the appropriate independent Python environments above. Repeat the CPU command for each fixed model. For two-host GPU export pass distinct `--shard`/`--shards`; CPU log partitioning uses `--log-shard`/`--log-shards` with separate output directories. No full result is valid with missing/duplicate tokens, failed rows, changed artifact hashes, or mixed protocols. CSV rows contain official submetrics, proposal/cache hashes and failure messages. Failed scores remain in the denominator as zero and invalidate a benchmark claim.

Two independent8GPU torchrun exporters can use `--shards 16 --shard-offset 0` and `--shards 16 --shard-offset 8`, preserving local process groups while assigning disjoint scene shards. After all CPU log partitions finish, `merge_scores --parts "$PART0" "$PART1" --index "$NAVTEST_CACHE_INDEX" --predictions "$EXPORT_ROOT/A0" --export-shards 16 --benchmark-navtest --output "$MERGED_A0"` checks full populations, common model/source/noise/evaluator identities and proposal provenance before writing one complete scene CSV. `compare_pdms --first "$MERGED_MODEL/scenes.csv" --baseline "$MERGED_A0/scenes.csv" --navtest --output "$PAIRED_REPORT"` reports paired submetrics and log-cluster intervals. The lightweight original decoder is shared with `infer.py`; exact parity was checked against the original source and128previous actual exports.

## Recovery

Read `CODEX_GOAL_STATE.md` and the live ledger before launching anything. Do not duplicate active jobs. Training checkpoints save optimizer/scheduler, sampling offset, presentation counts, Python/NumPy/Torch/CUDA RNG, and separate graph mask/time RNG. Resume with the same pinned source/data/config, add `--resume`, and use a new supervisor run ID with `--initial-step` equal to the checkpoint's saved step. A new segment charges only new updates and wall time. Epoch extension may change only the registered terminal epoch, preserving the original scheduler horizon. Batch/topology changes are not claimed to provide exact resume.

Current local recovery command, only after the recorded supervisor has exited, is the exact worker command in `public_foundation_continued_epoch8_to24_local8/supervisor.json` with `--resume` (retaining its `--continue-from` parent), wrapped in a new bounded supervisor segment. Model selection and publication await complete evidence; do not turn short-run loss decreases into an algorithmic PDMS claim.
