# Gate-2 pilot reproduction commands

All commands are run from the repository root on branch
`feature/action-effect-world-model`. Machine-local paths are resolved in this
order: explicit CLI argument, one-shot environment override, `env.local.sh`,
then the portable defaults in `env.sh`.

```bash
cd /mnt/workspace/project/DriveDreamer-Policy
source ./load_env.sh
git rev-parse HEAD
```

The executed pilot uses NAVSIM `train`, seed `20260821`, the 100k frozen
Qwen+DiT checkpoint, one PPU for Qwen/probe work, and CPU worker pools only for
official metric/geometry target construction.

## Phase 1: policy-local expert-anchor candidates

```bash
python scripts/action_effect/build_candidates.py \
  --config configs/action_effect/pilot_tiny.yaml \
  --datalist "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --split train \
  --anchor-type expert \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --seed 20260821
```

To add frozen policy anchors without confusing a path with checkpoint identity:

```bash
python scripts/action_effect/build_candidates.py \
  --config configs/action_effect/pilot_tiny.yaml \
  --split train \
  --anchor-type policy \
  --policy-prediction-root /absolute/path/to/frozen-baseline-predictions \
  --policy-anchor-id '<checkpoint-sha256>:<inference-config-hash>' \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/policy" \
  --seed 20260821
```

## Phase 2: official metric cache and consequences

```bash
python scripts/action_effect/build_metric_cache.py \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --processed-root "$DATA_ROOT" \
  --raw-log-root "$NAVSIM_PUBLIC_ROOT/navsim_logs/trainval" \
  --map-root "$NUPLAN_MAPS_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_tiny/train" \
  --split train \
  --workers 6

python scripts/action_effect/build_consequences.py \
  --config configs/action_effect/consequences_v2.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --metric-cache "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_tiny/train" \
  --map-root "$NUPLAN_MAPS_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --workers 6
```

The consequence command evaluates all accepted candidates under `log_replay`
and a deterministic 64-scene subset under NAVSIM-v2 IDM. It never calls IDM a
ground-truth counterfactual.

## Phase 3: robust scales and action-effect pairs

```bash
python scripts/action_effect/build_pairs.py \
  --config configs/action_effect/pairs.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert"
```

## Phase 4: data feasibility and Gate 1

```bash
python scripts/action_effect/run_data_diagnostics.py \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert" \
  --report-path reports/action_effect_world_model/data_feasibility.md \
  --output-dir reports/action_effect_world_model/data_feasibility_artifacts
```

## Phase 5a: frozen Qwen scene features

This uses one accelerator, batch 16, four image-loading workers, and never runs
DiT flow sampling. The checkpoint and source trees are hashed in the manifest.

```bash
python scripts/action_effect/cache_scene_features.py \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --checkpoint-run "$ACTION_EFFECT_BASELINE_RUN" \
  --model-iter "$ACTION_EFFECT_BASELINE_STEP" \
  --data-root "$DATA_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --split train \
  --device cuda \
  --batch-size 16 \
  --num-workers 4 \
  --qwen-forward-mode auto
```

## Phase 5b: consequence probe and collapse metrics

The config fixes three training seeds (`20260821`, `20260822`, `20260823`) and
five required controls. The evaluator uses 1,000 scene-clustered bootstrap
resamples.

```bash
python scripts/action_effect/train_world_probe.py \
  --config configs/action_effect/factual_only.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --output-dir "$ACTION_EFFECT_OUTPUT_ROOT/factual_only/pilot_tiny"

python scripts/action_effect/evaluate_world_probe.py \
  --config configs/action_effect/factual_only.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/factual_only/pilot_tiny" \
  --report-dir reports/action_effect_world_model/action_collapse_artifacts \
  --bootstrap-samples 1000
```

## Phase 5c: structured future target and probe

The target is `[3,7,32,32]` at 1/2/4 seconds. Six CPU workers build targets;
one accelerator trains the five single-seed diagnostic controls.

```bash
python scripts/action_effect/build_structured_future.py \
  --config configs/action_effect/structured_future.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --metric-cache "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_tiny/train" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/structured_future/pilot_tiny/expert_log_replay_32" \
  --workers 6

python scripts/action_effect/train_structured_future_probe.py \
  --config configs/action_effect/structured_factual_only.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --structured-future-cache "$ACTION_EFFECT_CACHE_ROOT/structured_future/pilot_tiny/expert_log_replay_32" \
  --factual-probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/factual_only/pilot_tiny" \
  --output-dir "$ACTION_EFFECT_OUTPUT_ROOT/structured_factual_only/pilot_tiny"

python scripts/action_effect/evaluate_structured_future_probe.py \
  --config configs/action_effect/structured_factual_only.yaml \
  --target-config configs/action_effect/structured_future.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --structured-future-cache "$ACTION_EFFECT_CACHE_ROOT/structured_future/pilot_tiny/expert_log_replay_32" \
  --factual-probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/factual_only/pilot_tiny" \
  --probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/structured_factual_only/pilot_tiny" \
  --report-dir reports/action_effect_world_model/structured_collapse_artifacts \
  --bootstrap-samples 1000
```

## Tests and source checks

```bash
source ./load_env.sh
pytest -q tests/action_effect
python -m compileall -q research/action_effect scripts/action_effect
bash -n load_env.sh env.sh
git diff --check
```

## Gate 2.5: action-path engineering audit

This audit uses the existing production trajectory encoder/fusion, 24 scenes
for candidate overfit, three seeds for equal-capacity controls, and 1,000
scene-bootstrap resamples.

```bash
python scripts/action_effect/run_gate2_5_audit.py \
  --config configs/action_effect/gate2_5.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --factual-probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/factual_only/pilot_tiny" \
  --output-dir "$ACTION_EFFECT_OUTPUT_ROOT/gate2_5/pilot_tiny_v1" \
  --report-path reports/action_effect_world_model/gate2_5_audit.md
```

## Structured-target action audit

The raw map tube remains diagnostic. The trajectory-aligned Phase-6 target is
`[3,9,32,32]` at 1/2/4 seconds and is always stored as target-only data.

```bash
python scripts/action_effect/build_effect_tube.py \
  --config configs/action_effect/effect_tube.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --metric-cache "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_tiny/train" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/effect_tube/pilot_tiny/expert_log_replay_32_v1" \
  --workers 6

python scripts/action_effect/audit_structured_targets.py \
  --config configs/action_effect/effect_tube.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_tiny/expert" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert" \
  --scene-feature-cache "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_tiny/qwen_dit_100k" \
  --raw-cache "$ACTION_EFFECT_CACHE_ROOT/structured_future/pilot_tiny/expert_log_replay_32" \
  --effect-cache "$ACTION_EFFECT_CACHE_ROOT/effect_tube/pilot_tiny/expert_log_replay_32_v1" \
  --raw-prediction "$ACTION_EFFECT_OUTPUT_ROOT/structured_factual_only/pilot_tiny/scene_action_probe/seed_20260821/predictions.npz" \
  --output-dir reports/action_effect_world_model/structured_target_artifacts \
  --report-path reports/action_effect_world_model/structured_target_audit.md
```

## Stratified LR/IDM agreement

```bash
python scripts/action_effect/analyze_lr_idm_agreement.py \
  --config configs/action_effect/agreement.yaml \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_tiny/expert" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_tiny/expert" \
  --output-dir reports/action_effect_world_model/lr_idm_agreement_artifacts \
  --report-path reports/action_effect_world_model/lr_idm_agreement.md
```

## Phase 6: scene-disjoint pilot_small data

The formal split is 4,200 train / 500 validation / 500 test scenes. Every
scene starts with 16 candidates. Scales and pair thresholds use the train list
in `split.json`; `turn_inner_outer_offset` is excluded only from training and
retained in validation/test.

```bash
python scripts/action_effect/build_candidates.py \
  --config configs/action_effect/pilot_small.yaml \
  --datalist "$NAVSIM_DATALIST_PATH" \
  --data-root "$DATA_ROOT" \
  --split train \
  --anchor-type expert \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --seed 20260821

python scripts/action_effect/build_metric_cache.py \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --processed-root "$DATA_ROOT" \
  --raw-log-root "$NAVSIM_PUBLIC_ROOT/navsim_logs/trainval" \
  --map-root "$NUPLAN_MAPS_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_small/train_phase6_v1" \
  --split train \
  --workers 12

python scripts/action_effect/build_consequences.py \
  --config configs/action_effect/consequences_phase6.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --metric-cache "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_small/train_phase6_v1" \
  --map-root "$NUPLAN_MAPS_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_small/expert_phase6_v1" \
  --workers 12

python scripts/action_effect/build_phase6_split.py \
  --config configs/action_effect/phase6_split.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_small/expert_phase6_v1" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/splits/pilot_small/phase6"

python scripts/action_effect/build_pairs.py \
  --config configs/action_effect/pairs.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_small/expert_phase6_v1" \
  --statistics-scene-file "$ACTION_EFFECT_CACHE_ROOT/splits/pilot_small/phase6/split.json" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_small/expert_phase6_v2"

python scripts/action_effect/build_effect_tube.py \
  --config configs/action_effect/effect_tube.yaml \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_small/expert_phase6_v1" \
  --metric-cache "$ACTION_EFFECT_CACHE_ROOT/metric_cache/pilot_small/train_phase6_v1" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/effect_tube/pilot_small/expert_log_replay_32_phase6_v1" \
  --workers 12
```

Frozen scene features are extracted from the tagged Gate-2 source rather than
the developer's dirty worktree. The explicit paths still obey the local path
configuration; the feature manifest hashes the checkpoint and clean source.

```bash
ACTION_EFFECT_GATE2_WORKTREE=/tmp/drive-action-effect-gate2-a41f081
git worktree add --detach "$ACTION_EFFECT_GATE2_WORKTREE" action-effect-gate2-v0
cd "$ACTION_EFFECT_GATE2_WORKTREE"
export PYTHONPATH="$ACTION_EFFECT_GATE2_WORKTREE:$ACTION_EFFECT_GATE2_WORKTREE/navsim_data_process:$NAVSIM_DEVKIT_ROOT"
python scripts/action_effect/cache_scene_features.py \
  --candidate-cache "$ACTION_EFFECT_CACHE_ROOT/candidates/pilot_small/expert_phase6_v1" \
  --checkpoint-run "$ACTION_EFFECT_BASELINE_RUN" \
  --model-iter 100000 \
  --data-root "$DATA_ROOT" \
  --cache-dir "$ACTION_EFFECT_CACHE_ROOT/scene_features/pilot_small/qwen_dit_100k_gate2_v1" \
  --split train \
  --device cuda \
  --batch-size 64 \
  --num-workers 8 \
  --qwen-forward-mode optimized
cd /mnt/workspace/project/DriveDreamer-Policy
git worktree remove "$ACTION_EFFECT_GATE2_WORKTREE"
```

## Phase 6: matched three-seed probes and Gate 3

`confidence_aee` is enabled only if the pilot_small stratified disagreement
report passes its configured support threshold. Otherwise the primary matrix
is factual-only, absolute, global separation, AEE, and the equal-capacity
scene-only control.

```bash
python scripts/action_effect/analyze_lr_idm_agreement.py \
  --config configs/action_effect/agreement.yaml \
  --consequence-cache "$ACTION_EFFECT_CACHE_ROOT/consequences/pilot_small/expert_phase6_v1" \
  --pair-cache "$ACTION_EFFECT_CACHE_ROOT/pairs/pilot_small/expert_phase6_v2" \
  --output-dir reports/action_effect_world_model/lr_idm_agreement_artifacts \
  --report-path reports/action_effect_world_model/lr_idm_agreement.md

python scripts/action_effect/train_phase6_world_probe.py \
  --config configs/action_effect/phase6.yaml \
  --output-dir "$ACTION_EFFECT_OUTPUT_ROOT/phase6/pilot_small_v1"

python scripts/action_effect/evaluate_phase6_world_probe.py \
  --config configs/action_effect/phase6.yaml \
  --probe-dir "$ACTION_EFFECT_OUTPUT_ROOT/phase6/pilot_small_v1" \
  --output-dir reports/action_effect_world_model/phase6_artifacts \
  --report-path reports/action_effect_world_model/phase6_probe_ablation.md \
  --gate-report-path reports/action_effect_world_model/gate3_decision.md \
  --bootstrap-samples 1000
```

Development stops at Gate 3. There is intentionally no Phase-7 Qwen+DiT
world-loss launcher and no planning-training or PDMS/EPDMS command in this
delivery.
