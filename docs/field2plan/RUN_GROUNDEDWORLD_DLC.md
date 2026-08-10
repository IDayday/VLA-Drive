# GroundedWorld-VLA DLC runbook

This runbook executes the algorithm in
`GroundedWorld_VLA_Revised_Research_Plan.md`. All formal train launchers enforce
one node, 16 processes, effective batch 32, strict cache manifests, and atomic
per-token cache writes.

## 0. What the complete algorithm does

At inference there is one model and no external teacher:

```text
history/current multi-view images
  ├─ VLM semantic path ───────────────┐
  └─ calibrated physical path         │
       ├─ current multi-scale geometry│
       └─ action-free dynamics/future │
                                      ▼
VLM + flow DiT ──> first trajectory ──> swept-tube read ──> one refiner ──> final trajectory
```

The first trajectory is an internal intermediate from the same jointly trained
model, not a cached trajectory and not a separately deployed baseline. VGGT and
Driving-JEPA exist only as offline training teachers. The consequence head also
exists only during training. No candidate reranking or EPDMS prediction is used.

## 1. Required environment

```bash
cd /mnt/workspace/project/DriveDreamer-Policy
source env.sh

export GROUNDEDWORLD_DATALIST_PATH="$PWD/train_meta.json"
export NAVSIM_EXP_ROOT="$DRIVEDREAMER_SHARED_ROOT/navsim_exp"
export GROUNDEDWORLD_GEOMETRY_CACHE="$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/vggt"
export GROUNDEDWORLD_DYNAMICS_PRIOR_CACHE="$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/driving_jepa_current"
export GROUNDEDWORLD_FUTURE_TARGET_CACHE="$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/future_student_ema"
export GROUNDEDWORLD_CONSEQUENCE_CACHE="$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/consequence"
export NAVSIM_METRIC_CACHE_ROOT="$DRIVEDREAMER_SHARED_ROOT/groundedworld_cache/navsim_metric_navtrain"
```

Driving-JEPA is an explicit external prerequisite. It is not present in this
workspace and is never downloaded automatically:

```bash
export GROUNDEDWORLD_DRIVING_JEPA_ADAPTER_FACTORY='my_drive_jepa_adapter:build_adapter'
export GROUNDEDWORLD_DRIVING_JEPA_REPO=/local/path/to/Drive-JEPA
export GROUNDEDWORLD_DRIVING_JEPA_CHECKPOINT=/local/path/to/checkpoint.pt
```

The adapter must use frames 0--3 only and return the declared
`CurrentPriorTeacherOutput[Th=4,V=3,C=96,Ht,Wt]`. Its patch grid is pooled into
a global current/history dynamics target; it is deliberately not treated as an
ego-BEV grid without an explicit spatial transform.

## 2. Offline cache jobs

Run these as separate DLC jobs. Each command fails on absent inputs or invalid
manifests.

```bash
bash scripts/grounded_world/00_cache_geometry_vggt.sh
```

```bash
bash scripts/grounded_world/00_cache_current_prior.sh
```

Build NAVSIM train metrics and six non-aggregate physical consequence labels:

```bash
bash scripts/grounded_world/00_cache_navsim_metrics_train.sh
bash scripts/grounded_world/00_build_consequence_labels.sh
```

The six outputs are clearance, TTC, collision, lane distance, progress, and
comfort. Logged traffic is declared as a non-reactive proxy; reactive
counterfactual behavior is not claimed.

## 3. First complete B5 run: 30k screening

This is a complete algorithm run at a shorter fine-tuning budget. It is the
recommended first end-to-end experiment; do not wait for 100k before checking
module learning and navtest checkpoints.

### Stage I: acquire current geometry and dynamics prior

```bash
GROUNDEDWORLD_EXPERIMENT=b5_full \
GROUNDEDWORLD_STAGE=stage1 \
GROUNDEDWORLD_RUN_SEED=42 \
MAX_TRAIN_STEPS=30000 \
RUN_ID=groundedworld-b5-stage1-30k-seed42 \
bash scripts/grounded_world/04_run_experiment.sh
```

Audit what was actually learned:

```bash
RUN_DIR="$NAVSIM_EXP_ROOT/groundedworld-b5-stage1-30k-seed42" \
GROUNDEDWORLD_AUDIT_STAGE=prior \
bash scripts/grounded_world/06_inspect_training_signals.sh
```

### Shared future target from the Stage-I EMA

Checkpoint files are flat under `checkpoints/steps_N_pytorch_model.pt`.

```bash
export GROUNDEDWORLD_STAGE1_CHECKPOINT="$NAVSIM_EXP_ROOT/groundedworld-b5-stage1-30k-seed42/checkpoints/steps_30000_pytorch_model.pt"
bash scripts/grounded_world/00_cache_future_ema.sh
```

All teacher controls must use this exact future-target manifest; do not build a
different future target for each control.

### Stage II: action-free future memory

```bash
GROUNDEDWORLD_EXPERIMENT=b5_full \
GROUNDEDWORLD_STAGE=stage2 \
GROUNDEDWORLD_RUN_SEED=42 \
GROUNDEDWORLD_STAGE1_CHECKPOINT="$GROUNDEDWORLD_STAGE1_CHECKPOINT" \
MAX_TRAIN_STEPS=30000 \
RUN_ID=groundedworld-b5-stage2-30k-seed42 \
bash scripts/grounded_world/04_run_experiment.sh
```

```bash
RUN_DIR="$NAVSIM_EXP_ROOT/groundedworld-b5-stage2-30k-seed42" \
GROUNDEDWORLD_AUDIT_STAGE=predictive \
bash scripts/grounded_world/06_inspect_training_signals.sh
```

### Stage III-A: learn reader/refiner without moving the planner

```bash
export GROUNDEDWORLD_STAGE2_CHECKPOINT="$NAVSIM_EXP_ROOT/groundedworld-b5-stage2-30k-seed42/checkpoints/steps_30000_pytorch_model.pt"
# The current env.sh RELEASE_MODEL was audited as framework=QwenOFT with a
# flat pytorch_model.pt. Pin a different path only if the baseline-lock
# experiment selects a different pure-trajectory checkpoint.
export GROUNDEDWORLD_BASELINE_CHECKPOINT="$RELEASE_MODEL/pytorch_model.pt"

GROUNDEDWORLD_EXPERIMENT=b5_full \
GROUNDEDWORLD_STAGE=stage3 \
GROUNDEDWORLD_STAGE3_PHASE=A \
GROUNDEDWORLD_RUN_SEED=42 \
MAX_TRAIN_STEPS=30000 \
RUN_ID=groundedworld-b5-stage3a-30k-seed42 \
bash scripts/grounded_world/04_run_experiment.sh
```

### Stage III-B: joint low-LR co-training

```bash
export GROUNDEDWORLD_STAGE3A_CHECKPOINT="$NAVSIM_EXP_ROOT/groundedworld-b5-stage3a-30k-seed42/checkpoints/steps_30000_pytorch_model.pt"

GROUNDEDWORLD_EXPERIMENT=b5_full \
GROUNDEDWORLD_STAGE=stage3 \
GROUNDEDWORLD_STAGE3_PHASE=B \
GROUNDEDWORLD_RUN_SEED=42 \
MAX_TRAIN_STEPS=30000 \
RUN_ID=groundedworld-b5-stage3b-30k-seed42 \
bash scripts/grounded_world/04_run_experiment.sh
```

In III-B the world losses reuse the differentiable baseline training forward,
so geometry/dynamics supervision can reach the shared visual encoder. The
separate trajectory sampler is detached.

```bash
RUN_DIR="$NAVSIM_EXP_ROOT/groundedworld-b5-stage3b-30k-seed42" \
GROUNDEDWORLD_AUDIT_STAGE=planning \
bash scripts/grounded_world/06_inspect_training_signals.sh
```

## 4. NAVSIM-v2 evaluation

Build each evaluator cache once:

```bash
EVAL_SUITE=navtest bash scripts/grounded_world/00_cache_navsim_metrics_eval.sh
```

For navhard, first download the official local assets using the vendored
NAVSIM instructions, then build processed inference metadata and the two-stage
metric cache:

```bash
OVERWRITE=0 bash scripts/grounded_world/00_prepare_navhard_metadata.sh
EVAL_SUITE=navhard_two_stage bash scripts/grounded_world/00_cache_navsim_metrics_eval.sh
```

Evaluate 10k/20k/30k on navtest as soon as each checkpoint exists:

```bash
export MODEL_DIR="$NAVSIM_EXP_ROOT/groundedworld-b5-stage3b-30k-seed42"
for step in 10000 20000 30000; do
  MODEL_ITER="$step" EVAL_SUITE=navtest \
  bash scripts/grounded_world/05_eval_checkpoint_navsim_v2_16gpu.sh
done
```

Evaluate the promoted checkpoint on two-stage navhard:

```bash
MODEL_ITER=30000 EVAL_SUITE=navhard_two_stage \
bash scripts/grounded_world/05_eval_checkpoint_navsim_v2_16gpu.sh
```

The navhard summary JSON contains `stage_one_score`, `stage_two_score`, and
`combined_score`. Any invalid/missing scenario or missing summary row fails the
job.

Same-checkpoint access removal (causal use test):

```bash
MODEL_ITER=30000 EVAL_SUITE=navhard_two_stage \
GROUNDEDWORLD_INFERENCE_INTERVENTION=disable_access \
bash scripts/grounded_world/05_eval_checkpoint_navsim_v2_16gpu.sh
```

Set `SAVE_DIAGNOSTICS=1` in a direct `4-infer.sh` job to save draft, final,
physical delta, source gates, tube validity, and tube points per token.

Aggregate only actual multi-seed results and compute paired-scene bootstrap:

```bash
cp artifacts/grounded_world/result_matrix.example.json /path/to/result_matrix.json
# Edit the copied matrix to point at actual *_summary.json files.
RESULT_MATRIX=/path/to/result_matrix.json \
REFERENCE_ARM=b0_pure_vlm_dit \
RESULT_REPORT_DIR=/path/to/groundedworld_report \
bash scripts/grounded_world/07_aggregate_results.sh
```

Missing declared summaries remain `MISSING`; they are not imputed as zero.

## 5. B0--B5 and attribution controls

Available arms:

```text
b0_pure_vlm_dit
b1_geometry_aux
b2_geometry_access
b3_current_world
b4_predictive_world
b5_full
```

Controls:

```text
control_real_sup_access
control_no_teacher_same_future
control_scene_shuffled_same_future
control_real_sup_noaccess
control_random_frozen_same_future
control_gt_task_mlp_same_future
control_generic_vjepa_same_future
```

Resolve an arm without allocating GPUs:

```bash
GROUNDEDWORLD_MATRIX_PRINT_ONLY=1 \
GROUNDEDWORLD_EXPERIMENT=b4_predictive_world \
GROUNDEDWORLD_STAGE=stage3 \
bash scripts/grounded_world/04_run_experiment.sh
```

B0 is the supplied pure VLM+DiT checkpoint and is evaluated, not retrained by
GroundedWorld. B1 uses direct III-B initialization because III-A has no
reader/refiner trainables. B2/B3 do not require Stage II. B4/B5 and the shared-
future controls require Stage I, the same future manifest, Stage II, III-A,
and III-B.

Use seeds 42/43/44 for ordinary arms and 42--46 for the final real/no-teacher/
shuffled/access comparisons. Promote by matched checkpoint rule, not best seed.

## 6. 100k expansion rule

Only expand the core arms after the 30k run satisfies all of the following:

- Stage-I real-vs-shuffled prior margin is positive and geometry coverage is nontrivial;
- Stage-II correct-time future similarity beats temporal shuffle;
- III-A/B tube coverage is valid and refiner delta leaves zero initialization;
- navtest real/access trends beat matched no-teacher/no-access controls;
- at least one promoted checkpoint completes navhard Stage 1 and Stage 2.

For 100k, omit `MAX_TRAIN_STEPS=30000` (formal default is 100k) and repeat the
same commands with new run IDs. A module-learning PASS is not a planning-gain
claim; the B0--B5 and same-checkpoint access comparisons are still required.
