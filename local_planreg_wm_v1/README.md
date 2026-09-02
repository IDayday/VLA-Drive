# PlanReg-WM-V1 audited experiment launchers

These launchers implement the audited protocol on
`fix/planreg-wm-v1-training-audit-20260902`. They use InternVL3-2B, 16
read-only planning registers, spatial thumbnail-query tile aggregation,
rank-32 InternViT Q/V LoRA, the unchanged 64-query/four-refinement trajectory
generator, and the exact DrivoR scorer lineage pinned in the source.

Every run has an explicit seed and output directory. Automatic checkpoint
discovery is disabled. A fresh bootstrap starts from `PLANREG_BASE_CHECKPOINT`;
forks require an explicit `BOOTSTRAP_CHECKPOINT`; lossless continuation requires
an explicit Lightning `RESUME_CHECKPOINT`.

## Required training sequence

Bootstrap each primary seed for two epochs with WM disabled:

```bash
for seed in 0 1 2; do
  PLANREG_TRAIN_TEST_SPLIT=navtrain \
    bash local_planreg_wm_v1/train_bootstrap_registers.sh "$seed"
done
```

Then fork E2 and E3 for eight epochs from the *same matching-seed bootstrap*:

```bash
BOOTSTRAP_CHECKPOINT=/absolute/bootstrap-seed0.ckpt \
  bash local_planreg_wm_v1/train_e2_from_bootstrap.sh 0
BOOTSTRAP_CHECKPOINT=/absolute/bootstrap-seed0.ckpt \
  bash local_planreg_wm_v1/train_e3_from_bootstrap.sh 0
```

E2 and E3 have the same epoch/step budget. E0/E2/E3 use seeds 0,1,2; E4-E7
and R1-R3 start with seed 0. Controls use `train_e4_from_bootstrap.sh` through
`train_e7_from_bootstrap.sh`. Register/tile ablations are:

- R1: bidirectional registers + mean tile aggregation;
- R2: read-only registers + thumbnail only;
- R3: read-only registers + thumbnail-query attention (main topology).

Bootstrap LRs are `2e-4` planning adapter, `1e-4` fusion, `2e-5` vision Q/V
LoRA, `2e-5` action/scorer, and `5e-6` Q-Former, with 5% warmup. Fork LRs are
`1e-4` planning/predictor, `5e-5` fusion/action/scorer, `2e-5` vision Q/V LoRA,
and `5e-6` Q-Former, with 3% warmup. All use per-step warmup-cosine, AdamW,
gradient clipping at norm 1.0, and no automatic batch-LR scaling. The full-run
default global batch is `2 x 8 = 16`.

## Audit and smoke commands

Dry-run resolves the exact command without requiring checkpoint files:

```bash
DRY_RUN=1 bash local_planreg_wm_v1/train_bootstrap_registers.sh 0
DRY_RUN=1 BOOTSTRAP_CHECKPOINT=/planned/bootstrap.ckpt \
  bash local_planreg_wm_v1/train_e3_from_bootstrap.sh 0
```

The real-data smoke uses 32 train/validation-filtered scenes, exactly two train
batches and one validation batch, requires valid batch coverage at all three
future horizons while retaining per-sample masks, rejects non-finite
losses/gradients, exports a student-only checkpoint, and performs a
pure-current-frame inference:

```bash
CUDA_VISIBLE_DEVICES=0 PLANREG_NUM_GPUS=1 \
  bash local_planreg_wm_v1/smoke_real_data.sh 0
```

Run directories record the git commit/status, redacted environment, launch
command, fully resolved Hydra config, train/validation log counts and SHA-256
hashes, overlap audit, optimizer groups, and logs under `run_metadata/`.

## Deployment and evaluation

Standard evaluation accepts only student-only checkpoints. Export first:

```bash
python scripts/export_planreg_student_checkpoint.py \
  /absolute/training-last.ckpt /absolute/planreg-student.ckpt \
  --resolved-config /absolute/resolved_hydra_config.yaml

bash local_planreg_wm_v1/evaluate_all.sh \
  e3_from_bootstrap_seed0=/absolute/planreg-student.ckpt
```

Evaluation forces `world_model.enabled=false` and `ema.enabled=false`; neither
the predictor nor EMA teacher is constructed. The precision contract is BF16
VLM plus FP32 action/scorer, not “full FP32”. `evaluate_b0_legacy.sh` provides
the matching legacy semantic-only baseline flow.

The environment can be overridden with `PLANREG_NAVSIM_LOG_ROOT`,
`PLANREG_SENSOR_BLOB_ROOT`, `PLANREG_TRAIN_METRIC_CACHE`,
`PLANREG_NAVTEST_METRIC_CACHE`, `NUPLAN_MAPS_ROOT`,
`PLANREG_BASE_CHECKPOINT`, `PLANREG_VLM_PATH`, and `PLANREG_RUN_ROOT`.

PlanReg-WM-V1 still does **not** implement multi-trajectory consequence
modeling: its predictor API carries a K dimension, but V1 supervision is K=1
and uses only the GT trajectory and real GT future images.
