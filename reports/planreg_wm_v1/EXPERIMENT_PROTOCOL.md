# PlanReg-WM-V1 audited experiment protocol

## Immutable identifiers

- Audit start: `feature/planreg-wm-v1-drivor-scorer` at
  `b1ebe1bfc0382d84a0715b0a6438175dceccb2b2`.
- Audit branch: `fix/planreg-wm-v1-training-audit-20260902`.
- Original DriveVLA-M0 base lineage:
  `d84bf2b39696050f715fe41c5f005d0d1115c0c1`.
- Exact scorer source: `valeoai/DrivoR` at
  `fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`.
- Baseline VLM: local trust-remote-code InternVL3-2B / 24-block
  `InternVisionModel` (`hidden_size=1024`), not Qwen3-VL.

Each run records the git SHA/status, redacted environment, exact command,
resolved Hydra configuration, optimizer groups, training log, and train/val
split audit. Evaluation additionally records the checkpoint SHA-256 and
precision contract.

## Fixed model contract

- Generator: 64 trajectory queries, four refinement stages, eight poses, three
  coordinates; unchanged from the audited base.
- Scorer: proposal coordinates are detached, flattened from `8x3`, re-embedded,
  and decoded by DrivoR's independent four-layer scorer. Its independent heads
  are NC/DAC/TTC/EP/DDC/Comfort with PDM weights `1/1/5/5/0/2`;
  `target_ttc==2.0` is masked and `b2d=false`.
- Diagnostic `pred_pdms = exp(log_pdm_score)/(TTC+EP+Comfort weights)`; this
  normalization does not change argmax/selected indices.
- Vision: `[CLS,16 registers,patches]`; every InternViT block receives rank-32
  Q/V-only LoRA. Base QKV, K, MLP, pixel shuffle/projector, LLM, and LM head are
  frozen.
- Read-only register attention lets register queries read all tokens, but masks
  register key columns from CLS/patch queries. It rejects flash attention.
  With zero LoRA B, legacy patch outputs match within `1e-5`.
- Registers are removed before patch reshape/pixel shuffle/`mlp1`/LLM.
- Main tile topology is `thumbnail_query_attention` using normalized
  `[cx,cy,width,height,is_thumbnail]`; the zero-initialized tile gate makes its
  initial output exactly thumbnail-only. Teacher/student topology is identical.
- Fusion target:
  `LN(planning + tanh(gate)*semantic)`; the runtime scene is
  `(1-rho)*semantic + rho*target`. Gate init is `atanh(0.5)`, and rho reaches
  one over 20% of optimizer steps. E0 semantic-only is an exact legacy bypass.
- Both the generator and scorer consume the same current `[B,16,256]` scene
  features. The scorer never consumes predicted future registers.

## Split and data contract

Defaults are:

```yaml
data_protocol:
  include_val_in_train: false
  require_disjoint_train_val: true
```

The no-cache train filter uses only configured train logs, while validation
uses only val logs. Cached training no longer concatenates validation by
default. Every launch prints counts, overlap count, and deterministic SHA-256
hashes for both log sets and writes them to `run_metadata`; non-empty overlap
fails.

`include_val_in_train=true` is explicit final-fit mode: it requires
`limit_val_batches=0`, removes score-monitored best checkpointing, saves only
`last`, and is forbidden for hyperparameter selection.

WM training is no-cache and uses the unique target name
`trajectory_target_planreg_wm_v1`. The current index is
`num_history_frames-1`; offsets `[1,3,6]` provide real front-view images at
0.5, 1.5, and 3.0 seconds:

```text
future_image_paths        [B,3,1024]
future_image_path_lengths [B,3]
future_valid_mask         [B,3]
```

Stale legacy target cache is rejected. Shuffled-future mode requires batch
size > 1.

## World-model numerical contract

The predictor API accepts `[B,K,8,3]` and returns `[B,K,3,16,256]`, but V1
trains only K=1 using the GT trajectory. It never consumes evaluator score,
future actor/map targets, or non-GT future images.

The trajectory feature is
`[x/30,y/10,sin(theta),cos(theta),speed/15,acceleration/8]`.
`v0=sqrt(vx^2+vy^2)` comes from current ego status; the first acceleration is
`(v1-v0)/0.5`.

All current/target states use affine-free layer normalization:

```text
pred_future_n = current_n + residual
wm_abs = weighted(1 - cosine(pred_future_n, target_future_n))
wm_delta = weighted SmoothL1(
  pred_future_n-current_n,
  target_future_n-target_current_n)
```

The residual output starts at zero. Horizon weights are `[1.0,0.7,0.4]` and
invalid horizons do not enter the denominator. `wm = wm_abs + 0.5*wm_delta`.
The total WM coefficient is zero for 5% of optimizer steps, ramps over the next
10%, then remains at 0.10. Step buffers resume continuously.

EMA includes only the vision model, Q/V LoRA, registers, planning neck, and the
same tile aggregator. It is copied after student checkpoint loading, is
frozen/eval/no-grad, and updates post-optimizer with cosine momentum
0.996→0.9999.

## Optimizer and scheduler

The full default global batch is `2 samples/GPU x 8 GPUs = 16`. LR is not
automatically batch-scaled. AdamW uses betas `[0.9,0.999]`, epsilon `1e-8`,
matrix decay `0.01`, and no-decay `0.0`. Every logical group is split into
decay/no-decay; biases, norms, tokens/queries/embeddings, semantic/tile gates,
and LoRA A/B are no-decay. Any unclassified trainable parameter fails.
InternViT and frozen-LLM activation checkpointing is enabled; only checkpoint
wrapper flags are put in train mode while frozen dropout/drop-path children
stay in eval mode.

| Logical group | Main-config peak LR | End ratio |
| --- | ---: | ---: |
| planning_adapter | 2e-4 | 0.10 |
| future_predictor | 2e-4 | 0.10 |
| fusion | 1e-4 | 0.10 |
| action_head | 1e-4 | 0.20 |
| scorer | 1e-4 | 0.20 |
| vision_qv_lora | 5e-5 | 0.10 |
| semantic_qformer | 1e-5 | 0.20 |

The resume-safe, per-step scheduler starts at 1% of peak, reaches peak after 3%
warmup, and cosine-decays to each row's end ratio. Total steps come from
Lightning's `estimated_stepping_batches`; optimizer and scheduler state come
from explicit `RESUME_CHECKPOINT`. Gradient clipping is norm 1.0.

## Staged experiment matrix

Bootstrap is two epochs with WM off: planning 2e-4, fusion 1e-4, vision LoRA
2e-5, action/scorer 2e-5, Q-Former 5e-6, 5% warmup.

Every fork is eight epochs from the same matching bootstrap: planning/predictor
1e-4, fusion/action/scorer 5e-5, vision LoRA 2e-5, Q-Former 5e-6, 3% warmup.
E2 and E3 have exactly the same step budget.

| ID | Register/tile topology | WM/control | Seeds |
| --- | --- | --- | --- |
| B0/E0 | registers off; exact semantic | off | 0,1,2 |
| Bootstrap | read-only + tile attention | off | 0,1,2 |
| E2 | read-only + tile attention | off | 0,1,2 |
| E3 | read-only + tile attention | correct future + GT action | 0,1,2 |
| E4 | read-only + tile attention | no action condition | 0 |
| E5 | read-only + tile attention | shuffled future | 0 |
| E6 | read-only + tile attention | repeated current | 0 |
| E7 | read-only + tile attention | correct; predictor-only WM routing | 0 |
| R1 | bidirectional + mean | correct | 0 |
| R2 | read-only + thumbnail only | correct | 0 |
| R3 | read-only + tile attention | correct | 0 |

## Preflight and smoke gates

```bash
python -m compileall navsim/agents/EpisodeDrive
pytest -q \
  tests/test_drivor_scorer_parity.py \
  tests/test_planreg_train_val_split.py \
  tests/test_planreg_lightning_hooks.py \
  tests/test_scene_fusion.py \
  tests/test_legacy_forward_parity.py \
  tests/test_read_only_register_attention.py \
  tests/test_register_patch_parity.py \
  tests/test_internvl_planning_registers.py \
  tests/test_vision_qv_lora.py \
  tests/test_future_register_predictor.py \
  tests/test_future_image_paths.py \
  tests/test_ema_register_target.py \
  tests/test_world_model_gradient_routing.py \
  tests/test_planreg_optimizer_groups.py \
  tests/test_planreg_scheduler.py \
  tests/test_student_checkpoint_export.py
python scripts/audit_drivor_scorer_parity.py
python scripts/smoke_planreg_wm_v1.py
```

The real-data gate uses 32 filtered scenes, two train batches and one validation
batch, requires overlap=0 and at least one valid sample at every future horizon
(ordinary missing samples remain masked), rejects non-finite losses/gradients,
exports a student checkpoint, and runs current-only inference:

```bash
CUDA_VISIBLE_DEVICES=0 PLANREG_NUM_GPUS=1 \
  bash local_planreg_wm_v1/smoke_real_data.sh 0
```

## Launch and deployment

```bash
bash local_planreg_wm_v1/train_bootstrap_registers.sh 0
BOOTSTRAP_CHECKPOINT=/absolute/bootstrap.ckpt \
  bash local_planreg_wm_v1/train_e3_from_bootstrap.sh 0

python scripts/export_planreg_student_checkpoint.py \
  /absolute/last.ckpt /absolute/student.ckpt \
  --resolved-config /absolute/resolved_hydra_config.yaml
bash local_planreg_wm_v1/evaluate_all.sh \
  e3_from_bootstrap_seed0=/absolute/student.ckpt
```

Student export strips EMA, predictor, WM step buffers, optimizer/scheduler, and
training callbacks and emits source/export hashes plus an architecture manifest.
Evaluation forces WM/EMA off, so training-only modules are not constructed.
The standard precision contract is BF16 VLM plus FP32 action/scorer—not full
FP32.

Report selected PDMS, all-64 proposal statistics, offline best-of-K, scorer
regret, candidate diversity, failure/coverage counts, checkpoint/config hashes,
split hashes, register diagnostics, WM horizon losses, and gradient norms.
Best-of-K is an oracle diagnostic, never deployable PDMS. The single-front-view
DriveVLA setting and DrivoR's four-camera reported 93.7 are not strictly fair
comparisons.

PlanReg-WM-V1 explicitly does **not** implement multi-trajectory consequence
modeling, candidate-specific future prediction, RGB prediction, or ranking
losses. Its K-shaped API is only an extension boundary; V1 remains K=1 with GT
future supervision.
