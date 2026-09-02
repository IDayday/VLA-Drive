# PlanReg-WM-V1 training audit — 2026-09-02

## Reproducibility identity

- Repository: `IDayday/VLA-Drive`
- Isolated worktree: `/mnt/project/DriveVLA-M0-planreg-audit-20260902`
- Branch: `fix/planreg-wm-v1-training-audit-20260902`
- Audit base: `b1ebe1bfc0382d84a0715b0a6438175dceccb2b2`
- DrivoR scorer source: `valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`
- Runtime: Python 3.9.25, PyTorch 2.5.1+cu124, Transformers 4.57.6,
  PEFT 0.17.1, Lightning 2.6.0
- VLM: InternVL3-2B, 24-block InternViT with vision width 1024; this is
  not Qwen3-VL.

The source worktree, `main`, and the feature branch were not modified. All
changes were made and tested in the isolated worktree above.

## P0/P1 resolution

| Item | Status | Audited outcome |
| --- | --- | --- |
| P0-1 train/validation leakage | fixed | train and validation filters use their own log sets; cached mode does not concatenate by default; overlap and hashes are enforced and recorded; explicit final-fit disables validation/best selection |
| P0-2 duplicated Lightning hook | fixed | one interval-gated `on_after_backward`; detached/no-grad register diagnostics; finite-gradient hooks do not walk every parameter per step |
| P0-3 semantic baseline/transition | fixed | `semantic_only` is an exact direct bypass; planning transition uses interpolation outside the planning target LayerNorm; default transition is 20% and gate is `atanh(0.5)` |
| P1 read-only registers | fixed | register queries read all tokens while CLS/patch queries cannot read register keys; flash attention fails explicitly; bidirectional remains an ablation |
| P1 spatial tile identity | fixed | normalized tile metadata and mean/thumbnail-only/thumbnail-query-attention reducers; main topology is thumbnail-query attention and is shared by student/EMA |
| P0-5 WM state space | fixed | affine-free normalization, residual prediction in normalized coordinates, normalized absolute/delta losses, scale-invariance and finite-gradient tests |
| Initial ego motion | fixed | first acceleration uses current status velocity; fixed x/y/speed/acceleration scales are applied |
| Optimizer/scheduler | fixed | seven exhaustive logical groups, decay/no-decay split, grouped AdamW and resume-safe per-step warmup cosine |
| Student deployment | fixed | exporter strips EMA/predictor/training state; evaluation disables WM/EMA and constructs neither training-only module |
| Scorer calibration | fixed | normalized diagnostic PDMS is separate from the unchanged DrivoR argmax value |

The 64-query generator, four refinement stages, independent four-layer DrivoR
scorer, six component heads, proposal detach boundary, TTC mask, PDM weights,
and frozen LLM remain unchanged.

## Parity and gradient boundaries

The final scorer audit passed against the pinned upstream source:

- all six component maximum absolute differences: `0.0`;
- PDM score maximum absolute difference: `0.0`;
- selected indices: identical (`[61,39]` in the seeded audit);
- state-dict shapes and head names: identical;
- proposal gradient: absent, norm `0.0`;
- scene-feature gradient norm: `18.2012882232666`;
- TTC invalid-target mask: exact expected loss, invalid gradient `-0.0`.

The resolved E0 configuration disables registers, vision Q/V LoRA, WM, and
fusion modules and takes the exact semantic legacy bypass. E0/legacy forward
parity and BF16 coverage pass at `max_abs_diff <= 1e-5`. With read-only
registers and zero-initialized Q/V LoRA B matrices, CLS/patch semantic output
also passes `max_abs_diff <= 1e-5`; patch count, pixel shuffle, and LLM token
count are unchanged. Planning/scorer losses retain nonzero gradients to the
registers and Q/V LoRA.

## Train/validation audit

The real-data smoke resolved 978 configured training logs and 214 validation
logs:

- overlap count: `0`;
- train log-set SHA-256:
  `1eb0d86f49940b9eb7feb7a2e82a0a6552788c424ce3bec1e2acecc74c76a682`;
- validation log-set SHA-256:
  `96f68962d59a044fd3aa490f820f9b8cd7fc368dfd76bfb4abc17b7f1e9fd64a`.

The audit is recorded in `run_metadata/train_val_protocol.json`. Default
training rejects a nonempty overlap. `include_val_in_train=true` is explicitly
marked final-fit, requires `limit_val_batches=0`, and saves only `last`.

## Optimizer and LR schedule

Default AdamW uses betas `(0.9,0.999)`, epsilon `1e-8`, matrix weight decay
`0.01`, and no-decay `0.0`. Biases, normalization parameters, tokens, queries,
embeddings, gates, and all LoRA A/B tensors are no-decay. Any unclassified
trainable parameter raises immediately.

| Logical group | Peak LR | Initial LR (1%) | Final LR | Final ratio |
| --- | ---: | ---: | ---: | ---: |
| planning_adapter | `2e-4` | `2e-6` | `2e-5` | `0.10` |
| future_predictor | `2e-4` | `2e-6` | `2e-5` | `0.10` |
| fusion | `1e-4` | `1e-6` | `1e-5` | `0.10` |
| action_head | `1e-4` | `1e-6` | `2e-5` | `0.20` |
| scorer | `1e-4` | `1e-6` | `2e-5` | `0.20` |
| vision_qv_lora | `5e-5` | `5e-7` | `5e-6` | `0.10` |
| semantic_qformer | `1e-5` | `1e-7` | `2e-6` | `0.20` |

The scheduler uses Lightning's estimated optimizer-step count, starts at 1%
of peak, reaches peak at the end of the 3% linear warmup, then cosine-decays to
the per-group final ratio. Unit tests prove the complete LR sequence is
identical after scheduler-state resume. No batch-size LR scaling is performed.
The full default global batch is `2 samples/GPU x 8 GPUs = 16`, and gradient
clipping is norm `1.0`.

The two-epoch bootstrap instead uses planning `2e-4`, fusion `1e-4`, vision
LoRA `2e-5`, action/scorer `2e-5`, Q-Former `5e-6`, WM off, and 5% warmup. All
eight-epoch forks use the same matching bootstrap and exact step budget, with
planning/predictor `1e-4`, fusion/action/scorer `5e-5`, vision LoRA `2e-5`,
Q-Former `5e-6`, and 3% warmup.

## World-model schedule and numerical definition

Current, current-target, and future-target registers are independently
normalized with affine-free LayerNorm. The predictor returns
`pred_future_n = current_n + residual`; therefore its zero-initialized output
has exactly zero predicted delta. The loss is:

```text
wm = weighted_horizon_mean(1 - cosine(pred_future_n, target_future_n))
   + 0.5 * weighted_horizon_mean(SmoothL1(
       pred_future_n - current_n,
       target_future_n - target_current_n))
```

Horizon weights are `[1.0,0.7,0.4]`, and invalid horizons are excluded from the
weighted denominator. The total coefficient is 0 before 5% of optimizer
steps, linearly reaches `0.10` at 15%, and stays at `0.10`. Restored step
buffers keep this schedule continuous on resume. V1 supervision remains one
GT trajectory (`K=1`) and real 0.5/1.5/3.0-second future images.

## Smoke and test evidence

Final synthetic smoke passed on CPU:

- scene registers `[2,16,256]`;
- proposals `[2,64,8,3]`;
- predicted future registers `[2,1,3,16,256]`;
- current-only inference trajectory `[2,8,3]`;
- register gradient norm `27.99070930480957`;
- vision Q/V LoRA gradient norm `1.3881393671035767`;
- no future key consumed at inference.

The real-data smoke passed on one NVIDIA A800-SXM4-80GB using 32 filtered
training scenes and 32 filtered validation scenes, exactly two train batches
and one validation batch. Every horizon had at least one valid future image in
each audited batch; missing individual samples remained masked. All checked
losses and gradients were finite. Frozen InternViT/LLM activation
checkpointing reduced observed single-GPU batch-2 peak allocation from an
initial out-of-memory attempt at about 78.6 GiB to about 34.1 GiB. Only the
checkpoint wrapper flags are put in train mode; frozen dropout/drop-path
children remain in eval mode.

Student export from the smoke checkpoint passed strict verification:

- source SHA-256:
  `e6fe99eba91d49a95bd9e34fa858d93da1d44532736b0422ea0cd02795fc6016`;
- exported SHA-256:
  `1adfc3bc7f314a00bcc203716089d5cb8a7e962b37163af9226809f8c499c35e`;
- removed state keys: `504`;
- retained state keys: `1114`.

The deployment smoke then loaded all `1114/1114` keys with no missing or
unexpected key, constructed neither EMA teacher nor predictor, consumed no
future key, and produced a finite `[1,8,3]` trajectory from current-frame data.
Its precision contract is BF16 VLM plus FP32 action/scorer, not full FP32.

Final verification commands and results:

- `python -m compileall navsim/agents/EpisodeDrive`: passed;
- the 16 required pytest files: `80 passed`;
- targeted E0/read-only parity rerun: `14 passed` (a subset of the 80);
- `python scripts/audit_drivor_scorer_parity.py`: passed;
- `python scripts/smoke_planreg_wm_v1.py`: passed;
- all experiment shell syntax and representative dry-runs: passed;
- `git diff --check`: passed.

No full eight-GPU, multi-epoch experiment or Navtest PDMS evaluation was run as
part of this code audit; those are long experiments, not smoke validation.

## Commands

Bootstrap and main E3 training:

```bash
bash local_planreg_wm_v1/train_bootstrap_registers.sh 0
BOOTSTRAP_CHECKPOINT=/absolute/bootstrap-seed0.ckpt \
  bash local_planreg_wm_v1/train_e3_from_bootstrap.sh 0
```

Student export and evaluation:

```bash
python scripts/export_planreg_student_checkpoint.py \
  /absolute/training-last.ckpt /absolute/planreg-student.ckpt \
  --resolved-config /absolute/resolved_hydra_config.yaml
bash local_planreg_wm_v1/evaluate_all.sh \
  e3_from_bootstrap_seed0=/absolute/planreg-student.ckpt
```

PlanReg-WM-V1 still does **not** implement multi-trajectory consequence
modeling. The predictor exposes a future-compatible K dimension, but V1 always
uses `K=1`, the GT trajectory, and GT future-image supervision; the scorer does
not consume predicted future registers.
