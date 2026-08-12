# Qwen Visual Action-Only Experiment

## Purpose

This experiment isolates one question: does allowing the Qwen3-VL visual
backbone to adapt directly to trajectory planning improve NAVSIM PDMS over the
same action-only planner with frozen visual features?

The implementation starts from the `feature/add-agent-query` action-only route,
but uses the `minimal` prompt rather than `minimal_agent`. It does not load a
baseline planner checkpoint, VGGT, DINO, Wan, PPD, GaussianSTORM, a reward
model, or any corresponding feature cache. The only pretrained model is the
Qwen3-VL base checkpoint containing the existing action tokens.

## Gradient path

```text
three current camera images
  -> Qwen visual patch encoder (trainable, LR 2e-6)
  -> image/deepstack tokens [N_visual, 2048]
  -> Qwen language backbone (trainable, LR 1e-5)
  -> 8 action-query states [B, 8, 2048]
  -> FlowmatchingActionHead (trainable, LR 1e-5)
  -> normalized trajectory [B, 8, 4]
  -> action flow-matching loss
```

The Qwen `lm_head` remains frozen and is bypassed by `QwenOFT`. The unused
agent-DINO projection head is also frozen. Visual block activation
checkpointing is enabled only in this opt-in experiment to control memory.

The released baseline remains unchanged: `framework.qwenvl.freeze_visual=true`
is the default, and visual features are produced under `torch.no_grad()`.
Cached Qwen image embeddings are rejected when visual tuning is enabled because
they would silently break the gradient path.

## Optimization

The formal default is one 16-PPU node:

```text
16 processes x micro-batch 1 x accumulation 2 = effective batch 32
max steps:       100000
warmup:          5000
checkpoint:      every 10000 steps
Qwen language:   1e-5
Qwen visual:     2e-6
Action DiT:      1e-5
weight decay:    1e-3
precision:       BF16
attention:       SDPA on PPU
```

The optimizer resolves overlapping module groups from the smallest child module
first. Therefore Qwen visual parameters occur exactly once at `2e-6`; the
remaining Qwen parameters occur at `1e-5`.

## One-command launch

Configure only the Qwen/data/output paths in the ignored `env.local.sh`, then
run from any directory:

```bash
bash /path/to/VLA-Drive/8-train_action-only-qwen-visual.sh
```

The launcher first runs a two-optimizer-step forward/backward smoke with the
same 16-PPU topology, full visual gradient path and effective batch. Formal
training starts only after that smoke succeeds. Set
`QWEN_VISUAL_RUN_SMOKE_BEFORE_FORMAL=0` only after the path has already been
validated on the same runtime.

Useful one-shot overrides:

```bash
RUN_ID=qwen-visual-action-only-seed42 \
MAX_TRAIN_STEPS=100000 \
VISUAL_LEARNING_RATE=2e-6 \
bash /path/to/VLA-Drive/8-train_action-only-qwen-visual.sh
```

Print the complete launch without loading data or models:

```bash
QWEN_VISUAL_TUNE_DRY_RUN=1 \
bash /path/to/VLA-Drive/8-train_action-only-qwen-visual.sh
```

## Fair frozen-visual control

Use the existing action-only launcher with the same effective batch and PPU
attention implementation:

```bash
RUN_ID=qwen-frozen-visual-action-only-seed42 \
PER_DEVICE_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=2 \
VLM_ATTN_IMPLEMENTATION=sdpa \
MAX_TRAIN_STEPS=100000 \
NUM_WARMUP_STEPS=5000 \
SAVE_INTERVAL=10000 \
ACTION_ONLY_FREEZE_MODULES=qwen_vl_interface.model.lm_head,agent_dino_head \
bash /path/to/VLA-Drive/8-train_action-only.sh
```

Both runs start from the same `BASE_VLM` and do not load planning weights.

## Training diagnostics and attribution

The trainer logs these additional values after backward:

- `qwen/visual_trainable`: must be `1` for the treatment and `0` for control.
- `qwen/visual_trainable_parameters`: static parameter-count contract.
- `qwen/visual_feature_grad_norm`: must be finite and non-zero; it proves that
  the action loss reaches Qwen image tokens.
- `learning_rate/qwen_visual`, `learning_rate/qwen_vl_interface` and
  `learning_rate/action_model`: verify the three optimizer schedules
  independently. The legacy `learning_rate` scalar remains for compatibility.
- `action_dit_loss`: confirms optimization, but is not enough to establish a
  planning benefit.

Interpret checkpoints in this order:

1. If visual feature gradient is absent, the experiment wiring is invalid.
2. If gradients exist but the action loss diverges, reduce visual LR before
   changing the architecture.
3. If optimization is healthy but PDMS does not improve, the result supports
   the hypothesis that single-trajectory imitation is insufficient to shape a
   better planning representation.
4. Compare identical 10k/20k/30k checkpoints and report PDMS components, not
   only aggregate PDMS. Use at least three seeds before claiming an effect.

This experiment tests end-to-end representation adaptation. It does not by
itself prove that an external geometry teacher is useful or unnecessary.
