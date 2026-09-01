# PlanReg-WM-V1 experiment protocol

## Immutable identifiers

- VLA-Drive base: branch `DriveVLA-M0`, commit
  `d84bf2b39696050f715fe41c5f005d0d1115c0c1`
- Scorer source: `valeoai/DrivoR`, commit
  `fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`
- Baseline VLM: local trust-remote-code **InternVL3-2B**, with 24-block
  `InternVisionModel` (`hidden_size=1024`); it is not Qwen3-VL
- PlanReg branch: `feature/planreg-wm-v1-drivor-scorer`

The run metadata directory must contain the exact VLA-Drive commit, dirty
status, resolved Hydra configuration, redacted environment, launch command,
training log, and—for evaluation—the checkpoint SHA-256.

## Fixed method contract

- InternViT input: `[CLS, 16 planning registers, patch tokens]`.
- Register output: per tile `[T,16,256]`; slot-wise mean `[B,16,256]`.
- Patch count and the legacy patch-to-`pixel_shuffle`-to-`mlp1`-to-LLM path are
  unchanged. Registers never enter `pixel_shuffle`, `mlp1`, or the LLM.
- Every InternViT block receives rank-32 Q/V-only LoRA; K, base QKV, and the LLM
  are frozen.
- Semantic Q-Former output is retained and fused into exactly `[B,16,256]`.
- Generator output remains `[B,64,8,3]`.
- The same current scene tokens drive generator and scorer; predicted future
  registers never drive the scorer.
- Scorer is the six-head, independent four-layer decoder implementation from
  the fixed DrivoR source. Proposal coordinates are detached before flattening
  and embedding. PDM weights are `1/1/0/5/5/2`, `b2d=false`, and TTC target
  `2.0` is masked.
- Predictor API accepts trajectories `[B,K,8,3]` and returns future registers
  `[B,K,H,R,D]`; V1 trains only `K=1` with GT trajectory and real GT future
  images at `0.5/1.5/3.0` seconds.
- EMA consists only of InternViT, its Q/V LoRA, registers, and register neck.
- Deployment deletes EMA and predictor and requires no future input.

No candidate-specific future image, non-GT future target, structured
consequence head, multi-trajectory consequence model, ranking loss, RGB
reconstruction, refinement, CEM/TOAD, retrieval/TTT, LLM fine-tuning, new
scorer, or new PDM component is in protocol scope.

## Experiment matrix

| ID | Register path | Vision Q/V LoRA | Future condition/target | Seeds |
| --- | --- | --- | --- | --- |
| E0 | disabled; semantic only | off | disabled | 0,1,2 |
| E1 | enabled | off | disabled | 0 |
| E2 | enabled | on | disabled | 0,1,2 |
| E3 | enabled | on | GT action + correct future | 0,1,2 |
| E4 | enabled | on | zero action + correct future | 0 |
| E5 | enabled | on | GT action + batch-shuffled future | 0 |
| E6 | enabled | on | GT action + repeated current | 0 |
| E7 | enabled | on | E3 target; WM gradients stop at registers | 0 |

All runs use explicit seeds and output directories. A run starts from the fixed
base checkpoint unless `RESUME_CHECKPOINT` is explicitly supplied. Automatic
checkpoint discovery/resumption is disabled.

## Preflight gates

Run from the isolated worktree with the project interpreter:

```bash
/mnt/project/DriveVLA-M0-env/bin/python scripts/audit_drivor_scorer_parity.py
/mnt/project/DriveVLA-M0-env/bin/python scripts/smoke_planreg_wm_v1.py
/mnt/project/DriveVLA-M0-env/bin/python -m pytest -q \
  tests/test_drivor_scorer_parity.py \
  tests/test_internvl_planning_registers.py \
  tests/test_vision_qv_lora.py \
  tests/test_future_register_predictor.py \
  tests/test_future_image_paths.py \
  tests/test_ema_register_target.py \
  tests/test_scene_fusion.py \
  tests/test_world_model_gradient_routing.py \
  tests/test_legacy_forward_parity.py \
  tests/test_planreg_optimizer_groups.py
```

Stop if any gate fails. Do not launch a long run on the basis of a partial or
manually edited result.

## Data contract gate

Training uses uncached current-frame features and the unique target builder
`trajectory_target_planreg_wm_v1`:

```text
cache_hidden_state=false
cache_mode=false
load_image_path=true
future_image_paths        [B,3,1024]
future_image_path_lengths [B,3]
future_valid_mask         [B,3]
```

The current index is `num_history_frames-1`; future offsets are `1,3,6`. A
small real-data run must report the count of valid targets per horizon and must
not silently replace a stale `trajectory_target` cache. For shuffled controls,
batch size must exceed one.

## Optimization and resume gate

The launcher must print five classified trainable groups and no unclassified
parameter:

```text
new_modules       lr=2e-4
action_head       lr=1e-4
scorer            lr=1e-4
vision_qv_lora    lr=5e-5
semantic_qformer  lr=1e-5
```

The LLM learning rate is zero and its parameters are frozen. Normal E3 WM loss
must yield nonzero gradients for registers and vision Q/V LoRA; E7 must yield no
WM-derived gradient into either while retaining predictor gradients. Scorer-only
loss must leave proposal coordinates unmodified but update the scene/register
path.

Only a Lightning `last.ckpt` supplied through `RESUME_CHECKPOINT` is treated as
a lossless continuation: it restores model, optimizer, global step, and the EMA
schedule buffers. A base/best checkpoint passed as `agent.checkpoint_path` is a
weight initialization and must not be described as lossless continuation.

## Launch sequence

First inspect all commands:

```bash
for script in local_planreg_wm_v1/train_e*.sh; do
  DRY_RUN=1 bash "$script" 0
done
```

Then run a small-data E3 smoke before full experiments:

```bash
SMOKE_SPLIT=1 SMOKE_SCENES=32 \
  bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh 0
```

Primary full runs:

```bash
for seed in 0 1 2; do
  PLANREG_TRAIN_TEST_SPLIT=navtrain \
    bash local_planreg_wm_v1/train_e0_semantic_exact_scorer.sh "$seed"
  PLANREG_TRAIN_TEST_SPLIT=navtrain \
    bash local_planreg_wm_v1/train_e2_register_qvlora.sh "$seed"
  PLANREG_TRAIN_TEST_SPLIT=navtrain \
    bash local_planreg_wm_v1/train_e3_register_qvlora_wm.sh "$seed"
done
```

Run E1 and E4-E7 at seed zero after the primary smoke gate. `trainval` may be
selected with `PLANREG_TRAIN_TEST_SPLIT=trainval`; the chosen split must be
recorded and kept identical within a comparison.

## Evaluation and promotion gates

Evaluation takes explicit label/checkpoint pairs, sets trainer precision to
FP32, and records each checkpoint hash:

```bash
bash local_planreg_wm_v1/evaluate_all.sh \
  e0_semantic_exact_scorer_seed0=/absolute/e0.ckpt \
  e2_register_qvlora_seed0=/absolute/e2.ckpt \
  e3_register_qvlora_wm_seed0=/absolute/e3.ckpt
```

Use a fixed, immutable candidate bank to report candidate mean/median,
diversity, selected PDMS, best-of-K, and scorer regret. Best-of-K is an offline
oracle diagnostic and is never reported as deployable PDMS. A configuration is
promoted only after:

1. all source/unit/smoke gates pass;
2. all three future horizons have real valid inputs in the real-data smoke;
3. held-out validation improves under identical evaluator settings;
4. primary seeds show a consistent effect (report dispersion and preferably a
   paired bootstrap confidence interval);
5. full Navtest covers the complete expected token set with no duplicates or
   failures and uses one explicit checkpoint per result.

Do not select a checkpoint on Navtest. Do not combine different candidate banks,
sensor contracts, scorer settings, or partial-scene outputs in one comparison.

The single-front-view DriveVLA-M0 setup and DrivoR's four-camera reported 93.7
setup are not a strictly fair comparison. Any table containing both must label
the sensor and training/evaluation differences explicitly.

## Required reporting

For each experiment report: git SHA, checkpoint SHA, full resolved config,
training split, seed, optimizer groups, effective global batch, training steps,
future validity by horizon, selected PDMS, offline best-of-K, scorer regret,
candidate diversity, and failure count. Include register effective rank,
pairwise cosine, standard deviation, horizon cosine/delta losses, and vision
LoRA/register gradient norms from training logs.

The reserved `[B,K,H,R,D]` predictor output is only an interface for later work.
This protocol does not implement or evaluate multi-trajectory consequence
modeling; V1 always supervises `K=1` from a GT trajectory.
