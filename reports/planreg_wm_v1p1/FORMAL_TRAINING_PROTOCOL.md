# V1.1 formal training protocol

Status: implementation and bounded numerical diagnostics; **no V1.1 formal
multi-day training has been launched**. Layout/convergence lock is not promoted.
All paths below refer to new artifacts, never old checkpoint/cache overwrites.

## Immutable scientific scope

BaseInit and Driving-VQAInit load only their standalone InternVL3-2B VLM. Their
new planning stacks share one seed-specific FP32 artifact, then initialize EMA.
No old M0 or epoch27/33 agent is used to initialize a new formal main result.
Old weights are used only for the separately labelled replay/probe diagnostics.

Single current front camera; 24 InternViT Q/V LoRA layers rank32; frozen base/K,
MLP, embeddings and LLM; 16 internal read-only registers; 8 global + 8 local
readout slots; planning-primary semantic cross-attention; 64×8×3 proposals;
four refinement stages; independent four-layer scorer and six heads. Fixed
scorer source is `valeoai/DrivoR@fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`.
Detach boundary, TTC invalid mask, NC/DDC mapping and 1/1/0/5/5/2 aggregation stay
unchanged. Both generator and scorer receive the same `[B,16,256]` scene.

Long-2 stays on. Default final-head-only supervision freezes head0..3, avoiding
their weight decay while retaining their checkpoint keys and forward/RNG sequence.
Opt-in light supervision is final + 0.2*mean(head1..3); head0 is excluded. No
five-head strong WTA, simple repulsion or increased candidate count.

## Input, world model and inference

Input-only cache may store paths, GT/long trajectories, status/commands, tokenized
prompts and tile metadata—not VLM hidden states, patches, semantic tokens,
planning registers or EMA targets. Build V1.1 prompts into a **separate cache root**.
The two-record real cache smoke passed; complete 103,288-record V1.1 cache is
NOT_BUILT in this task. Old caches remain intact.

WM stays enabled from step zero (min 0.01, provisional max 0.10, 10% ramp).
Correct real future images at 0.5/1.5/3.0 seconds, GT trajectory prefix only, K=1;
absolute cosine + 0.5*delta SmoothL1 in non-affine normalized state space;
horizon weights 1/.7/.4; invalid horizons excluded from the denominator.
EMA targets are online and stop-gradient. Same-batch unclipped diagnostics
compare planning loss to unweighted WM over identical visual parameter sets.
At interval500 they use cloned leaves and temporary non-reentrant checkpointing
to avoid DDP hooks, optimizer-gradient/RNG contamination and full activation
materialization. The ratio heuristic 5–15% is not a scientific PASS threshold.

The two-scene numerical checks do not justify a larger WM coefficient or a final
hyperparameter promotion. In particular a negative shared-register gradient
cosine is not an instruction to raise lambda. Keep the weaker default pending a
representative train-only log-split pilot. Correct/copy/action-only/shuffle-current
frozen diagnostics are task checks, not no-WM formal experiments.

Deployment strips EMA, predictor and training optimizer/scheduler state. Model
inference consumes only the current image and current/history ego input; no future
image, official evaluator or future annotations. Real deployment smoke used
BF16 VLM + FP32 action/scorer, **not full FP32**. The V1.1 benchmark launcher
explicitly requests full FP32 for matching audited candidate-bank comparisons.

## Budget and initial optimizer proposal

Old completed formal training actually used GB128, 807 steps/epoch, 21,789 total
steps—not the old conversational GB32/87k example.

V1.1 default candidate is GB64 (8×8 or 16×4); GB128 remains an efficiency candidate.
With 103,288 scenes and sampler padding, GB64 has 1,614 steps/epoch and 43,578
steps/27 epochs; GB128 has 807 and 21,789. Both expose 103,296 padded samples per
epoch. Compare short convergence at equal sample exposure and identical initial
weights, not equal optimizer-step counts. A 300-step throughput result cannot
prove convergence equivalence. Register/EMA diagnostics overhead must be included.

| Logical group | GB64 starting LR | Peak LR | Final LR |
|---|---:|---:|---:|
| planning adapter/readout, fusion | 2e-6 | 2e-4 | 2e-5 |
| generator, scorer | 2e-6 | 2e-4 | 2e-5 |
| semantic Q-Former | 1e-6 | 1e-4 | 1e-5 |
| future predictor | 1e-6 | 1e-4 | 1e-5 |
| vision Q/V LoRA | 3e-7 | 3e-5 | 3e-6 |
| LLM | 0 | 0 | 0 |

AdamW betas .9/.999, eps1e-8; matrices WD.01; norm/bias/register/query/embedding/
gates/all LoRA A/B WD0. Seven logical groups split into 13 actual optimizer groups
(LoRA has only no-decay). Global/local learned queries are explicitly no-decay.
5% warmup then step cosine to ratio.1, clip norm1, accumulation1. GB128 candidate
uses the configured square-root scaling relative to GB64 with existing caps;
this is an initial pilot rule, not an empirically optimal LR claim.

Sample-normalized EMA is unchanged: reference batch16, m=.996→.9999;
GB64 actual=.984095744256→.999600059996. Only accumulation dtype is repaired.
The formal endpoint is pre-registered epoch27; intermediate saves are for
recovery/curves, not Navtest epoch selection. Do not extend an exhausted old
scheduler and relabel it from-scratch training.

## Artifacts and commands

Shared initialization generated here:
`/mnt/project/DriveVLA-M0-formal-runs/v1p1_audit_20260905/shared_planreg_init_seed0.pt`
SHA256 `82d592643a0f2e08ae246a4e7716f48a16532feca03ca4d6863d5830babf50f7`.
Actual paired initialization is bitwise equal. `FORMAL_CONFIG_PAIR_AUDIT.json`
confirms only VLM identity, experiment name and output directory differ.

Generate a separate complete cache, then benchmark (not run here):

```bash
export PLANREG_PROMPT_VERSION=single_front_v1p1
python scripts/build_planreg_input_only_cache.py \
  --cache-root "$OLD_INPUT_SOURCE" --output-root "$NEW_V1P1_CACHE" \
  --tokenizer "$PLANREG_BASE_VLM_PATH" --jobs 16
export PLANREG_INPUT_CACHE="$NEW_V1P1_CACHE"
export PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt
bash local_planreg_wm_v1/benchmark_v1p1_layout.sh 8x8
bash local_planreg_wm_v1/benchmark_v1p1_layout.sh 16x4
```

After a representative train-only pilot and a new shared layout lock, explicitly
launch one chosen formal run. The lock must have `protocol_version=v1p1`,
`train_only_pilot_locked=true` and the exact dataset/step budget. Do not set that
flag without evidence. Missing lock/shared init/cache/VLM audits fail preflight;
the old V1 GB128 lock and prompt cache are rejected.

```bash
PLANREG_LAYOUT_LOCK=/absolute/v1p1_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt \
PLANREG_INPUT_CACHE=/absolute/v1p1_input_cache \
PLANREG_BASE_VLM_PATH=/absolute/InternVL3-2B-base \
bash local_planreg_wm_v1/train_formal_v1p1_base_init_wm.sh 0

# Same artifacts/layout; only the standalone VLM initialization differs:
PLANREG_LAYOUT_LOCK=/absolute/v1p1_layout_lock.json \
PLANREG_SHARED_INIT=/absolute/shared_planreg_init_seed0.pt \
PLANREG_INPUT_CACHE=/absolute/v1p1_input_cache \
PLANREG_VQA_VLM_PATH=/absolute/InternVL3-2B-driving-vqa \
bash local_planreg_wm_v1/train_formal_v1p1_vqa_init_wm.sh 0
```

Only explicit same-run `RESUME_CHECKPOINT` is accepted. No auto-resume to old runs.
Final launchers export epoch27 student-only weights. Evaluation is explicit:

```bash
PLANREG_FORMAL_EVAL_ROOT=/absolute/new_v1p1_eval \
PLANREG_BASE_VLM_PATH=/absolute/InternVL3-2B-base \
bash local_planreg_wm_v1/evaluate_v1p1_checkpoint.sh base /absolute/epoch27_student.ckpt
```

No complete Base/VQA V1.1 formal pairing or new full Navtest evaluation has been
run. No multi-trajectory consequence modeling, candidate-specific future, RGB
prediction, new scorer/ranking loss, CEM/TOAD or inference-time world model is
implemented. Predictor's future K API remains extensible, but V1.1 supervision
and formal training remain K=1 GT-only.
