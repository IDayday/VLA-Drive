# PlanReg-WM-V1

## Scope

PlanReg-WM-V1 adds action-oriented latent planning state to the existing
DriveVLA-M0 pipeline without changing its 64-query trajectory generator or
introducing a new scorer. The actual baseline in this repository is
**InternVL3-2B**, not Qwen3-VL. V1 therefore implements an explicit InternViT
adapter; the planning-register interface is isolated under
`layers/planning_registers/` so another vision-backbone adapter can be added
later without changing the action or world-model contracts.

The implemented path is:

1. prepend 16 trainable planning registers inside every InternViT forward,
   using read-only register attention so legacy CLS/patch queries cannot read
   register key columns;
2. adapt Q and V only in all InternViT attention blocks with rank-32 LoRA;
3. project the final registers from vision width 1024 to width 256 and combine
   tiles with thumbnail-query attention plus normalized tile geometry;
4. retain the original LLM-hidden-state to Q-Former semantic path;
5. mix planning-primary and semantic-residual scene features into exactly 16
   tokens of width 256;
6. feed the same mixed scene tokens to the unchanged 64-query generator and
   the DrivoR-derived scorer;
7. during training only, predict EMA future registers conditioned on one GT
   trajectory.

V1 deliberately does **not** implement candidate-specific future images,
non-GT visual targets, structured consequence heads, multi-trajectory
consequence modeling, ranking auxiliary losses, RGB reconstruction,
self-refinement, CEM/TOAD, retrieval/TTT, LLM full fine-tuning, a new scorer,
or a new PDM component. The scorer never consumes predicted future registers.

## Exact DrivoR scorer lineage

The scorer is adapted from `valeoai/DrivoR` at immutable commit
`fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a`. Source provenance is retained in
the adapted files. The parity script compares the frozen upstream files with
the local modules and verifies identical synthetic outputs.

The final proposal `[B,64,8,3]` is detached, flattened to `[B,64,24]`, and
re-embedded before an independent four-layer `TransformerDecoderScorer`.
There are six independent heads:

- `no_at_fault_collisions`
- `drivable_area_compliance`
- `time_to_collision_within_bound`
- `ego_progress`
- `driving_direction_compliance`
- `comfort`

The original sigmoid/log PDM aggregation is retained with weights
`NC=1, DAC=1, DDC=0, TTC=5, EP=5, Comfort=2`. TTC target `2.0` is excluded from
the component loss. `b2d` remains implemented as upstream but is disabled in
the PlanReg configuration. No pairwise, listwise, or ranking loss was added.
Scorer-only backward must leave proposal-coordinate gradients absent/zero and
must produce a nonzero scene-feature gradient.

Selection still uses the exact upstream log-space value. For calibration
diagnostics only, `pred_pdms = exp(log_pdm_score)/(5+5+2)` is compared with real
PDMS. This normalization never changes the argmax or selected trajectory.

## Tensor contract

| Value | Shape | Consumer |
| --- | --- | --- |
| InternViT input | `[tiles,1+16+P,1024]` | InternViT encoder |
| per-tile planning registers | `[tiles,16,256]` | slot-wise tile reducer |
| tile metadata | `[tiles,5]` | spatial tile aggregation |
| scene planning registers | `[B,16,256]` | fusion and WM loss |
| original patch/LLM features | `[B,L,1536]` | semantic Q-Former |
| semantic scene tokens | `[B,16,256]` | scene fusion |
| mixed scene tokens | `[B,16,256]` | generator and scorer |
| proposals | `[B,64,8,3]` | scorer (detached at its boundary) |
| predictor trajectory input | `[B,K,8,3]` | future-register predictor |
| predicted future registers | `[B,K,H,16,256]` | training loss only |

The InternViT sequence is strictly `[CLS, REGISTERS, PATCHES]`. After all
vision blocks, registers are sliced from positions `1:17`; only patch tokens
continue through square reshape, `pixel_shuffle`, `mlp1`, and the LLM. Registers
never enter those operations.

In the default `read_only` attention mode, register query rows can read CLS,
register, and patch keys, while CLS/patch query rows are masked from register
key columns. Therefore, when Q/V LoRA B matrices are zero, the patch output is
identical to legacy InternViT within `1e-5`. `bidirectional` remains available
only as an ablation. Read-only mode explicitly rejects flash attention.

Each tile carries normalized `[cx,cy,width,height,is_thumbnail]` metadata. The
main `thumbnail_query_attention` reducer uses each thumbnail register slot as
its query and crop registers plus a tile-position MLP as K/V, then returns
`thumbnail + tanh(tile_gate) * residual`. `tile_gate` starts at zero, so the
initial result is exactly `thumbnail_only`. Jointly permuting tiles and their
metadata leaves the result unchanged. `mean` and `thumbnail_only` remain
explicit ablations; missing thumbnails fail clearly.

The scene mixer computes, for `planning_plus_semantic`,

```text
planning_target = LayerNorm(
    planning + tanh(semantic_gate) * semantic)
scene = (1-rho) * semantic + rho * planning_target
```

`semantic_gate` starts at `atanh(0.5)=0.549306`. During training, `rho` rises
linearly from 0 to 1 over the first 20% of optimizer steps; validation and
inference use 1. `semantic_only` is an exact bypass: it returns semantic tokens
without constructing or using a gate or normalization. `planning_only` fixes
`rho=1` and excludes the semantic residual. Fusion never concatenates tokens.

## Vision adaptation and gradient routing

Each of the 24 runtime InternViT attention `qkv` modules is wrapped as:

```text
q = q_base + Bq(Aq(x))
k = k_base
v = v_base + Bv(Av(x))
```

Rank is 32, A matrices use Kaiming initialization, B matrices start at zero,
and the base QKV—including K—is frozen. A missing or structurally incompatible
`vision_model.embeddings`, `vision_model.encoder`, or
`encoder(inputs_embeds=...)` raises immediately. The generalized VLM PEFT path
is disabled, and the LLM is frozen.

The optimizer rejects any trainable parameter that is not classified into one
of these groups:

| Group | Learning rate |
| --- | ---: |
| planning adapter | `2e-4` |
| future predictor | `2e-4` |
| fusion gate/norm | `1e-4` |
| trajectory decoder and heads | `1e-4` |
| scorer decoder and six heads | `1e-4` |
| vision Q/V LoRA | `5e-5` |
| semantic Q-Former and queries | `1e-5` |
| LLM | frozen (`0`) |

Every logical group is split into `decay` (`weight_decay=0.01`) and
`no_decay` (`0.0`) subgroups. Biases, norms, register/query/embedding tokens,
semantic/tile gates, and every LoRA A/B tensor are no-decay. Unclassified
trainable tensors fail immediately. AdamW uses betas `(0.9,0.999)`, epsilon
`1e-8`, and does not automatically scale LR with batch size.

The per-step, resume-safe scheduler linearly warms from 1% of each group LR to
its peak over 3% of total optimizer steps, then cosine-decays. Final LR ratios
are 0.10 for planning/predictor/fusion/vision-LoRA and 0.20 for
action/scorer/Q-Former. Lightning's `estimated_stepping_batches` defines the
step count; scheduler state is restored on resume.

Activation checkpointing is enabled for the InternViT encoder and frozen LLM
decoder to make the audited per-GPU batch of two executable. Only checkpoint
wrapper flags enter train mode; frozen attention dropout and InternViT drop-path
children remain in eval mode, so this does not introduce stochastic VLM drift.

The normal WM setting backpropagates through current planning registers and Q/V
LoRA. `predictor_only=true` detaches current registers at the WM boundary, so
the WM loss updates only the predictor; the base trajectory/scorer losses retain
their ordinary routing. Scorer proposals are detached, while scorer scene
features remain differentiable.

## Future data and loss contract

The target builder uses `current_idx = num_history_frames - 1` and frame offsets
`[1,3,6]`, corresponding to `[0.5,1.5,3.0]` seconds. It emits fixed-size,
losslessly decodable path tensors:

```text
future_image_paths         [3,1024] uint8
future_image_path_lengths  [3]      int64
future_valid_mask          [3]      bool
```

Future supervision uses cache name `trajectory_target_planreg_wm_v1`; stale
legacy target caches are rejected. Training requires `load_image_path=true`,
`cache_hidden_state=false`, and `cache_mode=false`.

The EMA teacher contains only the vision model (including Q/V LoRA), planning
registers, and planning neck. It excludes the LLM, Q-Former, generator, scorer,
and predictor. It is deep-copied only after the student checkpoint has loaded,
is always frozen/eval/no-grad, and updates after optimizer steps with a cosine
momentum schedule from `0.996` to `0.9999`. Its progress buffers are restored on
Lightning resume.

For V1 training, the GT trajectory is expanded to `K=1`. Point features are
`[x,y,sin(theta),cos(theta),speed,acceleration]` at `dt=0.5`. Initial speed is
`sqrt(vx^2+vy^2)` from current ego status, so the first acceleration is
`(v1-v0)/dt`. Fixed scales are `x/30`, `y/10`, `speed/15`, and
`acceleration/8`; sine/cosine are unscaled. The causal trajectory transformer
samples indices `[0,2,5]`; horizon embeddings and a two-layer register
transformer predict three residual register states.

```text
current_n = layer_norm(current, affine=false)
target_current_n = layer_norm(target_current, affine=false)
target_future_n = layer_norm(target_future, affine=false)
pred_future_n = current_n + residual
wm_abs = weighted_mean(1 - cosine(pred_future_n, target_future_n))
wm_delta = weighted_mean(SmoothL1(
    pred_future_n - current_n,
    stopgrad(target_future_n - target_current_n)))
wm = wm_abs + 0.5 * wm_delta
loss = drivor_loss + current_wm_weight * wm
```

Horizon weights are `[1.0,0.7,0.4]`, with invalid horizons excluded from the
weighted denominator. `current_wm_weight` is zero for the first 5% of optimizer
steps, ramps linearly over the next 10%, and remains at `0.10`. Its
optimizer-step buffers are checkpointed, so the schedule is continuous after
resume. Legacy configs with a fixed `weight` remain loadable.

The four controlled target modes are `correct`, `no_action_condition`,
`shuffled_batch`, and `repeated_current`. `shuffled_batch` uses a cyclic batch
shift and fails explicitly for batch size one. No mode reads evaluator scores,
future actor annotations, map targets, or non-GT future imagery.

## Training data protocol

By default, training uses only `cfg.train_logs`, validation uses only
`cfg.val_logs`, and overlap is forbidden. The launcher records log counts,
overlap count, and deterministic SHA-256 hashes for both sets. Cached mode no
longer concatenates validation into training. Explicit
`include_val_in_train=true` is final-fit only: it requires zero validation
batches, disables score-monitored best checkpointing, saves only `last`, and
must not be used for hyperparameter selection.

## Checkpoint migration

Loading is strict. A legacy checkpoint may omit only explicitly whitelisted
PlanReg modules; every other missing, unexpected, or shape-mismatched key is an
error, and a complete audit is printed. Legacy PEFT deltas are folded into the
frozen corresponding base weights before the new Q/V-LoRA adapters are trained.
With planning disabled, the original forward path is called unchanged; the
synthetic parity tolerance is `max_abs_diff <= 1e-5` (including BF16 coverage).

A PlanReg training checkpoint includes predictor and EMA state. The EMA topology
is created before strict restore so optimizer/global-step/EMA momentum state can
resume continuously. Supplying a Lightning `last.ckpt` through the explicit
`RESUME_CHECKPOINT` launcher option restores optimizer and trainer state; merely
setting `agent.checkpoint_path` is a weight warm start, not a lossless resume.

## Inference contract

Inference accepts current-frame inputs only. Future path tensors are neither
required nor read. `scripts/export_planreg_student_checkpoint.py` removes EMA,
predictor, optimizer/scheduler/callback state, and WM step buffers, and emits a
SHA-256/provenance manifest. Evaluation forces `world_model.enabled=false` and
`ema.enabled=false`, so it never constructs those training-only modules.
Trajectory generation and scorer selection use only current scene features.

## Experiment matrix

| ID | Scene features | Q/V LoRA | WM target/control | Initial seeds |
| --- | --- | --- | --- | --- |
| E0 | semantic only | off | off | 0, 1, 2 |
| E1 | planning registers | off | off | 0 |
| E2 | planning + semantic | on | off | 0, 1, 2 |
| E3 | planning + semantic | on | correct future + GT action | 0, 1, 2 |
| E4 | planning + semantic | on | correct future, zero action condition | 0 |
| E5 | planning + semantic | on | shuffled future | 0 |
| E6 | planning + semantic | on | repeated current | 0 |
| E7 | planning + semantic | on | correct future, predictor-only WM routing | 0 |
| R1 | bidirectional + mean tiles | on | correct future | 0 |
| R2 | read-only + thumbnail only | on | correct future | 0 |
| R3 | read-only + thumbnail attention | on | correct future | 0 |

E0/E2/E3 are the primary comparison; controls test whether the future signal,
action condition, and visual-gradient path matter independently.

## Evaluation gates and interpretation

1. Source/unit gate: frozen DrivoR parity, TTC mask, state-dict head names,
   scorer gradient boundary, register/LoRA/EMA/path/legacy tests, and synthetic
   end-to-end smoke must all pass.
2. Data gate: a small real-data smoke must show valid future paths at all three
   horizons and finite losses before a long run.
3. Candidate gate: report candidate-bank mean/median, diversity, offline
   best-of-K, selected PDMS, and scorer regret. Best-of-K is an offline oracle
   upper bound, not deployable performance.
4. Validation gate: promote only improvements on held-out validation, using the
   same immutable scorer/evaluator settings and reporting seed dispersion or a
   paired bootstrap interval.
5. Navtest gate: run full Navtest with BF16 VLM plus FP32 action/scorer (not
   full FP32), explicit student-only checkpoint paths, checkpoint SHA-256
   records, and an explicit seed. Do not mix partial-scene runs into the final
   metric.

DriveVLA-M0 here uses a single front camera. DrivoR's reported four-camera 93.7
setting has a different sensor/input contract, so the two values are **not a
strictly fair comparison**. Claims must compare matched sensors, data, evaluator,
checkpoint selection, and split.

## Future extension boundary

The predictor API intentionally accepts:

```text
trajectories                    [B,K,8,3]
predicted_future_registers      [B,K,H,R,D]
```

This only reserves a shape-compatible extension point. PlanReg-WM-V1 always
uses `K=1`, supplies only the GT trajectory during training, and never ranks or
models candidate-specific consequences. Multi-trajectory consequence modeling
is explicitly not implemented.
