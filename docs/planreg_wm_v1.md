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

1. prepend 16 trainable planning registers inside every InternViT forward;
2. adapt Q and V only in all InternViT attention blocks with rank-32 LoRA;
3. project the final registers from vision width 1024 to width 256;
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

## Tensor contract

| Value | Shape | Consumer |
| --- | --- | --- |
| InternViT input | `[tiles,1+16+P,1024]` | InternViT encoder |
| per-tile planning registers | `[tiles,16,256]` | slot-wise tile reducer |
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

The scene mixer computes, for `planning_plus_semantic`,

```text
LayerNorm((1-rho) * semantic
          + rho * (planning + tanh(semantic_gate) * semantic))
```

`semantic_gate` starts at zero. During training, `rho` rises linearly from 0 to
1 over the first 10% of optimizer steps; validation and inference use 1.
`semantic_only` fixes `rho=0`, while `planning_only` fixes `rho=1` and excludes
the semantic residual. Fusion does not concatenate tokens.

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
| planning registers, neck, predictor | `2e-4` |
| trajectory decoder and heads | `1e-4` |
| scorer decoder and six heads | `1e-4` |
| vision Q/V LoRA | `5e-5` |
| semantic Q-Former and gate | `1e-5` |
| LLM | frozen (`0`) |

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
`[x,y,sin(theta),cos(theta),speed,acceleration]` at `dt=0.5`. The causal
trajectory transformer samples indices `[0,2,5]`; horizon embeddings and a
two-layer register transformer predict three residual register states.

```text
wm_abs = mean(1 - cosine(pred_future, stopgrad(target_future)))
wm_delta = SmoothL1(
    pred_future - current_registers,
    stopgrad(target_future - target_current),
)
loss = drivor_loss + 0.25 * (wm_abs + 0.25 * wm_delta)
```

The four controlled target modes are `correct`, `no_action_condition`,
`shuffled_batch`, and `repeated_current`. `shuffled_batch` uses a cyclic batch
shift and fails explicitly for batch size one. No mode reads evaluator scores,
future actor annotations, map targets, or non-GT future imagery.

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
required nor read. Before prediction, EMA teacher and future predictor modules
are removed; trajectory generation and scorer selection use only the mixed
current scene features. The serialized training checkpoint may retain those
modules for resumption, but the deployed inference object does not execute them.

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
5. Navtest gate: run full Navtest in FP32 with explicit checkpoint paths and
   checkpoint SHA-256 records and an explicit seed. Do not mix partial-scene runs into the final
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
