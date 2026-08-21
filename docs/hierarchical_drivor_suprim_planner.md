# QwenPI-DrivoRSuprim hierarchical planner

`QwenPI-DrivoRSuprim` is a single training task and a single model. Only Qwen
loads pretrained parameters; it is permanently frozen. The scene encoder,
Flow-DiT, DrivoR scorer, and both DriveSuprim stages start from random
initialization and are optimized by one AdamW optimizer, one scheduler, and one
backward pass.

## Architecture and tensor contracts

```text
multi-view images + navigation text
  -> frozen Qwen (one forward, no_grad)
  -> final 16 layerwise full sequences [B,L,2048] for existing Flow-DiT
  -> final full sequence [B,L,2048]
  -> 4-layer GlobalSceneQFormer
       dense memory  [B,L,2048]
       global tokens [B,16,2048]
  -> scene-conditioned Flow-DiT
       training: Flow-Matching velocity loss
       sampling: 64 independent normalized trajectories [B,64,8,4]
  -> TrajectoryCodec -> physical NAVSIM proposals [B,64,8,3]
  -> 4-layer DrivoR scorer (256-d query / 2048-d memory)
  -> dynamic Top-M (M ramps 64 -> 32)
  -> deterministic 8-to-40 interpolation
  -> 8192 static + M dynamic candidates [B,8192+M,40,3]
  -> shared DriveSuprim coarse scorer (256-d query / 2048-d global memory)
  -> one global Top-256
  -> 3-layer fine refiner (256-d query / 2048-d dense memory)
  -> final original candidate [B,40,3] -> [B,8,3] -> normalized [B,8,4]
```

Every asymmetric decoder layer owns its own `nn.MultiheadAttention` with
`embed_dim=256`, `kdim=2048`, and `vdim=2048`. There is no shared scene-memory
projection to 256 dimensions. The 8192/8224 candidate states remain 256-wide;
only the 16 global tokens and dense Qwen memory are 2048-wide.

The Q-Former receives the complete final Qwen sequence, not action tokens,
pooled features, per-camera registers, or a pseudo-spatial 4x4 map. Its mask is
converted from Qwen's `True=valid` convention to PyTorch attention's
`True=ignore` convention. No DINO or additional image encoder is instantiated.

## Responsibilities and gradient routing

| Component | Responsibility | Trainable | Receives gradients from |
|---|---|---:|---|
| Qwen | Image/text representation | No | none (`no_grad`, detached, forced eval) |
| Global Q-Former | Dense/global 2048-d scene memory | Yes | Flow and, by default, scorer losses |
| Flow-DiT | Flow-Matching generator and 64 proposal sampler | Yes | Flow loss only |
| DrivoR | Six-metric dynamic proposal pre-score | Yes | DrivoR BCE only |
| DriveSuprim coarse | Shared static/dynamic score and global Top-256 | Yes | coarse PDM + imitation loss |
| DriveSuprim fine | Three-layer Top-256 refinement | Yes | three intermediate fine losses |

Dynamic proposal geometry is detached inside `DrivoRDynamicScorer` and again
before joining the static vocabulary. Dynamic sampling for labels runs in
`torch.no_grad()`. Consequently no DrivoR or DriveSuprim loss can update
Flow-DiT. `detach_scene_for_scorer=true` is an ablation that additionally cuts
scorer gradients at the Global Q-Former output; it never changes proposal
detachment.

## Trajectory representation

Flow actions are
`[x_norm, y_norm, sin(relative_heading), cos(relative_heading)]`. The shared
codec uses exactly the dataset constants:

```text
x_mean=10.172484  x_std=8.805105
y_mean=0.360762   y_std=2.277741
```

`flow_to_navsim` restores physical `[x,y,heading]`. Dynamic points are at
0.5-second intervals through 4.0 seconds. Upsampling prepends the zero pose at
time zero, linearly interpolates x/y and unwrapped heading at 0.1-second
intervals, then wraps heading to `[-pi,pi]`. Downsampling uses indices
`[4,9,14,19,24,29,34,39]`.

## Static and dynamic metric caches

Training needs two kinds of external labels. Neither is used by inference.

- DriveSuprim static labels: the official `navtrain.pkl` contains eight metric
  vectors for every scene token and all 8192 vocabulary trajectories. The DLC
  launcher can download it from `alkaid-2000/DriveSuprim`. Because the pickle is
  about 15 GiB and would otherwise be unpickled independently by every rank,
  `tools/split_drivesuprim_static_scores.py` converts it once to lazy per-token
  NPZ shards. The converter needs enough host RAM to load the official pickle.
- Dynamic labels: `DynamicMetricSupervisor` evaluates the current rank's 64
  detached proposals against an official NAVSIM metric cache. It returns NC,
  DAC, TTC, EP, DDC, comfort, lane keeping, traffic-light compliance,
  history comfort, and aggregate score. Ranks do not all-gather proposals.

The dataset continues to carry only its light token string. Missing tokens,
cache files, fields, or incompatible vector sizes fail explicitly. The launcher
can generate a NAVSIM `navtrain` metric cache before starting training when the
cache is absent.

## One-task curriculum

- 0-10% completed optimizer steps: Flow + static-only DriveSuprim; DrivoR is
  not executed.
- 10-20%: sample K=64, ramp DrivoR weight from 0 to 1, and reduce M from 64 to
  32 while adding dynamics to the joint pool.
- 20-100%: K=64, M=32, and all four loss weights equal one.

The schedule follows completed optimizer steps, not accumulation microsteps.
`find_unused_parameters=true` supports the static-only interval. There is no
stage checkpoint hand-off or module reinitialization.

## DLC training and resume

The launcher is non-interactive and resolves the repository from its own path
unless `VLA_PROJECT_ROOT` is explicitly supplied. Required machine-specific
paths are environment variables rather than shared-YAML constants.

```bash
cd /path/to/VLA-Drive && \
QDS_ASSET_ROOT=/path/to/ddp-drs-assets \
SUPRIM_VOCAB_PATH=/path/to/ddp-drs-assets/drivesuprim/test_8192_kmeans.npy \
bash train_qwenpi_drivor_suprim_dlc.sh
```

The launcher defaults to 8 accelerators, micro-batch 8 per accelerator, and
gradient accumulation 1, preserving effective batch 64. Override
`QDS_LOCAL_PROCESSES`, `VLA_BATCH_SIZE`, or `QDS_TARGET_EFFECTIVE_BATCH` only
when intentionally changing the topology.

To resume the one joint checkpoint, point at an Accelerator state directory;
the completed step restores the curriculum position:

```bash
VLA_RESUME_CKPT=/path/to/run/checkpoints/steps_5000 \
QDS_RUN_ID=qwenpi-drivor-suprim-resume \
bash train_qwenpi_drivor_suprim_dlc.sh
```

Useful path overrides are `QWEN_VLM_PATH`, `DATA_ROOT`,
`OPENSCENE_DATA_ROOT`, `NAVSIM_DATALIST_PATH`, `NAVSIM_METRIC_CACHE_ROOT`,
`SUPRIM_STATIC_SCORE_CACHE`, and `VLA_OUTPUT_ROOT`. Setting
`SUPRIM_STATIC_SCORE_CACHE` to a prepared shard directory skips download and
conversion.

## Inference

Build `QwenPI-DrivoRSuprim` with the same YAML, restore the joint Accelerator
checkpoint, and call `predict_action(examples=...)`. It runs only learned
modules and returns `normalized_actions` for the legacy evaluator plus physical
8/40-point trajectories and candidate-source metadata. It does not load a
metric cache, ground truth, or an evaluator.

## Ablations

- no scene tokens in DiT: `framework.action_model.use_global_scene_tokens=false`
  (requires constructing the corresponding fresh action model);
- static-only selection: use the static-only curriculum path;
- all 64 dynamics without prefilter: set Top-M to 64;
- Top-32 / Top-16: set the curriculum end and final Top-M consistently;
- detached scorer scene: `detach_scene_for_scorer=true`;
- fine global-memory ablation: set
  `framework.hierarchical_scorer.refinement.memory_source=global_scene_tokens`.
  Dense Qwen memory is the production default.

There is no diversity loss, RL, scorer-guided generator gradient, source
embedding/bonus/quota, goal conditioning, memory retrieval, or learned
trajectory interpolation.
