# PlanReg-WM-V2

## Implementation boundary

V2 lives in `navsim/agents/EpisodeDrive/planreg_v2/`. V1 configs, checkpoint loaders, Q-Former, action decoder and loss remain available. The only shared production-module change is an optional decoder memory padding mask; `mask=None` retains exact frozen-source scorer parity.

V2 uses InternVL3-2B, one current front-camera image and existing numeric ego/history/navigation information. It does not deploy future images, EMA, a WM predictor, privileged labels or a second visual backbone. There is no RL, CEM, TOAD, ranking objective, coordinate correction, or newly invented consequence label/head.

## Current-only graph

```
current front image -> dynamic tiles -> frozen InternViT + Q/V LoRA + 16 internal registers/tile
                                | patches only                  | projected visual content
                                v                               v
                    pixel_shuffle + frozen mlp1          padded tile×16 memory + geometry
                                |                               | Q
          text prefix + 16 learned soft embeddings -> LLM ------| K/V semantic fusion
                                        language q/k/v/o LoRA   |
                                              shared rich scene memory
                                               /              \
                                  4 generator layers       physical proposals.detach()
                                  one shared head          -> original scorer decoder/heads
                                  normalized -> physical         -> original log argmax
```

The actual loaded language model is **Qwen2ForCausalLM, 28 layers, hidden size 1536** inside InternVL3-2B, not a substituted Qwen3 VLM. Dimensions are read at runtime. Each sample's 16 task queries follow its valid prefix; no new vocabulary IDs or logits are generated. Both planning and WM can backpropagate through frozen language layers into visual LoRA. Native causal language attention is retained.

Default scene memory length is `16 × actual tile count`, padded across samples. Geometry is `[cx,cy,width,height,is_thumbnail]`. The semantic gate starts at sigmoid probability 0.2. The explicit compact-memory control averages corresponding register slots only after fusion. The inactive V1 tile gate/MLP and Q-Former are not constructed or optimized in V2.

The generator has exactly one 256→1024→1024→24 MLP instance shared by four scene-reading layers. Each stage has weight 0.25. There is no scene-free head0 and no previous-loss recursion. Only final physical candidates receive PDM labels and scorer loss.

## Coordinates, time and normalization

Raw GT is current-rear-axle-relative metres/radians. Heading is anchored and temporally unwrapped for statistics; external trajectories are wrapped. `TrajectoryNormalizer` stores measured FP32 `[8,3]` means/stds with a reported std floor, token hash, unique count, split and coordinate convention. Zero position or a stopped vehicle is valid. Unsupported statistics timestamps raise instead of rounding. Statistics never include online candidates or Navtest.

Physical WTA matching is `mean_t(|dx|+|dy|+|wrapped dtheta|)`. After selection, regression is `mean_t(sum_d(abs(error)/std[t,d]))`; no extra division by three. Original and long targets may choose different candidates. Long-2 uses actual same-log poses through approximately five seconds, resampled to eight output slots. It is explicitly a retimed target, never paired with WM future images. Missing long data produces a mask without dropping a scene.

V2 future offsets are **1/3/8**, meaning **0.5/1.5/4.0 seconds** after `num_history_frames-1`. Timestamp tolerance is explicitly 20ms for frame coverage; actual measured times feed motion encoding. Real logs contained values such as `0.4997, 1.4988, 3.9988`, so exact-floating-point endpoint equality is invalid. A three-second frame cannot pass the four-second check.

`GTLogMotionBuilder` requires logged vx/vy/ax/ay and an explicit source vector frame. It never infers that frame from `in_global_frame` (a pose flag), or substitutes differences. It converts logged vectors once into current-anchor axes. The smoke used the declared ego-vector convention; production data versions must document/verify their producer convention.

`CandidateKinematicsCodec` is a real `[B,K,T,3]` API for all candidates, including GT without an identity flag. It calculates physical interval-average velocity at midpoint times, and causal acceleration using the current logged velocity at t0 for the first previous value. It then applies declared motion scales. It does not call interval velocity an instantaneous endpoint velocity.

The action encoder consumes all valid points in intervals 0→0.5, 0.5→1.5, 1.5→4.0 (1/2/5 points at nominal 2Hz), including absolute time, relative time and duration. Short/invalid coverage is masked; no uniform-velocity suffix is invented.

## World model

Teacher: InternViT + visual Q/V LoRA + registers + norm/projection only. One batched visual call encodes current plus three true future frames, using **the current frame's tile layout** for all four images. Content targets exclude the added geometry/time embeddings.

The shared AC predictor has two independently initialized 256-wide, 8-head blocks, FFN1024 and zero dropout. MHA packed input weights are explicitly initialized. Block-causal masking prevents earlier transitions, including their semantic-prefix path, from reading later states/actions. Geometry, real time/duration and register-slot identity are separate; registers are not represented as a fake spatial grid.

TF consumes online z0 plus detached real EMA prefixes. RO recursively consumes its own predictions without detach or teacher replacement. Every state is in the same affine-free LayerNorm space. Both losses are L1, first averaged over each sample/horizon's valid tokens/channels, then over globally valid sample/horizon pairs. DDP reductions include the correct gradient scaling for unequal validity counts and all-invalid ranks. Missing middle images invalidate dependent TF prefixes, not independently observable RO targets.

`L = L_trajectory + L_exact_scorer + lambda*(L_TF+L_RO)`.
The first transition participates in both TF and RO, deliberately. Lambda starts at 0.01 and reaches 0.10 over the first 10% of optimizer steps. This is an initial engineering setting, not a validated optimum.

The default WM branch uses **independent GT-log motion**. `motion_mode=trajectory_kinematics` is an explicit alternative using GT through the same candidate codec. `rollout_candidates` actually supports K=1/8/64, including chunking, but this does **not** implement or validate a multi-candidate labeled consequence model.

## Precision and resume

All 26,414,111 trainable parameters are FP32. The frozen VLM base is BF16. VLM activations use BF16 autocast; action/scorer operations are FP32. All actual AdamW moments are checked, not inferred from YAML.

EMA keeps 101 persistent FP32 master tensors for tracked visual trainables; frozen base weights are copied once. Dtype conversion cannot reduce master precision. Optimizer post-step hooks ensure exactly one EMA update per real optimizer step, not per accumulation batch or GradScaler-skipped step. Momentum is cosine 0.996→0.9999, raised to `actual_global_batch/16`; actual batch includes accumulation. Exhausted schedules raise instead of silently extending/clamping.

Complete resume saves model/normalizers, optimizer, scheduler, EMA/master, per-rank RNG, shuffled sampler seed/epoch/offset, token-manifest hash and locked schedule. CUDA deterministic algorithms and cuBLAS workspace are explicitly configured. V1 warm start is a separate audited migration, not complete resume; old BF16 EMA increments are unrecoverable. Compatible final physical output weights are transformed by `W/std` and `(b-mean)/std`; only head-level physical equivalence is claimed.

The unchanged source loss performs in-place NC/DDC target conversion. V2 supplies an FP64 label boundary so each source `.to(logit_dtype)` creates an independent FP32 tensor. This avoids autograd version-counter aliasing without modifying label values, the six BCE terms or scorer source.

## Training and deployment commands

Use the already verified environment; do not upgrade the shared environment implicitly:

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages"
PY=/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PLANREG_BASE_VLM_PATH=/path/to/base-InternVL3-2B
export PLANREG_VQA_VLM_PATH=/path/to/dense-driving-VQA-InternVL3-2B
export PLANREG_V2_NORMALIZER=/new/cache/normalizer.json
```

Build a **new** input-only cache from an authorized training token list (no Navtest tokens):

```bash
$PY scripts/build_planreg_v2_input_cache.py --logs /data/logs/trainval \
  --sensors /data/sensor_blobs/trainval --metric-metadata /cache/metadata.csv \
  --tokens /protocol/trainval_tokens.json --source-vector-frame ego \
  --split trainval_final_fit --output /new/cache/v2
$PY scripts/create_planreg_v2_shared_init.py \
  --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --seed 0 --output /new/shared_v2_seed0.pt
export PLANREG_V2_SHARED_INIT=/new/shared_v2_seed0.pt
$PY scripts/audit_planreg_v2_pair.py \
  --base-config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --vqa-config navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml \
  --output /new/pair_audit.json
```

Profile V2 with a bounded `--smoke-steps 32` run before creating a GB128 layout lock. A single-GPU microbatch-one smoke is **not** a formal-layout validation. Prefer smaller microbatches plus accumulation to reach GB128; do not disable language adaptation, rich memory or WM to fit memory.

```bash
# One explicitly authorized full run, 27 epochs and no internal best-epoch selection:
export PLANREG_V2_MANIFEST=/new/cache/v2/manifest.json
export PLANREG_V2_LAYOUT=/new/measured_v2_layout.json
LAUNCH_FORMAL=1 OUTPUT_DIR=/runs/v2_base bash local_planreg_wm_v2/train.sh

# VQA uses the same architecture and shared trainable initialization:
LAUNCH_FORMAL=1 OUTPUT_DIR=/runs/v2_vqa \
  V2_CONFIG=navsim/planning/script/config/common/agent/planreg_wm_v2_vqa.yaml \
  bash local_planreg_wm_v2/train.sh

LAUNCH_FORMAL=1 OUTPUT_DIR=/runs/v2_base RESUME_CHECKPOINT=/runs/v2_base/last.ckpt \
  bash local_planreg_wm_v2/resume.sh

$PY scripts/export_planreg_v2_student.py --input /runs/v2_base/epoch_27_final.ckpt \
  --output /runs/v2_base/epoch27_student.ckpt
# Official selected Navtest score, using the original NAVSIM evaluator:
PLANREG_V2_STUDENT=/runs/v2_base/epoch27_student.ckpt \
  PLANREG_NAVTEST_LOGS=/data/logs/test PLANREG_NAVTEST_SENSORS=/data/sensors/test \
  PLANREG_NAVTEST_METRIC_CACHE=/cache/official_navtest EVAL_OUTPUT=/eval/v2_official \
  LAUNCH_NAVTEST=1 bash local_planreg_wm_v2/navtest.sh
```

Explicit legacy migration is a separate bounded diagnostic, never implicit initialization of a formal main result:

```bash
$PY scripts/migrate_planreg_v1_to_v2.py --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --input /old/v1.ckpt --output /new/v1_to_v2_warm_start.pt
$PY scripts/train_planreg_v2.py --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --manifest /new/cache/v2/manifest.json --output /new/warm_start_replay \
  --warm-start /new/v1_to_v2_warm_start.pt --smoke-steps 32 --microbatch 1 --accumulate 1
```

Gradient accumulation uses global validity counts across the complete optimizer batch for trajectory and TF/RO terms. Step timing covers all microbatches and data wait; per-step logs report the optimizer-batch mean, current group LRs and EMA momentum. Low-frequency update reports measure actual parameter deltas rather than equating finite gradients with updates.

Standalone V2 training uses fully resolved OmegaConf configs with explicit inheritance; it does not mutate V1's Hydra launch path. `planreg_wm_v2_student.yaml` is also a Hydra factory entry for the normal NAVSIM evaluator. Student construction contains no teacher or predictor and embeds the normalizer statistics in the checkpoint, so it does not need the training statistics file.

`evaluate.sh` is a separate all-64 candidate diagnostic using the preserved **training PDM scorer / fixed reference progress** protocol; its report is marked `official_navtest_result=false`. Do not equate that diagnostic with the official `navtest.sh` result. Both entries use the same deployed policy, and neither is automatically launched. The official entry defaults to the existing sequential worker for an unambiguous selected-score reference; distributed/batch scoring requires its own protocol parity validation.

For 103,288 scenes at GB128, the exact sampler pads eight exposures per epoch: 807 steps/epoch and 21,789 steps at the fixed epoch27 endpoint. The launcher computes this from the actual manifest/layout and refuses silent schedule changes. Checkpoints are last plus epochs 5/10/15/20/25/27. No final-fit internal validation is used to choose a checkpoint.

The current no-WM and compact-memory configs are explicit controls, never auto-launched. A different trainable topology needs its own versioned initialization artifact. No-WM has no unused predictor parameters.
