# Field2Plan Phase 2 Report

Date: 2026-08-08 UTC

## Completed scope

- Added a frozen baseline-draft cache generator with checkpoint/config/data
  hashes, deterministic seeds and inference steps, resume support, atomic
  entries, strict manifests, and fail-fast dataset binding.
- Added a CPU converter for the repository's existing DA3 metric-depth files.
- Added confidence-masked metric depth residual, relative geometry,
  occupancy, and free-space supervision.  Auxiliary losses do not receive a
  future trajectory or action.
- Added supervision-by-access switches and deterministic real, random,
  shuffled, equal-capacity, and current-state-only GT-MLP controls.
- Added geometry coordinate/field visualizers and draft/final BEV overlays.
- Added a pinned official VGGT-1B offline adapter and one-node/16-PPU cache
  launcher.  Training imports neither VGGT nor DA3; it reads only validated
  cache tensors.
- Validated the formal cached-proposal path: the frozen language/action planner
  is not recomputed during training, while the current Qwen visual tower is run
  once to construct the field.
- Kept all new behavior opt-in.  The legacy `QwenOFT` framework and baseline
  configuration remain unchanged; the Field2Plan main config still defaults
  to `geometry.teacher_type=none` and supervision disabled.
- Validated the three completed formal train caches and added eight explicit
  one-experiment-per-DLC launchers documented in `PHASE2_EXPERIMENTS.md`.

## VGGT metric-depth decision

Official VGGT depth is scale ambiguous.  A first implementation metricized it
from the three known camera baselines.  Real five-token PPU inference exposed
scale factors from approximately `7.9` to `183.6`; because the NAVSIM lateral
camera baselines are only about `0.13–0.28 m`, pose noise is strongly amplified.
That mode remains explicit `camera_rig` for diagnostics and is not the formal
default.

The formal VGGT cache instead uses `da3_scale_anchor`: current-frame VGGT depth
is aligned to current-frame DA3 metric depth and receives one robust
log-median scale per sample.  No future image, future action, or GT future
trajectory is used.  VGGT supplies its learned multiview geometry structure
and confidence; DA3 supplies only metric scale.  The cache manifest records
both teachers and hashes both source files.

On the five-token PPU check, depth medians stabilized to `17.6–37.3 m`, versus
`4.8–146.8 m` with the short-rig estimate.  This is an engineering and scale
sanity check, not evidence that the VGGT hybrid improves planning.

## External assets actually validated

- Official repository:
  `/mnt/data_and_weight/VLA_Group/LLM_weight/facebookresearch/vggt`
- Repository commit:
  `a288dd0f14786c93483e45524328726ab7b1b4ce`
- Public VGGT-1B checkpoint:
  `/mnt/data_and_weight/VLA_Group/LLM_weight/facebook/VGGT-1B/model.safetensors`
- Hugging Face revision:
  `860abec7937da0a4c03c41d3c269c366e82abdf9`
- Checkpoint bytes: `5,026,367,224`
- Checkpoint SHA256:
  `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`

No package or checkpoint download occurs from model forward, dataset loading,
or any training launcher.

## Modified and added files

- `starVLA/model/modules/field2plan/geometry_teachers.py`
- `starVLA/model/modules/field2plan/geometry_supervision.py`
- `starVLA/model/modules/field2plan/controls.py`
- `starVLA/model/modules/field2plan/__init__.py`
- `starVLA/dataloader/field2plan_cache.py`
- `starVLA/dataloader/navsim_dataset.py`
- `starVLA/dataloader/__init__.py`
- `starVLA/model/framework/QwenOFT_Field2Plan.py`
- `starVLA/config/training/cfg_field2plan_mvp.yaml`
- `starVLA/config/training/cfg_field2plan_mvp_debug.yaml`
- `tools/field2plan/cache_baseline_drafts.py`
- `tools/field2plan/cache_geometry_da3.py`
- `tools/field2plan/cache_geometry_vggt.py`
- `tools/field2plan/visualize_fields.py`
- `scripts/field2plan/02_cache_baseline_drafts.sh`
- `scripts/field2plan/03_cache_geometry_da3.sh`
- `scripts/field2plan/04_debug_geometry.sh`
- `scripts/field2plan/05_train_geometry.sh`
- `scripts/field2plan/06_cache_geometry_vggt.sh`
- `scripts/field2plan/07_run_phase2_experiment.sh`
- `scripts/field2plan/train_p2_*.sh`
- `tests/field2plan/test_launchers.py`
- `tests/field2plan/`

## Actual tensor contract

- Frozen normalized proposal: `[B,M,8,4]`.
- Frozen physical proposal: `[B,M,8,3]`.
- DA3/VGGT depth, confidence and validity: `[B,V,Hd,Wd]`; `V=3` in
  `cam_f0,cam_l0,cam_r0` order.  VGGT formal cache uses `[3,144,256]` per
  sample; existing DA3 source uses `[3,140,252]`.
- Camera intrinsics / ego-to-camera / image size: `[B,V,3,3]`, `[B,V,4,4]`,
  `[B,V,2]`.
- Per-view/per-height supervision targets and predictions:
  `[B,V,Z,Ny,Nx]`; main uses `Z=3`, `Ny=Nx=64`.
- Geometry field: `[B,256,64,64]` in the main config.
- Tube readout: `[B,M,8,256]` in the main config.
- Initial refiner output remains exactly zero, so `final=draft` before
  learning.

## Commands actually executed

```bash
pytest tests/field2plan -q
python -m compileall -q starVLA/model/modules/field2plan \
  starVLA/model/framework/QwenOFT_Field2Plan.py \
  starVLA/dataloader/field2plan_cache.py tools/field2plan
bash -n scripts/field2plan/*.sh
git diff --check

# Validate the existing 396-entry DA3 mini conversion.
python tools/field2plan/cache_geometry_da3.py \
  --source-root "$DATA_ROOT/meta/mini" \
  --datalist "$PWD/mini_meta.json" --split mini \
  --output-dir "$PWD/field2plan_cache/debug_geometry_da3_mini" \
  --max-samples 396 --validate-only

# Actual pinned official VGGT inference/cache validation on local PPU.
VGGT_ALLOW_NONFORMAL_TOPOLOGY=1 VGGT_NUM_PROCESSES=1 \
VGGT_SPLIT=mini VGGT_MAX_SAMPLES=5 \
FIELD2PLAN_DATALIST_PATH="$PWD/mini_meta.json" \
FIELD2PLAN_VGGT_CACHE="$PWD/field2plan_cache/debug_vggt_da3_anchor_mini" \
bash scripts/field2plan/06_cache_geometry_vggt.sh

# Actual cached-draft + VGGT-cache training forward/backward.
FIELD2PLAN_DEBUG_USE_DRAFT_CACHE=1 \
FIELD2PLAN_DEBUG_MAX_SAMPLES=2 \
FIELD2PLAN_DEBUG_NUM_PROCESSES=2 \
FIELD2PLAN_GEOMETRY_TEACHER_TYPE=vggt \
FIELD2PLAN_GEOMETRY_CACHE="$PWD/field2plan_cache/debug_vggt_da3_anchor_mini" \
RUN_ID=field2plan-vggt-cache-smoke-20260808 \
bash scripts/field2plan/04_debug_geometry.sh

# Actual full-cache binding/count/hash QA plus a deterministic 512-token sample.
# The corresponding commands used DraftCacheReader/GeometryCacheReader against
# train_meta.json and checked all three 103,288-entry directories.

# Actual two-PPU optimizer step using the formal train caches.
FIELD2PLAN_DEBUG_USE_DRAFT_CACHE=1 \
FIELD2PLAN_DEBUG_DRAFT_CACHE="$PWD/field2plan_cache/baseline_drafts_193514_seed20260808_steps10" \
FIELD2PLAN_DATALIST_PATH="$PWD/train_meta.json" \
FIELD2PLAN_DEBUG_SPLIT=train FIELD2PLAN_DEBUG_MAX_SAMPLES=32 \
FIELD2PLAN_DEBUG_NUM_PROCESSES=2 FIELD2PLAN_GEOMETRY_TEACHER_TYPE=vggt \
FIELD2PLAN_GEOMETRY_CACHE="$PWD/field2plan_cache/geometry_vggt_1b_da3_anchor_v1" \
RUN_ID=field2plan-formal-cache-vggt-smoke-20260808 \
bash scripts/field2plan/04_debug_geometry.sh
```

## Test results

- PASS: all current CPU tests in `tests/field2plan`, including launcher
  selection, manifest pinning, and the eight one-container wrappers.
- PASS: compileall, all Field2Plan launcher syntax checks, whitespace check,
  and static scans for bare `except:` and unconditional `.cuda()` in new paths.
- PASS: strict missing/corrupt/mismatched cache and datalist binding tests.
- PASS: actual official VGGT checkpoint load and PPU inference, atomic cache
  generation, manifest construction, resume, and validation.
- PASS: actual two-PPU Accelerate/DeepSpeed ZeRO-2/BF16 one-optimizer-step
  training with cached baseline drafts and VGGT geometry.  Effective batch was
  `2 devices * 2 batch * 8 accumulation = 32`; finite results included
  `plan_loss=0.0096841`, `geometry_depth_loss=43.6817`,
  `geometry_valid_ratio=0.32060`, `delta_norm=0`, `model_time=1.676 s`, and
  `wall_time=2.094 s`.  Exit code was zero; PCCL abort messages occurred only
  during successful process-group teardown.
- PASS: formal draft, DA3, and VGGT cache manifests all bind to the same
  `103,288` ordered train tokens and datalist SHA. Each train directory has
  exactly `103,288` NPZ files and zero atomic-write temporary files. Draft and
  DA3 generation logs performed full `validated_entries=103288` readback;
  all 16 VGGT shard summaries cover their exact rank partition.
- PASS: actual two-PPU full-cache training smoke with effective batch 32.
  Finite step-1 values included `plan_loss=0.0294438`,
  `geometry_depth_loss=17.6601`, `geometry_valid_ratio=0.322917`, and
  `delta_norm=0`; wall time was `4.789 s` for eight accumulated microsteps.
- PASS: deterministic 512-token QA. Draft decode parity was exact; DA3/VGGT
  entry median-depth medians were `24.35/24.92 m`. VGGT per-entry maximum
  depth reached `897.24 m`, but training masks samples beyond the configured
  `200 m` supervision range.
- NOT RUN: full 100,000-step Phase-2 training, probe comparison, NAVSIM-v2
  inference, or PDMS evaluation.

## Known risks

- Phase-2 scientific acceptance is still open.  Geometry probes must beat
  random/equal-capacity controls and planning must be compared across the full
  supervision-by-access matrix.
- The VGGT hybrid is not an independent metric teacher because DA3 supplies its
  scale.  It must be reported as `VGGT structure + DA3 scale`, and compared
  against plain DA3; hiding this dependency would invalidate the ablation.
- VGGT confidence is positive and uncalibrated.  The cache applies the recorded
  monotonic transform `c/(1+c)`; calibration quality remains an experiment.
- VGGT full-cache consolidation validated all shard/token/checksum coverage,
  while this development-machine audit decompressed a deterministic 512-token
  sample rather than every VGGT NPZ. Runtime readers still fail fast on every
  entry used by training.
- No Field2Plan checkpoint or PDMS score exists yet.  A target such as PDMS
  above 92 cannot be claimed from unit/smoke tests.

## Required full DLC work before Phase 3

The full caches are complete. Run the eight separate seed-42 jobs listed in
`docs/field2plan/PHASE2_EXPERIMENTS.md`. Each thin script locks one scientific
arm and delegates to the same audited one-node/16-PPU launcher, preserving the
baseline effective batch `16 * 2 * 1 = 32` and refusing implicit restart from
scratch.

Do not proceed to the Phase-3 dynamics implementation until the complete
Phase-2 probes and planning evaluation provide positive evidence.  A possible
RL stage remains deferred until after Phase 4 outcome grounding and Phase 6
closed-loop evaluation; optimizing total PDMS without a reliable interactive
rollout would create a high reward-hacking risk.
