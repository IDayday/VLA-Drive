# Structured-world FGTR round one — implementation milestone

This branch contains executable code and real data/training/deployment checks.
**The six formal training groups have not started. No planning improvement is
claimed by this milestone.** Engineering scores from four updates on a debug
subset are explicitly marked as such and cannot establish method quality.

## Source and preservation

Base result archive: `e5c12fa52184a5d77389d8e6693c2aefffb087af`.
The actual S/C implementation and necessary dependencies were recovered file
by file; see `SOURCE_RECOVERY.json` (39 files, including the unchanged canonical
scoring core). No branch merge, historical objective, training controller or
Navtest observer was restored. Historical weights/data/caches remain outside
Git and unchanged. Original three-view/eight-step defaults remain available.

`third_party/LOCK.json` and `third_party/ADDENDUM_LOCK.json` pin upstream commits,
file hashes, licenses, extraction/compatibility changes and validation. Original
small operators and their licenses are retained under `third_party/*/upstream`.
Reference Torch 1.9/MMCV 1.4 and Qwen Torch 2.5/MMCV 2.2 environments are isolated.
Nine official/ported forward and backward comparisons passed at the registered
numerical tolerance; the largest absolute difference was 3.34e-6.

## Implemented path

One current Qwen forward produces the original action representation and W.
The original FM supervision, repeat eight, and native independent-noise
ten-step proposal are retained. Proposal sampling uses no gradients, temporarily
restores deterministic eval/dropout behavior, and restores all training modes.
Separate FM noise/time, proposal and query generators are checkpointed.

Current BEV comes from the actual ResNet50/FPN/DepthNet/RCSample/BEV neck path,
never from reshaping W. A current-only wrapper corrects projection pixel/depth
centers, supports the source-camera ROI and disables calibration-cache reuse.
TokenLearner, physical time conditioning, configurable MLN, shared latent
decoder and TokenFuser generate each Bt from B0, H_A and detached model q0.
The original deformable attention reads the corresponding Bt at each q0 point;
an H_A way-decoder supplies the per-time queries. One initially zero residual
in the original four-channel normalized action space produces q_final.

Current road distance/occupancy and future full-occupancy or conditional
enter/release heads share those same fields. Event occupancy uses predicted p0.
Full-body reads use a rectangle grid with at most 0.5 m point spacing. The local
query mixture is 50% proposal neighborhood, 25% road/occupancy/change boundary,
25% global, K=1024; neighboring times, fallback counts and out-of-range flags
are explicit. Auxiliary terms are calculated once per original scene.

NAVSIM supports three current views and eight future points. nuScenes supports
six current views and six future points. No map, boxes, LiDAR, GT depth or future
images enter deployment. The standalone current input reader checks the strict
field whitelist, geometry file hash and (new exports) source RGB hashes.
FP32 native ZeRO master reconstruction is strict; TF32 is off for inference.

## Data and differences requiring disclosure

- NAVSIM retains the original 101592 train / 1696 development / 12146 Navtest
  manifests. Whole-train vehicle coverage fits x=[-20,80], y=[-40,40], 0.5 m;
  no map-size search was performed.
- One original NAVSIM PCD is truncated and affects two scenes. The old complete
  build attempt is preserved as INCOMPLETE. The corrected builder keeps both
  scenes and original ego supervision; affected auxiliary frames become
  unknown/ignore255 rather than free space. The full corrected cache is being
  built; it must finish before formal common preparation.
- nuScenes official metadata, maps and CAN bus are downloaded under
  `/mnt/project/datasets/nuscenes-v1.0-trainval`; ten raw trainval archives are
  still downloading with resumable verification. Engineering checks use real
  intact sample files extracted from a partial official archive in a separately
  marked provisional directory; these are ineligible for formal preparation.
- Mature VAD six-successor eligibility gives 23230 train and 4969 validation
  samples across the official 700/150 scenes, requiring a legal previous pose
  and complete future. These are nominal 2 Hz annotated keyframes; actual
  timestamp jitter is stored, and this is **not exact physical-time resampling**.
  Future scene labels now accept only annotations within 60 ms of their nominal
  physical time; larger jitter is unknown255. Native VAD ego targets/evaluation
  remain unchanged. This approximation and per-time valid coverage are reported.
  Whole-NAVSIM timing QA likewise found 678 jump-frame scenes. Their native
  complete eight-point ego labels remain in training; future scene labels with
  mismatched physical times are unknown rather than excluding those scenes.
- NAVSIM origin is the canonical Pacifica rear axle. The nuScenes wrapper uses
  rotated current LiDAR axes (forward/left = LiDAR y/-x), exact inverse transform,
  and the native UniAD 4.084 x 1.85 m body with +0.5 m offset. Current state uses
  only previous/current poses. VAD navigation is a fixed high-level command
  derived from logged future endpoint left/right thresholds; this protocol
  condition is disclosed and common to both groups, with no coordinates input.
- Maps/frames produce derived SDF, box occupancy and endpoint events, not native
  risk GT. Height validity is conservative local LiDAR ground evidence with at
  most 2 m fill; missing support is unknown. Upright boxes and planar map yaw
  remain approximations. A named wrapper corrects mature bottom-center drift
  under pitch/roll while retaining the original transformations/rasterizer.
  Real eight-scene gravity-center error is at most 1.28e-5 m.

## Actual checks and remaining gates

Contract tests passed, including original ten-step proposal identity,
6/8-time-point gradient paths, 3/6-camera contracts, event identities,
unknown labels, clone invariance, native UniAD metric AST equality and tilted
box center semantics. The historical S0 strict read-only replay is bitwise
identical for its recorded scene. A real full-Qwen probe verified one current
forward, enabled branch gradients, GT-key poisoning invariance and unchanged
predictions after stripping only supervision heads.

24 actual eight-rank reduction comparisons passed across all four objectives,
6/8 horizons, global batches 32/24/30 and an entirely auxiliary-empty rank.
Maximum gradient discrepancy from a full-population native objective was
5.97e-8. Independent current exports cover all 1696 development scenes. Native
GT replay on four development scenes successfully traversed the canonical
scoring adapter; these privileged GT results are not model performance.
The canonical lock is taken from completed S0 scoring, not the older metric
cache builder's package tree; see `CANONICAL_EVALUATION_LOCK.json`.

Fixed-grid geometric approximation measurements are retained in
`GEOMETRY_POLYGON_RASTER_BODY_QA.json`: axis-aligned straight roads are exact at
cell centers; a rotated synthetic road has at most 0.25 m point error. The
body-point proxy can miss small boundary/hole incursions (four of 512 hole poses
and three of 512 rotated-road poses); exact polygon containment and this proxy
are reported separately. It is not an exact DAC or a collision probability.

Actual full-eight-A800 optimizer profiles completed for NAVSIM and nuScenes:
global batch 32, FM repeat 8, full ten-step proposal, geometry, all Bt, FGTR,
all losses, FP32 masters and actual 10x new-module LR. Observed NAVSIM microbatch
four updates took about 4.2 s; nuScenes six-view updates took 8.1–9.2 s. These
profiles preceded final loss calibration and are not a final cost commitment.
FP32 nuScenes deployment on eight real scenes and native primary scoring ran.

Common geometry debug fitting learned depth/current occupancy and overall road
distance, but boundary error did not improve; this needs checking after full
preparation. It was 64-scene engineering fitting, not five full train epochs and
is forbidden as formal initialization. Raw per-objective gradient measurements
show the unit-weight future loss dominates FM+DINO; weights will be calibrated
on a fixed training subset after full shared preparation, jointly for controls.

Eight-rank checkpoint restoration reproduces model, FP32 masters, Adam moments,
LR, all RNG and data position **exactly at the restored boundary**. Geometry's
subsequent short continuation passed the registered floating tolerance. VLA's
continuation has 5022 out-of-tolerance floating elements out of 11.25 billion,
max 2.32e-4 (Adam first moments), with exact discrete/RNG state. That continuation
test is retained as a failure; CUDA atomic/dropout/kernel rounding is not hidden
by claiming bitwise future updates. Functional consequences need assessment.

Still pending: full data hashes/label builds; remaining full-model/restore
checks; full common
geometry five epochs; common weights and final G1/G3 profile; clean frozen
source/run registration; all six formal seed42 trainings; common endpoints,
development/final evaluation, paired failure and mechanism analysis, costs and
second-seed limitations. Existing six YAMLs are implementation probes and are
guarded against formal launch until the common recipe is frozen.

## Executable entry points

Run from the task repository, using the isolated Qwen Python for model/data
tools and `/root/miniconda3/envs/navsim/bin/python` for canonical NAVSIM scoring.
All scripts below exist. `--help` lists actual required asset arguments; outputs
must be new directories or identity-matching resumes.

```bash
QPY=/tmp/structured-world-round1/envs/qwen/bin/python
ART=/mnt/project/structured-world-fgtr-round1-artifacts/20261008
$QPY -m pytest -q tests/structured_world
$QPY tools/structured_world/validate_reference_modules.py --help
$QPY tools/structured_world/download_nuscenes.py --help
$QPY tools/structured_world/build_cache.py --help
$QPY tools/structured_world/build_nuscenes_cache.py --help
$QPY tools/structured_world/build_current_dino.py --help
$QPY tools/structured_world/train_geometry.py --help
$QPY tools/structured_world/calibrate_objectives.py --help
$QPY tools/structured_world/train_vla.py --help
$QPY tools/structured_world/verify_training_recovery.py --help
$QPY tools/structured_world/build_navsim_current_inputs.py --help
$QPY tools/structured_world/export_current_inputs.py --help
$QPY tools/structured_world/infer_checkpoint.py --help
$QPY tools/structured_world/score_nuscenes.py --help
/root/miniconda3/envs/navsim/bin/python tools/structured_world/score_navsim.py --help
```

Formal training uses `train_vla.py --scope formal` only after its initialization,
cache, clean-source and frozen-configuration gates are satisfied. Restart uses
the **same frozen source and arguments** with `--resume`; changing source hashes
or common recipe is rejected. No automatic legacy control process is started.
Only this feature branch is to be pushed; no automatic merge.
