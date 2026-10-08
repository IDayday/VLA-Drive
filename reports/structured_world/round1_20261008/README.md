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
  unknown/ignore255 rather than free space. The corrected v5 cache is complete
  for all 101592 scenes with zero errors (5074 seconds, 103.74 GB). Common
  preparation finished on the complete eight-A800 zt2 host: five full epochs,
  15875 updates, 507960 scene exposures, 12.964 training GPU-hours. It trained
  current geometry only; no planner or Qwen was involved. The shared identity is
  `e07c1a00d8961e10f88d71799957bcd647bc4a58009cd6cfbae22f42dfa9fecd`.
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

The new dedicated NAVSIM current-C1 cache is complete for all 101592 scenes.
It copies the locked legacy current targets bitwise, verifies original chunk
hashes, and contains no future target dependency. It took 393 CPU wall seconds
and zero GPU-hours. Eight real samples loaded successfully with a nonexistent
future teacher index. Formal training rejects the legacy mixed current/future
cache path; historical assets remain intact. A separate actual file-access
audit on eight current exports found zero non-current data reads and bitwise
identity of every current image, state, calibration and geometry input.
nuScenes retains 85.46% of nominal future scene-label times at the shared 60 ms
tolerance; every original 23230/4969 planning sample remains eligible. Exact
per-time coverage is stored in `NUSCENES_WHOLE_POPULATION_AUX_TIME_COVERAGE.json`.

Actual full-eight-A800 optimizer profiles completed for NAVSIM and nuScenes:
global batch 32, FM repeat 8, full ten-step proposal, geometry, all Bt, FGTR,
all losses, FP32 masters and actual 10x new-module LR. Observed NAVSIM microbatch
four updates took about 4.2 s; nuScenes six-view updates took 8.1–9.2 s. These
profiles preceded final loss calibration and are not a final cost commitment.
FP32 nuScenes deployment on eight real scenes and native primary scoring ran.

Common geometry's five complete training epochs learned depth, occupancy and
overall road distance. On a fixed 64-scene training diagnostic, overall road MAE
fell from 4.65 to 0.777 m, projected depth-bin expectation MAE from 34.6 to
1.375 m, and current occupancy IoU reached 0.438 (precision 0.505/recall 0.767).
Boundary MAE did not improve (0.686 to 0.772 m); a near-zero initial distance
predictor is a weak boundary-only reference. This limitation remains explicit,
and these are training diagnostics, not generalization or planning results.
Old debug-only geometry checkpoints remain ineligible for formal initialization.
Unit-weight future gradients dominated FM+DINO in early probes. The shared
training-only calibration rule is now preregistered in `COMMON_CALIBRATION_RULE.json`;
G1/G3 component medians selected one recipe common to all controls, without dev
PDMS: geometry 0.1161248667, future semantics 0.1054616027, query relations
0.3871265716 and refine 1.0, with the common 1000-update refine warmup. Both
datasets use these scalar weights; class weights come from each training
population. See `COMMON_LOSS_CALIBRATION_SELECTED.json`. Complete profiles and
small-set learning with this recipe remain prerequisites for formal freezing.

Eight-rank checkpoint restoration reproduces model, FP32 masters, Adam moments,
LR, all RNG and data position **exactly at the restored boundary**. Geometry's
subsequent short continuation passed the registered floating tolerance. VLA's
continuation has 5022 out-of-tolerance floating elements out of 11.25 billion,
max 2.32e-4 (Adam first moments), with exact discrete/RNG state. That continuation
test is retained as a failure; CUDA atomic/dropout/kernel rounding is not hidden
by claiming bitwise future updates. Functional comparison on eight real current
inputs also failed its strict predeclared tolerance: maximum coordinate
difference 5.77 mm and heading difference 0.00355 rad. The failure is retained
in `VLA_RECOVERY_FUNCTIONAL_DIAGNOSIS.json`. Two independent uninterrupted runs
are measuring ordinary native CUDA variation. Torch 2.5's warn-only determinism
does not select deterministic Flash backward; this is an observed source of
variation, not proof that it explains every difference. The new current-only
run passed exact eight-rank restoration with every checkpoint file verified by
SHA256, and completed its remaining two updates. No failed continuation check
is relabeled as a pass. Two independent uninterrupted four-update controls also
failed the same floating-state tolerance: 19.39 million elements out of 11.25
billion, maximum absolute difference 0.00839 in a BatchNorm running variance,
with exact RNG/discrete state. This establishes ordinary CUDA continuation
variation but does not prove functional equivalence or erase the recovery
failure. See `AA_UNINTERRUPTED_NUMERICAL_ANALYSIS.json`.

All 1696 development scene labels were built using the same unchanged geometry
core in a separate population that cannot enter common training preparation.
`score_scene_fields.py` evaluates complete prediction populations on dense
valid cells and an identical logged-GT body neighborhood for every model. It
reports endpoint events/timing, stationary raster-track retention, temporal
changes, road boundary errors, full-body relations and explicit unknown/OOR
coverage. Camera support is a calibration/ROI proxy, not annotated object
visibility. Synthetic perfect, delayed/missed event, unknown-label and OOR
checks passed for six/eight points. Real label-to-field identity checks passed
on 128 NAVSIM/nuScenes samples as privileged GT fixtures, not model results.
Real-model scene metrics remain pending.

nuScenes archive 02 is fully downloaded, hashed and extracted. After observing
progress lost during internal curl retries, the owned downloader was restarted
with a new If-Range request from the latest file size after each error. Completed
archives, partial bytes and the pinned object identities are preserved.

Explicitly authorized cleanup removed only verified `gpu_stress.py` process
trees on local/zt2 after they occupied 75 GiB per GPU and caused a calibration
startup OOM. Unrelated training and the old-objective pause locks remain intact.
Physical UUID comparison found that local and recovery containers share the
same eight GPUs; they are not counted as independent training servers.

Still pending: nuScenes full data hashes/label builds and shared preparation;
remaining full-model/restore checks; final G1/G3 profile and small-set learning; clean frozen
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
$QPY tools/structured_world/build_navsim_dev_labels.py --help
$QPY tools/structured_world/build_nuscenes_cache.py --help
$QPY tools/structured_world/build_current_dino.py --help
$QPY tools/structured_world/slice_navsim_current_c1.py --help
$QPY tools/structured_world/train_geometry.py --help
$QPY tools/structured_world/calibrate_objectives.py --help
$QPY tools/structured_world/train_vla.py --help
$QPY tools/structured_world/verify_training_recovery.py --help
$QPY tools/structured_world/build_navsim_current_inputs.py --help
$QPY tools/structured_world/export_current_inputs.py --help
$QPY tools/structured_world/infer_checkpoint.py --help
$QPY tools/structured_world/score_nuscenes.py --help
$QPY tools/structured_world/score_scene_fields.py --help
/root/miniconda3/envs/navsim/bin/python tools/structured_world/score_navsim.py --help
```

Formal training uses `train_vla.py --scope formal` only after its initialization,
cache, clean-source and frozen-configuration gates are satisfied. Restart uses
the **same frozen source and arguments** with `--resume`; changing source hashes
or common recipe is rejected. No automatic legacy control process is started.
Only this feature branch is to be pushed; no automatic merge.
