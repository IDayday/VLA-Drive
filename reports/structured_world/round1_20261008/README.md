# Structured-world FGTR round one — implementation milestone

This branch contains executable code and real data/training/deployment checks.
**Formal NAVSIM G1/G3 are running on two distinct physical eight-A800 hosts;
G0/G2 are queued. nuScenes official assets and complete train/validation labels
are ready; its own five-epoch shared perception preparation is complete.
Formal nuScenes G1/G3 are now updating on vla-zt3/vlawm-zt, respectively, from
the prescribed generic/random driving initialization and own shared perception.
No planning improvement is claimed.** NAVSIM training/evaluation source remains
frozen at `0c54f1c`; nuScenes is frozen at `41da7c1`. Each run records
the full SHA and its `f052e2f` recovery ancestry. `FORMAL_TRAINING_STATUS_LAUNCH.json` is a timestamped launch snapshot,
not a live counter or a completed result. Historical engineering scores remain
explicitly separate from the formal population and endpoint.

The initial `e0e2592` implementation unintentionally ran the frozen Qwen vision
tower in FP32, outside the language autocast context. Its G1/G3 prefixes were
saved with complete native recovery at updates 390/391 and preserved as a
superseded implementation version. All four current formal groups use the
corrected common BF16 vision/language compute from the prescribed initialization;
they do not initialize from those prefixes or profiling checkpoints.
`STEP_TIME_DIAGNOSIS_20261009.json` records the S0/S3 full-run timing comparison,
real CPU data probe, live stack sample and eight-GPU full-chain profiles.
Correcting vision autocast reduced steady profile time from about 4.89 seconds
to 3.46/3.42 seconds for G1/G3. These eight-update profiles are cost measurements,
not completed planning experiments. IO-only execution improvements subsequently
reduced the 16-update profiles to 3.172/3.157 seconds. The original activation
recomputation, precision, losses, proposal steps, effective batch and RNG remain
unchanged. Actual G1/G3 formal training resumed from updates 728/492 with exact
native model/master/Adam/LR/RNG/data-position verification before the next
update; both have continued updating. G0/G2 are queued with the same execution
mode and their prescribed common initialization, not those parent weights.

`EXECUTION_SPEED_AND_DEPLOYMENT_20261009.json` contains the acceptance evidence,
timestamped recovery/progress snapshot and actual deployment costs. The real
64-scene prefetch check passed byte and RNG identity. Full-model q0/q_final,
RNG and buffers matched bitwise in completed controls. Strict gradient checks
remain **failed**, including native BF16 reference-versus-reference replay.
The FP32 control has a maximum 9.06e-6 discrepancy, within the mature-operator
tolerance registered before this work, but above the separate strict 1e-6
threshold. IO acceptance does not promise bitwise future CUDA updates. The
candidate that disables activation recomputation is prohibited in formal runs.

The following single-A800, batch-one costs use 32 real current NAVSIM development
inputs after four warmup scenes. All use one native ten-step proposal; the new
model adds one residual. Weight loading is excluded from per-scene latency.

| Executed model / precision | Model median | Current-file read + model median |
| --- | ---: | ---: |
| Historical S0 Qwen + original DDP, FP32 | 1.031 s | 1.177 s |
| Structured G3, FP32 | 1.307 s | 1.449 s |
| Structured G3, training AMP compute | 0.594 s | 0.743 s |

The historical row strictly reconstructs S0's 100000-update FP32 masters from
the pinned `1493ded` implementation. The new rows use update 492 only to measure
cost, not completed model quality. Historical FP32 Qwen encoding takes about
0.904 s and the original action sampler 0.127 s. New FP32 geometry, future-space
generation and FGTR together take about 0.126 s, but the new sampler also measures
0.271 s. Thus the observed total increase is 0.276 s (26.8%); it cannot all be
attributed to the three new branches. The additional sampler runtime is not yet
explained by an isolated control. These are measured implementations, not a
claim that the public DDP algorithm inherently requires a one-second VLM pass.
The AMP cost row is a separate precision experiment; planning-quality parity
with canonical FP32 has not been established. Canonical scoring remains FP32.

The two extra nuScenes hosts each have 23,230 complete training labels, current
C1 targets and 139,380 current JPEGs on local storage (72.69 GB of hashed assets).
Each passed full eight-GPU profiles, a 100-update fixed training subset with
exact native all-rank recovery at update20, random/prepared component gradient
checks and FP32 camera-only deployment plus locked native planning/scene scoring
on 32 engineering validation scenes. Those short checks are not formal results.
`NUSCENES_FULL8_QUALIFICATION_20261009.json` preserves costs and learning checks;
`NUSCENES_FORMAL_LAUNCH_20261009.json` records actual formal commands, source,
assets, LR groups and timestamped progress. Both groups use full official train700
eligibility, 24 epochs / 17,424 updates, batch32/repeat8, and fixed full4,969-val
observations at updates4,356/8,712/13,068/17,424. The shared geometry cost was
4.021 training GPU-hours for five epochs /116,150 exposures. The first64 train
perception check improved global road MAE, depth and occupancy, while boundary
road MAE worsened; that limitation is retained, not hidden as a planning gain.
The shared-host profile initially measured6.96s/step; its existing allocations
subsequently released themselves and the final20 small-fit steps measured4.48s.
No unrelated process was killed. Formal training began at about4.5s/step on
both hosts; future co-tenancy remains a source of runtime variation.

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
- nuScenes official metadata, maps, CAN bus and all ten raw trainval archives
  are downloaded, hashed and extracted under
  `/mnt/project/datasets/nuscenes-v1.0-trainval` (13 completed official assets).
  Complete labels contain 23230 training and 4969 validation samples with no
  build errors, occupying 38.85/8.29 GB. Earlier partial-archive engineering
  inputs remain separately marked and ineligible for formal preparation.
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
- Whole-nuScenes training metadata coverage was checked with the same mature
  VAD poses and shared full-body reader on all 23230 eligible samples. There
  were zero GT-body out-of-range scenes. Body extents were x=[-3.56,58.44] m,
  y=[-14.14,16.44] m within the common frozen grid; no map-size adjustment or
  validation/test selection was needed. See `NUSCENES_WHOLE_TRAIN_GRID_AUDIT.json`.
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
small-set learning with this recipe passed. Final full profiles took 4.94 s
per steady update and 31.90 GiB peak allocated memory per GPU. On the identical
64 training scenes over 100 updates, mean first-ten/last-ten FM losses fell
1.606/0.531 (G1) and 1.597/0.471 (G3). Raw refine losses fell 1.333/0.246 and
1.331/0.216. These are learning checks, not planning benefits; see
`COMMON_SMALL_LEARNING.json`. Actual complete training-forward substitution of
ego, DINO, road, depth and current/future occupancy targets passed on four
real scenes per group: losses changed, while q0/q_final remained bitwise equal.
See `TRAINING_TARGET_INVARIANCE.json`.

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

All 13 nuScenes official archives are now downloaded, hashed and extracted.
The earlier progress-loss diagnosis and monotonic If-Range retry records remain
preserved. Complete official label builds finished on both populations; older
partial-data engineering caches remain separate.

Explicitly authorized cleanup removed only verified `gpu_stress.py` process
trees on local/zt2 after they occupied 75 GiB per GPU and caused a calibration
startup OOM. Unrelated training and the old-objective pause locks remain intact.
Physical UUID comparison found that local and recovery containers share the
same eight GPUs; they are not counted as independent training servers.

Still pending: completion of nuScenes own shared perception preparation;
remaining full-model/restore checks; all six formal seed42 trainings; common endpoints,
development/final evaluation, paired failure and mechanism analysis, costs and
second-seed limitations. NAVSIM's four recipes are frozen after common
calibration, complete profiles and learning checks. nuScenes remains guarded
until full assets, its own perception preparation and population class counts
are complete. `run_navsim_pair.py` executes fixed development observations and
verified restoration on each complete eight-GPU host; it does not select
checkpoints with Navtest. Native CUDA continuation tolerance failures above
remain disclosed and are not relabeled as passes.

`prepare_nuscenes_labels.py` has completed the official 23230/4969 populations.
After the user's additional resource authorization, the owned waiting
perception controller was replaced with a pinned vla-zt3 route using source
`4aa8703`. Its 23230-scene label copy and public teacher/backbone assets are on
the host's local disk. Current-only C1 generation finished for all 23230 samples
in 440.71 seconds (0.979 GPU-hours), identity
`b0dc99cff8c2cb32905452087791d9c91540acfdcd64be4d8297b10ce6b1827e`.
The queue has automatically started five full epochs of nuScenes-only shared
geometry on all eight GPUs, with real updates and FP32 parameter changes,
using the same physical GPU locks as NAVSIM.
The isolated environment and eight-rank NCCL/deformable-operator readiness checks
passed. NAVSIM and unrelated jobs remain running. nuScenes final profile, recipe
registration and formal 24-epoch G1/G3 training remain required afterward;
perception preparation is not reported as those formal VLA runs.

`RESOURCE_ROUTING_20261009.json` records the additional host inspection and actual
dispatch. Recovery and local containers share the same physical GPUs, so they
are counted once. vlawm-zt has ample memory for sharing but its compute load
changed during continuous sampling; readiness checks alone are not evidence of
acceptable full-model training throughput. No unverified GPU workload was killed.

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
$QPY tools/structured_world/prepare_nuscenes_labels.py --help
$QPY tools/structured_world/prepare_nuscenes_perception.py --help
$QPY tools/structured_world/validate_training_target_invariance.py --help
$QPY tools/structured_world/validate_execution_equivalence.py --help
$QPY tools/structured_world/benchmark_deployment.py --help
$QPY tools/structured_world/benchmark_legacy_ddp.py --help
$QPY tools/structured_world/run_navsim_pair.py --help
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
the **same frozen source and arguments** with `--resume`. The registered IO-only
upgrade separately requires `--execution-mode io_preserving_v1`, the hashed
`--execution-acceptance` record and `--resume-origin-run`; it validates the whole
scientific contract and exact native recovery boundary. Arbitrary source or
recipe changes remain rejected. No automatic legacy control process is started.
Only this feature branch is to be pushed; no automatic merge.
