# Fixed road command-direction diagnostic implementation

Status: **IMPLEMENTED_CPU_FIXTURES_VERIFIED; REAL_DIRECTION_NOT_RUN**.
Independent correctness/resource review is pending. No scientific gain, policy
training, formal score, GPU use, real rollout reload or metric-cache reload is
claimed. The active feature pause is respected.

The design is the already written `consequence_proxy_evidence_v1/NEXT_DIRECTION_DIAGNOSTIC.md`,
not a new road reward or model. Its fixed 32 TRAIN scenes (31 logs), original C1
policy boundary4095, candidate0, and original ten-Euler generation are retained.
Every original already has DAC=1. There is no road-violation repair denominator.

## Implementation

- `road_margin_proxy.py` calls the original `decode_ego`, uses CPU float64 for
  the diagnostic derivative, reconstructs original rear-axle global transforms,
  official timestamp truncation/interpolation and periodic angular unwrap.
- `AnchoredDecode` explicitly uses stored physical origin plus the delta of the
  unchanged CPU decode formula. It preserves the exact saved original request,
  records raw origin, unanchored CPU decode, the complete constant correction,
  and periodic yaw correction. Its derivative does not include that constant.
  This adaptation is **pending independent approval**; it does not establish
  CPU/GPU bitwise equivalence. Real origin corrections have not been measured in
  this implementation and must be checked against pinned request/raw provenance.
- Four corners use actual vehicle half-length/width/rear-axle offset and original
  corner order. `RoadUnion.from_map` selects exactly ROADBLOCK, INTERSECTION,
  DRIVABLE_AREA, CARPARK_AREA. No repair, simplification, hole filling, centerline
  substitute or omitted disconnected region is allowed.
- Nearest-segment search has bounded1024-segment chunks, detached index choice,
  and differentiable projection/norm on the chosen fixed segment. Strict union
  containment provides the discrete sign. Nearest ties, projection regimes,
  boundary points, angular unwrap and canonical-any-polygon versus union seam
  disagreements are explicitly exposed. There is no smoothness claim at ties.
- The only cost is mean over all41×4 corners, `0.1*softplus((0.1-d)/0.1)`.
  Fixed t0 contributes no action gradient but is retained in the164 denominator.
  No safety-based mask or per-scene adaptive scale changes the objective.
- Gradient check uses predeclared raw finite-difference steps1e-4/1e-5 and the
  predeclared1e-4 relative/1e-6 absolute limits. Unstable regimes, zero gradients,
  singular sin/cos, invalid inputs or perturbation-cap failures return undefined;
  the caller must retain the scene and reason, not replace or resample it.
- Negative/positive/fixed random directions use the same0.005 raw L2 step.
  CPU RNG is local and deterministically bound to scene/selection/literal.
  Max41-time corner shift0.15m and angular shift0.025rad are geometric caps,
  not driving-safety assertions. No line search, clipping or GT targeting occurs.
- `score_diagnostic_variants` is a thin callable adapter to the existing
  `ConsequenceEngine`. It requires admitted engineering-control mode, explicit
  pre-call charge and persistence callbacks, validates original reference
  provenance and its exact physical hash, reuses the valid original, invokes the
  canonical backend separately for the three perturbations, verifies invariant
  reference/context, and preserves transient failures without retries. It cannot
  construct an admission, dispatch itself or alter the scorer/reward.

## Tests and measured scope

`TEST_FINAL.txt`: **30 passed**,7.756s wrapper wall,826116KiB peakRSS,<2GiB;
CUDA not initialized. Thread counts were1; outer timeout600s. The initial wrapper
attempt failed because `/usr/bin/time` is absent, then a Python resource/time
wrapper was used; both logs are preserved. The earlier28-test run and final30-test
run are separately recorded and not counted as distinct scientific replications.

Tests compare against actual installed official functions with geometry fixtures:
9 timestamp/origin/yaw-wrap combinations,1e-10 absolute pose/corner tolerance;
rear-axle offsets and corner ordering; sin/cos raw-chain derivative; exact stored
anchor; holes/disconnected maps/union seams;164 denominator; both finite-difference
scales; fixed equal-norm directions/RNG isolation; saturated safe zero-gradient;
ties/boundaries; NaN/shape/dtype/singular input refusal; invalid map refusal;
constants refusal; identical executed custom geometry; metadata-only preparation
and changed-selection rejection. The fake backend test checks charge-before-call
and no-retry behavior only, **not official score equivalence**.

CLI `--help` and `--mode prepare` executed. `PREPARATION.json` binds code hashes,
exact32-scene selection and constants, returns `PREPARED_NOT_EXECUTED`, and has
`dispatch_authorized=false`. It does not read scene/cache/model assets. It refuses
existing outputs and offers no execute, review-bypass or alternative-scene flag.
The parent admitted runner can later call the module after review; this change
intentionally does not add another experimental controller.

## Remaining gates and risks

1. Independent review of exact implementation and anchored decode adaptation.
2. Lead fresh resource/read admission: evidence audit already consumed62.76MiB of
   its64MiB budget; this implementation did **not** reuse or exceed that budget.
3. Before real scoring: revalidate pinned raw/request/cache identities; deserialize
   all32 cache schemas under a new bounded expansion/RSS allowance, verify cached
   actual vehicle and scorer footprint equality, check official interpolation and
   footprint parity for each real scene, and record CPU/GPU origin differences.
4. A separately admitted driver must durably cap backend invocations at128 incl
   retries, retain all32 statuses, persist full canonical failures/results, and
   enforce its CPU/wall/read/output bounds. This module is not a scheduler.
5. Conservative ties/regime checks can leave scenes undefined; do not widen
   tolerances or replace scenes. All-safe/saturated scenes can be uninformative.
6. Command-space gradient excludes LQR/bicycle; only a later paired real-execution
   measurement can test transfer. Custom executed margin is not DAC or PDMS.
   Canonical safety regressions must be reported even if custom cost improves.
7. No update to original policy, original decoder, canonical scorer, FPO loss,
   reward, training source, active job, dataset or feature-side work occurred.
