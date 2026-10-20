# Independent G0 primitive correctness review

Reviewer: `/root/effect_navsim_feedback`. This review covers the Lead's
intervention basis, geometric screens and controller-tracking transform. It
does **not** review my own feedback wrapper. Reviewed source is
`cdd33e240cfebbedcdbdae84cd824605e0858f08`, including the owner fixes in
`823adecb`; source content hashes are in the accompanying JSON. No owner
primitive source was edited by this reviewer.

Verdict: **PASS_FOR_REVIEWED_PRIMITIVES_ONLY**. This is not approval of the
as-yet-unreviewed G0 runner, its ledger, a real simulation pilot, formal expanded
G0 execution, or scientific repair effectiveness.

## Findings reproduced independently and corrected by owner

**G0-01: incorrect temporal support metadata.** Original owner commit
`00a5b754a6d772d5184697effedbe62be1762788` labelled an early quintic ramp's
active interval `[0,2]`, although its positional offset is held through 4 s.
On a curved base the held tangent-relative displacement also rotates as base
heading changes. Claiming this is an intervention restricted to the ramp
would corrupt temporal attribution. The independent historical test loads
the exact old basis source, verifies the old interval, reconstructs the current
candidate and verifies displacement remains at all sampled points 2.5–4 s.
The owner now stores `ramp_intervals_seconds=[0,2]` separately from
`active_intervals_seconds=[0,4]`. That preserves the intervention and corrects
its scientific metadata; no results or thresholds were changed.

**G0-02: low-speed curvature and turn-in-place acceptance.** Original geometric
screen computed curvature only where speed exceeded 1 m/s, replacing all other
curvatures with zero. A rear-axle circle of radius 0.1 m and yaw-rate 0.5 rad/s
passed every old screen with reported curvature 0, despite actual curvature
approximately 10 inverse metres. A stationary trajectory with yaw changing
0.05 rad per 0.5 s also passed. The independent test loads the exact historical
constraint source and reproduces both acceptances. Current source computes
curvature at translating steps above 1e-6 m/s and explicitly rejects yaw change
without translation, retaining defined/undefined step counts. Both independent
negative cases now reject. No bound was widened.

## Other checks performed

The independent test file contains 14 actual CPU cases:

- Historical ramp/hold-tail mismatch and corrected current metadata.
- Historical slow-circle and stationary-turn acceptance; current rejection.
- Global-map rear-axle execution transformed into ego(t0) for a curved path,
  four initial headings including both sides of the ±pi boundary, and exact
  0.5 s versus 0.1 s time-grid alignment. Maximum fixture XY error is below
  5e-13 m and heading error below1e-14 rad. These are numerical fixture checks,
  not registered real-simulation tolerances.
- Observed initial local velocity affects the first acceleration estimate;
  5 m/s matching straight motion yields zero acceleration, whereas 20 m/s
  produces 30 m/s² and is rejected.
- Exact-zero intervention preserves complete FP64 byte content, including
  signed zero and principal-angle yaw near ±pi, without aliasing original data.
- Longitudinal/lateral basis axes follow the base heading and remain orthogonal;
  a curved hold tail continues changing relative to global axes.
- Controller XY thresholds are applied in physical metres, with tests just
  below and above the declared 2 m threshold.
- Mismatched time grids and nonfinite execution raise exceptions; no valid
  zero reward is returned.
- Undefined original tangents retain base heading. The output is explicitly
  still `REQUIRES_REAL_CONTROLLER_REPLAY`, never a safety label.

Actual command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_interventions_independent_review.py
```

Actual result: **14 passed in 0.17 s**, tool wall 0.451526 s. Complete output is
`G0_INTERVENTION_INDEPENDENT_TESTS.txt`; source/output pins and execution scope
are in `G0_INTERVENTION_CORRECTNESS.json`. No old 76-test suite was rerun. No
GPU, real-data simulation, training, metric-cache scan or model load occurred.

## Integration requirements that remain in scope for G0

`initial_velocity_xy` must be an ego(t0)-local rear-axle-frame vector. A native
global-map vector cannot be passed without conversion. The implicit initial
heading 0 is correct **because the submitted trajectory frame is local**; it
would be wrong for global input. `tracking_screen` correctly removes the native
initial global translation and rotation and wraps only heading differences.

Geometric screens are declared finite-difference bounds, not exact continuous
acceleration/jerk certificates. The first jerk cannot be fully recovered without
an observed initial acceleration. Do not call the screen an official safety
metric. A small tracking error threshold is a declared engineering check, not
a proof that arbitrary pointwise geometry is dynamically achievable.

`reconstruct` documents undefined tangents but returns only a trajectory array;
it has no explicit per-time undefined-tangent diagnostic output. This is not a
blocker for the present real-controller-gated G0 because the runner must
retain the geometric screen and actual executed tracking separately. If future
learning uses a tangent-dependent target, it should expose this mask rather
than silently claiming the reconstructed yaw is fully identified.

Current primitive validation checks shape and finite physical values; canonical
feedback separately enforces principal yaw and `[8,3]` rear-axle poses. The G0
runner must not omit that canonical check or accept raw `[8,4]` sin/cos actions
as physical yaw. It must derive masks/status from real submitted and executed
trajectories, and never treat malformed candidates or simulation failure as
valid 0 PDMS. A physically expressible collision/road/TTC failure remains a
useful negative example and must not be filtered as corrupt data.

Actual intervention effect is a same-context comparison. First-event time is
an observed state/projection time, not the responsible control point. Correct
ramp/support metadata enables finite intervention analysis; it does not itself
prove causality or a globally valid gradient.

The G0 ledger/cache, true controller replay, official event consistency,
remaining-call accounting, pipeline resume, real throughput and resource
admission still require their affected-scope reviews and actual pilot. They
are not implicitly approved by this primitive report.
