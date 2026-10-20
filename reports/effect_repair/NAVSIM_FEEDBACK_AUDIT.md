# Official feedback audit for action-effect research

Audit date: 2026-10-20 UTC. Source starting point:
`36493208ba4d8c25bedfcc43935a4ce4527131cc`, branch
`feature/action-effect-feedback-20261020`. Owned paths are the three new
`tools/action_effect` modules, their tests, and this audit. The canonical
scorer, simulator, metric cache and existing consequence backend are unchanged.

## Actual implementation to reuse

`starVLA/model/modules/st_world/consequence.py` already provides the required
official single-candidate backend. Its source hash is
`eb96a59fa1945fef47085928d55e91e399f74d2edb457ba0808607ff0bcb837c`.
It pins original source commit `1493deda247107efe076b9294863a6753391298a`
and actual file hashes, installed NumPy/SciPy/Shapely versions, nuPlan Python
source identity and all NAVSIM Python sources. `protocol_identity` rejects
another source or a mixed v1/v2 import. `ConsequenceSpec` binds permitted metric
cache roots and campaign. `ConsequenceEngine` is private to one CPU process;
`ConsequenceService` spawns 1–4 isolated workers with bounded in-flight requests.
Its inherited OMP/OpenBLAS/MKL thread limits must all be **1** before spawn.

Each request runs unchanged `navsim_v1.1/navsim/navsim/evaluate/pdm_score.py`
with exactly `[the cache's official PDM reference, one requested trajectory]`.
GT never replaces the reference. A separate invocation is required for every
candidate. Thus appending other candidates does not change this call's progress
normalizer. Directly scoring a growing candidate pool would change that
normalizer and is not an allowed optimization.

`ConsequenceEngine.score` returns the complete native record, not only PDMS:

| Content | Actual meaning and availability |
|---|---|
| Requested trajectory | FP64 `[8,3]`, initial rear-axle local x/y/metres/yaw/radians, timestamps 0.5–4.0 s |
| Executed states | FP64 `[41,11]`, global map rear-axle poses and native dynamic states, timestamps 0.0–4.0 s at 10 Hz |
| Body coordinates | FP64 `[41,5,2]`: FL, RL, RR, FR corners and centre |
| Terminal metrics | PDMS, NC, DAC, normalized EP, TTC, comfort and DDC, all `[0,1]` with explicit valid masks |
| Canonical arrays | Two-proposal multiplicative metrics, weighted metrics and raw progress metres |
| Process events | Exact native-predicate collision contacts, responsibility/type/contribution/order; road flags; TTC projection hits |
| First events | Official first at-fault state time and first TTC state time; read-only road first flag time |
| Context/reference | Full cache hash, semantic context hash before/after; full reference input and reference/candidate execution arrays and hashes |
| Failures/cost | Distinct invalid prediction/context/contract/transient failure, bounded retry eligibility, actual CPU/wall/call count |

`observe_canonical` replays native predicates without modifying the scorer and
asserts the replay NC/DAC/TTC and first NC/TTC times exactly equal native arrays
and APIs. A failure is not a valid zero. The wrapper preserves this distinction.

## What the v1.1 predicates actually mean

Collision and responsibility are different. `_calculate_no_at_fault_collision`
queries ego-body polygon intersections at each executed state. It ignores red
light geometries and already-collided tracks. `get_collision_type` distinguishes
stopped ego, stopped track, active rear, active front-bumper and lateral contacts.
Native stopped-ego classification uses 0.05 m/s. Front/stopped-track contacts are
at fault; lateral contacts are at fault when ego occupies multiple lanes or
non-drivable area. At-fault dynamic-agent contact contributes NC=0; at-fault
non-agent contact contributes 0.5. A non-at-fault contact is added to the ignore
list; an at-fault contact is not, so later or repeated at-fault contacts remain.
First at-fault time is the minimum executed state index times 0.1 seconds.
These are the benchmark's responsibility rules, not a general legal finding.

DAC checks **all four body corners**, not only rear axle or centre. A state is
off-road when at least one corner is not inside the union of the configured
drivable polygon layers. Any such state makes terminal DAC=0. This is the exact
corner predicate; it is not proof that the whole polygon interior is on-road.

TTC is a binary projected-body intersection test, not a continuous
distance/speed time-to-impact scalar. At every executed state it projects ego
with its current speed and heading to offsets **0, 0.3, 0.6, 0.9 seconds** and
queries corresponding logged occupancy. It filters stopped ego below 0.005 m/s
and applies native ahead/behind/intersection/off-road conditions. Its reported
first time is the **current ego state** that fails the projection test. A failure
at ego 4.0 s may inspect occupancy at 4.9 s. The backend requires at least 50
native 10 Hz occupancy entries and rejects temporal aliasing/truncation.

The v1 metric cache builds logged objects at 2 Hz and interpolates them to
10 Hz (`MetricCacheProcessor._interpolate_gt_observation`). Background traffic
is fixed logged replay. It is not a candidate-conditioned reactive rollout.
Interpolation and collision/TTC loops are part of the pinned benchmark contract.

Progress is the clipped nonnegative difference of **body-centre** projections
onto the official centreline at start/end. It is not local x displacement.
The denominator is `max(raw_progress * NC * DAC)` over the fixed official
reference/candidate pair. If this maximum exceeds 5 metres, the effective
progress is divided by it; otherwise EP is one except for zero-multiplier
proposals. Terminal PDMS is:

`NC * DAC * (5*EP + 5*TTC + 2*comfort + 0*DDC) / 12`.

DDC is computed from oncoming travel over native windows, but has zero weight
in this locked v1.1 configuration. Direction improvement cannot be counted as
a PDMS improvement unless another scored component changes.

Comfort is a terminal conjunction over six native tests: longitudinal/lateral
acceleration, magnitude/longitudinal jerk, yaw acceleration and yaw rate. Native
code smooths/rounds derivatives and reduces each test over the full horizon.
The native comfort boolean is not a per-time event label. The new wrapper does
not fabricate a timestamp for it.

## Submitted versus executed trajectories

`transform_trajectory` converts rear-axle-local poses to global poses, adds the
actual initial ego state, and interpolates the 8-point request. Submitted
velocity/acceleration are not directly used. `PDMSimulator.simulate_proposals`
uses a batch LQR tracker to produce acceleration/steering-rate commands and a
rear-axle kinematic bicycle model to execute them at 0.1 s. The model uses first
order acceleration/steering delays and clips steering. The tracker estimates
velocity/curvature from the pose path and switches to a stopping controller for
a stopping reference. Therefore a geometrically plausible request is not yet a
verified executable repair: G0 must check both controller execution and actual
official feedback. Submitted poses and executed states remain separate fields.

The semantic context hash includes occupancy geometries, object attributes,
road geometry, centreline, initial ego state/time and vehicle parameters. Every
call checks it is unchanged. Native scorer arrays are reset each call. The
reference **input** is immutable identity; executed reference arrays are outputs
and may differ under native numerical layout/dispatch. The existing
`st_world_reference_input_and_execution_v2` contract deliberately preserves
both exact arrays/hashes without confusing executed output with common input.
Effect pairs compare reference-input hashes, not executed-reference hashes.

## v2 is present, but is a different protocol

The repository also contains `navsim/`, whose setup declares version **2.0.0**.
Source availability does not establish a working v2 data/cache/metric run.
Its `MetricCache` has log/time/scene type, human and past human trajectories,
past/current/future tracked objects and map parameters absent from v1.1.
`pdm_score` calls `traffic_agents_policy.simulate_environment` using the
candidate's executed ego states; `score_proposals` updates the observation with
the response. A candidate's traffic response must not be reused for another.

v2 has separate NC/DAC/DDC/traffic-light multiplicative metrics, lane keeping,
history comfort and two-frame extended comfort. The human-penalty filter may
modify component penalties using a separately simulated human trajectory.
`run_pdm_score.compute_final_scores` requires a nonmissing two-frame extended
comfort column before final aggregation; the lower-level `pdm_score` alone is
not the complete EPDMS protocol. The v2 progress branch also differs from v1.

The new adapter explicitly rejects v2. No v1 cache conversion, v2 labels,
cross-frame EPDMS, reactive equivalence or v2 real integration is claimed.
Status: **SOURCE_AUDITED; REAL_V2_INTEGRATION_NOT_RUN**.

## New shared API and immutable data boundaries

`tools/action_effect/feedback_schema.py` defines
`action_effect_feedback_v1` and `action_effect_pair_v1`.
`navsim_feedback_adapter.py` exposes:

```python
from_canonical_record(record, request, *, observation_id,
                      observation_sha256, canonical_receipt) -> dict
label_arrays(feedback, *, verify_receipt=True) -> dict
summary(feedback) -> dict
make_effect_pair(base_feedback, modified_feedback, *, basis_id, coefficients,
                 active_interval_seconds, geometric_distance_m, executable,
                 intervention_metadata=None) -> dict
validate_pair(pair, *, verify_receipts=True) -> Mapping
```

`canonical_receipt` is an independently checksummed absolute receipt path.
The caller must load the source record through the existing ledger's verified
`COMPLETE -> result checksum -> request identity` chain. The thin wrapper
does not unpickle that potentially large file again. It pins the complete
original record content and request association; it does not pretend that an
arbitrary supplied record was independently looked up in that file. The
upstream call site must retain that ledger verification.

Typed ndarray JSON stores exact dtype/shape/bytes, including FP64 values; no
BF16/FP16 conversion or quantization is involved. Wrapper identity binds all
content. First-event absence has a false time-valid mask, not a fabricated
zero-second observation. Failed predictions retain their raw nonfinite
evidence and failure reason, but supply no valid labels and cannot form pairs.

`label_arrays` returns ordered terminal labels and raw progress metres, first
event times with individual masks, and 41-step native collision/road/TTC flags.
Speed, native acceleration and yaw rate are explicitly **derived from executed
states**. Continuous clearance, continuous TTC, per-time centreline progress,
and per-time comfort are **not yet available** in the original record. They
are explicitly absent/masked, not guessed. Adding these requires a separately
audited read-only geometry/derivative extension; this implementation does not
claim the full continuous-risk part of the research task complete.

`make_effect_pair` requires two valid, fully observed records from the same
scene/log/observation, runtime, protocol, semantic context and official
reference input. It preserves coefficients, basis identity, 0–4 s intervention
interval, metadata and an executable-check outcome. The geometric distance is
**mean pointwise XY L2 in metres** and is recomputed from submitted trajectories.
Signed deltas are modified-minus-base and only valid for common valid labels
of the same units. First-time deltas are masked if either event is absent;
event repair/introduction and temporal flag deltas are separate labels.
An event timestamp never becomes `credit[t]`. A controlled local intervention
effect is not declared a globally valid gradient.

These APIs are offline label/data tools. Route A/B deployment must receive only
legal observation/context features and model predictions, never these labels,
metric caches or the canonical record.

## Tests and actual execution scope

The full command and output are retained in `FEEDBACK_TEST_OUTPUT.txt` and
`FEEDBACK_TEST_RECEIPT.json`. New tests cover lossless storage, masks, event
timing, score reconstruction, same-context pairing, correct signed deltas,
event repair, zero intervention, physical distance, failure handling, stale
receipt/hash/runtime rejection, v2 rejection, no mutation and CLI failure exit.
Existing native-predicate/reference/receipt contract tests are also run.

An initial broader invocation with `OMP_NUM_THREADS=2` correctly failed the
existing worker-spawn guard (72 passed / 1 environment rejection). It was
rerun with all three required limits equal to 1; the guard was not weakened.
The initial 21 new contract tests passed in 3.53 s. The subsequent complete
regression result is recorded in the final receipt. All these cases use
explicitly marked synthetic geometry/records. They are not real images/logs,
real controller-execution evidence, or scientific repair results.

Actual new real simulation calls: **0**. Actual new GPU work: **0**. No models,
production source, active jobs or old weights were modified.

## Proposed first real pilot, for Lead dispatch only

Use the Lead's one-time bound S0 and immutable TRAIN-only cohort. Start with
**32 scenes**, independently fixed before intervention scoring. For each scene:
one original prediction, one exact repeated prediction, one zero intervention,
and six preregistered smooth local interventions. This is **288 primary calls**.
Allow at most eight separately admitted infrastructure retries across the
whole pilot (maximum296 calls), two CPU workers, all BLAS threads1, CUDA hidden,
an outer900-second bound plus cleanup, and ≤1 GiB output. Check actual context
sizes and per-scene cost before raising this budget. No training/DEV/Navtest.

The original backend call surface, inside the registered G0 runner, is:

```python
with ConsequenceService(spec, workers=2, max_calls=296) as service:
    # The existing durable ledger records admissions before calls and reuses
    # only checksum-verified completed results. "requests" are preregistered.
    records = ledger.score(requests, policy_version=policy_version,
                           service=service)
```

Verify all requested scenes remain in the denominator; exact repeated native
metrics/events, immutable common reference input/context, and identity zero
intervention; then compare batched service order/chunk/reordering. Keep exact
native execution arrays and report numerical differences without relaxing
tolerances after inspection. Measure actual child CPU/wall/read/output bytes.
These are a proposed finite dispatch budget, **not executed results**. The Lead
must supply actual bound input paths and the new registration, instead of
silently reopening an old diagnostic run or using guessed cache paths.

No-simulation artifact validation is executable now:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.validate_feedback \
  --input /absolute/path/to/the/actual/feedback-or-pair.json --max-bytes 16777216
```

The final path is an explicit caller input, not a discovered artifact claim.
The command validates provenance and returns nonzero on invalid/stale content;
it performs no simulation.
