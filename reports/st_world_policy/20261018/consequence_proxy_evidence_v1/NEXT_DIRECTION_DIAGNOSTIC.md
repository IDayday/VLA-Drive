# Next diagnostic: road command-space direction, not a new training branch

**First limitation: all 32 fixed original trajectories already have canonical DAC=1.**
This set cannot establish repair of an existing road violation, improved overall
PDMS, or a useful training reward. It can test the mathematical direction of one
proxy, transfer to a separately defined executed-road margin, and preservation of
existing safety. Do not replace these scenes after seeing their scores.

Status: **assets located; direction untested; independent review pending; no
dispatch authorized by this evidence card**. No new scoring, rollout generation,
GPU work, policy update, C5, learned consequence head, or deployment selection was
run. Existing C2/C3 results at 4096 updates do not validate this proxy.

## Fixed assets and what was actually verified

The read-only audit in `NEXT_DIRECTION_DIAGNOSTIC.py` completed on 2026-10-18.
Its exact paths, hashes, 32-scene selection and source files are recorded in
`NEXT_DIRECTION_DIAGNOSTIC.json`.

- C1 run: `/mnt/project/st-world-policy-20261016/formal/consequence/runs/C1_fulltrain_seed42_official_4096_v1`.
- Runtime: `/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017`, commit
  `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`.
- Selection: `rollouts/boundary_004095/rank_{0,1,2,3}.pt`, saved local scene order,
  candidate index **0**. This is 32 distinct TRAIN scenes from 31 logs. It is the
  **4095 old-policy snapshot**, hash
  `0b0fa1f63de40e5663666ea0b09177d7bc22eeeac927189c26823a8c07df5695`,
  not the final post-update 4096 weights.
- Four rollout payload checksums and receipts match. Schema `fpo_rollout_v1` has
  `initial_noise` and `raw_actions`, each `[8,4,8,4]` per rank, 10 Euler steps,
  `normalized_xy_sin_cos`; all raw tensors finite. No GT bridge was substituted.
- All 32 are present in the hashed `student_train_v1/index.json`, identity
  `1045890740e1a4ec8c11b802dede2b6a42d0c0b7131d7f581081e6da1f3b7b7d`.
  Source log joins match. No validation/test asset was accessed.
- Official request/record joins match by candidate ID, scene, policy, numerical
  runtime, protocol, trajectory and cache hashes. Saved physical command is
  float64 `[8,3]`; saved executed state is `[41,11]`. The physical command already
  records the original GPU decode result. Do not replace it with a CPU re-decode
  while claiming bitwise identity: future gradient code must first quantify that
  decoding boundary or use an explicitly recorded zero-offset correction at the
  original point. Any such correction needs review; no correction was applied.
- All 32 actual metric-cache files exist and their complete compressed SHA256
  matches the original requests. Total compressed bytes: **11,780,860**.
- One cache was fully deserialized for the schema probe: `MetricCache` with
  71 valid polygons, map types `[0,1,11,17]`, 228,601 decompressed bytes. Scorer
  Pacifica and cached actual vehicle agree for this probe: half-length 2.588m,
  half-width 1.1485m, rear-axle-to-center 1.461m, wheelbase 3.089m. The other 31
  schemas/vehicle dimensions must be checked before a later diagnostic executes;
  their historical official records and current compressed hashes are verified.
- The 176,439,631-byte enriched `TRAINING_MANIFEST.json` was **not** scanned or
  rehashed. Its registered hash is retained as provenance, not freshly verified.
  Independent TRAIN index membership plus the actual admitted rollout is the
  bounded membership evidence here.

The first audit attempt stopped at the 30th cache (`09dd0a5d0ec15e6d`) because its
decompressed object exceeded a 2MiB per-cache safety cap. No scoring was attempted.
The successful audit limited deserialization to the first cache and checked all
32 compressed hashes. The whole task's file-read accounting includes a
conservative 35MiB reserve for discovery and the stopped attempt: **62,762,236B
<64MiB**. This is an upper-bound accounting, not a measurement of physical disk IO.
No further large-file reads were performed. Python import IO is outside the asset
read ledger; tensor/map payload reads are included. Successful audit: 2.45s,
CUDA not initialized, official calls 0, train updates 0.

## Official execution and DAC contract

Primary-source checks used the [official v1.1 scorer](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py)
and [official v1.1 trajectory entry](https://github.com/autonomousvision/navsim/blob/v1.1/navsim/evaluate/pdm_score.py).
The executable authority is the **local pinned source and its SHA256**, not current
GitHub `main`; formatting and versions need not be byte-identical to a public tag.
The JSON records hashes for these exact local functions:

1. `transform_trajectory`: rear-axle-local x/y/yaw at .5..4s becomes global poses
   relative to `initial_ego_state.rear_axle`. Initial state is prepended. Command
   velocity/acceleration fields are not an alternative control input.
2. `get_trajectory_as_array`: official interpolation to t=0,.1,...,4.0 seconds,
   41 states including fixed t0, with the original angular interpolation and
   timestamp rounding. A custom torch interpolation needs value comparison with
   this function before its derivatives are used.
3. `PDMSimulator.simulate_proposals`: LQR tracks command states, then the bicycle
   propagates the actual state over 40 intervals. Actual cached vehicle parameters
   are used by the motion model. This is **not direct scoring of command poses**.
4. `state_array_to_coords_array`: translate rear axle to center and use the
   oriented vehicle's four corners plus center. The scorer's footprint parameters
   must be bound separately from the simulator's parameters; check agreement.
5. `_calculate_ego_area` uses ROADBLOCK, INTERSECTION, DRIVABLE_AREA and CARPARK_AREA.
   `_calculate_drivable_area_compliance` returns zero if any corner, at any of the
   41 **executed** states, lies outside every selected drivable polygon.

Canonical DAC therefore checks four footprint corners, not just a center point.
It is also not an exact test of every interior point of the vehicle rectangle
against a nonconvex road union. Do not silently strengthen or weaken this rule.
The native map uses strict polygon `contains`: [boundary points are excluded](https://shapely.readthedocs.io/en/stable/reference/shapely.contains.html).

There is **no canonical continuous road-margin metric or official road gradient**
in that contract. Both margin quantities below are explicitly our diagnostic
measurements. They cannot replace DAC, change metric weights, or be called PDMS.

## One hypothesis and one proposed proxy

Hypothesis: for these fixed generated commands, a small negative gradient step of
a command-space road cost reduces the **same custom geometric cost measured on
actual executed poses** more than a positive-gradient or equal-norm random step,
without introducing canonical safety violations. Failure or an uninformative
all-safe set is an acceptable outcome. This tests no policy-learning claim.

For each cached scene, form the union U of exactly the four drivable map types
above. Preserve holes, multipolygons and coordinates. Reject invalid map geometry;
do not repair it with `buffer(0)`, discard holes, or substitute a centerline.
Let d(p,U) be positive distance to the union boundary inside U and negative outside.
Internal overlapping-polygon seams are absent from this union distance: at exact
boundary coincidences its sign can differ from the canonical any-polygon strict
contains predicate. Report such cases explicitly.

Decode the original 32D raw command a, perform the official-equivalent angular
interpolation/coordinate transform, and construct all four oriented corners
c[t,k](a), t=0..40, k=1..4. Define just one cost, in meters:

```
J_command(a) = mean_(41 times,4 corners) [s * softplus((m-d(c[t,k](a),U))/s)]
m = 0.10m; s = 0.10m.
```

t0 remains the cached initial state and contributes no action gradient. All
corners and yaw/length/width/rear-axle offset remain present. Corner distance is a
surrogate reflecting the official corner test; it does not certify full continuous
body containment or behavior between the sampled times.

Minimal derivative implementation: extract fixed polygon-ring segments once;
evaluate signed point-to-nearest-segment distance using torch float64 arithmetic.
Nearest-segment choice and inside/outside sign are locally discrete. Keep the
selected segment coordinates fixed within the local derivative; differentiate
projection distance and the original raw-to-corner chain. Check finite differences
only where closest-segment, sign, angular unwrap and segment-projection regimes
remain stable. Record discontinuities and zero/near-zero gradients, not fabricated
smooth derivatives. Shapely is a geometry oracle for maps/signs, **not autograd**.

This cost excludes LQR/bicycle from its derivative. `J_executed` is the same custom
corner cost evaluated after actual simulation; it is a diagnostic measurement,
not its derivative. Neither is a canonical gradient. The existing DFQL paper/code
reference does not solve this transfer question.

## Proposed fixed perturbation and evaluation budget

This is a **reviewable proposal**, not a retrospectively registered successful
test. Independent review must lock the constants and implementation hash before
any new metric result is read.

- Work in the original 32D raw representation. No robot action clipping, GT target
  interpolation, action-space reparameterization or learned reward.
- Four variants: original; `a-epsilon*g/||g||2`; `a+epsilon*g/||g||2`; and
  `a+epsilon*u/||u||2`, with g=grad J_command(a), epsilon **0.005 raw L2**.
  `u` is one deterministic CPU normal vector, seed from SHA256(scene ID,
  selection hash, literal `road_direction_v1`). No repeated random-direction search.
- Finite raw values and initial sin/cos radius >=0.25 are numerical prerequisites.
  Require each variant to preserve finite poses, principal yaw, 8 points/.5s,
  fixed t0 and frame. Before scoring, assert maximum corner displacement <=0.15m
  and maximum angular change <=0.025rad versus original at all 41 command times.
  These are *geometric perturbation caps*, not driving-safety certificates.
- The raw epsilon is identical for negative/positive/random directions. No
  per-variant reward line search, adaptive epsilon, yaw normalization or clipping.
  A cap violation, singularity, gradient norm <=1e-12 or failed finite difference
  is a recorded diagnostic failure/undefined direction for that fixed scene.
  Do not replace it, manufacture a direction or hide it from the denominator.
- CPU numerical prerequisites: official interpolation parity (including yaw wrap),
  exact vehicle-coordinate parity, sign tests near holes/overlaps, finite
  differences at raw steps 1e-4 and 1e-5. Away from discrete switches require
  `||g-fd||2/max(||g||2,||fd||2,1e-12)<=1e-4` or absolute max <=1e-6.
  Tolerances are proposed now; failures cannot retroactively relax them.
- At most **32 scenes × 4 unique commands = 128 candidate records**. Existing
  valid originals may be reused only with the complete identity checked above.
  Therefore normally 96 new official calls; hard cap **128 backend invocations
  including retries**, never more than one accepted result per variant. Each
  invocation independently evaluates `[fixed official reference, candidate]`;
  max256 simulated proposal traces, no changing candidate-pool normalization.
- Proposed resources: CPU only, 2 workers, 1 numerical thread/worker, 8GiB total
  RSS ceiling, 30-minute wall deadline, 1GiB output ceiling, <=32MiB/s aggregate
  asset reads. No active-training GPU or model load. Re-check current authorization
  and resource policy at dispatch. These are caps, not measured cost estimates.
- Reuse the existing `ConsequenceEngine`/request and receipt machinery; each call
  gets reset state and the original cached reference/context. Add only an offline
  diagnostic driver and geometry function in an owned module after review.
  No shared trainer, reward, deployment policy or automatic C5 edit is required.

## Outputs and interpretation locked before execution

Preserve all 32 IDs, per-scene status and all four variants. Report differentiable
regime coverage and all exclusion reasons. Report `J_command` and `J_executed`
changes separately; minimum signed corner distance and outside-corner-time counts
are also explicitly custom geometry diagnostics. The primary paired direction
contrasts are negative minus positive and negative minus random in `J_executed`.
No choosing the best of those variants for deployment or for a headline score.

Alongside those contrasts report official DAC/NC/TTC, PDMS, progress, comfort,
direction compliance, zero scores and all changes relative to original. DDC has
zero aggregation weight in this protocol. Never silently treat margin improvement
or DDC improvement as PDMS gain. Report command/executed displacement, velocity,
acceleration, curvature and tracking mismatch; numerical geometric validity is not
equivalent to dynamic feasibility.

With original DAC=1 for32/32, there is **no road-repair denominator**. Any later
positive result can at most support a limited local-direction/safety-preservation
observation on this set. No change or saturated gradients implies
`INCONCLUSIVE_DIRECTION_SIGNAL`; an executed-cost reversal falsifies transfer for
those cases; canonical regressions are reported even if custom margin improves.
No statistical significance or deployment guarantee is implied by32 scenes/31logs.

## Prerequisites and exact current scope

1. Independent correctness review of this fixed proposal, including physical
   perturbation caps and original-decode boundary.
2. Implement/test the small geometry/derivative driver. It is **not yet implemented
   or approved**; this task was evidence and feasibility only.
3. Before any scoring, validate all32 deserialized geometry schemas and footprint
   parameters within a newly authorized bounded read budget. The earlier2MiB
   expansion cap failure is a memory-guard finding, not evidence of corrupt data.
4. Lead separately authorizes and records a <=128-call CPU diagnostic, if useful
   despite the all-DAC1 limitation. No new scene selection is implied here.

Read-only reproduction (new output path required):

```bash
cd /mnt/project/VLA-Drive-st-consequence-proxy-evidence-20261018
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python \
 reports/st_world_policy/20261018/consequence_proxy_evidence_v1/NEXT_DIRECTION_DIAGNOSTIC.py \
 --output /tmp/NEXT_DIRECTION_DIAGNOSTIC_independent_review.json
```

This command audits assets only. It does not implement or execute the proposed
direction experiment, and its success must not be presented as independent
approval or a formal scientific PASS.
