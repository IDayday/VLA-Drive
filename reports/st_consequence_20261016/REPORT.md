# Canonical NAVSIM v1.1 consequence observer

Status: **ENGINEERING PASS — NEEDS_INDEPENDENT_REVIEW**.
This is a bounded CPU instrumentation qualification. No policy training,
GPU/model inference, dev/Navtest evaluation or scientific-gain claim occurred.

## Scope and provenance

Implementation branch: `agent/st-consequence-20261016`.
Base commit: `1493deda247107efe076b9294863a6753391298a`.
The nine audited official scoring, simulation, observation and conversion
files are unchanged, and match both base1493 and
`7ffd77d9d0d6acbe6f7d39852e57eccd02160ae9` byte for byte.

`starVLA/model/modules/st_world/consequence.py` contains the entire runtime
change. It calls the original `navsim.evaluate.pdm_score.pdm_score` with
exactly `[fixed cache PDM reference, one requested candidate]` on every call.
The cached reference is never replaced by GT, and other requested candidates
never enter that call's progress-normalization pool.

The isolated-worker/reference pattern follows the already existing
`starVLA/rl/safe_flow_improvement/v1_reward.py` at
`6edb05d576b40a9718421e86c06d560f184a11c6`, file SHA256
`33a8c9e4742a9b5158f9a7fc01a18bdca2c067f3fe410f8769147d8f65517fd1`.
The event loops follow pinned canonical `pdm_scorer.py`, SHA256
`02ab075017d4edb1c8bba6476ef1674b84d43dbca0f599df87cfe052b2dd5211`.
The scorer itself, its aggregation, original LQR and bicycle dynamics are
unchanged. No old geometric MIN labeler, trajectory refiner, or differentiable
surrogate was restored.

## Observable contract

The request carries campaign/scene/log/candidate/policy-version identity,
checkpoint and inference-runtime hashes, the trajectory hash, complete metric
cache checksum and protocol hash. A stale policy version, wrong scene path,
wrong source, wrong cache, or wrong reference is rejected. The protocol binds
all local NAVSIM Python sources, installed nuPlan source digest, NumPy/SciPy/
Shapely/Python versions and exact canonical scorer configuration. Production
policy requests require their checkpoint hash. Engineering controls require
explicit opt-in, a generator source hash and **no pretend checkpoint**.

Input: finite float64 rear-axle-local future absolute poses `[8,3]`, x/y in
meters and principal yaw in radians, at0.5 through4seconds. Invalid yaw is
rejected, not silently converted or wrapped. Official resampling produces
41 states includingt0 and the native11-dimensional state schema.

Output contains all seven official metrics and validity, raw simulated
states/body coordinates, canonical component arrays, raw centerline progress,
fixed-reference identity and context hash, plus these read-only observations:

| Field | Exact interpretation |
|---|---|
| D | Native four-body-corner road support at each10Hz state; a supported center alone is insufficient. This retains the native corner rule, not a new polygon-interior proof. |
| NC contacts | Every post-filter native contact in native loop order, object/type, collision type, responsibility, contribution0/0.5/1 and whether it lowers the cumulative minimum. Repeated at-fault contacts survive; non-at-fault contacts enter only a local ignore list. |
| NC summary | Minimum contribution over all events, per-bin minimum, first at-fault state time. A0.5 collision can precede a later0 collision; neither event is discarded. |
| TTC contacts | Native projected intersections at offsets0/3/6/9steps, base state index, projected observation index and violation predicate. This is not a distance/speed approximation. |
| TTC time | The ego-state time of the first violation. Last-state TTC reads observation49 at4.9seconds; truncated/reindexed observations are rejected. |
| Progress | Nonnegative raw centerline meters; reference/candidate effective progress afterNC×DAC; fixed-pair denominator; exact>5m versus small-distance aggregation branch. |

Eight physical-time bins use `[0,.5],(.5,1],...,(3.5,4]`. t0 and late events
are retained. Absence of an event is valid with `time=null`, never an infinite
regression target. The observer checks its NC/DAC/TTC aggregates and first
times against canonical arrays for both reference and candidate.

The context is the locked logged environment; this version does not claim
reactive traffic. It checks relevant environment state before and after each
call. A worker owns its private scorer, simulator and at most one loaded
metric cache. Official reset/LQR update run on every candidate. There is no
mutable scorer shared by concurrent threads.

Successful true zero reward is `VALID_SCORED`, `valid=true`, `score=0`.
Invalid prediction, missing/incompatible context, contract mismatch and
transient IO/timeout failure have `valid=false`, `metrics=null` and all metric
validity false. Unexpected programming/native errors propagate. No error is
converted into reward0 or silently removed from an update denominator.

## Current tests and real evidence

Final command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=. /root/miniconda3/envs/navsim/bin/python -m pytest \
  tests/st_world/test_consequence.py -v --durations=10
```

Result: **24 passed**,14 upstream matplotlib/pyparsing deprecation warnings,
2.78seconds. Full log:
`/mnt/project/st-world-policy-20261016/audits/consequence/implementation_v1/pytest_final_full.log`.
An initial worker test exposed an unavailable `threadpoolctl` dependency.
It was removed; the service now requires inherited BLAS=1 **before spawn and
NumPy import**, instead of trying to change it too late. No environment upgrade.

Covered behaviors include body-corner off-road and reentry, static0.5 then
dynamic0 with repeated contacts, ignored prior/non-at-fault/redlight contacts,
TTC state40/observation49, stopped/behind filters, native small-progress
semantics, pooled-EP negative control, stale policy/trajectory/source/cache
rejection, no-event/null times and spawned-worker failure denominators/budget.

Real smoke was registered before scoring at:
`/mnt/project/st-world-policy-20261016/audits/consequence/implementation_v1/real32/REGISTRATION.json`
SHA256 `c66c5469bba20bb338940b63846f9d2ec9c42f0c1e7d3b4453f0e6fbcbc949fa`.

- 32 scenes /32 logs, selected by fixed hash order from complete
 `diagnostic_random_v1`; every scene is in the verified original train index.
- OriginalGT and a continuous current-speed/current-heading control for each
 scene. Both are explicitly engineering controls, not model candidates or
 scientific performance results. No future input enters the constant-speed
 control. No candidate changes were made after seeing scores.
- Four CPU workers maximum, inherited BLAS1, no GPU, local
 `training-vla-zt-worker-0`, canonical Python3.9.25 environment.
- Exactly192 native calls: two controls × three executions ×32scenes.
 Phase0 plain, phase1 observed/reversed, phase2 observed/shuffled duplicates.
 Chunk sizes64/7/3; phase2 used three workers because only three jobs were
 concurrently admitted. No extra real calls were used for negative tests.
- **192/192 valid scores**, full seven metrics and41×11 executed states exactly
 equal for every control across all three executions. Body coordinates,
 raw-progress/component arrays, reference/context hashes and duplicate events
 also exactly equal. Immutable context check passed every time.
- 64 distinct controls contain20 valid zero scores,9 withNC events,16 withTTC
 events and13 withD events. Negative requests are separate invalid records.
- All worker processes exited after the bounded test. Original source hashes
 remain unchanged. No ongoing training or other task was signaled.

| Execution | Calls | Phase wall including worker startup (s) | Mean per-request worker wall (s) | p50 | p95 |
|---|---:|---:|---:|---:|---:|
| Plain |64|4.192|0.15025|0.14377|0.27762|
| Observer, reversed |64|4.674|0.15309|0.15197|0.26726|
| Observer, shuffled duplicate |64|6.289|0.15329|0.14752|0.26814|

Per-request time includes cache loading/checksums/context hashing and scoring;
phase wall uses different chunking. These are engineering measurements, not a
controlled scalability or pure observer-overhead benchmark. CPU user totals
were9.399/9.600/9.597seconds, and system totals0.179/0.163/0.174seconds.
Stored phase artifacts total1,379,591 compressed bytes. No GPU-time estimate.

Full immutable phase outputs retain states/events/metrics. Per-scene parity,
negative requests, source audit and final summary are beside registration.
Only compact evidence is committed; raw logs, images and metric caches are not.

## Public call entry points

```python
from starVLA.model.modules.st_world.consequence import (
    ConsequenceSpec, ConsequenceRequest, ConsequenceService, protocol_identity,
    trajectory_sha256,
)

protocol = protocol_identity(navsim_root)
spec = ConsequenceSpec(navsim_root, maps_root, (metric_cache_root,),
                       campaign_id, protocol['sha256'])
request = ConsequenceRequest(
    campaign_id=campaign_id, scene_id=scene_id, source_log=source_log,
    candidate_id=candidate_id, policy_version=current_policy_version,
    generator_checkpoint_sha256=policy_checkpoint_sha256,
    inference_runtime_hash=runtime_sha256,
    trajectory_sha256=trajectory_sha256(physical_poses),
    metric_cache_sha256=cache_sha256, score_protocol_hash=protocol['sha256'],
    metric_cache_path=cache_path, trajectory=physical_poses,
)
# Set OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 before starting.
# Put process startup in an if __name__ == '__main__' guarded caller.
with ConsequenceService(spec, workers=4, max_calls=registered_call_budget) as service:
    records = service.score(requests, expected_policy_version=current_policy_version)
```

`require_valid=True` is the default and raises `ConsequenceBatchError` with
all returned rows when an expected failure occurs. A distributed trainer must
collectively reject an invalid update; this CPU module does not create DDP
collectives or own optimizer state. Unexpected native/worker exceptions also
abort the call. The parent trainer must not catch them and substitute zeros.

`ConsequenceEngine.score` is available for one isolated sequential CPU worker;
`observe_canonical` audits an already completed fixed-pair scorer.

## Exact bounded smoke commands

Run from `/mnt/project/VLA-Drive-st-consequence-agent-20261016`:

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=.
/root/miniconda3/envs/navsim/bin/python -m tools.st_world_policy.consequence_smoke plan \
  --output /mnt/project/st-world-policy-20261016/audits/consequence/implementation_v1/real32 \
  --campaign-id st_consequence_cpu_smoke_20261016 \
  --cache-receipt /mnt/project/flow-safe-improvement-20261013/cache_builds/diagnostic_random_v1/receipt.json \
  --train-manifest /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1/index.json \
  --raw-logs /mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval \
  --maps-root /mnt/navsim/maps \
  --navsim-root /mnt/project/VLA-Drive-st-consequence-agent-20261016/navsim_v1.1/navsim \
  --max-scenes 32 --workers 4
/root/miniconda3/envs/navsim/bin/python -m tools.st_world_policy.consequence_smoke run \
  --output /mnt/project/st-world-policy-20261016/audits/consequence/implementation_v1/real32
/root/miniconda3/envs/navsim/bin/python -m tools.st_world_policy.consequence_smoke verify \
  --output /mnt/project/st-world-policy-20261016/audits/consequence/implementation_v1/real32
```

These completed artifacts are reused without new simulation calls. An admitted
but incomplete phase is deliberately not replayed automatically: its consumed
call budget must be audited first. The smoke CLI has no full/dev/Navtest mode.

## Remaining review boundary

The implementation and real parity result require independent correctness and
systems review before formal data/optimizer integration. The small fixture
counterexamples and32 real scenes do not prove all possible geometry cases.
No proxy gradient, FPO training, parameter update, DDP model test, late-event
credit rule, production scaling or scientific benefit was tested here.
Read-only replay currently repeats geometry queries and hashes relevant
environment inputs; later optimization must retain this reference comparator.
