# G0 simulation and shared training datasets

Implementation starts from Lead `4ff98740`, then inherits its exact collector,
strict config and source-closure changes through `42970881`. Shared collector,
config, official scorer and feedback schema were not edited in this change.
Primitive review `4df3b9d1` remains historical independent scope; the new runner
requires its own independent review and a Lead-dispatched real pilot.

Actual implementation evidence: `tests/action_effect/test_g0_pipeline.py` and
`G0_RUNNER_TEST_OUTPUT.txt`. Tests use explicitly synthetic canonical records;
no real simulation, GPU, image/model load or scientific opportunity claim is
made by these tests. The final source/test pins are in `G0_RUNNER_RECEIPT.json`.

## Call path and immutable identities

`simulate_interventions.run(cfg, max_samples=64, max_simulation_budget=2048,
workers=4, resume=False, max_seconds=3600)` uses the actual strict config,
once-bound campaign, immutable TRAIN cohort and completed context manifest.
It reuses the collector's strict shard/head/observation/runtime verification.
The first N contexts must remain the first N preregistered scenes, with their
original source-log fit/validation roles. It never reads DEV or Navtest.

Per scene it reconstructs the fixed 24-grid proposals, including identity, and
adds the two original-S0 alternate-noise samples already extracted by the
collector. Exact duplicate trajectories are counted but not rescored. Nonzero
geometrically inadmissible proposals have null metrics and explicit status;
they are not valid zero scores. The identity proposal is always submitted to
the official backend even if its geometric screen fails. All collisions,
road departures and TTC failures from valid native simulation remain legitimate
negative effect examples.

The original private `ConsequenceEngine` is reused within each spawned CPU
worker. It still scores one requested candidate against the fixed official
reference. Actual initial native ego velocity supplies the geometric screen;
actual executed global rear-axle states supply the tracking screen. Reconstruct,
official scoring, event extraction and physical units are unchanged.

Artifacts are under `g0/population_000064/` (or the explicitly requested N),
with immutable registration, resolved config, per-scene flock/SCENE/COMPLETE
receipts and distinct invocation records. Registration pins the source,
context/cohort manifest, Base, numeric runtime, protocol, budgets and rules.
The shared `g0/candidate_cache/<exact request identity>/` holds canonical
gzip-JSON/receipt/admission chains. A 64-scene result can therefore be reused
under a separately registered larger population without scoring it again,
provided the exact request, trajectory/context/runtime/protocol match.

The call ledger reserves a durable global slot under a short blocking flock
before writing ADMITTED and before simulation. A crash between counter reserve
and ADMITTED may overcount an unused slot; it cannot undercount a simulation.
That reservation gap is explicitly reported. Incomplete ADMITTED is never
re-executed as the same attempt. A distinct attempt requires the explicit
global retry allowance; at most one retry per candidate is implemented.
Invalid predictions/context/contract failures are not treated as transient.
Unexpected native errors retain ERROR evidence and require diagnosis.

Each canonical COMPLETE binds its ADMITTED checksum, the exact full original
record content hash and compressed file hash. Feedback wrappers independently
pin that receipt. `verified_feedback(row)` follows this complete chain and
checks wrapper content against the original canonical bytes. Existing marker
files alone never establish successful completion. Any unresolved candidate,
resource/time/call exhaustion, missing scene or failed current verification
returns PARTIAL/nonzero, preserving all completed candidate calls for resume.

## Frozen target rule

Candidate targets require NC/DAC/TTC all full, total score at least 0.8,
raw progress at least 0.9 times the original, no comfort regression, passing
geometry and passing real-controller tracking. Among eligible **local** grid
repairs, select the smallest mean pointwise XY L2 distance, then higher score,
then stable trajectory hash. These are research thresholds, not safety guarantees.

Original high-quality/safe scenes receive identity preservation. Larger-support
grid candidates and extra policy samples are recorded separately; they cannot
be called local repairs or silently become local supervised targets. No verified
repair within this budget does not mean mathematical irreparability. Event
timestamps are observations, not causal-credit labels.

## Stable dataset API for G1 / EGR / IEG

`build_effect_pairs.build(g0_population_root)` validates complete fixed population
receipts and generates two immutable JSON manifests:

`PAIR_DATASET.json`: one compact entry per successfully simulated controlled
nonidentity intervention (including harmful/ineffective cases). Policy resamples
are excluded from local-intervention training and remain in the full G0 bank.
Fields include scene/log/role, context shard/offset, observation hash, physical
base/modified trajectories, eight coefficients/basis/support, labels/masks/units,
signed differences, 41×3 temporal event differences/masks, event transitions,
executable/target eligibility and independently verifiable feedback pins. Memory
is referenced once by shard/offset, never replicated into every pair.

`REPAIR_DATASET.json`: one entry per requested scene. It includes
`repair_status`, `trainable`, `target_kind`, original/target physical trajectories,
coefficients, original/selected metrics and label arrays, feedback and canonical
receipt pins, context reference and modification distance. Missing target means
`trainable=false` and `target_kind=unavailable`, not a fabricated supervision.
Identity-preservation and verified-local-repair targets remain distinct.

Label names are exactly:

```text
score, no_at_fault_collisions, drivable_area_compliance, ego_progress,
time_to_collision_within_bound, comfort, driving_direction_compliance,
progress_raw_meters, first_at_fault_time_seconds,
first_ttc_state_time_seconds, first_road_time_seconds
```

The first seven have units 0–1, progress has metres and first times have seconds.
Absent first events are masked; zero placeholders are never valid zero-second
events. G1 losses must normalize physical units with the correct individual
masks. The training sampler must sample scenes uniformly, then a valid pair
within that scene; candidate-rich scenes must not be implicitly oversampled.
Validation roles never fit. These manifests are supervision, not deployment input.

`analyze_repair_opportunities.analyze(root)` records all requested denominators,
invalid/duplicate counts, local versus support opportunity, per-kind new NC/DAC/
TTC and high-to-zero failures, score tradeoffs, distance and support/first-event
observations. Native continuous clearance/margin remains explicitly unavailable.
The proposal-slot budget curve includes geometric rejections and is not an
actual simulator-call count. Actual calls/reservations/retries and CPU/wall are
reported separately, with historical cache reuse excluded from new-stage cost.
Invocation records also measure parent and waited-child CPU; child maximum RSS
is a Linux high-water mark, not summed process-tree memory or fabricated GPU cost.

## Real commands, executed only by the Lead after affected-scope review

From the integrated main research worktree, with actual completed 64-context
manifest and fixed config:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.simulate_interventions \
 --config configs/action_effect/main.yaml --max-samples 64 \
 --max-simulation-budget 2048 --workers 4 --max-seconds 3600

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.simulate_interventions \
 --config configs/action_effect/main.yaml --max-samples 64 \
 --max-simulation-budget 2048 --workers 4 --max-seconds 3600 --resume

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.build_effect_pairs \
 --config configs/action_effect/main.yaml --max-samples 64

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.analyze_repair_opportunities \
 --config configs/action_effect/main.yaml --max-samples 64
```

Expanded G0 uses a separate 1024 population registration and its original full
32768-call ceiling after measured pilot/evidence review; existing exact cached
64 requests remain read-only reusable. This implementation did not execute the
pilot, expanded simulation or any training. No new science result is filled in.
