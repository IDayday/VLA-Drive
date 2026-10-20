# Action-effect physical intervention campaign

User-authorized 2026-10-20; source start36493208ba4d8c25bedfcc43935a4ce4527131cc.
Reference decodable study e0a6561e0708ec4f53a974e2b3ffd4c10406dfc9 is read-only;
no new teacher/representation training is prerequisite. Existing S0 architecture,
DINO/W, normalizer, action dimensions, source sampler and inference are audited,
not replaced. All source and scientific identities are independent of old ST runs.

## Questions

G0: Does a controlled, physically executable local intervention repair actual S0
failures while preserving high-quality behavior? Minimum repair means minimum
within the explicitly enumerated budget, not a global optimum. Event time is an
observation; causal credit requires matched physical interventions.

G1: At matched capacity/scenes/labels/updates, does learned conditional delta
feedback rank interventions better than terminal PDMS, absolute full-feedback or
trajectory-only models? Context shuffle and privileged upper bounds are separate.

A: Can a small no-simulator residual refiner use frozen predicted effects to
improve held-out actual trajectories over a plain same-capacity refiner?

B: Can original Euler10 generation internalize these effects without an additional
completed-trajectory refinement? Compare ordinary repair FM (B0), shared-path
direction learning (B1-trajectory), and B1-effect plus preserve on identical data.
A checkpoint never initializes B. True validated A trajectories can be labeled
B supervision with an explicit lineage, not an independent-data claim.

## Invariants and execution sequence

1. Bind mature original S0 once, strictly load FP32 masters, fixed real TRAIN8
   replay with existing scene seed42, compare historical raw/physical hashes.
2. Reuse canonical unmodified v1 reference-pair simulator/consequence observer.
   Submitted8x3 and executed41x11 are separate; official and derived labels carry
   masks/source. Unverified v2 adapters reject rather than silently map v1 labels.
3. Fixed-hash TRAIN cost pilot64, before reading scores. Measure real encoder,
   storage, simulation costs; then lock expanded TRAIN cohort and per-scene budget.
   Include all quality strata; no DEV/navtest failure case influences data/radii.
4. Shared schema action_effect_feedback_v1 / action_effect_pair_v1, identical
   scene/context/reference INPUT/numeric identities for every before/after pair.
   Retain improvements, ties, regressions and repair-with-new-failure pairs.
5. Freeze immutable data content and log-group fit/validation assignments before
   fitting. Existing fixed DEV1696/16logs remains final reporting population.
6. G1/A/B each get fixed formal budgets and independent correctness/efficiency
   review after a bounded real smoke. No blind100k or full hyperparameter search.
7. A/B implement independently; only verified data/G1 enables B1-effect training.
8. Actual single-/two-GPU, accumulation, finite/missing labels, native resume,
   deployment information boundary and Euler10 endpoint checks. Never claim
   unmeasured bitwise equivalence or mock integration as PASS.
9. Fixed common-endpoint DEV comparisons; meaningful finalists get three training
   seeds only after data/engineering/mechanism evidence warrants expansion.
10. Preserve all failures and costs. Push source/reports on the independent branch;
    no ACTIVE update, new Navtest, unrelated termination or asset cleanup.

## Initial resource budget

The first admission is audit/data only: TRAIN8 replay0 optimizer/0 simulation;
TRAIN64 cost pilot with a proposed bounded intervention grid (final count/radii
locked before its scoring). Initial own artifact cap16GiB and shared-free floor
340GiB, measured before launch. Current shared disk is hundreds ofGiB free;
there is no new multi-TB visual cache. Context W/A per scene is about1.19MiB in
FP32, measured actual dimensions prevail. Subsequent train/simulation budgets
are explicitly registered after real cost/coverage checks. Pressure monitor and
original worker identity are captured before exact owned worker release; preserve
real GPU owners and let retained monitor requalify released GPUs when work exits.

## Reports / states

reports/effect_repair: BASELINE_AUDIT, RELATED_WORK_AND_REUSE,
NAVSIM_FEEDBACK_AUDIT. reports/action_effect: G0_REPAIR_OPPORTUNITY,
G1_EFFECT_MODEL, G2_EXPLICIT_REFINEMENT, G3_INTERNALIZED_FLOW,
FINAL_COMPARISON and machine-readable source/data/config/checkpoint receipts.
States: NOT_RUN, RUNNING, PASS, FAIL, BLOCKED, INCONCLUSIVE. Engineering and
scientific statuses are separate. No result-derived threshold changes.
