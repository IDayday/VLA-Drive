# No-VQA E35 Wave-12 five-fold predeclaration

Created before any Wave-12 fold training result or Navtest result existed.

## Question

Can the scorer improvement survive five disjoint physical-log domains rather
than only the official 101-log/61-log split that repeatedly over-predicted
Navtest gains?

## Locked data split

All 103,288 valid Navtrain scenes from all 162 physical logs are assigned to
five validation folds. Each log appears in validation exactly once and never
appears in both train and validation within a fold. Assignment seed is
`20260902` and balances four Navtrain-only quantities:

- scene count;
- scenes containing an unsafe NOC/DAC/TTC candidate;
- unsafe candidate count;
- candidate score-span sum.

Validation fold sizes are 20,082–20,842 scenes and 32–33 physical logs. No
Navtest input participates in fold construction.

## Locked scorer and training rule

Every fold uses exactly the same model and optimization:

- frozen No-VQA E35 trajectory generator and its immutable 64 candidates;
- scorer-private four-camera spatial tokens;
- proposal-to-observation point attention;
- released M0 candidate feature as an internal M0 residual stream;
- full-coverage current-frame actor auxiliary supervision;
- Top-32 conservative Base-relative objective;
- physical-log equal sampling with within-log risk multiplier 4;
- seed 2, 8 epochs, final epoch 7 retained.

The final epoch is fixed in advance. Per-fold best epochs are diagnostic only
and cannot select the all-log refit stop time.

## Robust gate and refit

The configuration may be refit on all 162 logs only if, at fixed epoch 7, all
five folds have:

1. positive selected-PDMS delta versus their own frozen Base selections; and
2. positive physical-log bootstrap 95% lower bound.

If the gate passes, architecture, objective, seed, scheduler horizon, stop
epoch and deployment policy are frozen; a fresh deterministic model is then
trained on all 103,288 scenes. Only that provenance-locked refit is eligible
for complete 12,146-scene FP32 Navtest and online/cache parity. Navtest cannot
select a fold, epoch, sampler strength, residual scale, gate or ensemble.

## Prospective deployment-policy amendment

Added at `2026-09-02T11:25:39Z`, while fold training was running but before
any locked epoch-7 result or deployment-threshold sweep existed. At amendment
time only the default-policy epoch-1 metrics had been observed; those metrics
were not used to choose the grid or ordering below. This timing is recorded
explicitly rather than retroactively describing the amendment as part of the
original predeclaration.

The phrase “deployment policy is frozen” is made stricter as follows: every
fold first retains its independently trained epoch-7 weights. The same 192
deployment policies are then evaluated on all five held-out folds:

- gain quantile: q10 or q50 (q90 is excluded as non-conservative);
- minimum predicted gain: `-0.01, 0, 0.0025, 0.005, 0.01, 0.02`;
- maximum safety-worse probability: `0.02, 0.05, 0.10, 0.20`;
- minimum safe-improvement probability: `0.50, 0.70, 0.80, 0.90`.

A common policy is eligible only when all five point deltas and all five
physical-log bootstrap 95% lower bounds are positive, and each fold's NOC,
DAC and TTC delta is at least `-0.0005`. Among eligible policies the immutable
priority is: maximize worst-fold delta, then combined 162-log bootstrap lower
bound, then scene-weighted delta, then minimize switch rate. Per-fold policy
selection is forbidden. If no common policy passes, no all-log refit or
Navtest evaluation is allowed. The fixed original q50/0/0.10/0.70 policy is
retained as a separate diagnostic.
