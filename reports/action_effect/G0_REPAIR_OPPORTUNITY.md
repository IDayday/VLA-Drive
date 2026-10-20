# G0: actual local physical-repair opportunity

Status: **complete real1024-scene TRAIN intervention analysis**. This establishes
some verified physical repair opportunity; it does not establish learned policy
generalization. Full provenance/thresholds/budget/timing and seven compact endpoint
audits are in [G0_PHYSICAL_ANALYSIS_FULL1024.md](../effect_repair/G0_PHYSICAL_ANALYSIS_FULL1024.md)
and its [JSON](../effect_repair/G0_PHYSICAL_ANALYSIS_FULL1024.json). The full46,360,978B
private analysis is pinned there; raw feedback, images and checkpoints are not in Git.

| Population | Requested scenes/logs | Fixed original failures | Selected verified local repairs | Pair scenes |
|---|---:|---:|---:|---:|
| fit |791/405|6|6|758|
| held-out TRAIN validation |233/116|1|1|225|
| all |1024/521|7|7|983|

Failure was fixed before learning: PDMS<.8 or NC/DAC/TTC<1, tolerance1e-8. The
pre-score hash-selected TRAIN cohort has4 zero,3 positive<.8,457 in[.8,.95),560≥.95.
Base previously saw its source TRAIN population; the holdout is only new effect/
refiner training. Independent DEV remains1696 scenes/16 disjoint source logs.

Four original zeros are DAC failures (3fit/1validation). Small verified lateral
changes restore DAC; their raw-progress change is only−.0106..+.0360m, so the
large normalized score recovery is not a claim of greatly increased distance.
One fit TTC failure improves .519633→.904818 while raw progress5.478669→4.990015m,
within the predeclared90% progress condition. Two fit cases improve EP only:
.799206→.815062 and.798264→.816219; their hard safety was already full. No original
NC failure exists, so there is **no collision-repair-positive support** here.

The selected minimum-distance targets change mean pointwise XY by.121875m on
average (range.046875–.15625m). With1/5/9/17 registered proposal slots,0/2/4/7
of the seven failures are repaired; slots include identity/geometric rejection
and are not simulator-call counts. This is finite-grid minimum feasible repair,
not proof of a globally minimal edit or infeasibility outside the queried support.

Retaining Base on all other1017 scenes yields the **offline minimum-target oracle**:

| Population | Base PDMS/100 | Verified-target PDMS/100 | Gain, points |
|---|---:|---:|---:|
| all1024 |94.341247|94.759831|+.418584|
| fit791 |94.503553|94.919014|+.415461|
| validation233 |93.790243|94.219428|+.429185|

These are selected supervised-target opportunities, not a trained model result
or the highest-score oracle over the entire grid. Base zeros4→target zeros0 is
not automatically a learned repair success.

The20,678 valid effect pairs retain harmful cases. Among original-high light
lateral perturbations,331 pairs across172 unique scenes become zero; joint changes
produce109 such pairs/69 scenes. These are valid low-quality actions, not failed
scores.2,889 geometric proposal rejections are separately counted and never
converted into valid zero labels. Actual negative-event diversity is materially
richer than the6 unique positive fit repairs; G1 can study harm prediction without
claiming broad learned repair support.

Native event repair starts after basis support onsets1.2/1.5s earlier for DAC
cases and2s earlier for TTC. This is measured temporal association for tested
interventions, not proof that all that lead time is necessary or that an event
index identifies action responsibility. Ramp support and held tail are recorded;
controller-executed influence is a separate object.3,491 matched same-scene
longitudinal/lateral/joint contrasts retain nonlinear interactions; finite
differences are not a global simulator gradient. Continuous road clearance and
full causal credit identification remain unavailable/unproven.

The1024 registration references11,757 newly charged completed native admissions
plus11,978 lawful earlier admissions,0 retries. The analysis itself creates0
native calls/forwards/updates. Strict inventory verification incurred716.8s wall,
mainly Lustre metadata waits, not simulation time; public records preserve that
cost rather than recasting it as model inference. The complete pair/repair source
inventory covers96,992 files, with compact bytes pinned by the evidence index.

Conclusion: there is real but **small positive repair support**, including DAC/TTC,
and substantial harmful-intervention evidence. That justifies the finite G1 and
A/B tests, but cannot support a universal repair/generalization claim. All learned
conclusions require the frozen policies' actual held-out outputs and risks.

## Actual subsequent learning outcome

The registered learning/evaluation stages are now complete. A0/A1 fit the six fixed training repairs strongly, but all six main seed42 learned methods leave the sole held-out repair scene zero. This does not negate the target's genuinely simulated repair opportunity; it distinguishes available supervision from a learned generalizable function. The final full1696-scene/16-log development comparison, not the finite target oracle, governs policy-quality conclusions. See [FINAL_COMPARISON.md](FINAL_COMPARISON.md) and [the predeclared direct-subset audit](PREDECLARED_FIT_VALIDATION_SUBSET_RESULTS.json).
