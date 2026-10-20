# Interpretation of the first real pilot

This is the fixed pre-score hash sample of64 original S0 TRAIN observations,
61 logs, with45 fit/19 validation scenes. It measures physical data integrity,
controlled search opportunity and cost; it is not the final1696-scene DEV result.

1576 native calls completed in112.86s with4 bounded CPU workers, zero retries,
zero unfinished admissions,88 proposals rejected geometrically before scoring.
The immutable bank contains1384 controlled effect pairs, excluding policy
resamples;806 non-joint pairs are eligible for G1 fitting,405 pairs held out by
log, including71 joint combinations never fitted by G1. Two extra original S0
noise samples per scene are support diagnostics, not local-effect supervision.

Only1 original failure was observed; it has a verified local repair with
mean-pointwise XY displacement0.15625m. All19 held-out scenes are already
qualified keeps, so this pilot cannot identify held-out failure repair. Reporting
1/1 as universal repair success would be misleading. Original S0 has seen the
TRAIN population. The prelocked1024 extension remains necessary.

Harmful interventions are retained: small lateral changes produced16 high-to-zero
candidate outcomes, and longitudinal3, joint6, larger-support10. These are
candidate-event counts, not distinct-scene repair rates. There is no evidence
that arbitrary local changes are safe. Eligibility means meeting the declared
constraints; on an already good scene it is not a newly repaired failure.

The new G1 delta model has performed32 real GPU optimizer updates in a separate
engineering namespace, with finite value/ranking/event/temporal/uncertainty/repair
losses and gradients, preserving the1000-update scheduler axis. This is not a
scientific positive result, a fully trained effect model, or an evaluated refiner.
