# No-VQA E35 Wave-11 predeclaration

Created before any Wave-11 validation or Navtest result was available.

## Motivation

The locked domain-shift audit shows that the No-VQA Base scorer has similar
Best-of-64 ceiling on Navtrain and Navtest, but Navtest regret is 1.4623 times
the held-out Navtrain-validation regret. Navtest also has materially lower
candidate NOC/TTC prevalence. Wave-11 therefore tests a training-distribution
change rather than another unrestricted scorer-capacity increase.

## Locked sampling rule

Every physical training log keeps exactly equal total sampler mass. Within a
log, a scene receives deterministic hardness between 1 and `m`, where `m` is
2, 4, or 8. Hardness is the maximum of:

- Base-top-K PDMS headroom, clipped at 0.05;
- presence of both safe and unsafe NOC/DAC/TTC choices in Base top-K;
- an unsafe Base-selected candidate.

Only training labels determine sampling frequency. These labels are not model
inputs, are not serialized into the deployable artifact, and are unavailable
to Navtest inference. Candidate generation, current-image caches, optimizer,
epochs, seed, and held-out physical-log split stay fixed.

## Predeclared variants

Seven 8-epoch variants are run on GPUs 1-7 after the Wave-7 strict evaluation
releases them:

1. path-local Top-32, sampler maximum 2;
2. path-local Top-32, sampler maximum 4;
3. path-local Top-32, sampler maximum 8;
4. path-local Top-16, sampler maximum 4;
5. pooled spatial Top-32, sampler maximum 4;
6. path-local plus released-context Top-16, sampler maximum 4;
7. path-local Top-32 plus current-actor CV candidate-relative relabeling,
   sampler maximum 4.

All variants use the full 103,288-scene current-frame actor supervision table.
The actor target is training-only. The frozen No-VQA trajectory generator and
all 64 proposals are unchanged.

## Promotion and benchmark rule

An artifact is promoted only if its selected-PDMS delta has a positive
physical-log bootstrap 95% lower bound on the locked 61-log validation split.
Every promoted artifact, not merely the validation winner, must then complete
12,146-scene/136-log/64-candidate FP32 Navtest, proposal-cache lineage checks,
and four-scene real-agent/cache parity. Navtest is not used to select sampler
strength, architecture, epoch, scale, gate, or ensemble.
