# Consequence focus after the feature-side pause

The user's 2026-10-18 instruction pauses feature-side development and execution.
The existing single-trajectory DiT/Flow consequence campaign continues. This
note does not authorize another training matrix, change rewards, or select
examples from DEV failures.

## Evidence already available

C0–C4 completed 4096 optimizer updates each. On the same 1696 DEV scenes,
C1 (total PDMS feedback) gains 0.04696 percentage points over BC; its paired-log
95% bootstrap interval includes zero. C3 also has a small, inconclusive gain.
C2 and C4 are slightly negative. These results do not establish a practical
held-out improvement.

The fraction of training scene presentations with a nonzero group-relative
advantage is 61.75% for C1, 0.6005% for C2, 0.0298% for C3 and 0.3998% for C4.
Thus sparsity is a plausible limitation of the component rewards; it cannot
by itself explain C1. See the unchanged complete-result table in
`../consequence4096_complete_results_v1/RESULTS.csv`.

## Two bounded questions before further training

1. **Does total-score feedback supply a material, useful gradient next to BC?**
   Reconstruct checkpoint 3584 and its actual following four-rank, 32-scene
   batch. Preserve the original BC repeat8/dropout/RNG and FPO MC50 definition.
   Measure separate gradient norms, cosine and weighted joint contributions.
   No optimizer update, new feedback, coefficient search or checkpoint choice
   follows from this one boundary. Separate BF16 backward with FP32 summation
   is an objective diagnostic, not an exact reproduction of ZeRO2/Adam.

2. **Does a command-space road-margin gradient have the claimed direction
   after the official simulator executes the command?** Use the already fixed
   32 TRAIN scenes and compare original, negative-gradient, positive-gradient
   and fixed random perturbations. Preserve the registered perturbation size,
   geometry tolerances, scene denominator and original metric context. Keep
   undefined gradients and failed scores in the report. This proxy does not
   differentiate through the official simulator or equal its DAC gradient.
   All original trajectories have DAC=1, so the repair denominator is zero;
   no road-repair claim is identifiable on this diagnostic population.

If gradients are weak or conflict, report that evidence before changing their
weights. If the road direction fails, diagnose the proxy/simulator mismatch
before using it as policy supervision. Passing either diagnostic does not
establish held-out driving improvement. No result-dependent scene replacement,
tolerance widening, new feature training, or automatic full training is part
of these checks.

## Execution identities

Both diagnostic sources are frozen at
`83679937fe9702be39f0184154fdf9ebdc4eb98a` in
`/mnt/project/VLA-Drive-st-consequence-diagnostics-runtime-v2-20261018`.

- Road registration SHA256:
  `d6ad347867b77a274de976d62e07a0ab9c142eb46336c7e7de9e0198185c4152`.
- Gradient registration SHA256:
  `715ccad8839c9486f5a0b40f934da23ff4036247c8c448516788e77c7e490aab`.

Independent review and actual execution receipts determine their status.
This plan is not an execution-completion or scientific-success receipt.
