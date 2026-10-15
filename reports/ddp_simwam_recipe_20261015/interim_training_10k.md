# Matched first10k optimizer updates: training evidence only

Both variants have completed this common window. Each contributes 2,500 fresh
global collections, 20,000 scene exposures and 160,000 new candidates. The
four inner optimizer updates reuse a collection and are not counted as new
environment samples. All eight rank journals are included. The locked hard
population and data ordering are common; the two learned policies subsequently
produce different candidates. This is not a fixed-policy before/after evaluation.

| First10k window | R_SOURCE | R_CANONICAL |
|---|---:|---:|
| Nonzero-PG groups | 19,925 /20,000 (99.625%) | 19,934 /20,000 (99.670%) |
| All-full canonical-score groups | 15 | 14 |
| All-zero canonical-score groups | 30 | 27 |
| Other canonical-score ties | 30 | 25 |
| Progress-only groups | 13,275 (66.375%) | 14,177 (70.885%) |
| Groups with an NC/DAC/TTC difference | 4,950 (24.750%) | 4,563 (22.815%) |
| Random-policy canonical PDMS /100 | 82.5605 | 82.4275 |
| Random-policy NC /100 | 99.7584 | 99.7809 |
| Random-policy DAC /100 | 94.9538 | 95.0750 |
| Random-policy TTC /100 | 97.0369 | 97.5594 |
| Random-policy EP /100 | 68.6630 | 67.4573 |
| Fully passing NC/DAC/TTC candidates | 92.2713% | 92.8806% |
| Positive-advantage unsafe candidates | 438 | 257 |
| Above, with a safe alternative in that group | 340 | 198 |

These are **G8 stochastic training-policy averages on the fixed difficult TRAIN
population**, not native single-ODE scores. No candidate max is substituted
for a policy score. Source training rewards and canonical metrics are separate
readouts from one simulation; their differently weighted means are not compared
as if they were the same metric.

The difficult-scene recipe has made zero-PG groups uncommon. That establishes
available update signal under the complete new recipe, not attribution to one
module or evidence that the signal improves deployment. Roughly two thirds of
source groups still differ only in progress. Compared over this common window,
source10:5:2 has 1.206 points higher EP but 0.523 points lower TTC and 0.609
percentage points fewer fully safe candidates; its canonical mean is only0.133
points higher. This is an observed progress/safety tradeoff in the collected
policies. It does not establish either variant as a better deployed planner.

Every observed first inner recomputation has ratio exactly1 and zero PPO clip
fraction, consistent with the saved behavior path. Later inner updates retain
the original old probabilities and change the ratio. The raw/clamped ratios,
per-inner clipping, global reward scale and costs are retained in
[R_SOURCE_training_snapshot_0533.json](R_SOURCE_training_snapshot_0533.json) and
[R_CANONICAL_training_snapshot_0533.json](R_CANONICAL_training_snapshot_0533.json).

At the sealed cost snapshots, source had17,600 updates in15,052 seconds and
canonical11,000 in14,978 seconds. Update/backward/optimizer phases dominate;
scoring wait is approximately4.0% /3.1% of rank0 elapsed time. Host contention
differs, and G2 later joined the source host, so these rates are not algorithmic
speedup estimates or guaranteed completion times. The different current
endpoints are not compared as matched efficacy results.

The failed milestone control route has been repaired and both update5000
exports dispatched for complete DEV1696 and fixed TRAIN512 evaluation. Each
worker uses the original raw-observation deployment and separately produces
E_NATIVE and E_SHIFT5 with seed2101. Full-population paired results, later
milestones and final locked Navtest remain required. The source transport
failure and recovery are documented in [controller_recovery.md](controller_recovery.md).

## Fixed update5000 saved-batch gradients

Both diagnostics reuse collection1250, the final complete fresh batch at the
registered5000 checkpoint, after its four inner updates. This adds no candidate
or scoring call. All eight paired scene identities and frozen conditions were
checked. This single real batch diagnoses update arithmetic; it is not a
population estimate or a checkpoint-selection criterion.

| Random step | Source PG norm | Source beta50-anchor norm | Cosine | Canonical PG norm | Canonical beta50-anchor norm | Cosine |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 0.02930 | 0.00205 | -0.161 | 0.02901 | 0.00239 | -0.099 |
| 8 | 0.03676 | 0.00524 | -0.126 | 0.03414 | 0.00665 | -0.275 |
| 9 | 0.07641 | 0.03246 | -0.402 | 0.06070 | 0.03702 | -0.281 |

The last random step contributes the largest per-step gradient in this batch.
The anchor opposes part of the PG direction, with anchor/PG norm ratios of
0.425/0.610 at step9. Neither a complete suppression of learning nor an
optimal anchor strength follows from this one batch. Post-four-update step9
PPO clip fractions are32.81% /45.31%; these are not the first-inner numerical
ratio error or the full-training average. Raw-Gaussian and source-clamped PPO
clip fractions happen to agree on this batch, without removing the source
tail-clamp operation. Per-channel mean/velocity shifts, logical anchor
contributions and all96 effective LoRA updates are in
[R_SOURCE_gradient_update005000.json](R_SOURCE_gradient_update005000.json) and
[R_CANONICAL_gradient_update005000.json](R_CANONICAL_gradient_update005000.json).
