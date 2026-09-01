# Independent scorer experiment status

Updated: 2026-09-01 21:02 UTC.

## Resource contract

- New scorer jobs use only `rl-zt3` and `rl-zt4`.
- No new GPU allocation was made on `vla-zt` or `vla-zt2`.
- Existing unrelated processes were not stopped.

## Complete Navtest results

Every row uses all 12,146 Navtest scenes, 136 segment logs, the same 64 frozen
M0 proposals, FP32 inference, and zero invalid scenes. Official candidate
scores are joined only after selection.

| Selector | Navtest PDMS | Delta from public M0 |
|---|---:|---:|
| Public M0 scorer | 0.909594 | 0.000000 |
| Factor-heavy, epoch 6 | 0.895126 | -0.014468 |
| Factor-heavy, final saved epoch 9 | 0.895823 | -0.013771 |
| Three-default-seed equal-score ensemble | 0.897006 | -0.012588 |
| Factor-only all-64, epoch 3 | **0.897539** | **-0.012055** |
| Factor-only all-64, validation-best epoch 9 | **0.900880** | **-0.008714** |

The current independent method has not improved public M0 on Navtest. Its
best change is a real improvement over earlier independent checkpoints, not a
new test-set best.

## Running on rl-zt4

| Run | GPU | State | Purpose |
|---|---:|---|---|
| low-res factor-only all-64 | 2 | epoch 4 complete; validation 0.923593 | isolate conflicting-loss effect |
| high-res 960-token factor-only (`v2_clean`) | 7 | training | test whether spatial visual detail is limiting |
| low-res factor-only final+epoch-3 replay | 5 | training | test fixed proposal-distribution overfitting |
| M0-native four-view cache, shards 0--3 | 0/1/3/4 | exporting 103,288 trainval scenes | replace DINO with released M0 vision features |
| Q-Former + current-actor auxiliary target | 6 | loading/training | test whether explicitly supervised dynamic queries improve ranking |

The high-resolution run is
`m0_independent_dino_highres960_factoronly_keep48_seed2_v2_clean`. Its manifest
has been checked against the live process: batch 20, evaluation batch 40,
48 sampled training candidates, top-16 fine configuration, and 10 epochs.

## Run-integrity incident

Two previously scheduled wait wrappers were initially mistaken for exited
processes because they later replaced their shell command with Python. A second
pair briefly launched against the same output names. The duplicate PIDs were
stopped before any epoch completed. The original multi-replay process and its
manifest agree and remain valid. The first high-resolution directory had a
manifest/process mismatch, so that process was stopped and the directory is
excluded from all results; it was not deleted. High resolution was restarted
in the separate `v2_clean` directory with a verified manifest.

## Current diagnosis

- The scorer uses current vision: cross-log scene shuffling lowers held-out
  PDMS by 0.029946 and zeroing scene tokens lowers it by 0.033757.
- Factor-only training transfers positively relative to the earlier
  independent scorer, but still loses mainly ego progress and DAC on Navtest.
- Equal-score ensembling improves safety factors but further lowers progress,
  so ordinary variance reduction does not solve the calibration problem.
- High-resolution perception and proposal-distribution replay are the two
  currently running causal checks.

## M0-native scorer-private representation

An 8-scene smoke export passed on `rl-zt4`. It uses the released M0 checkpoint
and its own frozen InternVL visual encoder on current `CAM_F0/L0/R0/B0` images.
The cache has 80 fixed camera-block slots of width 1536 (48 valid in the smoke
sample), contains no proposal, future, evaluator score, or factor field, and
records checkpoint SHA256
`7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`.
Four full-data shards are running on `rl-zt4` GPUs 0/1/3/4 at approximately
3.6 scenes/s/GPU.

A pinned-code follow-up (`0c3f4b7`) is queued behind those exporters. After a
full 103,288-token cache validation, GPU 0/1 will train a parameter-matched
pair with and without current-actor auxiliary supervision using the released
six-head BCE, while GPU 2/6 will run the corresponding continuous-progress
regression pair. All four use the same factor-ranking auxiliary. GPU 3/4 will
export the complete 12,146-scene M0-native Navtest observation cache. This
queue does not reserve GPU memory while the current exporters are active. The
earlier idle watchers were replaced before they started any training or
created any output.

The previous dynamic/static/signal query banks had different parameters but
no semantic supervision. A new optional training-only auxiliary head now
teaches the dynamic queries to recover the 16 nearest current actors in the
current-ego frame (presence, type, position, velocity, heading, and size).
These labels never enter the model forward or deployment artifact. The target
store covers 45,377 balanced scenes; unmatched replay rows are explicitly
masked. A Q-Former control is running on GPU 6, and the same loss will be tested
on the M0-native four-view cache after export completion.

All repository tests pass: 194 passed. Warnings are dependency deprecations and
the pre-existing Shapely numerical warning; there are no test failures.

The low-resolution factor-only all-64 run reached a best held-out-log PDMS
of 0.931604 at epoch 9 (the public M0 selector on the same fold is 0.951612).
Its final validation-selected factor checkpoint reaches complete-Navtest PDMS
0.900880, an improvement of +0.003341 over the previously tested epoch 3 but
still -0.008714 below public M0. The physical-log bootstrap 95% interval for
the delta from public M0 is [-0.013556, -0.003615].

Source inspection identified an important training-semantic difference: the
released EpisodeDrive loss uses ordinary BCE for NOC/DAC/TTC, whereas the
independent campaign so far upweighted rare violations by 10. The resulting
Navtest signature (better safety, worse progress/DAC) is directionally
consistent with that change. A full 2x2 Q-Former diagnostic now tests
unweighted BCE with factor-ranking and current-actor supervision independently.
The inspection also found that the independent loss had excluded progress from
BCE and replaced it with `2 * SmoothL1`; released EpisodeDrive includes the
continuous progress target directly in the same six-head BCE. The new explicit
`episode_drive_bce` mode is unit-tested against the source-equivalent formula,
while the former behavior remains available only as a named ablation.

## Q-Former actor supervision and conservative-gate Navtest

With the frozen released 16-token Q-Former observation, the continuous-progress
factor scorer reached held-out-log PDMS `0.949755` at epoch 3 with current-actor
auxiliary supervision, versus `0.947291` for its no-actor control at the same
epoch.  The gain came mainly from ego progress and DAC.  The same actor model's
conservative reference gate reached `0.954641`, or `+0.003029` over Base on the
held-out fold with a positive physical-log bootstrap interval.  The no-actor
gate reached `0.955196`, or `+0.003585`.

Both validation-positive gates were immediately evaluated on all 12,146
Navtest scenes.  The result reversed sign: the actor gate obtained `0.894955`
(`-0.014639` from public M0, 95% CI `[-0.017602, -0.011590]`) and the no-actor
gate obtained `0.899366` (`-0.010228`, 95% CI
`[-0.012328, -0.008003]`).  Both audits pass the 64-candidate completeness,
oracle identity, and selected-score reconstruction checks.  These gates are
therefore rejected; the positive trainval result is not a deployable planning
improvement and provides direct evidence of a severe validation-to-Navtest
shift for post-hoc switching policies.

The no-actor continuous-progress factor ranker itself later became a strict
held-out promotion at epoch 6: validation PDMS `0.953213`, delta `+0.001601`,
with bootstrap lower bound `+0.000010`.  Its complete all-64 Navtest result is
`0.901797`, delta `-0.007797` with 95% CI
`[-0.010752, -0.004674]`.  This audit also passes all 12,146-scene completeness
and reconstruction checks.  Correcting factor weighting and adding a ranking
objective therefore improves the independent Q-Former control, but does not
close the public scorer gap or support a 0.93 claim.

An all-physical-log refit path is now available for validation-selected
architectures.  It accepts only an artifact whose held-out physical-log
bootstrap lower bound is strictly positive, then locks its model configuration,
loss weights, optimizer settings, seed, selected stop epoch, and original
scheduler horizon.  The refit itself reports no validation metric and cannot be
used as a new selection artifact.  This permits a pre-specified 103,288-scene
refit without using Navtest feedback for model or epoch selection.

The current-actor auxiliary ranker later reached a strict held-out promotion
at epoch 7: PDMS `0.954557`, delta `+0.002945`, with physical-log bootstrap
95% interval `[+0.000697, +0.004724]`. Its complete 12,146-scene Navtest audit
reversed sign again: PDMS `0.897208`, delta `-0.012386`, interval
`[-0.016207, -0.008677]`. The audit has 64 candidates per scene, zero invalid
scenes, exact selected-score reconstruction, and the unchanged M0 candidate
bank upper bound `0.984112`. Current-actor supervision therefore improves the
held-out trainval fold but worsens the present Navtest selection; it is not a
deployable improvement.

Two additional M0-native four-view experiments are queued on `rl-zt3` GPUs
5/6 behind completion of the immutable 103,288-scene observation cache. Both
score all 64 candidates and retain the released six-factor BCE auxiliary. One
optimizes regret-weighted pairwise ranking; the other adds listwise, top-set,
expected-regret, and top-regret objectives. These runs test direct candidate
ranking separately from the four parameter-matched factor-only controls on
`rl-zt4`.

A second queued design keeps the same M0-native scorer-private visual encoder
but makes the released M0 factor logits and aggregate score an explicit
calibration anchor. Its factor-correction and utility-correction heads are
zero initialized, so before training its 64-way output and selected candidate
are exactly identical to public M0. It then learns candidate-relative factor
and utility residuals from the current four-view representation. This path
uses M0 inference tensors only—no DrivOR representation or weight, future
annotation, MetricCache value, or official PDM input. Hybrid-residual and
factor-residual variants are queued behind the direct-ranking controls on
`rl-zt3` GPUs 5/6. The complete repository suite passes `198` tests, including
zero-init Base identity, candidate-permutation equivariance, inference-schema
checks, and finite end-to-end residual training loss.

`rl-zt4` is now explicitly available for this campaign. Its GPUs 0/1/3/4 are
finishing the immutable 103,288-scene released-M0 four-view cache, GPU 5 is
running the validation-locked no-actor all-log refit, and the existing watcher
will use GPUs 0/1/2/6 for the four controlled scorer runs plus GPUs 3/4 for the
12,146-scene current-only Navtest cache. The cache exporter remains unchanged
near completion rather than being repartitioned and duplicated.

The M0-native Navtest path now supports both independent rankers and the
Base-calibrated residual architecture. A strict promotion manifest reads the
artifact selected on held-out physical logs and requires both positive mean
delta and a bootstrap 95% lower bound above zero. Only promoted artifacts are
evaluated. Inference consumes current M0 visual tokens, current state, frozen
M0 proposals, and—only for the calibrated residual—M0's own deployable factor
logits and scorer values. The official `[12146,64,7]` candidate matrix is
joined only after all selected indices are fixed. Full audits are serialized
on an otherwise idle GPU, validated with the reusable Navtest skill, and
compared scene-by-scene with public M0. The complete repository suite now
passes `200` tests.

The validation-locked no-actor all-log refit has now completed and passed the
strict 12,146-scene audit. It reaches Navtest PDMS `0.897837`, delta
`-0.011757` from public M0 with log-bootstrap 95% interval
`[-0.015066, -0.008519]`. Its unchanged candidate-bank upper bound is
`0.984112`; the loss is selection-only. The refit increases progress by
`+0.003679` but degrades collision by `-0.005475`, DAC by `-0.008315`, TTC by
`-0.019595`, and DDC by `-0.005022`. Training on all physical logs therefore
does not repair the held-out-to-Navtest reversal and is rejected.

The separately validation-locked current-actor all-log refit also completed
and passed the same strict audit. It reaches Navtest PDMS `0.895842`, delta
`-0.013752` from public M0 with log-bootstrap 95% interval
`[-0.017315, -0.010182]`. Its factor deltas are collision `-0.006957`, DAC
`-0.009880`, TTC `-0.022641`, DDC `-0.003828`, and progress `+0.004004`.
Together, the two pre-specified all-log refits show that simply exposing a
post-hoc fixed-bank scorer to all 103,288 training scenes does not remove the
distribution shift: both trade measurable safety for small progress gains.

M0-native deployment artifacts now use the real
`M0NativePrivateScorerAgent`, not a cache-only surrogate class. The online
agent obtains F0/L0/R0/B0 from the current frame, extracts private tokens with
the released M0 vision encoder, and supports either the independent ranker or
the Base-calibrated residual scorer. The artifact pins the Base checkpoint,
source-ranker SHA256, camera order, crop policy, pooling grid, wrapper chain,
and exact scorer class. Every promoted architecture is scheduled for a
four-scene same-device online/cache parity test with `1e-6` tolerance in
addition to full Navtest. Remote validation now calls the repository's audited
validators directly instead of assuming `/root/.codex/skills` is mounted on
every worker. The complete suite passes `202` tests.

The immutable four-view trainval cache is now complete for all 103,288 scenes.
`rl-zt4` GPUs 0/1/2/6 are training the four factor-loss/current-actor controls,
GPU 5 is training the Base-calibrated hybrid residual, and GPUs 3/4 are
exporting the matching full Navtest observation cache. `rl-zt3` GPUs 5/6 run
the two direct-ranking controls and GPU 3 runs the Base-calibrated factor
residual. Strict full-Navtest promotion remains serialized on `rl-zt4` GPU 7;
no job uses `vla-zt` or `vla-zt2`.
