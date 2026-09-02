# M0-owned scorer campaign: current strict status

## Outcome

The active requirement is **not yet complete**. A deployable result must use
only M0-owned current-observation representation and scorer weights and must
exceed `0.93` PDMS on the complete FP32 Navtest audit. No artifact satisfying
all of those conditions has yet been measured above the threshold.

The previously measured `0.932614` selector is deliberately excluded from the
success claim: it scores the frozen M0 proposal bank with the independently
released DrivOR SimScale-134k representation and scorer. It remains useful as
an analysis of representation quality, but it is neither an M0-owned scorer nor
an eligible solution to the current objective.

| Selector on the public M0 64-proposal bank | Complete Navtest PDMS | Eligible |
| --- | ---: | --- |
| Public M0 Base | 0.909594 | baseline |
| DrivOR-reference gate r2 on M0 proposals | 0.923998 | no: external DrivOR representation/reference |
| DrivOR original representation + scorer | 0.929291 | no: external model |
| DrivOR SimScale-134k representation + scorer | 0.932614 | no: external model |
| Best-of-64 offline oracle | 0.984112 | no: non-deployable oracle |

The `0.923998` result is a strict complete-Navtest measurement, but its exact
lineage is `DrivORReferenceGateRanker`: it uses DrivOR current-observation
registers and the released DrivOR factor choice as its reference.  It is
therefore a useful diagnostic/control, not an M0-only result.  The best
strictly measured result that remains fully within the M0-owned constraint is
still the released M0 Base at `0.909594`; completed replacement scorers have
not yet improved that full-Navtest score.

## Native proposal-bank comparison

M0 and DrivOR must be compared on their own native proposal banks before
reasoning about scorer quality. The accepted strict audits contain 12,146
Navtest scenes from 136 logs, 64 candidates per scene and zero invalid scenes.

| Native released system | Selected | Best-of-64 | Candidate mean | Candidate median | Scorer regret |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.909594 | 0.984112 | 0.795276 | 0.835676 | 0.074518 |
| DrivOR original | 0.936907 | 0.993342 | 0.797153 | 0.849673 | 0.056436 |
| DrivOR SimScale-134k | 0.945829 | 0.994094 | 0.804264 | 0.861690 | 0.048265 |

The M0 bank already has a high oracle ceiling, but its released scorer leaves
substantially more regret. DrivOR also has a modestly stronger native bank;
its advantage is therefore not purely a scorer-head effect. The strict paired
decomposition shows that about 34% of DrivOR-original's selected-score gap and
28% of SimScale-134k's gap comes from the higher oracle ceiling; the remaining
66% and 72%, respectively, comes from lower scorer regret. M0 has the largest
within-bank pairwise ADE (`1.877 m`), so lack of raw geometric diversity is not
the explanation. The complete selected/oracle factor and cross-bank geometry
audit is recorded in
[`NATIVE_M0_DRIVOR_64_COMPARISON.md`](NATIVE_M0_DRIVOR_64_COMPARISON.md).

## What the DrivOR diagnostic established

Using DrivOR features to score M0 proposals was a controlled diagnostic only.
On 18,179 held-out scenes from 61 physical logs, shuffling the DrivOR scene
registers reduced PDMS from `0.949809` to `0.834258`, and zeroing them reduced
it to `0.891486`. Thus the useful unit is the scorer-oriented visual
representation, proposal interaction and calibrated factor head together—not
a detachable scalar head. This evidence motivates an M0-owned scorer-private
perception path but cannot be used in the final model.

## Current M0-only experiments

The two full-log M0 context-fusion tests finished and were rejected before
Navtest promotion. On the fixed 18,179-scene / 61-physical-log validation set,
`v7` scored `0.922752` against Base `0.951612` (`-0.028860`, bootstrap CI fully
negative), while the cleaner no-future single-variable `v8` scored `0.928892`
(`-0.022720`, CI fully negative). Simply concatenating an additional M0 context
feature does not produce a useful scorer-private representation.

A separate temporal-consequence scorer is undergoing whole-log five-fold CV
over 1,192 segment logs and 103,288 scenes. Three discovery folds have
completed; folds 3 and 4 are still running. Raw selection results are:

| Fold | Base PDMS | Raw best PDMS | Delta | Regret reduction | Raw best epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.963681 | 0.968600 | +0.004919 | 31.09% | 9 |
| 1 | 0.965542 | 0.969641 | +0.004099 | 25.57% | 9 |
| 2 | 0.964957 | 0.970005 | +0.005048 | 32.35% | 3 |

Using epoch 9 as a provisional common epoch across the three completed folds
gives a scene-weighted delta of `+0.004671`, a worst-fold delta of `+0.004099`,
`29.56%` regret recovery and `0.9413` pairwise accuracy for score differences
at least 0.02. All three fold-specific raw improvements have positive
physical-log bootstrap lower bounds. These are Navtrain held-out-log
diagnostics, not Navtest results.

The current deployment sweep is substantially more conservative than the raw
ranker. Its retained deltas for folds 0--2 are `+0.001752`, `+0.002087` and
`+0.001925`, recovering only 11--13% of Base regret. The raw model consistently
improves progress while modestly lowering TTC and DDC/DAC. Factor-only scoring,
hybrid factor/residual scoring and denser absolute safety thresholds recovered
only small or inconsistent extra value. The principal remaining problem is
therefore safe override accuracy, rather than absence of a learnable ranking
signal.

Source inspection confirmed that EpisodeDrive's `tr_out` is the actual
trajectory-conditioned hidden state consumed by its scorer and is already
shaped by the per-proposal agent/area auxiliary losses during joint training.
A strict single-variable five-fold campaign that exposes this native M0
candidate hidden state to the temporal scorer is queued on `rl-zt4`. Three
fold-0 pilots are queued on `rl-zt3`: Base candidate features, a learned
candidate-vs-Base relative-safety head, and a Base-relative utility head. All
use current M0 inputs only; no DrivOR representation, future annotation or PDM
input is present at inference.

## Cross-validation lineage correction

An audit found that the original five-fold helper could select a common epoch
from metric histories while materializing weights retained at each fold's
independent pairwise-best epoch. Fold 0 already demonstrates why this matters:
epoch 2 has the best PDMS so far, while epoch 3 has the best pairwise accuracy.

The corrected process is two-stage:

1. run all five discovery folds and choose one epoch using only whole-log
   Navtrain validation;
2. deterministically replay every fold with `--retained-epoch` and sweep one
   common deployment policy on weights from exactly that epoch.

Artifacts and sweep rows now record their weight epoch. Policy
materialization fails if it differs from the common epoch. Discovery outputs
remain preserved but cannot be promoted directly. The resumable
`watch_temporal_locked_replay_campaign.sh` controller performs the locked
replay, three full-data seeds and the eight-artifact Navtest audit only after
those checks pass. Replay also preserves the discovery run's 12-epoch cosine
scheduler horizon even when it stops immediately after the locked epoch; this
keeps the learning-rate sequence and retained weights exactly reproducible.

## Strict acceptance rules

- FP32 inference, deterministic CUDA setup and exact checkpoint class;
- 12,146 unique Navtest scenes, 136 logs, 64 candidates and zero invalids;
- future annotations/images and official PDM values are training/evaluation
  targets only and never inference inputs;
- official candidate factors are joined only after a selected index exists;
- positive whole-log validation bootstrap before Navtest;
- artifact SHA coverage for every promoted method;
- same-device online/cache score error at most `1e-6`;
- native proposal quality and scorer selection quality reported separately.

## Current conclusion

The native-bank audit and three completed temporal folds both support continued
M0-owned scorer work. They also narrow the failure mode: M0 has sufficient
candidate-bank headroom, the temporal scorer learns a repeatable ranking
signal, but the current safety-aware override policy discards most of it.
However, the requested M0-only `>0.93` complete-Navtest result remains
unproven. The final conclusion will be updated only after common-epoch
five-fold replay and strict Navtest evaluation of every promoted artifact.
