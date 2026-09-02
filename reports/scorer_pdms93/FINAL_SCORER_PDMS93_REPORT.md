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
| Previously reported best M0-only gate/control | 0.923998 | below target |
| DrivOR original representation + scorer | 0.929291 | no: external model |
| DrivOR SimScale-134k representation + scorer | 0.932614 | no: external model |
| Best-of-64 offline oracle | 0.984112 | no: non-deployable oracle |

The `0.923998` row is retained as a historical campaign summary while its
artifact-SHA aggregation is refreshed. It does not meet the target under
either interpretation.

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
its advantage is therefore not purely a scorer-head effect.

## What the DrivOR diagnostic established

Using DrivOR features to score M0 proposals was a controlled diagnostic only.
On 18,179 held-out scenes from 61 physical logs, shuffling the DrivOR scene
registers reduced PDMS from `0.949809` to `0.834258`, and zeroing them reduced
it to `0.891486`. Thus the useful unit is the scorer-oriented visual
representation, proposal interaction and calibrated factor head together—not
a detachable scalar head. This evidence motivates an M0-owned scorer-private
perception path but cannot be used in the final model.

## Current M0-only experiments

Two full-log, 103,288-scene context-fusion experiments are running:

- `v7`: M0 context fusion added to the masked candidate-consequence ranker;
- `v8`: the same context fusion as a single-variable addition to the strongest
  no-future residual baseline.

Both use only current M0 features at inference. Positive held-out-log results
must still pass full Navtest and same-device online/cache parity.

A separate temporal-consequence scorer is undergoing whole-log five-fold CV.
The discovery pass uses 1,192 logs and 103,288 scenes. Fold 0 contains 20,658
validation scenes from 238 disjoint logs. Its first four raw epochs are:

| Epoch | Validation PDMS | Delta vs Base | Pairwise accuracy | Regret reduction |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.967619 | +0.003938 | 0.9405 | 24.9% |
| 1 | 0.967941 | +0.004260 | 0.9285 | 26.9% |
| 2 | 0.967953 | +0.004272 | 0.9424 | 27.0% |
| 3 | 0.967669 | +0.003988 | 0.9430 | 25.2% |

These are Navtrain held-out-log diagnostics, not Navtest results. The raw
selector also trades route progress against TTC/DDC, so a common safety-aware
deployment policy is required before promotion.

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
remain preserved but cannot be promoted directly.

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

The proposal ceiling supports continued scorer work, and the first temporal
fold shows meaningful held-out-log regret reduction. However, the requested
M0-only `>0.93` complete-Navtest result remains unproven. The final conclusion
will be updated only after common-epoch five-fold replay and strict Navtest
evaluation of every promoted artifact.
