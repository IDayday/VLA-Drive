# Gate 2.5 action-path audit

**Decision: PASS.** The observed Gate-2 collapse is not explained by a broken action branch.

Source commit: `a41f081077d457bc5fd2d6c0550cba8c3c8dc880`; source tree hash: `691d0f1c252fc0f530577143aa39aa23c91f588b13fc8d92480e345129b1c1d9`.

## Synthetic action fitting

- Correct normalized MSE: 0.033826.
- Shuffled normalized MSE: 0.142211.
- Shuffled/correct ratio: 4.20.
- Scene-bootstrap shuffle-gap 95% CI: [0.070483, 0.149015].

## Candidate consequence overfit

- Scenes/candidates: 24 / 374.
- Correct decoded MAE: 0.109190.
- Mean-control MAE: 0.517603.
- Shuffled decoded MAE: 0.202246.

## Action-path engineering audit

- Optimizer contains every action/fusion parameter: True.
- Action encoder gradient norm: 2.718706e+00.
- Output-to-trajectory Jacobian norm: 7.912444e+01.
- Within-scene action embedding variance: 3.104962e-01.
- Fusion action contribution ratio: 0.823975.
- Physical candidate range `[x, y, heading]`: min `[0.0, -12.777761459350586, -1.1237819194793701]`, max `[56.68881607055664, 10.060880661010742, 0.7767719030380249]`.
- Normalized candidate range `[x, y, sin(yaw), cos(yaw)]`: min `[-0.7358095049858093, -28.744935989379883, -0.9017416834831238, 0.4322752356529236]`, max `[4.760293483734131, 22.59608268737793, 0.7009808421134949, 1.0]`; std `[1.2634340524673462, 4.616204261779785, 0.21602657437324524, 0.07936064153909683]`.

## Equal-capacity controls and calibrated false-safe metrics

Thresholds are fit on accepted fit-scene candidates and frozen before held-out evaluation. Intervals resample scenes, never individual candidates or pairs.

| Control | Unsafe prevalence | Balanced accuracy | AUROC | AUPRC | False-safe rate |
|---|---:|---:|---:|---:|---:|
| scene_only_probe | 0.0641 [0.0430, 0.0892] | 0.5117 [0.4873, 0.5453] | 0.5004 [0.3688, 0.6216] | 0.0878 [0.0368, 0.2501] | 0.9653 [0.9020, 1.0000] |
| trajectory_only_probe | 0.0641 [0.0430, 0.0892] | 0.5724 [0.4975, 0.6510] | 0.5644 [0.4326, 0.6788] | 0.0928 [0.0486, 0.1827] | 0.7083 [0.5439, 0.8631] |
| scene_action_probe | 0.0641 [0.0430, 0.0892] | 0.5058 [0.4802, 0.5412] | 0.4807 [0.3476, 0.6031] | 0.0680 [0.0341, 0.1735] | 0.9618 [0.8950, 1.0000] |
| shuffled_action_probe | 0.0641 [0.0430, 0.0892] | 0.5119 [0.4889, 0.5445] | 0.4695 [0.3402, 0.5914] | 0.0714 [0.0335, 0.2019] | 0.9583 [0.8961, 1.0000] |

## Interpretation

The action branch can fit a deterministic trajectory function and memorize candidate-level consequences, with non-zero gradients and Jacobian. Poor factual-only candidate sensitivity therefore represents a statistical shortcut under single-future supervision rather than an action wiring failure. Calibration metrics replace the earlier uncalibrated 0.5 false-safe number.
