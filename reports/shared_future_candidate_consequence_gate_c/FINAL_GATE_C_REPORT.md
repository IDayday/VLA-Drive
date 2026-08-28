# Shared Future–Candidate Consequence Gate C Final Report

## Outcome

- Gate C0 reproduction: PASS (32 scenes, exact max error 0)
- Gate C1 oracle dynamic incremental value: FAIL
- Gate C2 current-observation predictability: NOT RUN
- Gate C3 real-candidate planning gain: NOT RUN
- Final route: **Route E** under the predeclared decision rule

Route E does not mean the measured dynamic signal is exactly zero. It means the
expanded, log-balanced and real-proposal evidence does not satisfy the minimum
support required to spend model-training budget or claim a shared-future world
model. A narrower physical-risk distillation study would be a different method.

## Environment and data

- Base commit: `6e96cf7321b134c42c2cf0fbbc315cd61c925b11`
- Branch: `feature/shared-future-candidate-consequence-gate-c`
- Legal split: `trainval`
- Selected/scanned scenes: 45,378/103,288
- Selected logs: 1,192; maximum 50 scenes/log
- Five log-disjoint folds: {'0': 9076, '1': 9076, '2': 9076, '3': 9075, '4': 9075}
- Candidate bank: randomized controlled K=16 and frozen EpisodeDrive K=16
- Traffic setting: non-reactive candidate-conditioned relabeling of trainval logged future
- Reactive/navtest response cache: NOT RUN (forbidden for training/tuning)
- Synthetic follow-up data: discovered in environment audit but not mixed into any Gate C result
- EpisodeDrive checkpoint SHA256: `7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d`
- Candidate/target coverage: 99.998%/99.998%
- Oracle completed prefix: 45,377/45,378
- Audited construction failure: [{'error': 'GEOSException: TopologyException: side location conflict at 664469.44046492805 3997510.1089300122. This can occur if the input geometry is invalid.', 'log_name': '2021.06.09.19.40.26_veh-12_01241_01510', 'scene_token': 'f47b259046405a8d'}]

## Formal oracle decomposition

| Group | Pairwise | Top-1 regret |
|---|---:|---:|
| O0 | 0.7413 | 0.0740 |
| O1 | 0.7300 | 0.1486 |
| O2 | 0.7607 | 0.0796 |
| O3 | 0.8476 | 0.0413 |
| O4 | 0.8574 | 0.0333 |
| O5 | 0.9022 | 0.0199 |
| O6 | 0.8466 | 0.0406 |
| O7 | 0.8748 | 0.0257 |
| O8 | 0.8748 | 0.0263 |
| O9 | 0.8751 | 0.0256 |
| O10 | 0.8347 | 0.0384 |
| O11 | 0.8082 | 0.0514 |
| O12 | 0.7921 | 0.0656 |
| O13 | 0.8413 | 0.0388 |

- O8−O3 dynamic gain: 0.0271
- Equal-log point estimate: 0.0327
- Equal-log bootstrap 95% CI: [0.0301, 0.0352]
- Statistical note: the Gate threshold uses the mean of five fold-level pairwise accuracies; the bootstrap gives every log equal weight, so these two point estimates need not coincide.
- Top-1 regret reduction: 36.38%
- O9 state/recomputed-risk gain retention: 1.011
- Held-out candidate-family mean/worst gain: 0.0492/0.0146

| Gate C1 criterion | Result |
|---|---|
| `bootstrap_ci_lower_above_zero` | PASS |
| `collision_or_ttc_improved` | PASS |
| `cross_scene_shuffle_gain_disappears` | PASS |
| `dynamic_pairwise_gain_at_least_0p03` | FAIL |
| `every_heldout_candidate_type_has_positive_gain` | PASS |
| `random_dimension_control_fails` | PASS |
| `repeated_static_control_fails` | PASS |
| `state_recomputed_risk_retention_at_least_0p40` | PASS |
| `top1_regret_reduction_at_least_20pct` | PASS |
| `within_scene_shuffle_gain_disappears` | PASS |

## Frozen EpisodeDrive proposal evidence

- O3/O4/O5/O8/O9 pairwise: 0.7536 / 0.7239 / 0.8417 / 0.7670 / 0.7699
- O8−O3 gain and 95% CI: 0.0133, [0.0119, 0.0339]
- Raw-state/direct-risk/recomputed-risk gains: -0.0297 / 0.0880 / 0.0162
- Original scorer / O5 risk ranker / O8 full-dynamic ranker / best-of-K mean score: 0.9626 / 0.9337 / 0.9210 / 0.9842

## Required questions

1. **Does 0.764 reproduce on more logs?** Yes for the absolute O8 metric. The formal O0 trajectory-only and O8 values are 0.7413 and 0.8748; the more stringent conditional dynamic increment over O3 is 0.0271. The predeclared +0.03 increment criterion is FAIL; the overall Gate C1 result also includes the independent controls below.
2. **Where does the gain come from?** Static-map O3 adds +0.0869 over O2; raw actor state O4 adds +0.0098, direct collision/TTC-adjacent physical risk O5 adds +0.0546, and future signal O6 adds -0.0010 over O3. Direct risk is therefore reported separately from raw actor state rather than packaged as a generic world-model gain.
3. **How much survives without direct collision/TTC?** O9 retention is 1.011; O9 recomputes risk from actor state and masks rather than ingesting official factors.
4. **Can current vision predict dynamic consequence?** INCONCLUSIVE / NOT RUN because Gate C1 failed.
5. **Is shared-future prediction better than direct prediction?** INCONCLUSIVE / NOT RUN.
6. **Does the GT image anchor improve planning?** NOT RUN; it is excluded from the final method.
7. **Does the consistency verifier identify unreliable consequences?** NOT RUN.
8. **Are actual EpisodeDrive proposal choices improved?** No deployable model was trained. The O8 logged-future oracle ranker mean 0.9210 does not beat the original scorer 0.9626.
9. **Does the result depend on fixed candidate templates?** Candidate parameters/order/GT index were randomized. Held-out-family worst gain is 0.0146; the corresponding Gate criterion is PASS.
10. **Route?** Route E. Stop shared-future model integration under this protocol.

## Leakage and terminology

All official aggregate/factor values are physically isolated as offline targets.
O0–O13 receive no official score, future image or candidate-type label. The
structured current-actor features and HD-map relations in the oracle table are
explicitly oracle-only because EpisodeDrive does not consume them directly.
No deployable model was allowed to inherit those fields. The
construction is a non-reactive candidate-conditioned relabeling of one shared
logged future; it is not a true counterfactual future or true multi-agent
response. One audited invalid map-geometry scene remains a reported failure and
was not repaired by modifying source data.

## Primary blocker

dynamic_pairwise_gain_at_least_0p03
