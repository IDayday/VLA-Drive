# Direct Scorer Rehabilitation V3

## Outcome

The preregistered Direct Baseline Quality Gate passes on the fresh 512-scene holdout. The scientific output is the raw three-seed hybrid-current ensemble: its independent six-factor selected score is `0.590544`, versus `0.489983` for the frozen WoTE base-anchor selector. The paired gain is `+0.100561` with 95% scene-bootstrap CI `[+0.074379, +0.125778]`; top-1 regret falls by `25.77%`.

This is an offline fixed-bank ranking result. It is not navtest PDMS and no claim of a 93-point navtest result is made.

## Data contract

- Candidate bank: the released, fixed 256 base anchors; no offsets, additions, removals, score-conditioned sampling, or forced oracle candidate.
- Labels: `independent_wote_labels_4s_six_factor.v2`, order `[NC, DAC, DDC, EP, TTC, Comfort]`, evaluated over all 256 candidates at once.
- Features: frozen WoTE current-only exports. Every direct input was exported before the latent-world-model call. Future/effect fields are absent and `label_source=none`.
- Holdout: `original_test[712:1224]`, disjoint from train, val, the old Oracle Effect test/dev set, and the sealed future-effect reserve.
- The sealed 512-scene reserve had zero accesses. The access audit passes.

## Architecture improvement

The repaired scorer is trained from scratch on frozen features and fixed trajectories; it is not a fine-tune of the original WoTE reward head. It combines:

1. deterministic per-waypoint kinematics (position, heading sine/cosine, speed, acceleration, curvature);
2. the complete frozen 8x8 current-BEV token grid through candidate-to-BEV cross-attention;
3. the frozen candidate-current feature from WoTE's trajectory/ego encoder;
4. path-aligned bilinear and 3x3 tube sampling from the current BEV along every candidate;
5. six independent factor heads, plus unbounded ranking utility and hard-safety heads.

The objective ablation selected the pure independent six-factor objective (`O0`). The hybrid-current representation then won the preregistered representation ablation and was confirmed across seeds 0, 1, and 2.

## Holdout metrics

| Output | Selected score | Regret | Mean rank | False-safe | Zero score |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen WoTE selector | 0.489983 | 0.390217 | 30.771 | 9.18% | 6.25% |
| Raw Direct ensemble (scientific) | 0.590544 | 0.289656 | 19.826 | 8.59% | 5.08% |
| Safe fallback (diagnostic) | 0.514779 | 0.365421 | 26.770 | 8.20% | 5.47% |

The safe fallback uses the development-locked factor floor `0.6` and predicted margin `0.3`; it overrides WoTE in `14.65%` of scenes. Because it assigns sentinel values to fallback candidates, pairwise and NDCG metrics are intentionally not reported for that policy.

## Gate interpretation

`DIRECT_BASELINE_UNDERFIT` is repaired: both preregistered alternatives pass (`score gain >= 0.05` and `regret reduction >= 20%`), and raw false-safe does not regress. This does not retroactively modify the old Oracle Effect report. The action-effect hypothesis remains untested under the repaired direct backbone.

The next and only in-scope experiment is to rerun A–L with the same hybrid current backbone, the same auxiliary encoder/fusion capacity, and input masking as the sole variant difference. Train/val/dev effect assets may be used; this fresh holdout remains frozen. The experiment must stop at the Oracle Effect Gate and must not train a forward effect predictor, inverse model, VLA, trajectory generator, or refiner.

## Reproducibility

The immutable result is `experiments/cf_effect_wote_direct_rehab/direct-rehab-v1-20260901/final-holdout-evaluation-v1.json` (SHA256 `6444d3bbc5555050dd0eefb80ef2ca30a5c49db196fe57965cd7543a5865cd97`). Asset and split hashes are in `ASSET_MANIFEST.json`; the one-pass output contract is in `HOLDOUT_RESULT.json`; legacy report hashes remain unchanged.
