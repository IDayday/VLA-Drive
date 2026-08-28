# NAVSIM v2 Reactive and Synthetic Extension Audit

## Reactive traffic policy

- v2 devkit/version/commit: `True` / `2.0.0` / `9fe1459b8f6ab69a15274450ec301d541209bedd`
- IDM traffic policy code present: `True`
- Simulated object types: `VEHICLE`
- Remaining object types merged from logged future: `True`
- Deployed v2 metric-cache split: `navtest`
- Eligible mini/trainval reactive cache available: `False`
- Candidate-level reactive empirical comparison run: `False`
- Blocker: Only deployed v2 cache records train_test_split='navtest'; test/navtest labels are excluded.

Consequently, this deployment supports the reactive mechanism in code, but the audit does not report actor endpoint/speed/braking deltas as measured training evidence.  Running those metrics on the only configured `navtest` cache would violate the split constraint.  The implementation is vehicle-only; pedestrians and other types remain log replay.

## Synthetic follow-up scenes

- CSV/pickle scenes: 204 / 204
- Unique corresponding original scenes: 16
- Follow-ups per original (min/median/max): 9 / 11.5 / 19
- Pickle load success: 100.000%
- Four history frames / annotations / track tokens: 100.000% / 100.000% / 100.000%
- Extended tracks / traffic lights with at least 8 steps: 100.000% / 100.000%
- Referenced synthetic camera / LiDAR / combined sensor file coverage: 100.000% / 0.000% / 88.889%
- Same-original groups with non-identical synthetic current poses: 100.000%
- Corresponding original log availability in the allowed trainval log directory: 0.000%

The synthetic scene current pose agrees with its CSV `viewpoint` (maximum measured XY error `0 m`), but different follow-ups mapped to one original have different synthetic current states and image paths.  Their start-state offset from the unavailable corresponding original warmup log cannot be reliably computed in this deployment.  These are synthetic follow-up scenes suitable for neighborhood-state augmentation or weak multi-future supervision, not real counterfactual futures from one identical current observation.
