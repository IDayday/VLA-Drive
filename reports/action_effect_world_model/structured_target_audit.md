# Structured target action-dependence audit

The original raw map tube is retained as a diagnostic only. Main action-collapse and Phase-6 representation metrics must use channels classified as action-dependent below; action-invariant channels are excluded rather than allowed to dominate raster averages.

Within-scene candidate variance and target Action Gap measure target action dependence. Between-scene variance quantifies the competing scene prior. If predictions are supplied, the sensitivity ratio and shuffle gap use the corresponding scene-action probe.

## Retained Phase-5 raw map diagnostic

| Channel | Within variance | Between variance | Action ratio | Target AG | Pred/target | Target shuffle | Prediction shuffle gap | Class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| drivable_area | 0.014496 | 0.168300 | 0.079302 | 0.194451 | 0.054805 | 0.027642 | -0.000052 | action_effect/action-dependent |
| lane_or_connector | 0.015344 | 0.169316 | 0.083094 | 0.198262 | 0.052923 | 0.029277 | -0.000052 | action_effect/action-dependent |
| route | 0.010668 | 0.149331 | 0.066674 | 0.164025 | 0.069572 | 0.020207 | -0.000042 | action_effect/action-dependent |
| dynamic_occupancy | 0.013048 | 0.035476 | 0.268902 | 0.158865 | 0.070502 | 0.025757 | -0.000005 | action_effect/action-dependent |
| relative_longitudinal_velocity | 0.000754 | 0.001785 | 0.296844 | 0.037941 | 0.200417 | 0.005288 | -0.000001 | action_effect/action-dependent |
| relative_lateral_velocity | 0.000026 | 0.000052 | 0.334663 | 0.004459 | 0.637182 | 0.000341 | 0.000000 | action_effect/action-dependent |
| dynamic_clearance | 0.000927 | 0.074667 | 0.012258 | 0.050258 | 0.136066 | 0.018823 | 0.000013 | action_effect/action-dependent |

## Trajectory-aligned effect tube

| Channel | Within variance | Between variance | Action ratio | Target AG | Pred/target | Target shuffle | Prediction shuffle gap | Class |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| candidate_relative_dynamic_occupancy | 0.009450 | 0.029615 | 0.241908 | 0.131027 | n/a | 0.018452 | n/a | action_effect/action-dependent |
| drivable_area_sdf | 0.002614 | 0.170965 | 0.015060 | 0.081054 | n/a | 0.029215 | n/a | action_effect/action-dependent |
| lane_sdf | 0.003648 | 0.150145 | 0.023723 | 0.090139 | n/a | 0.034200 | n/a | action_effect/action-dependent |
| route_sdf | 0.002189 | 0.155153 | 0.013914 | 0.069240 | n/a | 0.025900 | n/a | action_effect/action-dependent |
| relative_longitudinal_velocity | 0.000522 | 0.001519 | 0.255595 | 0.029298 | n/a | 0.003234 | n/a | action_effect/action-dependent |
| relative_lateral_velocity | 0.000394 | 0.001091 | 0.265444 | 0.022911 | n/a | 0.002921 | n/a | action_effect/action-dependent |
| dynamic_clearance | 0.001012 | 0.102523 | 0.009778 | 0.050003 | n/a | 0.018277 | n/a | action_effect/action-dependent |
| dynamic_collision_field | 0.014820 | 0.161188 | 0.084203 | 0.176645 | n/a | 0.028305 | n/a | action_effect/action-dependent |
| ego_swept_footprint | 0.000524 | 0.003027 | 0.147502 | 0.025776 | n/a | 0.001051 | n/a | action_effect/action-dependent |

## Target contract

The effect tube contains candidate-relative dynamic occupancy, map signed-distance fields, occupied-agent relative velocity, dynamic clearance/collision fields, and the ego swept footprint at 1/2/4 seconds on a 32×32 candidate-aligned grid. All logged future actors remain target-only under `log_replay`; true interactive response remains unknown.

## Pilot-small learned-target confirmation

The formal scene-disjoint Phase-6 run uses 78,688 valid targets from 5,300
scenes. Per-channel controls use the declared loss-aligned primary metric and
1,000 whole-scene bootstrap resamples. A channel passes only when AEE beats the
scene-only, train-mean, and zero controls and within-scene action shuffling
significantly harms prediction.

| Channel | Primary metric | Beats all controls | Shuffle gap [95% CI] | Pass |
|---|---|---:|---:|---:|
| candidate-relative dynamic occupancy | balanced BCE | False | 0.002398 [0.001883, 0.002946] | False |
| drivable-area SDF | normalized L1 | True | 0.000098 [0.000072, 0.000126] | **True** |
| lane SDF | normalized L1 | True | 0.000098 [0.000071, 0.000127] | **True** |
| route SDF | normalized L1 | False | 0.000085 [0.000053, 0.000118] | False |
| relative longitudinal velocity | masked L1 | False | 0.000030 [0.000008, 0.000054] | False |
| relative lateral velocity | masked L1 | False | 0.000011 [-0.000002, 0.000023] | False |
| dynamic clearance | normalized L1 | False | 0.000048 [0.000037, 0.000061] | False |
| dynamic collision field | balanced BCE | False | 0.000270 [0.000207, 0.000340] | False |
| ego swept footprint | Dice | False | 0.000485 [0.000381, 0.000606] | False |

Thus at least two structured channels carry learnable candidate action effects;
Gate-3 failure is not caused by every structured target being exogenous. The
small absolute shuffle effects also show that scene priors remain strong, so
action-invariant and action-dependent channels must continue to be reported
separately.
