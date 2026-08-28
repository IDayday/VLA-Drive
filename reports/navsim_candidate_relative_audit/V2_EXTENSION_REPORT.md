# NAVSIM v2 Extension Audit

## Reactive traffic policy

- Available: **True**
- Official cached scope: **128 scenes / 1894 candidates**
- Captured-track rerun: **32 / 32 scenes**
- Mean reactive-vs-replay actor endpoint change: **0.6732543938109112 m**
- Candidate-dependent vehicle response nonzero rate: **0.010670731707317074**
- NAVSIM IDM simulates `VEHICLE` only. Pedestrians and all remaining types are merged from logged replay by the abstract policy.

These are reactive-policy simulated consequences, not observed multi-agent reactions and not causal counterfactuals.

## Synthetic follow-up scenes

- Deployed: **True** at `/mnt/data_and_weight/Public_Space/navsim/navhard_two_stage/synthetic_scene_pickles`
- Scene files: **5462**; sampled metadata: **512**
- Legal train synthetic root deployed: **False**
- Camera file coverage in sample: **1.0**
- Extended tracks coverage in sample: **1.0**

Resolved data is NAVHARD/two-stage challenge data. Metadata was audited, but annotations are not used for training or supervision.

At most neighborhood-state augmentation or weak multi-future supervision; not same-current-state, different-action ground truth.
