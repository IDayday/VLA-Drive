# Randomized Controlled Candidate Audit

- Formal scenes/logs: 45,377/1,192
- Candidates: 726,032/726,048 (99.998%)
- Candidates per successful scene: 16
- GT appeared in 16 different candidate indices
- Global seed: 20260828; every scene uses a stable SHA256-derived subseed
- Families: {'brake_timing': 90754, 'different_prefix_similar_endpoint': 90754, 'gt': 45377, 'lateral_offset': 136131, 'progress_shape': 90754, 'same_endpoint_mid_curve': 90754, 'same_prefix_different_tail': 90754, 'speed_change': 90754}

All non-GT candidates use continuous scene-specific parameters, smooth temporal
profiles and deterministic de-duplication. Candidate order and GT position are
independently shuffled per scene, so candidate index does not identify a fixed
behavior. These are controlled perturbations, not true futures.
