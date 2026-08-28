# Candidate-relative Target Construction

- Successful scenes: 500/500 (100.000%)
- `C_full`: trajectory-derived candidate motion plus world-relative relations.
- `C_environment_only`: 15 relationship features; no waypoint copy, candidate type/index, official final score or official aggregate factor.
- Dynamic actor tensor: nearest 16 per candidate/horizon, deterministically sorted by center distance then stable token hash.
- Shared logged-world tensor: up to 64 dynamic actors per horizon in global coordinates.
- Time horizons: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] s at 2 Hz.

The construction is a **non-reactive candidate-relative consequence**: every candidate is related to the same logged future actors, traffic-light records and static map. It is not a candidate-specific ground-truth image, true multi-agent response, causal effect or complete counterfactual future.

The deployed training MetricCache has no named `future_tracked_objects` field. Actor identity/type/velocity therefore comes from official `NavSimScenario.get_tracked_objects_at_iteration`, backed by Scene annotations; Gate A verifies its global polygons against MetricCache's 10 Hz logged occupancy.
