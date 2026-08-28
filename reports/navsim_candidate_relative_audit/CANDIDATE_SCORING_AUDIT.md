# Candidate Scoring Audit

## Gate B: **PASS**

- legal_candidate_success_rate_gt_98pct: **PASS**
- gt_coordinate_alignment: **PASS**
- deterministic_repeat: **PASS**
- candidate_factor_difference_exists: **PASS**
- state_horizon_alignment: **PASS**
- candidate_order_preserved: **PASS**

- Traffic setting: `non_reactive`
- Official scoring success: **768 / 768 (100.000%)**
- Scenes with at least one differing PDM factor: **55 / 64**
- Repeat state max error: `0.0`
- Cached-vs-repeat state max error: `0.12127027381211519`
- Cached-vs-repeat after declared float32 storage cast: `0.0`

The deployed pdm_score API evaluates a candidate plus the fixed PDM reference. Candidates remain individually scored because proposal-set progress normalization makes all-K scorer batching a different protocol; the simulator itself is batch-capable.

Each cached candidate was scored with the same PDM-closed reference proposal. No across-scene max-progress normalization is used as a training label. Official ego_progress remains an official within-call normalized factor.

The Parquet table keeps non-reactive and reactive labels in distinct runs/columns through the `traffic_policy` field. These are official simulated consequences, not true causal counterfactuals.
