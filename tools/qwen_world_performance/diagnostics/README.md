# TRAIN diagnostics used in this campaign

These are byte-for-byte copies of the actual diagnostic sources. Their original
paths and hashes are in
`reports/qwen_world_performance_20261103/results/DIAGNOSTIC_SCRIPT_SOURCES.json`.
They use the registered NAS data and frozen campaign source; they do not change
the training manifest, priorities, optimizer, actor inputs or NAVTEST protocol.

- `probe_train19.py`: actual IL70, true full G16 and true full G32, 15 fixed
  TRAIN B1-zero cases and four high-score controls; 912 actual PDM queries.
- `summarize_review.py`: CPU-only evidence checks and per-scene classification
  of that completed exploration diagnostic.
- `audit_verified_gt.py`: timestamp/coordinate checks, true TRAIN GT scoring
  and comparison with fixed TRAIN2048 B1. Invalid GT timing excludes only that
  teacher, never the scene from policy training.
- `audit_roundtrip_gt.py`: actual ActionCodec-roundtrip GT scoring on the sealed
  first S6 collection, including the seven scenes lacking high-quality samples.
- `audit_synthetic_progress.py`: explicitly synthetic path and progress
  proposals on those seven TRAIN scenes, followed by real canonical scoring;
  this is candidate discovery, not a deployed policy or a PPO rollout group.
- `dispatch_train2048_pair.py`: finite CPU admission owner for exported S0u400
  and S5u800. Default only prints the plan; `--run` waits for the registered
  GPU3/5 lanes, binds UUIDs and exact native exports, and dispatches the unchanged
  canonical B1 evaluator. It admits no new task after 23:00 UTC on November 2.

These diagnostics measure TRAIN behavior. Their scores are not NAVTEST results.
GT endpoints must additionally pass ActionCodec roundtrip scoring before any
future use as training teachers; the raw GT audit does not establish that check.
