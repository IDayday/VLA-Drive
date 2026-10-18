# `road_failure_direction_selection_v1`

Top-level fields consumed by this adapter:

|Field|Meaning|
|---|---|
|schema/status|`road_failure_direction_selection_v1` / `BOUND_BEFORE_PROXY`|
|domain/seed|`st-road-failure-direction-v1` /42|
|source_run|The original `C1_fulltrain_seed42_official_4096_v1` absolute run|
|source_runtime/source_runtime_commit|Original canonical runtime and exact6714b1e9 SHA|
|frame|Exact FRAME32 path/file SHA/byte count|
|source_evidence/source_run_artifacts|Original evidence and C1 registration/identity/result references|
|train_index/train_index_identity|Original TRAIN index file and semantic identity|
|selection_count/selected|1–32unique TRAIN cases; actual FRAME32 contains32|
|selection_sha256_before_proxy|SHA256 of canonical JSON `selected`, sorted keys, compact separators, allow_nan=False|
|sources|Original pinned canonical source file references|

Each selected row preserves FRAME32 order and contains `scene_id`, `source_log`,
`role=train`, `boundary`, `rank`, `slot`, `policy`, `rollout_id`,
`observation_hash`, `condition_hash`, `artifacts`, `failed` and `safe_sibling`.
`boundary` equals frame `policy_version`, and `slot` equals frame `local_slot`;
the original `global_scene_slot` must equal `rank*8+slot`.

`artifacts` has exactly the used immutable original references:
`requests`, `results`, `complete`, `admitted`, `rank_complete`, `rollout`.
Each reference includes `path`, `sha256`, positive `bytes`; paths must correspond
to the original run/ledger/boundary/rank and hashes must match the frame.

`failed` and `safe_sibling` each contain:

- Actual candidate index0..3 and complete original canonical `request` identity.
- Saved FP32 `raw_sha256`; original request contains physical trajectory hash.
- Exact `context_hash`, `reference_input_sha256`, `native_t0` boolean.
- `valid=true`, `score_units=0..1`, full named `metrics` and `metrics_valid`.

The failed candidate must have DAC0; the sibling DAC1 and native t0=false.
Both have the same scene, boundary/rank/slot, generator identity, cache, context,
score protocol, runtime and reference input. An unexpected original t0=true is
retained for explicit undefined handling, not swapped for a different case.

Candidate ordering uses SHA256 of UTF-8 NUL-joined
`st-road-failure-direction-v1`, `42`, `candidate`, `candidate_id`.
Choose the first DAC0 and first DAC1 by that key, independently of total score or
proxy results. Runtime rechecks this rule against all four actual records.

Selection JSON is produced by the actual bounded binder, not manually filled
from aggregate means. No raw trajectory, context array or private scene image is
committed to git. A future real selection is an external runtime artifact.
