# Full Query Base navtest — actual checkpoint measurement

**Checkpoint step14000: official NAVSIM v1.1 PDMS = 85.2969/100**, all **12146 scenes /136 logs**, valid12146, invalid0. Log-cluster bootstrap95% interval **[84.1555, 86.4466]**, 2000 resamples, seed42. This is an absolute Base quality measurement, not incremental IQE gain or a converged/maximum achievable score.

| Metric | Value, displayed on0–100 scale |
|---|---:|
| PDMS | 85.2969 |
| comfort | 100.0000 |
| drivable_area_compliance | 94.7061 |
| driving_direction_compliance | 97.9829 |
| ego_progress | 81.2679 |
| no_at_fault_collisions | 96.6615 |
| time_to_collision_within_bound | 91.6186 |
| zero-score percentage | 8.4802% (1030/12146) |
| nonfinite raw/physical trajectories | 0 |

This is a single Query producing one full trajectory, with no Scorer, argmax or extra experts. It was trained using Query trajectory IL **plus the original retained DINO/future-clip/interaction auxiliaries**, so it is not a pure-IL ablation. Original3 current cameras1024×576, Qwen3-VL-2B, ego inputs and normalizer are preserved. No learned DiT/driving checkpoint initialized this Query Base. Raw normalizedXY/sincos converts through the existing physical trajectory conversion (8 steps,0.5s, rear-axle ego frame).

Checkpoint: `/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_014000.pt`; SHA256 `a65c68c4bcb562cbd134c31007a38255c0bbaf489a8a0f5c960c74288de00e93`. It was complete and checksum-verified before the evaluation began. The current training run continued unchanged on training-vla-zt2. Dataset hash `9051de59a26a65e3817ce0236066249354625e0d0287a63f0736c4eb2bcbfa25`, score protocol `570cb579994926588ef536162c81e573180e5661c9e8ab9efe0ed58a661c2966`, IQE Python source hash `7cac27c53e6d926788fa2c744cf8970b1bb67660994058eeea2503a687d29e4b`. Full lock/resolved config, input image/cache hashes and per-scene raw/physical trajectories are retained under `/mnt/project/iqe-review-runtime-1e876da/navtest/step_014000`; the dataset is `/mnt/project/iqe-review-runtime-1e876da/navtest/dataset/DATASET.json`.

The user explicitly replaced additional stage_val measurement with direct checkpoint navtest monitoring. The4-scene13k diagnostic is not used for this result. Training/dev and navtest scene/log intersections are empty. No GT, future labels or metric context enters the model's inference input. The official reference is metric_cache.trajectory, official nonreactive traffic; each candidate gets its own context/simulator/scorer. Internal labels remain[0,1]. All per-log values are included in [the result](evidence/NAVTEST_NAVTEST_RESULT.json).

A scoring-environment mismatch was caught during execution: the first persistent-import process selected installed nuplan. Its 4319 intermediate labels were preserved under superseded_installed_nuplan_scores and **none entered this result**. The corrected wrapper pins the exact repository nuplan source used by independent reference calls. Four independent reference scores and nine repeated/reordered persistent calls match total and named components exactly. [Runtime identity](evidence/NAVTEST_SCORING_RUNTIME_IDENTITY.json) records an explicit canonical source-hash algorithm; the earlier ad-hoc correction hash is retained but its unrecorded serialization is not compared to the canonical hash. No model source or checkpoint was modified by this correction.

Eight idle A80080GB GPUs on training-vlawm-zt performed independent batch1 shards; no Base GPU was used. Each rank's model parameter/buffer hash is unchanged. Observed peak allocated memory was9,038,349,312 bytes/rank. Measured image-preprocessing→trajectory/CPU-copy p50/p95 **0.674457/0.702186s**, batch1, encoderBF16 autocast withFP32 master/head, TF32off, no warmup samples excluded. These timings include current_observation, shared VLM, projections and decoder, and synchronize on CPU copy; they exclude image checksum/model loading/official simulation. Full shard elapsed time (including construction/loading/hash checks) ranged **1319.00–1325.66s**. Corrected scoring wall time **745.92s** with8 CPUworkers includes waiting for remaining GPU outputs; it is not pure scoring CPU time. Persistent imports reuse no mutable scene/scoring state.

Before observing this score, subsequent checkpoint steps were fixed to **20000,50000,100000** from the same training-run identity. The finite observer waits for complete checksum-valid checkpoints and idle authorized hosts. Its current status is in SCHEDULE_STATUS.json, and its launch/hold/recovery commands are in [RUNBOOK](RUNBOOK.md). Those future scores are **PENDING**, not forecast. No additional full training, calibration, IQE bank or serving activation was started. Navtest now serves checkpoint observation and is not described as an untouched final test.

Formal Base acceptance rules were not preregistered; this result is not automatically converted into a formal LOCKED_S0 or expert-training release. No further validation-set run is needed to obtain or interpret this requested navtest result. IQE scientific gain remains UNTESTED.
