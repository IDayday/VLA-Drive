# Query Base evaluation and qualification

The active Base is Query+IL with the inherited DINO/future-clip/interaction auxiliaries. It is not a pure-IL ablation. Inference emits exactly one complete trajectory and uses no expert selector or learned Scorer.

Before the user redirected evaluation to navtest, a fixed **4-scene/4-log stage_val** diagnostic evaluated complete `step_013000.pt` (SHA256 `d81f1ad49493fea7db7a1caac32f1f6deefecb938e03699f72ec538195c62a44`). The checkpoint was selected when it was the latest completed snapshot; training continued independently.

| Metric | Actual diagnostic value |
|---|---:|
| raw IL loss | 0.1599680539 |
| physical ADE/FDE, meters | 0.7471365 / 1.4037250 |
| circular yaw MAE, radians | 0.01414815 |
| official PDMS, [0,1] | 0.95750794 |
| NC / DAC / TTC / comfort / direction | 1 / 1 / 1 / 1 / 1 |
| ego progress | 0.89801906 |
| valid official scores / nonfinite / zero rate | 4 / 0 / 0 |

This small diagnostic is **not a full benchmark or an achievable-score estimate**. Model parameter/buffer hash did not change. Input path contains only actual current cameras, observed ego history and navigation; GT is read only after prediction for IL/ADE. Actual NAVSIM1.1 official reference context and all per-log/scene scores are in [BASE_EVALUATION.json](evidence/BASE_EVALUATION.json). The result was reused only after checkpoint/manifests/protocol identity checks, including training-log disjointness.

No historical Base quality thresholds were found. `BASE_RULES_DRAFT.json` was written before these results, expressly **DRAFT_NOT_PREREGISTERED**: proposed meanPDMS≥0.8, zero/NC/DAC violation rates0, ≥128 scenes and ≥10 logs. These proposals were not silently converted to a preregistration. `qualify-base` returned **BASE_QUALITY_UNTESTED**, exit2. An actual full-mode `freeze-base` call returned **BASE_UNQUALIFIED**, exit2, and created no LOCKED_S0. No requirement for exactly100000 steps was introduced.

**Latest user steering:** do not continue validation-set measurement; directly evaluate suitable complete checkpoints on navtest. The pending128-scene validation cache job was stopped with its partial evidence preserved. A separate navtest-monitoring entry was added without weakening formal freeze or using navtest to train/calibrate IQE. The 14000-step checkpoint was selected before seeing its navtest score. See NAVTEST_REPORT for that full benchmark; the earlier4-scene score must not be substituted for it. No same-framework comparable single-policy baseline or formal Base acceptance protocol has yet been bound.

The requested full14000-step navtest has now completed: **PDMS 85.2969/100**,12146/12146 valid,136 logs, zero rate8.4802%. This is real measured Base quality; the remaining formal block is an unregistered acceptance protocol, not absent evaluation. See [NAVTEST_REPORT](NAVTEST_REPORT.md).
