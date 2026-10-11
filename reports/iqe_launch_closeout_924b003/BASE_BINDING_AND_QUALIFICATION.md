# Actual formal incremental start

The original S0 framework was migrated to a single Query full-trajectory head. This is neither DiT output equivalence nor a strict reproduction of original M0. Qwen3-VL-2B, original three cameras/resolution, preprocessing, normalizer and auxiliary training objectives remain unchanged. No old learned DiT driving weights were substituted.

Selected complete checkpoint:

`/mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_022000.pt`

SHA256: `42320ea3e6fb09c6cd2b955dde7398a4033338a98a68bc4bac111fe8e5e01e93`.

This checkpoint was produced by the ongoing full Base job, not the one-step diagnostic Base and not our 32-step performance profile. Strict loading and real image K=1/clone checks passed. Six protected source files remain identical; old checkpoint and contract were not rewritten.

The prospective selection protocol fixed this one checkpoint and a stable 32-scene/32-log stage_val list before evaluation. Rules require complete valid official labels, finite planning outputs/statistics, correct independent input/fit-log isolation and a read-only evaluation. No arbitrary PDMS convergence threshold is asserted. This grants engineering eligibility for a resampling_only incremental experiment; scientific baseline sufficiency and deployment remain UNTESTED. The protocol/rules timestamps are actual, not backfilled. Historical observed navtest scores were disclosed before registration and were not used to choose score thresholds or failure cases.

Fixed manifest `/mnt/project/iqe-launch-runtime-924b003/stage_val_fixed.json`, SHA256 `82744d0162849ded6e259bfc946ab3977c6db5d0b5162f6549ce15952e948c11`. Query Base fit IDs and logs are disjoint. This does not change the inherited data-role partition.

Results (32/32, not full navtest): PDMS **0.90916554** = **90.9166/100**; raw IL 0.19935959; ADE 0.820110m; FDE 1.738936m; periodic yaw error 0.024813rad. NC 0.96875, DAC 1, TTC 0.9375, EP 0.85699730; one zero-score scene, no invalid trajectory. These observations do not establish convergence or incremental gain.

Selected qualification `175b61961441ccd3005a6b794ed5443198e470ff7f7cb88b612774255772ddb6`, checksum `580956c1b43ec66f7129b10f1e3eca8f72f0a12052d7a7a53841ac283721ada3`. Actual preflight and formal freeze passed for `configs/iqe/formal/round1_locked.yaml`. Immutable lock: `outputs/iqe/formal_round1_22k/LOCKED_S0.json`. Old draft `review/formal_posttrain.yaml` remains a historical draft and is not the launch config.

Full roles audited: incremental_fit 90,892 scenes/1,056 logs; stage_val 5,563/56; selector_cal 5,137/64; dev_report 1,696/16. Unique observations 103,288; source groups 1,192. No final_test training access. The formal bank has not been generated. Original train-cache objects are not directly compatible with official single-reference scoring; `prepare-metric-contexts` must construct/verify official contexts before banking. Bounded real contexts are verified separately; no cache alias or scoring-weight change hides that dependency.

User confirmation was received at2026-10-11T03:11:03UTC: “采用 22k 工程资格起点（推荐）”. It is recorded separately in `evidence/USER_BASE_SELECTION_CONFIRMATION.json`; no earlier registration or evaluation artifact was rewritten.
