# Structured world V1 goal state

Status: bounded implementation/experiments complete. engineering_status=READY; research_status=INCONCLUSIVE.
Objective: reports/structured_world_v1/OBJECTIVE.md. Final evidence: reports/structured_world_v1/FINAL_REPORT.md and STATUS.json.

- Branch feature/structured-world-v1-20260926. Latest valid implementation commit 4ed9bbfe709e920c59e4c14f09272e565f2a7665; final evidence is committed separately. Final branch HEAD is reported in the delivery message and verified against origin.
- Baseline source0ecd2ae1f616844641a6d94cfafb50e0f32c26fe; checkpoint SHA2569445f9da577a8e3c6b7c636c60a98d714d602668f9a8033b0891c703bc40210f. Three current front cameras, one candidate, original DiT, no scorer.
- Artifact root /mnt/project/structured-world-v1-artifacts/20260926. Locked budget ledger contains11639/12000 consumed optimizer steps,361 remaining; no further training planned. Both hyperparameter/convergence repair rounds exhausted. Correctness fixes and actual GPU update tests are charged.
- Historical A1/A2/B/C/D/E seed42 and B/C seed43 each1000 steps; corrected C/E/adapter each400 steps. A0 and all11 trained variants evaluated on the same1696 scenes:20352 score rows,0 failures. Old adapter stopped at201 steps and excluded from comparison.
- FOV polynomial foldback corrected; current caches targets_v6_train8192 and targets_v6_dev_full. Historical v4/v5 remain immutable and explicitly labeled; no silent replacement of old training results. All perception diagnostics use full corrected dev GT.
- Actual BEV geometry and external/cache/online path pass complete-policy parity. Eight CPU tests, FP32/disabled/target-independence/padding/vision-update tests, strict checkpoint keys, real2-GPU empty-rank/accumulation and joint Qwen/world/DiT resume pass. Fast provider continuation failure retained; explicit deterministic CPU geometry backward plus actual GPU CNN training passes exact4 vs2+2 continuation.
- Final provider is from-scratch, weak (22/1043 current train targets matched); no pretrained BEVDet claim. Corrected planning C90.7067/E90.4992/adapter91.2704, baseline93.1446. Paired intervals cross0; no promoted model. Navtest NOT_RUN.
- Original dirty workspaces/data/checkpoints untouched. All experiment GPU jobs finished. Local0-3 and remote0-7 placeholder scripts restored; original local4-7 job untouched. See RESOURCE_RESTORATION.json.

## Recovery / reproduction

cd /mnt/project/VLA-Drive-structured-world-v1-20260926
source /mnt/project/structured-world-v1-artifacts/20260926/campaign.env
"$WORLD_PYTHON" tools/structured_world/resume_run.py --checkpoint "$WORLD_ARTIFACTS/C_support_v6_seed42/checkpoint.pt"

This is tested and returns already_complete without new steps. Do not repeat existing runs or exceed reservations. Full commands and NOT_RUN boundaries: docs/STRUCTURED_WORLD_V1_QUICKSTART.md. A future full corrected matrix requires a new bounded plan; do not spend navtest for tuning.
