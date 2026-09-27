# Structured World V1.1 signal rehabilitation — ACTIVE

Objective: reports/structured_world_v1p1/OBJECTIVE.md (full new attachment).
Base: 393c53bbd685c77694d342496a0b5cd7aa1ae735.
Branch: feature/structured-world-v1p1-signal-rehab-20260927.
Worktree: /mnt/project/VLA-Drive-structured-world-v1p1-20260927.
New artifacts: /mnt/project/structured-world-v1p1-artifacts/20260927.
Old artifacts: /mnt/project/structured-world-v1-artifacts/20260926 (immutable).

## Completed

- P0: independent maximum-cardinality-then-distance current matching, strict <2m, optional same-class constraint. Legacy matcher/metrics retained; old reports unchanged. New evaluate.py uses independent metrics, old rediagnose.py explicitly legacy, old summarizer rejects new schema instead of silently reporting wrong numbers.
- Full12 old banks ×1696 scenes re-audited from NPZ with full v6 targets:20352 rows,0 failures, no VLM/PDM rerun. Figures/sums/fingerprints in METRIC_REAUDIT.*. C_support and adapter have zero objectness-positive predictions, all108544 centres in support; raw recall1.3743/2.1539%. Genuine no-object and localisation collapse alongside old metric defect.
- Actual64-scene dataset actions decoded through released decoder agree with independent ego(t0) SE2 (max2.65e-6m/2.36e-7rad). Raw time/current cameras/poses checked. Initial audit used matrix ZYX yaw and failed; correct NAVSIM Quaternion.yaw_pitch_roll gives max4.44e-16rad. Failed audit retained; training labels/tolerances unchanged. A1/A0 factor decomposition and39 high-to-zero scenes saved.
- P1 append_tail implemented, legacy_pre_action default retained. Actual64 real GPU scenes: native prefix embedding/mRoPE/mask/hidden/action and final trajectories EXACT for append_tail and actual gate0 bridge. Legacy untrained insertion shifts up to6.88284m. No optimizer updates.
- Nine targeted CPU tests pass (metrics, brute-force matching, append positions/padding, gate gradients). Do not repeat old full V1 validation suite.
- Reports/commands/status/index committed. Latest implementation c2f00248d204a6f10522dce9165516c3189d7b37 (P1 check); later commits contain evidence/summary guard. Determine current HEAD with git.

## Next required work (not complete)

P2 first: implement and test coordinate references/residual decode, current-reference-based no-object eligibility, displacement-only motion loss, W_PRE/W_POST head readout and real accumulation trainer. Hypotheses in WORLD_LEARNABILITY.md. These are not implemented yet and must not be reported as learned.
Run matched W_PRE/W_POST first16 then full64 using original overfit_tokens.json and v6 labels. No GT selection/token count. Each variant <=1000 updates, preferred batch8. Freeze original Qwen/vision/DiT; frozen Qwen forward must propagate Reader gradients. No full-Qwen graphs retained over accumulation. Only if W_PRE learns and adequately trained W_POST fails may one LoRA bottleneck probe be added. Engineering target ≥80% geometry recall/≥50% precision on full64; class-correct and failures also report.
P3: at most one genuine external pretrained feature provider, or explicit NOT_TESTED. Old random CNN is not such a provider.
P4 gated on world learnability: matched current vs current+motion, each >=4 passes8192 scenes, then P_CAPACITY/P_CURRENT/P_FUTURE with original frozen driving model, zero gate and same exposure. Original1696 dev protocol; no navtest tuning. Include all runs/intermediates, not best checkpoint cherry-picking.
Required remaining deliverables: WORLD_LEARNABILITY, PROVIDER_AUDIT, PLANNING_PILOT, new trainer/resume/configs/viz, run ledger, final four independent statuses. Final push only new branch and verify SHA.

## Budget / resources

New cap24000 optimizer steps AND48 GPU-hours. Used0 steps and0.039005458884769015 GPU-hours (P1 inference charged conservatively). Old11639 steps sealed. New ledger at artifacts/budget_ledger.json; no further jobs live after P1 completed. Both repair rounds available; CPU reference-audit correction is not a failed learning run.
P1 paused only verified local gpu_stress.py parent168455, its0–3 workers; local4–7 original placeholders untouched. Restored0–3 via same gpu_stress.py command, verified parent663431/workers663535–663538 (session63132); inspect placeholder_restore_verified.log/current nvidia-smi before future pause. Initial detached shell launch failed and was retried; see RESOURCE_STATE.json. No other training task stopped. Local and training-vla-zt2 allowed; inspect real process ownership before use.

## Recovery commands / paths

cd /mnt/project/VLA-Drive-structured-world-v1p1-20260927
export PYTHONPATH=.:/mnt/project/DriveVLA-M0/nuplan-devkit
export WORLD_PYTHON=/root/miniconda3/envs/ddp/bin/python
export DEPTH_MODEL_CKPTS=/mnt/project/DriveDreamer-Policy/depth_model_ckpts
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$WORLD_PYTHON" -m pytest tests/structured_world_v1p1 -q

Full executed commands: docs/STRUCTURED_WORLD_V1P1_QUICKSTART.md. Do NOT rerun completed audits into their immutable paths.
Baseline /mnt/project/DriveDreamer-Policy/models/DriveDreamer-Policy; VLM sibling Qwen3-VL-2B-WorldAction.
Data old artifacts/dataset_v1; manifests old artifacts/{overfit,train,dev}_tokens.json.
Targets old artifacts/targets_v6_train8192 (capacity64, overflow recorded); full dev old artifacts/targets_v6_dev_full (capacity999). For small-set full-GT diagnostic regenerate independently if capacity64 truncates; never silently delete targets to meet80% goal.
Raw logs /mnt/project/onevl_navsim_data/navsim_logs/trainval.
No applicable AGENTS.md in worktree/parents; optional CODEX_GOAL_STRUCTURED_WORLD_V1P1_393c53b.md not found in worktree/attachments.
