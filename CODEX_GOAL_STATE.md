# Structured world V1 goal state

Full objective: reports/structured_world_v1/OBJECTIVE.md (all sections retained).
Status: active, P0 replay and P1 real 64-scene bbox/motion training complete; P2 provider adaptation running.
Engineering: PARTIAL. Research: NOT_RUN.

## Authoritative state
- Isolated branch: feature/structured-world-v1-20260926.
- Candidate source baseline: 0ecd2ae, recovered stable QwenOFT training implementation.
- Candidate weights: local DriveDreamer-Policy released pytorch_model.pt.
- Released baseline loads with zero missing keys and no unexpected keys except explicitly disabled Wan. Four real training scenes replayed with finite outputs; full manifest pending.
- Agent-query reference 9dd6b71 is available locally; not merged.
- Original /mnt/project/DriveVLA-M0 and DriveDreamer-Policy dirty worktrees untouched.
- Allowed hosts: local and training-vla-zt2. Only identified placeholder GPU jobs may be paused.
- Budget ledger: /mnt/project/structured-world-v1-artifacts/20260926/budget_ledger.json; 400 completed steps (bbox 200 + motion 200), provider adaptation 400 reserved/running. Always read live ledger before launch.

## Next actions
1. Bind checkpoint/config/tokenizer/runtime to a reproducible Qwen baseline.
2. Save fixed-noise original real-scene outputs before editing QwenOFT.
3. Implement target contracts, t0 geometry, track-identity matching and masked losses.
4. Integrate Reader/post-Qwen heads and camera-only geometric BEV provider.
5. Execute objective P0–P4 and all twelve scoped tests within execution_budget.yaml.

No hard blocker. World-disabled real GPU exact replay and frozen-Qwen query gradients passed. Six focused CPU tests and real 2-GPU NCCL head/loss equivalence with an empty rank passed. Full Qwen DDP, accumulation, resume, cache parity, visual unfreeze, planning pilots remain outstanding.

Fixed dev manifest found: DriveDreamer-Policy-perf/reports/ddp_flow_grpo_paired/full_target/dev_tokens.json (1696 /16 logs). Task train manifest excludes all dev logs. Temporary overfit run was aborted during loading, 0 steps, because two temporary scenes overlapped dev. Evidence in SPLIT_AUDIT.json. Task-owned dataset_v1 contains all 9888 train+dev metadata, no missing images; target caches generating for 8192 train /1696 dev.

Running handles: provider_adapt log; overfit_motion_eval log; targets_v3_train8192 log. All under artifact root. Geometry provider is trained from scratch, NOT a pretrained prior. No pilot launched yet.
Baseline source: 0ecd2ae. Latest valid task commit: 7e4ab7a; follow-up implementation uncommitted.

Next: finish target caches, evaluate provider perception/64 overfit; implement regression/resume checks; launch matched 1000-step A1/A2/B/C/D/E pilots using same base and fresh seed42 modules; preserve 12000 total cap. Provider D/E must load verified adapted provider checkpoint. Do not use overfit model initialization for core variants unless matching all controls.
