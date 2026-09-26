# Structured world V1 goal state

Full objective: reports/structured_world_v1/OBJECTIVE.md (all sections retained).
Status: active, P0 baseline discovery; no experiment has been launched.
Engineering: PARTIAL. Research: NOT_RUN.

## Authoritative state
- Isolated branch: feature/structured-world-v1-20260926.
- Candidate source baseline: 0ecd2ae, recovered stable QwenOFT training implementation.
- Candidate weights: local DriveDreamer-Policy released pytorch_model.pt.
- Baseline selection remains provisional until strict key audit and real replay.
- Agent-query reference 9dd6b71 is available locally; not merged.
- Original /mnt/project/DriveVLA-M0 and DriveDreamer-Policy dirty worktrees untouched.
- Allowed hosts: local and training-vla-zt2. Only identified placeholder GPU jobs may be paused.
- Budget: 0 / 12,000 optimizer steps consumed; full cap remains.

## Next actions
1. Bind checkpoint/config/tokenizer/runtime to a reproducible Qwen baseline.
2. Save fixed-noise original real-scene outputs before editing QwenOFT.
3. Implement target contracts, t0 geometry, track-identity matching and masked losses.
4. Integrate Reader/post-Qwen heads and camera-only geometric BEV provider.
5. Execute objective P0–P4 and all twelve scoped tests within execution_budget.yaml.

No current hard blocker. A GPU replay has not yet been run.
Latest valid source commit: 0ecd2ae (task commits pending).
