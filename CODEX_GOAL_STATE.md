# Masked Joint Trajectory World Model — ACTIVE

Latest user objective replaces the previous V1.1 scope restriction against joint multi-agent generation. Fixed three current front views; VLA+WM with random actor-trajectory masking, useful BEV representation, and real planning transfer. No RL/scorer/PDMS leakage.

Branch feature/masked-trajectory-world-20260927; source96ff2ee; original worktrees and data untouched. Design and budgets: reports/joint_world/DESIGN.md and execution_budget.yaml. Previous V1.1 runs terminal,2596 updates/1.67101 GPU-hours; planned box4 repair NOT launched. Previous diagnostic artifacts remain immutable in /mnt/project/structured-world-v1p1-artifacts/20260927. New artifacts must use /mnt/project/joint-world-artifacts/20260927.

Next: implement/test joint masked flow model and prediction-only planner bridge, run actual baseline GPU integration and bounded true-data learning; add BEV task objectives; matched controls/holdout/development planning evaluation and final report. Do not describe unit tests as planning success. Future context may only appear in explicitly privileged training reconstruction, never planner or predict_action input.

Resource snapshot: local GPUs0–3 free after own previous experiments;4–7 existing unrelated workers untouched. Verify before use. No new training launched yet.
