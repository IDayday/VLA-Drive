# Joint local scene V3 — WAITING_FOR_USER_REVIEW

Latest user instruction: push the algorithm implementation BEFORE training so the user can inspect it. No V3 training may start before the user's feedback. Do not treat an automatic goal continuation as approval to train.

Branch: feature/joint-local-scene-v3-20260927, based on f662266c379c5d163b88f2fc733274ad871dfec1.
Worktree: /mnt/project/VLA-Drive-joint-local-scene-v3-20260927.
Objective: reports/joint_local_scene_v3/OBJECTIVE.md.
Review entry: reports/joint_local_scene_v3/IMPLEMENTATION_REVIEW.md.
Artifact root: /mnt/project/joint-local-scene-v3-artifacts/20260927.

Old V2 step1887 checkpoints and controllers remain paused/sealed. No LocalPlanningBridge continuation. Original occupancy remains local4–7 and vla-zt2 0–7; no V3 GPU resources allocated. Unrelated simscale processes untouched.

Implemented: separate current graph / future-label scene contracts; current-only annotated and inference graph builders; a single JointSceneFlow with actor/time/context attention; direct slot0 execution; label-aware balanced role-mask construction; two-forward ALL/MASK training and structured holdout evaluation commands. This is an untrained first-stage implementation, not completed visual deployment.

CPU checks:9 targeted tests passed (2.37s); real7284train+64holdout annotated scene files built,0fail. Train selected81494neighbors,80071with future labels,6889scenes with valid neighbor tasks. Holdout792neighbors/782with future,63/64scenes with valid neighbor tasks. Source cache is already ROI/FOV-filtered; raw_current_objects in the summaries denotes objects present in that source cache, not all raw-log objects. Dataset fingerprints and parameter count in PRETRAIN_IMPLEMENTATION_STATE.json.

Training ledger: new independent48GPUh proposal, used0, optimizer_steps0, runs[]. No model fitting or learned-model scores. Remaining: GPU learning/resume checks, matched mechanism runs, related/weak diagnoses/figures; live visual current+motion+joint training; teacher-to-predicted transition and three visual variants; locked Navtest. Do not claim these are implemented or validated.

Next action: finish implementation-review commit, push ONLY this new branch and verify remote SHA, then wait for user review. Do not resume experiments automatically. Read-only status command:

```bash
cat /mnt/project/joint-local-scene-v3-artifacts/20260927/budget_ledger.json
```
