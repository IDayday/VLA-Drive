# V3 pretraining correction — CORRECTION_COMPLETE_WAITING_REVIEW

Completed the bounded correction from b4f08d05c0202fb483bc3247cab97abe59694d16 on isolated branch `fix/joint-local-scene-v3-pretrain-review-20260927`, worktree `/mnt/project/VLA-Drive-v3-pretrain-review-20260927`.

Latest implementation commit: 4893159c25dc7dfad6f718f8173dd46cb9533295. Main numerical/GPU/resume evidence commit: 4de523bdefb99e7490d358754e5c464ecef70686. Subsequent delivery commit contains reports/state only. Final remote SHA is verified after push in the delivery response; do not infer upload from this file alone.

Implemented modeled-state policy and safe padding, strict active-input validation, decision_local/nearest current graph selection and bounded risk context, schema4 corpus identities, cross-call role scheduler, fixed paired completion queries, separated all-hidden exports, weighted two-forward trainer, full resume state and enforced run budget.

53 CPU tests pass. CPU and CUDA each passed 4 continuous vs2+2 synthetic trainer updates (16 TOTAL synthetic updates). CUDA forward/backward checks and4 real-scene/21-query no-update checks pass. Final export/cache/output-argument refinements have CPU coverage; no final-commit GPU/resume rerun is claimed. New train7284/holdout64 caches completed with0 failures, token/log intersections0; prior ROI/FOV filtering and missing raw current population explicitly reported. No private artifacts committed.

Real optimizer updates0; synthetic16/20, remaining4; GPU0.008588/2h. All task runs complete, no live task training process. Leftover budget is NOT an instruction to run more tests/training. Old V2step1887/controller remain sealed; old48GPUh campaign is not activated. No ALL/MASK long experiment, full visual training, Navtest or oldV2 restart occurred. Other tasks/occupancy processes were not stopped.

Artifact root `/mnt/project/v3-pretrain-review-artifacts/20260927`; ledger `budget_ledger.json`. New `annotated_{small,train,holdout}_v4`; old `annotated_train_v1` unchanged. Read `reports/joint_local_scene_v3/pretrain_review/RESULTS.md`, AFTER.json and BUDGET_LEDGER.json for exact identities, counts and limitations. READY/learned planning gain is not claimed: full camera/Qwen integration remains unfinished.

Stop after correction delivery. Next action is user code review, not automatic training. Safe zero-update reproduction from this worktree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q tests/joint_scene
```

Do not rerun the8-update/device resume harness against the remaining4-update allowance, create replacement ledgers to bypass caps, or resume a checkpoint under a different source identity. Further structured mechanism training awaits the next reviewed authorization.
