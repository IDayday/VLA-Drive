# COMMON4096 frontend: independent narrow correctness review

Reviewer: `/root/spatial20k_correctness`. **The corrected Lead frontend passes this code review. Actual COMMON4096 analysis remains waiting for completed inputs.** No analysis, bind, dispatch, model load, CSV read or bootstrap computation was performed.

Original implementation: author commit `0c4243714acdc81658b483a8a5e7445ebced6438`, frontend SHA256 `13ddeb7e51b40297c9349eac9cc7580b35d641a7559df3647976dde225ea3736`. That version has the lock inheritance finding below and is not the approved executable.

Reviewed corrected executable:
`/mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/run_common_once.py`

SHA256: `3cccc594f984cb293f7bd68b74696e0db30379370d81b439baf9cf6610feff1c`.

## Finding and verified local correction

The original `run()` held both a shared `SERIAL.lock` and a per-arm `RUN.lock`, but passed only the latter into the unchanged `run_pair`/monitor. The monitor inherits exactly that FD in its child. If the controller exits abruptly while the child remains alive, the shared serial lock disappears, allowing another arm onto the same CPUs while the original child still runs.

Two tiny actual subprocess/file-lock tests reproduced this boundary without launching campaign work. Passing the arm FD leaves SERIAL acquirable after closing parent copies; passing SERIAL keeps the cross-arm lock held until the child exits. Lead changed the single `run_pair` argument from `lock` to `serial` and retained both parent locks. An AST equality check confirms this is the only executable difference. Original monitor and statistical implementation were not modified.

The new actual SHA above, rather than the author's superseded original SHA, is the scope of the pass result.

## Retained behavior checked

- All four fixed C0 versus C1–C4 commands retain original argv, environment, cwd, plans and limits. Only COMMON_ENDPOINT step 4096 is accepted; old 1024 and incomplete scoring receipts are rejected.
- Fixed population remains 1,696 scenes / 16 logs. No new endpoint, metric, selection rule or scientific threshold is introduced.
- The native completion verifier is AST-identical to the original except for its stage literal and module qualification. Its full payload checks remain delegated to the original analyzer at execution time, not replaced by the small metadata checks used in this review.
- The retained CPU monitor is the original reviewed `8674aea1...` file in the frozen c4-cpu6 workspace. This is a path relocation to identical bytes, not acceptance of the changed file at the old moving Lead path.
- Analyzer runtime remains clean at `6714b1e9fb2e0c8059b90b8b4d48cd40796fe345`; `evaluate_campaign.py` remains SHA256 `3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3`.
- Each pair retains a 1 GiB whole-process logical read cap including binding, 2 GiB aggregate RSS, 1,800-second limit plus 40-second cleanup, and 32 MiB total output/control cap. The same guards and monitor are reused; actual resource usage has not been measured for 4096 here.
- Consumed session markers reject another execution. Existing or partial outputs are not overwritten. Controls use the new COMMON4096 namespace; analytical output remains each original preregistered COMMON_ENDPOINT namespace. No old 1024 run is dispatched.

## Current test and input evidence

Current corrected frontend metadata/source fixtures: **9 passed**, 0 failed, 0 skipped, 0.065 seconds. Independent actual FD boundary tests: **2 passed**, 0 failed, 0 skipped, 0.020 seconds. Full logs are retained beside this report. These are code/metadata checks, not new model, scoring or statistical results.

At review time, file-presence inspection found C0's official COMPLETE receipt and no C1–C4 official COMPLETE receipts. This is a presence snapshot, not certification of C0 scoring or any input payload. Lead must bind the actual complete export/score receipts for the chosen pair once ready, verify current resource availability, and invoke the corrected executable only. The review does not authorize reruns or bypass any incomplete-input check.

Reproduce the bounded checks from this review worktree:

```bash
python -B reports/st_world_policy/20261018/consequence4096_frontend_correctness/test_lock_boundary.py
python -B /mnt/project/VLA-Drive-st-world-policy-20261016/reports/st_world_policy/20261018/consequence4096_bounded_preparation_v1/test_preparation.py
```

`review.py` records the exact pinned source and AST-delta checks into an immutable local `REVIEW.json`; running it against an already present receipt intentionally refuses to overwrite evidence.
