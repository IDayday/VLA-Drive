# A/V Navtest parallel execution — 2026-10-07

At14:11UTC all six requested exact-checkpoint jobs were running and emitting real trajectories. This is execution progress; no75k/80k full benchmark has completed yet.

| Arm | Host |75k physical GPUs |80k physical GPUs |75k predictions at snapshot |80k predictions at snapshot |
|---|---|---|---|---:|---:|
| A_ACTION | training-vlawm-zt2 |0–3|4–7|11594/12146|300/12146|
| A_NO_MAE | training-vlawm-zt |0–3|4–7|186/12146|329/12146|
| V_QUERY | training-vla-zt |0–3|4–7|265/12146|348/12146|

These are24 inference allocations on GPUs also running the three existing8-card trainers. Live inventory found approximately16–17GiB training allocation per80GiB GPU and64GiB free before launch. Each inference process acquires the existing Navtest/research lease, checks the exact GPU UUID, at least30000MiB free, host RAM and the exact registered trainer argv/run ID. Foreign or invisible GPU processes cause a retry; no trainer, pressure script or unrelated process is evicted. The eight original authorized hosts were inventoried; the three hosts above had suitable visible ownership and headroom. vla-zt3/rl-zt4 had high occupancy with unavailable process ownership and were not forcibly cleared.

The prior75k attempt emitted11425 A_ACTION predictions and11425 official CPU rows before all four exporters exited by SIGKILL (native−9/wrapper247). Their origin remains UNVERIFIED. The failed controller was waiting for its own scorer and serially blocked80k. Only exact owned evaluation argv/PGID groups were stopped to replace that execution. Trajectories and official rows remain in their original banks, including all initial OOM/failure logs. Confirmed dead native meters preserve their original JSON and charge through the observation time as a conservative cost upper bound; no failed cost or real updates are erased.

The first parallel recovery retained those banks but requested8 CPU workers, which the native cached CPU identity rejected because the original bank binds16. This failed attempt is preserved. The75k allocation now explicitly uses16 workers; new80k banks use8. Four shared CPU scoring leases bound simultaneous score processes (up to64 workers if all are16). No scoring implementation, runtime, metric cache or precision changed. Inference and CPU scoring can overlap.

The nine-task80k/90k/100k observer no longer waits for75k to finish. It dispatches up to three independent future checkpoint evaluations, one active checkpoint per arm, while its separate fast preservation loop retains exact completed checkpoints outside training directories. All three80k snapshots already exist.90k/100k remain WAITING until those exact checkpoints are published. No old C/S or unrelated evaluation observer was restarted.

Execution source: `4b566a6c6c6d3b4b6212ed19d7af018485d5298d`, clean frozen worktree `/mnt/project/VLA-Drive-planning-interface-navtest-parallel-source-4b566a6`. Original registration/controller evidence source `c9e101096b08b828ee7362faa12f2cd6f635a449` remains immutable. Native checkpoint restoration/export/scoring sources remain `f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882` for A arms and `7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c` for V_QUERY. No training method/source/optimizer was changed.24 targeted queue, allocation, retention, compatibility and cost-evidence tests passed. Source was pushed and remote SHA checked before launch.

Exact protocol remains FP32 optimizer-master restoration, FP32 inference, TF32 off,10 original Euler steps, one candidate, scene-bound sampling seed42, full12146 scenes/136 logs and the original full-precision official environment. Four native scene partitions remain unchanged when a bank moves host/cards. Each complete result requires zero missing/failed scenes, native bank/content identity checks, the applicable full official CPU replay qualification and eight new distinct-log unmodified official spots within1e-8 for all seven metrics. Partial counts above are not PDMS. This is one sampling seed at intermediate checkpoints, not final five-run or training-seed stability evidence.

Budget remains48 evaluationGPU-hours for75k plus144 for the nine future tasks (16 per task); the original future observer48-hour deadline remains. No unlimited allocation or training budget is introduced. Evaluation optimizer updates remain0. Existing training exposure/checkpoint hashes and full execution receipts are in the aggregate evidence; weights, raw scene rows, image/metric caches remain outside Git.

75k active controller PID444712; future observer PID443350. Recovery commands below must only run after verifying the corresponding live owner has exited. Their leases reject a duplicate owner.

```bash
cd /mnt/project/VLA-Drive-planning-interface-navtest-parallel-source-4b566a6
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.evaluate_fixed_navtest run --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_75k_20261007_v1/registration.json --execution /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_parallel_20261007_v1/execution75_resume.json
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.watch_navtest watch --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_80_90_100k_20261007_v1/registration.json --execution /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_parallel_20261007_v1/execution.json
```

Snapshot evidence: [NAVTEST_AV_PARALLEL_20261007.json](evidence/NAVTEST_AV_PARALLEL_20261007.json). Earlier serial startup and recovery reports remain historical evidence, superseded for current allocation by this report.
