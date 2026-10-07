# A/V exact75k evaluation and finite80k/90k/100k queue

The requested evaluations are deployed. Results are **PENDING**, not completed benchmark measurements. The existing50k results and all historical artifacts remain intact. No training recipe, training source, model weights or input protocol was changed.

| Experiment |75k |80k |90k |100k |
|---|---|---|---|---|
| A_ACTION | Four native export ranks launched | WAITING for exact checkpoint | WAITING | WAITING |
| A_NO_MAE | Queued in75k controller | WAITING | WAITING | WAITING |
| V_QUERY | Queued in75k controller | WAITING | WAITING | WAITING |

All three75k checkpoints contain `completed=75000`, with2,399,816 actual training scene presentations each. These are the existing original-GT A/V runs, not the separate optimized-trajectory campaign. Exact60k/70k checkpoints were removed by rolling retention before this request; they are neither evaluated nor replaced by nearby steps.

| Experiment | Native training/inference source |75k checkpoint identity SHA256 |
|---|---|---|
| A_ACTION | `f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882` | `ed7e80df12f013b38dd913886febd5e868e10b4202b8403daf5e366dc1f62740` |
| A_NO_MAE | `f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882` | `c6d0c20f67202c0a45c672d6e51f07ceb243ab7276787af4df7baea93a09cfa3` |
| V_QUERY | `7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c` | `09ebc6e8821b1dee0e3949edfe986af16ed97b37fb7ab5b129d6e94682e1fa88` |

Controller/observer source is `c9e101096b08b828ee7362faa12f2cd6f635a449`, committed, pushed and remote-SHA verified before launch. Its immutable worktree is `/mnt/project/VLA-Drive-planning-interface-navtest75-source-c9e1010`. This differs from each model's unchanged native inference source. Later report commits are not execution sources.

The75k controller is RUNNING PID428366, bootstrap428365, registration `15c99d9ccd30c48773297604c6937f8d3c507371164fb6729d33afb5d8bce66a`. Artifacts: `/mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_75k_20261007_v1`.

The finite future observer is RUNNING PID428393, bootstrap428392, registration `b1aee9f28642e344c4806b80ae1f960956d3851dc1f14efe604c2f586d3520d2`. Artifacts: `/mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_80_90_100k_20261007_v1`. It contains exactly nine tasks: each of the three arms at80k,90k,100k. It is independent of all earlier C/S observers; no old controller was restarted.

The observer polls every five seconds and hard-links only COMPLETE exact checkpoint files into external immutable snapshots before rolling GC. It preserves all requested checkpoints while the75k evaluation runs. After75k completes, the queue evaluates each ready arm independently, without waiting for the other arms or for training to finish. One subsequent checkpoint evaluation uses the four GPU slots at a time; hashing/loading/scoring cannot block the fast preservation loop. Source run, copied identity, registration, actual step, every file size and complete checkpoint content hashes are checked before export. Missing steps are recorded as MISSED_CHECKPOINT, never silently substituted. A failed task retains its artifacts and does not turn the other tasks into completed results.

Resource policy is unchanged: only physicalGPU4–7 on `training-vla-zt3`, UUID checked immediately before launch; export rank mapping is `[7,4,5,6]`. Existing GPU/research leases are respected. No unrelated job or pressure process was signaled, and physicalGPU0–3 are excluded. CPU scoring uses the already qualified canonical host, at most two16-worker scorers during75k and one16-worker scorer for subsequent single-arm jobs; nested BLAS/OpenMP threads are one. Live inventory found128 available CPU-affinity cores and approximately13 runnable-load units at registration.

Every evaluation preserves the existing50k protocol: full canonical12,146scenes/136logs, FP32 optimizer-master reconstruction, FP32 compute, TF32off, original10-step Euler, one executed ego candidate, scene-bound samplingseed42. W is retained; all auxiliary heads and GT/teacher inputs are removed. Official NAVSIMv1 full-precision caches retain all traffic categories. No scorer, alternate environment, best-of-N, auxiliary-loss ranking or Navtest-driven training selection is introduced.

The applicable independent full official CPU replay qualification is reused, not rerun or claimed as a new full per-checkpoint reference. Each new bank additionally receives eight fresh distinct-log checks through the unmodified official submission evaluator, all seven score fields within the existing1e-8 raw tolerance, plus full token/log, shard, receipt and content validation. PDMS, NC/DAC/TTC/EP/Comfort/DDC, exact zero-score count, failures and egoADE/FDE will be accepted only after these checks finish.

Finite budgets:75k maximum48 evaluationGPU-hours; subsequent nine tasks maximum16GPU-hours each,144 total. Combined limit192GPU-hours; the future observer window is48hours from registration. The completed same-protocol50k three-arm evaluation consumed14.573910GPU-hours, suggesting about58.3GPU-hours for all twelve new evaluations before contention/failure overhead. This estimate is not a completion promise. Model loading, export and saves are metered; evaluation optimizer updates are zero. Original training remains RUNNING. Checkpoint hard-links are outside Git and keep the optimizer/RNG files needed for correct FP32 reconstruction.

Nineteen targeted queue/compatibility tests passed, including exact-step retention surviving simulated trainer GC, missing-step rejection, independent preservation after a corrupt task, STOP behavior, duplicate observer exclusion, foreign-run rejection and completed single-arm resume without duplicate exports. Both real CLIs were executed; the75k registration verified all three actual full checkpoints, and the nine-task observer registration and WAITING states exist on disk. This is startup evidence, not learning or PDMS evidence.

Recovery commands below reuse the frozen source and artifacts. Run them only after verifying the corresponding existing controller/observer has exited; a live owner holds the exclusive lease.

```bash
cd /mnt/project/VLA-Drive-planning-interface-navtest75-source-c9e1010
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.evaluate_fixed_navtest run --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_75k_20261007_v1/registration.json
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.watch_navtest watch --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_80_90_100k_20261007_v1/registration.json
```

`STOP_SCHEDULING` in the future artifact root prevents new evaluations while preserving incoming checkpoints. Failed-task recovery is explicit through the observer's `task --registration ... --task A_ACTION_080000` entry, after inspecting retained child evidence and existing leases. Active workers are never terminated by this script. Deadline/budget pauses and incomplete results remain visible.

Aggregate startup evidence: [NAVTEST_AV_75_TO_100K_STARTED_20261007.json](evidence/NAVTEST_AV_75_TO_100K_STARTED_20261007.json). Prediction banks, private scene/log rows, weights and metric caches remain outside Git.
