# A/V Navtest resource expansion and completed75k/80k results

All six requested75k/80k evaluations are COMPLETE:12146 scenes,136 logs, no missing or failed benchmark rows. Each result reuses the qualified official NAVSIMv1 backend and has eight new distinct-log official reference checks with maximum error0. These are trainingseed42, samplingseed42 fixed-checkpoint measurements, not five-run averages or final100k comparisons.

| Model |75k PDMS |80k PDMS |75k zero scenes |80k zero scenes |
|---|---:|---:|---:|---:|
| A_ACTION |88.87072694 |89.02944299 |580 |558 |
| A_NO_MAE |89.06341450 |89.33965719 |541 |510 |
| V_QUERY |89.00584454 |89.41242363 |570 |511 |

The pinned original S3@80k reference is89.21488181. No exact S3@75k reference is available; do not interpolate one. This report does not establish an isolated method advantage or training-seed stability.

## Deployed allocation

The user explicitly permits existing authorized resources and GPU sharing with trainers. Model inference is unchanged: FP32 optimizer-master loading, FP32 compute, TF32 off, original10-step Euler, one ego candidate, scene-bound seed42, current observations only. Auxiliary heads remain stripped and W remains present. Native bank world_size4 is preserved.

The finite resource companion is RUNNING, PID467428, source `23b66776efce7585ae76140c4e05667e4983a8bb`, immutable worktree `/mnt/project/VLA-Drive-navtest-expanded-source-v3`. It services only the six already registered A_ACTION/A_NO_MAE/V_QUERY90k/100k banks. It does not capture checkpoints, start another Navtest observer, change the training recipe or modify live native source.

| Arm | Existing native host / GPUs | Additional helper allocations |
|---|---|---|
| A_ACTION |training-vlawm-zt2 /4–7 |same host0–3; training-vlawm-zt3 /0–7 |
| A_NO_MAE |training-vlawm-zt /4–7 |same host0–3; training-vlawm-zt4 /0–7 |
| V_QUERY |training-vla-zt /4–7 |same host0–3; training-vla-zt2 /0–7 |

There are36 registered additional helper slots across six eligible hosts. Every launch rechecks UUID, free memory, GPU process ownership and inference leases. The other inventoried hosts currently lack verified ownership or sufficient headroom; no foreign job is evicted. New checkpoints are pending, so these36 slots are registered capacity, not36 currently active inferencers.

Training reservation leases on the three additional hosts were verified by owning PID and exact command-line SHA256. Shared evaluation uses its own exclusive per-GPU lease while preserving the trainer's reservation. The known trainer and every visible GPU process must still match at launch. No trainer, pressure manager or unrelated process was signaled. Actual FP32 native prediction parity was executed on each additional host while its trainer was active: all three maximum pose errors were0.

Helpers load and verify native inference while original exporters continue. After loading, only manifest-bound native evaluation writers are temporarily paused; missing scenes are partitioned disjointly, completed rows remain immutable, and original writers resume to verify rows and publish their original four completion receipts. Independent bounded recovery resumes these exact writers after a controller failure. Every helper first recomputes a retained native trajectory and must pass the1e-6 pose tolerance. Evaluation changes neither optimizer state nor training data.

## Current-task acceleration and cost

The first tail-expansion attempt requested28 GPUs. Four helpers acquired leases and passed exact native parity;24 declined because the existing training reservation held the exclusive lease. This was corrected through verified shared reservations in the new source. The original75k/80k banks completed during helper loading; no supplemental scene generation is attributed to this attempt. Its loading/parity cost was0.165588GPUh and is retained. Three subsequent cross-host shared-inference parity checks cost0.044367GPUh. No speedup is inferred from those checks.

V_QUERY75k received an independent32-worker official CPU scoring process, PID465931. Its output stays separate from the canonical16-worker resume cache. Both complete results agree exactly:89.00584454, zero failures, fresh official spot maximum error0. This additional CPU cost is logged rather than treated as free. Existing scoring outputs and qualification remain intact.

At the snapshot, native75k charged GPU time including loading and prior failed attempts is31.028028GPUh; native80k is22.298391GPUh. CPU work is logged separately. Supplemental runs have zero optimizer updates. The detailed checkpoint, CSV, source, result and execution identities are in [the aggregate evidence](NAVTEST_AV_RESOURCE_EXPANSION_20261007.json); raw scenes and weights remain outside Git.

## Actual commands and continuation

Existing observerPID443350 continues exact80k/90k/100k capture and evaluation;75k controllerPID444712 owns its completed results. Do not duplicate either owner. The allocation-only companion command is:

```bash
/root/miniconda3/envs/ddp/bin/python -u \
  /mnt/project/VLA-Drive-navtest-expanded-source-v3/tools/planning_interface_transfer/accelerate_navtest_tail.py watch \
  --plan /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_expanded_future_20261007_v1/plan.json
```

This command is already running and holds a singleton lock. Use it for recovery only after verifying the old companion has exited. Its finite list has six requested tasks and a24-hour registration window; each supplemental inference window is bounded at3600seconds with native continuation and fail-safe recovery. There is no added checkpoint-selection or Navtest-based optimization loop. Nine targeted partition, layout and existing parallel-execution tests passed.

The new resource implementation and reports are committed only to `feature/planning-interface-mae-action-query-20261005`; native A/V inference sources remain `f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882` / `7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c`.90k/100k scores and training-seed replication remain unfinished.
