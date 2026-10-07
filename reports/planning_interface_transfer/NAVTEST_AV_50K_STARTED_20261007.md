# A/V exact50k Navtest evaluation, 2026-10-07

The user explicitly requested complete Navtest for A_ACTION, A_NO_MAE and V_QUERY at the exact50,000-update checkpoint, using vla-zt3. Previous91.3684 V_QUERY@50k is a development score and is not a Navtest result. No A/V Navtest score is available in this startup record.

All three checkpoints were hashed and registered, and the finite background controller is actually running. Native A_ACTION export PID416039 acquired physicalGPU7 on training-vla-zt3; the other immutable scene partitions wait for physicalGPU4–6, currently leased by an existing S0@70k evaluation. Existing training/evaluation processes and sources are preserved. The prior GPU4–7 authorization is enforced using UUID-bound leases.

| Model | Exact updates | Native inference source | Checkpoint SHA256 |
|---|---:|---|---|
| A_ACTION | 50000 | f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882 | 0629886b6e48a3d88622cec742e8c0265a2fbef1a1bd92e50c00d2300cd11dbf |
| A_NO_MAE | 50000 | f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882 | 0a30de13ae669c04be4d5c87994ac50e560206979cf33b5420fbaa87a78bed11 |
| V_QUERY | 50000 | 7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c | 1309e3964d1f78cb775fb30c98d00c43b2a1b9833e393b6f90f6c0615c3bd2d3 |

Controller source: `0f8257eda763527362444efdda575244e8897ab1`; registration identity: `79b62d28ba61c441e6acca4cf3206c54dd8cf2420e1e0ffa674028b11c4cab65`. Independent immutable controller checkout: `/mnt/project/VLA-Drive-planning-interface-navtest-source-0f8257e`.

Protocol: complete canonical12,146scenes/136logs; original full-precision metric cache; NAVSIMv1; FP32 optimizer-master reconstruction and FP32 inference; TF32off; original10-step Euler; one executed ego candidate; scene-bound samplingseed42. Retain W and strip all auxiliary heads. No DINO, MAE, GT future or auxiliary action condition enters inference. CPU scoring runs on the canonical host with at most two16-worker tasks, nested BLAS threads1. Existing full independent official CPU replay qualification has exact matching runtime/source fingerprints; eight distinct-log official submission spots and full population/content-hash checks are required for every result.

This is a finite3-checkpoint request, capped at48evaluation GPU-hours with resumable immutable scene partitions. No training updates, new automatic milestone observer, model selection or best-of-N. Partial results cannot be reported as complete Navtest. Aggregate/per-factor scores, zero counts and original-GT ego ADE/FDE will be written to each arm’s result.json.

Artifacts: `/mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_50k_20261007_v1`. Actual launch and frozen registration are retained there; large checkpoints, scene trajectories, caches and private rows stay outside Git.

Resume only after checking the controller has exited; the exclusive controller lock rejects duplicates:

```bash
cd /mnt/project/VLA-Drive-planning-interface-navtest-source-0f8257e
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.evaluate_fixed_navtest run --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_50k_20261007_v1/registration.json
```
