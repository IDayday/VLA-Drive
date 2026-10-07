# A/V exact50k Navtest results, 2026-10-07

All three requested new checkpoints are COMPLETE. Each covers all12,146 canonical scenes/136 logs with0 inference/scoring/ego-metric failures. These are Navtest scores; development results are separate. All new models and matched old controls use original GT training labels, not the optimized trajectory campaigns.

| Arm | Meaning | PDMS | Zero scenes | Ego ADE m | Ego FDE m |
|---|---|---:|---:|---:|---:|
| S0 | A_W: frame sequence, MAE reads W | 88.243106 | 650 | 0.914155 | 1.885540 |
| A_NO_MAE | Same frame-sequence recipe, no MAE loss | 88.879682 | 559 | 0.902930 | 1.880331 |
| A_ACTION | Same frame-sequence recipe, MAE reads action queries | 88.373808 | 636 | 0.924638 | 1.905882 |
| S2 | V_NONE: video, no GT action condition | 88.329542 | 639 | 0.913501 | 1.885755 |
| S3 | V_MEMORY: video, action memory only | 88.503739 | 625 | 0.907363 | 1.889075 |
| V_QUERY | Video, action memory plus time-query injection | 88.713521 | 603 | 0.935463 | 1.927868 |

Matched comparisons retain scene weighting and bootstrap136 log clusters with10,000 draws/seed20260928; all comparisons are within trainseed42/inference seed42. PDMS differences and intervals below are percentage points, not relative percentages. These are exploratory midpoint measurements.

| Comparison | Delta PDMS points | Paired log95% interval | Role |
|---|---:|---|---|
| A_ACTION-S0 | +0.130701 | [-0.205622, +0.469279] | primary |
| A_ACTION-A_NO_MAE | -0.505874 | [-0.895061, -0.149931] | primary |
| V_QUERY-S3 | +0.209782 | [-0.155365, +0.588574] | primary |
| A_NO_MAE-S0 | +0.636576 | [+0.193464, +1.116300] | supplementary_exploratory |
| V_QUERY-S2 | +0.383979 | [-0.013817, +0.800991] | supplementary_exploratory |

A_NO_MAE has the highest mean of the three new models at50k. Moving MAE supervision to action queries improves the mean over oldS0/A_W slightly, but is below A_NO_MAE. This midpoint provides no positive independent MAE benefit; it does not establish that the teacher or whole research direction is ineffective. V_QUERY has a small mean gain over oldS3/V_MEMORY and oldS2/V_NONE, with uncertainty reported above. Retain the registered100k endpoint and separate training-seed confirmation; no recipe or test-seed selection is made from these results.

| Arm | NC | DAC | TTC | EP | Comfort | DDC |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 98.209287 | 96.303310 | 94.944838 | 82.412322 | 99.991767 | 98.435699 |
| A_NO_MAE | 98.287502 | 97.027828 | 94.796641 | 83.174780 | 99.991767 | 98.419233 |
| A_ACTION | 98.081673 | 96.550305 | 94.648444 | 82.688438 | 99.991767 | 98.295735 |
| S2 | 98.126955 | 96.451507 | 94.854273 | 82.474540 | 99.991767 | 98.275152 |
| S3 | 98.147538 | 96.591470 | 94.936605 | 82.710364 | 100.000000 | 98.316318 |
| V_QUERY | 98.221637 | 96.723201 | 95.101268 | 82.876225 | 99.991767 | 98.332784 |

Precision and scoring: FP32 optimizer-master reconstruction, FP32 inference, TF32off, original10-step Euler, one executed ego candidate, scene-bound samplingseed42. W144 is retained; auxiliary heads and all teacher/GT-future paths are removed. Every current-input/protocol/evaluator fingerprint and every scene metric-cache hash matches the reused controls. The canonical full-precision official NAVSIMv1 environment includes all traffic categories. No best-of-N, scorer or alternate PDMS cache.

Scoring verification distinguishes source qualification from per-model checks: the backend reuses a complete independent unmodified official submission replay with identical source/runtime fingerprints; each new model has eight fresh distinct-log official submission spot checks with maxerror0. No new full per-model official replay is claimed. All requested prediction shards and content hashes are complete.

| Arm | Training / native inference source | Exact checkpoint SHA256 |
|---|---|---|
| A_ACTION | f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882 | 0629886b6e48a3d88622cec742e8c0265a2fbef1a1bd92e50c00d2300cd11dbf |
| A_NO_MAE | f0ed4c3608b8ae78ddb1a0e7a5cca78f9a4b9882 | 0a30de13ae669c04be4d5c87994ac50e560206979cf33b5420fbaa87a78bed11 |
| V_QUERY | 7c5231c514f2a327dabf934fbe3bd9a30f9b7c6c | 1309e3964d1f78cb775fb30c98d00c43b2a1b9833e393b6f90f6c0615c3bd2d3 |

Controller source: `0f8257eda763527362444efdda575244e8897ab1`; frozen registration: `79b62d28ba61c441e6acca4cf3206c54dd8cf2420e1e0ffa674028b11c4cab65`. Analysis uses the existing paired-log implementation from `a839fdcc209a3221697477dab7f474c3a7ed14f0`; later report commits are not training sources.

GPU inference used only authorized physicalGPU4–7 on training-vla-zt3 under UUID leases, preserving existing tasks. CPU scoring used the canonical host. Evaluation optimizer updates0.

| Arm | Evaluation GPU-hours | Peak allocated GiB per exporter | Peak reserved GiB per exporter |
|---|---:|---:|---:|
| A_ACTION | 4.869290 | 11.956 | 12.373 |
| A_NO_MAE | 4.868473 | 11.956 | 12.373 |
| V_QUERY | 4.836147 | 11.956 | 12.373 |

Total metered exporter cost: 14.573910 GPU-hours, including loading, export and saves. Lease queue wait is not GPU occupation; CPU scoring is reported separately in the evidence. The finite controller and bootstrap exited successfully. Do not restart it or create duplicate milestone observers.

Artifacts: `/mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_50k_20261007_v1`. All source score CSVs, prediction banks, checkpoint receipts, shard identities, official spots and private scene/log pairing CSVs remain there outside Git. Aggregate evidence: [NAVTEST_AV_50K_RESULTS_20261007.json](evidence/NAVTEST_AV_50K_RESULTS_20261007.json). Startup/history evidence is preserved.

The completed run can be inspected idempotently with the existing command; it does not require a new export:

```bash
cd /mnt/project/VLA-Drive-planning-interface-navtest-source-0f8257e
/root/miniconda3/envs/ddp/bin/python -u -m tools.planning_interface_transfer.evaluate_fixed_navtest run --registration /mnt/project/planning-interface-transfer-artifacts/20261005/navtest_AV_50k_20261007_v1/registration.json
```
