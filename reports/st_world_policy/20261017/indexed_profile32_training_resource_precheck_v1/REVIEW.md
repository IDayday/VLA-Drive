# PROFILE32 / 64-update resource precheck

**PRECHECK_ONLY_NOT_TRAINING_ADMISSION.** This is a read-only local snapshot for the forthcoming exact launcher review. No output directory, claim, lease, registration, ledger entry, training process, or GPU allocation was created.

## Completed metadata preparation

Actual factory receipt SHA256: `6407b37cd7f457c57f60717c71ef47bcedc9e2d40c090e51239cdc2fcf8be6d1`.
It records COMPLETE, exit 0, 9.223396 seconds, sampled RSS 642,953,216 bytes, child logical `rchar` 3,020,067,533 bytes, zero GPU allocation, zero new teacher tensor bytes, no training, and no remaining owned child. Adding the reserved PCA payload and 16 MiB parent allowance gives 3,038,435,613 bytes, below 4 GiB. The counter's zero `torch_load_calls` is not proof of zero tensor reads: the factory legitimately uses the explicitly budgeted existing PCA.

The whole existing factory parent, including original evidence and the new PROFILE child, now occupies **151,552 charged bytes**. This is the root-inclusive after-receipt snapshot, distinct from the controller's 144,253-byte before-receipt snapshot. It remains within the original 32 MiB allowance. No new metadata reservation is needed; effective administrative plus ledger metadata remains 1,264 MiB of 4 GiB.

Actual training config SHA256: `e72f767f4dfdc349508befb64665bb11f4e407b8e670fc3a31dd983f8c0c7676`. It specifies one GPU, microbatch 1, global batch 2, the fixed 32-scene population, stop after 64 updates, the original 20,000-update schedule/warmup 1,000, max_seconds 1,500, and save_every 64. This is a repeated-population PROFILE, not full-population throughput or formal training qualification.

## Local resource facts

Snapshot time: Unix `1792225327.062954` (2026-10-17 08:22:07 UTC).

- Host `training-vla-zt-worker-0` is currently allowed by the actual dispatch policy and campaign authorization; policy SHA256 is recorded in `OBSERVATION.json`.
- GPU 0: A800 80GB, UUID `GPU-f679bff0-a4a2-ae4f-0b85-74c519b45a9c`, 0 MiB used, 0% utilization, and no reported compute process.
- Existing `/tmp/foresight_gpu_0.lock` has no matching active holder in `/proc/locks`. The reviewer did not acquire it. This observation is not a future lease; the launcher must check ownership again and acquire this exact lock before dispatch.
- GPU 1–4 each have about 13 GiB allocated and active utilization. Their tasks are protected. No signals, pause, pressure removal, or process reassignment was performed.
- Available RAM: 897,274,200,064 bytes. Load average was 20.02 / 19.72 / 19.94. These observations do not establish a per-job performance guarantee.
- Target `/mnt/project/st-world-policy-20261016/profile/world_indexed_provider_profile32_updates64_v1` does not yet exist.

## Checkpoint capacity

| Item | Bytes |
|---|---:|
| Existing full commitments, 2,103 GiB | 2,258,079,055,872 |
| Additional PROFILE allowance, 80 GiB | 85,899,345,920 |
| Combined full commitments, 2,183 GiB | 2,343,978,401,792 |
| Additional free floor, 1 TiB | 1,099,511,627,776 |
| Required free capacity | 3,443,490,029,568 |
| Actual filesystem available bytes | 7,340,766,330,880 |
| Capacity above that requirement | 3,897,276,301,312 |

This conservatively retains the entire original nine-run 1,983 GiB and prior 120 GiB smoke allowance even after files have been written. No old checkpoint is deleted or subtracted. The new 80 GiB is separate from the existing factory 32 MiB and the cache ledger's 768 GiB cap. It is an advisory capacity commitment, not an OS quota; actual shared quota remains **unverified**.

The handoff's observed native checkpoint size is 37,365,928,655 bytes. One normal checkpoint plus one simultaneous partial totals 74,731,857,310 bytes, leaving 11,167,488,610 bytes within 80 GiB for metadata and ordinary allocation overhead. The exact launcher must preserve that two-slot plan, check space before writes, and not silently add additional saved checkpoints.

The existing 2,103 GiB receipt expires at **08:44:09 UTC** and had only 1,322 seconds remaining at this snapshot. It cannot cover a fresh 1,500-second PROFILE plus checkpoint completion and abnormal cleanup. Before any training admission, issue a fresh conservative receipt covering **2,183 GiB + 1 TiB floor** throughout the exact launcher's full outer/cleanup window. Do not relabel the old receipt's timestamp or count future reservations as already consumed bytes.

## Exact launcher review still needed

Use the established one-run/one-GPU controller and two CPU threads, no background cache construction or increased CPU scoring concurrency. The prior two-update controller hard-codes a GPU 1 lock and a 1,020-second timeout; those cannot be inherited accidentally for this GPU 0 / 1,500-second PROFILE. The final launcher must bind the actual config, descriptor, source, GPU UUID, correct lock, fresh capacity receipt, and an outer save/cleanup window consistent with the 64-update budget. Check for an existing identical run/output before launch and retain one-shot failure evidence.

The factory's 4 GiB read cap does **not** apply to native model training: the native checkpoint alone is tens of GiB. Keep the previous bounded, single-process data/provider route and original RAM cache limits; report actual startup/checkpoint I/O and supplier waits. Do not claim an explicit training I/O bandwidth cap if the reused controller only limits concurrency/threads. No new concurrency or data scope is approved by this precheck.

No exact launcher was available for this report, so training admission remains **WAITING_LAUNCHER_AND_FRESH_COMMITMENTS**, despite sufficient observed capacity and an idle GPU. This is a concrete pending artifact, not a new model, numerical, or scientific gate.
