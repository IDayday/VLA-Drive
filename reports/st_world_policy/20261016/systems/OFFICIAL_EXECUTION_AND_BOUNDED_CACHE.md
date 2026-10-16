# Official execution and bounded teacher storage

The user selected official DriveDreamer-Policy execution and rejected multi-TB cache prebuilding. This change applies to new registered ST runs. Existing parent weights, running unrelated experiments, canonical evaluation and old evidence remain intact.

The inspected official source is `youngzhou1999/DriveDreamer-Policy@f35fe7836625434dc4f67176a3967bf42b0d446d`. Its actual launch uses BF16 execution with FP32 Adam masters/moments, Torch AdamW handed to ZeRO2, no offload, FlashAttention2 and 500,000,000-element communication buckets. The apparent gradient-checkpointing config is not consumed by that official path; the effective setting adopted here is false. This is not whole-model FP32 training or low-precision Adam state. The ST research phase retains its explicitly declared 20k/4096 horizons and study losses; it does not claim to reproduce the complete official 100k experiment.

## Storage decision

This campaign has a **768 GiB total cache cap** including registered old-cache reservations, new payload, indexes and staging, plus a **1,024 GiB filesystem free-space floor**. The current conservative reservation is 692,754,203,494 bytes (about 645.18 GiB). The floor is an admission safeguard, not an allocation or a verified user quota. Checkpoint storage is budgeted separately. No old cache/checkpoint is automatically deleted.

These are frozen DINO teacher targets, not the historical multi-TB Qwen visual-output cache. The current three-variant train target estimate is 629,335,619,712 bytes (about 586.11 GiB); metadata and temporary files are reserved above that. Additional time points/layers are not automatically prebuilt. They require a bounded plan within the same cap, or on-demand computation with an explicit execution identity. The storage decision does not change targets, preprocessing, sample order or tensor precision.

`cache_storage_budget.py` uses a campaign lock and durable byte claims. New registered writers require the policy, reserve output/staging/metadata before writing, refuse conflicting claims, and fail on cap/free-space violations. Existing immutable writers are honestly recorded as prebounded legacy reservations; they were not hot patched or retroactively claimed to use the new guard.

## Current evidence

- Official-runtime integration: 155 passed, 1 intentionally skipped opt-in test, 27.90 seconds. This includes inherited tests; it is not 155 new tests.
- Cache guard integration: 42 passed, 4.57 seconds. An earlier wrong test filename produced exit 4 and is retained as a failed invocation.
- Final pause/timing regression: 67 passed, 5.76 seconds.
- Independent final boundary review: 47 passed, 6.23 seconds, including six concrete counterexamples. Overlap between suites is not counted as additional coverage.
- Real guarded cache test: 256 scenes / 768 actual images, two CPU workers, 151,039,872 payload bytes; every FP32 tensor and mask equals the previously computed identical pooled targets bit for bit. Planning, derivation, finalization, full-record verification and equality checks took 31.329 seconds. It performed no new teacher GPU forward and is not a GPU cache-throughput result.
- Both official-execution real-image smoke runs completed two 8-GPU updates, global batch32 / micro4, with 64 scene presentations. All eight ranks have actual FP32 master and Adam moment receipts. C1 performed 256 valid official single-candidate evaluations; its frozen encoder hash was unchanged. These are engineering checks, not planning gains.

The two-step runs used `9c87d8fa`. Their pause-at-epoch-end exposed duplicate same-step native saves. Final `fdf25f99` checks the stop before advancing an unconsumed epoch and reuses a just-committed boundary by a rank-0 decision broadcast. Old duplicate files are preserved. A W checkpoint is about 37.37 GB, and a C checkpoint about 11.47 GB; avoiding duplicate writes saves real storage without reducing recoverable state.

New 240-update profiles at `fdf25f99` are running in separate namespaces. They use two 20-warmup/100-steady windows and do not contribute formal initialization weights. Their 64-scene compute working set does not establish full-population cold-storage throughput. Steady timing, current topology choice and the new Torch/FA2 full-model native-resume qualification remain pending in this snapshot; no formal approval is manufactured here.

All actual artifact paths and checksums are in the adjacent JSON. No model, private image, cache or raw process inventory is committed.
