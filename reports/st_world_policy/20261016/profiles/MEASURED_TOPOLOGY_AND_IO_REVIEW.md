# Actual topology and shared-I/O review

Independent reviewer: `/root/st_systems`. These are bounded PROFILE observations, not a formal approval or scientific result. No active Base, unrelated task, or pressure worker was signaled by this reviewer. Every PROFILE weight is ineligible for formal initialization.

## Scope and identities

The model is the original DiT/Flow S0 continuation with S-P2 current spatial supervision, not the former Query/IQE scorer experiment. Original DiT, repeat8, three views, input preprocessing, BF16 execution, FP32 master/Adam, ZeRO2, deterministic/TF32-off flags and activation recomputation remain enabled. Each task has global batch 32 and the full 20,000-update scheduler; `stop_after=240` limits this invocation without shortening cosine or warmup.

Timing source is `58f85044b01f1336b8063a36d34fb16623dfdedb`, immutable worktree `/mnt/project/VLA-Drive-st-profile-runtime-20261016`. The separately executed trace uses `ab2184bf66894c24b3d911f0e01c9b3c986d956d`, adding only instrumentation to that source. Explicit registrations, config, input, teacher, environment and GPU lease hashes reside in each run directory.

The compute workset is exactly 64 real scenes, repeatedly sampled for 240 updates / 7,680 presentations per task. This is not a full-population I/O or convergence experiment. Its 384-channel PCA is fixed from the previously registered train-only statistics. No new PCA fit, sample reduction, or loss removal was used to claim faster throughput.

## Same-host topology comparison

Host `training-vla-zt-worker-0`, 8 A800 SXM4 80GB with NV8 interconnect. Two timing windows are updates 21–120 and 141–240, preceded by their 20-update warmups. Numbers below use whole-update wall on the slowest rank, including data, forward/backward, reductions and synchronous logging; periodic checkpoint time is separate.

| Task | microbatch / accumulation / ranks | Mean update s | p50 s | p95 s | Peak allocated bytes |
|---|---:|---:|---:|---:|---:|
| 8 GPU single task | 4 / 1 / 8 | 2.045829 | 2.025881 | 2.238474 | 13,353,963,008 |
| Concurrent 4 GPU A | 2 / 4 / 4 | 4.536525 | 4.524525 | 4.773552 | 18,564,829,696 |
| Concurrent 4 GPU B | 2 / 4 / 4 | 4.539314 | 4.526995 | 4.776285 | 18,564,829,696 |

Eight-GPU window throughput: 15.603141 and 15.663159 scenes/s (combined 15.633092). Actual concurrent 2×4 throughput: 14.157260 and 14.032241 scenes/s (combined 14.094473). Its windows overlap for 451.489284 and 455.905312 seconds. This was two real simultaneous jobs, not a solo result multiplied by two. The measured 8-GPU topology is approximately 10.9% faster in steady node throughput. Smaller microbatch2 is a qualification of this conclusion: it does not prove every possible 2×4 configuration is slower. A separately approved 4-GPU micro4×accum2 32-update SHORT_SCREEN is reported separately; it cannot establish concurrent 2×4 sustained throughput.

Each job completed exactly 240 updates and all losses/gradient norms are finite. Global scene multisets match between topologies at every update. A and B, with identical 4-GPU topology, have identical loss streams. Cross-topology rank partition and random streams differ; no bitwise-continuation claim is made.

## Complete allocation and checkpoint cost

| Task | Whole allocation s | Final native checkpoint s | Checkpoint payload bytes | GPU allocation seconds |
|---|---:|---:|---:|---:|
| 8 GPU | 738.000153 | 45.244939 | 37,366,164,082 | 5,904.001221 |
| 4 GPU A | 1,321.171401 | 52.921877 | 37,366,028,042 | 5,284.685605 |
| 4 GPU B | 1,317.651071 | 49.786024 | 37,366,028,042 | 5,270.604284 |

These allocations include loading, optimizer construction, warmups, measured windows, final save/checksum and teardown. Two concurrent four-GPU jobs amortize their independent setup concurrently; their complete allocation must not be conflated with steady update throughput or compared to one eight-GPU job without normalizing presentations. At the unchanged save interval of 1,000, the observed eight-GPU save would add approximately 0.045245 s/update; this is an amortization estimate from a single final save, not an observed periodic mean. Twenty 20k-run checkpoints would occupy approximately 747.3 GB per run before extra milestones/atomic temporary copies. No cleanup was performed.

The existing training `slowest_rank_seconds` is an inner timer and omits later reductions/logging. Its eight-GPU steady mean is about 1.96045 s, versus 2.04583 s whole update. It is not interchangeable with the performance wall metric.

Rank peak RSS sums are 211.05 GB for eight ranks and 225.53 GB across both four-rank jobs. These are sums of per-process historical peaks, not simultaneous PSS or actual resident unique bytes. CPU self accounting includes the threaded loader; waited-child accounting excludes any unobserved live child process. This runner uses thread loaders, not hidden multiprocessing DataLoader workers.

## Independent 256-scene shared-I/O measurement

This test uses 256 unique current observations not recycled from the 64-scene timing workset. Eight actual CPU processes consume disjoint `rank::8` indices once, each with two ordered loader threads, four prefetched records, two Torch threads and BLAS1. It executes the real image decode/current ego/native teacher/FP32-PCA Dataset path and checks exact scene coverage and finite targets.

All eight processes exited zero. Common union wall: 4.349727 s; actual aggregate: **58.854269 scenes/s**, rather than a multiplied single-process estimate. CPU user 48.718640 s + system 5.329884 s across processes; ~12.43 effective CPU cores during this window. Sum process peak RSS upper bound: 5,906,400 KiB. `/proc/self/io` deltas sum to 11,041,639,467 logical read characters and 4,681,236,480 read bytes. These counters do not uniquely measure Lustre wire traffic. All 256 IDs are unique, with exact disjoint manifest coverage and no training sampler reorder.

The run may benefit from OS cache and does not claim physically cold storage. It supports a bounded current-only input capacity above the measured 15.63-scene/s consumer under this observation, but cannot establish full-population cold-read throughput or capacity while other jobs saturate Lustre. Full-population cache coverage and final formal source/manifest binding remain Lead/data review dependencies. CPU and GPU profile windows did not overlap this I/O test.

## Resource and qualification boundaries

Live inventory checked visible cgroup CPU quota, affinity, memory, NV8 topology, existing processes and shared `/mnt/project` Lustre mount. Container-visible CPU quota is unbounded and affinity has 128 logical CPUs; hidden ancestor limits are not observable. No finite cgroup memory limit was visible. Filesystem free space at the later Tsinghua check was 11,431,696,781,312 bytes. `lfs quota` is unavailable, so filesystem free space must not be described as a verified per-project quota.

Recommended measured topology for the currently audited compute workload is 8 GPUs / micro4 / global32. S-P3 W96 has a separate registered 240-update profile because it lengthens the language input. S-P0/1 keep W48 but change teacher/readout shape; they may reuse topology evidence only with actual cell-specific gradient/memory/cache checks, and must not inherit the S-P2 throughput number as their own measurement. Temporal/history and FPO are not covered by these timings.

Formal approval is not signed here. It must bind the final clean execution SHA, actual complete population and teacher manifests, resource lease and correctness/evidence reviews; these measured artifacts can support that concrete review.

Evidence: `/mnt/project/st-world-policy-20261016/audits/systems/TOPOLOGY_MEASUREMENTS.json`, three per-topology timing summaries, `sp2_pca384_concurrent_2x4_timing_summary.json`, `loader256_8proc_v1/{REGISTRATION,SYSTEMS_LAUNCH,SUMMARY}.json`, and per-rank results. Actual launch, resource UUIDs and native receipts remain under the corresponding `profiles/` run namespaces.
