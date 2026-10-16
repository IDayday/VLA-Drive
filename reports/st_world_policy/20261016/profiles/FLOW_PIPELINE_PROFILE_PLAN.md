# C-line full-pipeline efficiency: one bounded proposal

Status: **PROPOSED, NOT LAUNCHED**. This is an independent Systems review, not a formal training approval. It preserves the original S0 action-only C1 FPO definition; it does not apply the unrelated Query/IQE runtime or the S-P2 throughput result to C1.

The source inspected was Lead revision `f29965c76a79e750e0b109fec902e680f757db16`, with ongoing FP32-master/checkpoint fixes. The Flow owner explicitly confirms that those fresh-master changes have no final validated SHA yet, and that real production MC chunk5 update/resume has not been demonstrated. Accordingly this proposal is **not** a valid `REGISTRATION.json`; it cannot launch until final source, exact data manifest, environment and a fresh resource lease are pinned. No remote resource was acquired or existing task stopped.

## What the actual C1 smoke establishes

Run: `/mnt/project/st-world-policy-20261016/smoke/flow_C1_real32_v1/runs/C1_real32_updates2_v1`, source `73d1be9b42b88f76a1f85e6e823158b80d0d1e6c`. It used one GPU, global scene batch2, microbatch1 and MC chunk1. The pool contained32 real train scenes, but two updates consumed only4 scene presentations and16 actual rollout trajectories. Three scene groups had zero centered advantage and one group had a real nonzero comparison. That is a correct observation about this smoke, not evidence of full-population reward diversity or formal efficiency.

| Measured inner region | Update1 s | Update2 s |
|---|---:|---:|
| Data | 2.5446 | 0.6824 |
| Shared encoder | 0.8380 | 0.3712 |
| Old snapshot plus rollout | 2.0448 | 2.0158 |
| Official CPU feedback | 6.9346 | 1.2283 |
| Old CFM MC50 | 4.9718 | 5.1141 |
| Current BC/FPO, backward and optimizer | 15.8024 | 15.7232 |
| Inner complete update | 33.1416 | 25.1384 |

Current CFM/backward plus old CFM dominate the two observed steps. First-call CPU scorer initialization is conspicuous. The existing update timer ends before `all_gather_object`, JSON logging and atomic status writing, so it is not the requested whole-update wall metric. Neither two samples nor mixed setup/steady values establish sustained throughput.

The run saved an11,473,245,232-byte native payload. Native save timings were10.2837s at step0,27.1969s at step1,25.0230s at step2 and24.5307s for its budget boundary. These include the diagnostic `save_every=1` pattern, which is not proposed for the timing candidate. Peak reserved memory was23,855,104,000 bytes. The old smoke is preserved; the pending master fix must not relabel its rounded step0 initialization as newly verified.

## One candidate, no topology search

| Item | Locked proposal |
|---|---|
| Cell/backend | C1 / original BF16 execution + FP32 master/Adam + ZeRO2, deterministic and TF32-off |
| GPUs/global scene batch | 8 /32 |
| Scene microbatch/accumulation | 1 /4 per rank |
| Rollouts/Euler | 4 actual trajectories per scene /10 original Euler steps |
| FPO samples | 50 coupled old/current MC samples per action |
| MC chunk | 5, a same-objective execution batching candidate requiring real numerical/resume evidence |
| BC | Original repeat8, dropout and weight1; policy weight1 |
| LR | Original4096-update cosine, warmup100, lr1e-5 to5e-7; no short-run cosine restart |
| Adam | betas0.9/0.95, eps1e-8, wd0.001, clip1 |
| CPU | 4 global official scoring workers,2 ordered data threads per rank, Torch2, BLAS1 |
| Snapshot/reuse | Old snapshot refreshed every optimizer update, feedback reuse1 |
| Velocity checkpointing | Enabled; no language gradient checkpointing because the shared encoder is frozen |
| Native saves | Initial and final; `save_every=512` during the bounded profile |

This is one topology with a16-update initial screen followed, only if justified and separately registered, by a fresh240-update timing run. **Total candidate budget is256 new updates**, not32+240. The long run uses two20-warmup+100-measurement windows, exactly the established protocol. PROFILE checkpoints cannot initialize the later formal run. Short and long identities must be distinct and the long run must start from the same valid bound S0, not from the screen's learned weights.

Each update has32 original GT scenes,256 BC noise presentations,128 actual rollout actions,6,400 old MC points and6,400 current MC points. MC points are not128×50 new simulated trajectories. Official primary feedback calls are2,048 for the16-update screen and30,720 for240 updates; one explicitly bounded retry reserve raises the respective upper bounds to4,096 and61,440.

Both candidate JSON configs passed the current strict `flow_training_config.validate_config` on2026-10-16 with no GPU/model execution. Schema validity alone does not make a run launchable. Exact config copies and proposal are indexed in `MEASUREMENT_EVIDENCE.json`.

## Real64 data, fixed before new rewards

Use the completed256-entry/199-log official **training** cache receipt:

`/mnt/project/flow-safe-improvement-20261013/cache_builds/diagnostic_random_v1/receipt.json`

SHA256: `f8fee35f08a47db1b6e555ce95ee25997c44bc707cd844e0d37690d784dd95e1`.

The existing `train_flow_policy prepare` entry selects by `seeded_hash_before_any_scores`, seed42. Its first64 contains the earlier fixed first32, without selecting scenes according to their new rewards. One previously seen nonzero group does not guarantee new rollout noise will produce nonzero advantage. The short screen must report exact nonzero-group counts, reward ranges, clipping, actual policy gradients and master updates. If it has no useful comparison signal, report that limitation; do not synthesize rewards, alter the4-rollout definition, or silently reselect favorable scenes.

The checked real preparation CLI is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.train_flow_policy prepare --base-binding /mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json --data-root /mnt/project/ddp-foresight-artifacts/20260928/student_train_v1 --cache-receipt /mnt/project/flow-safe-improvement-20261013/cache_builds/diagnostic_random_v1/receipt.json --navsim-root /mnt/project/VLA-Drive-st-flow-smoke-runtime-20261016/navsim_v1.1/navsim --output /mnt/project/st-world-policy-20261016/audits/systems/flow_pipeline_profile_candidate_v1/real64 --scope PROFILE --max-scenes 64 --seed 42
```

Run only from the final clean immutable runtime after Lead binds it; this command is **provided, not executed**. It hashes original observations/ego and canonical contexts without inference or scoring. All later training must use that exact resulting manifest and its unchanged score-protocol identity. This64-scene workset is sufficient for controlled compute and true-feedback profiling, not complete training-population coverage or cold-I/O qualification.

## Measurement and stop conditions

Short wall cap:1,800s. Proposed long cap:10,800s. A low-confidence estimate is100–200 minutes for240 steps (13.3–26.7 GPU-hours), extrapolated from the one-GPU chunk1 smoke; it is **not a measured speed**. Chunk5,8-rank accumulation and4-worker CPU throughput can substantially change it. If the measured short cost plus setup/save projects beyond the long cap, stop with `BLOCKED_COST` and obtain a new explicit budget decision. Do not reduce MC50, rollouts, Euler steps, BC repeats, scene exposure, objective weights or the learning-rate horizon to fit a timer.

Before long timing, verify the fixed-master code and native resume, finite losses/gradients/actual FP32 master updates, old/current matching chunk5, all original counts, frozen encoder retention and one real nonzero advantage group. The Flow owner reports an existing real-DiT toy/reference chunk5 gradient test; this does not replace real production update/resume evidence.

Timing must capture both per-rank inner regions and whole-wall completion boundaries. The smallest trainer wiring is existing `PipelineProfiler.begin_update` immediately before the first per-update stop/data collective, and `end_update` after status/logging, with one untraced tail boundary at the next update start. It must not insert extra synchronization inside every region in timing mode. An independent read-only status observer can cross-check rank0 completion intervals, but its50ms sampling precision is insufficient to replace exact all-rank wall instrumentation. Record allocations from launch to all-child exit, initial/final checkpoint and official-cache setup separately.

For a few diagnostic steps only, use the already present `flow.*` `record_function` regions with a bounded profiler; all throughput windows must be profiler-free. Record slowest rank, GPU allocated/reserved, CUDA work/overlap, CPU scorer calls/active processes/time, I/O counters and per-scene reward validity. CPU `RUSAGE_CHILDREN` only counts reaped children; capture live worker PID statistics to avoid pretending still-running scorer CPU time is zero.

The current pipeline waits synchronously for official feedback before old CFM. There is no measured GPU/CPU feedback overlap. Computing the already-bound old CFM while CPU scores are pending is a potential scheduling change, but it is **not implemented or required in this proposal**. Any later version must keep the fixed old-policy boundary, validate complete feedback before backward/optimizer commit and repeat its own correctness/timing review. Do not claim that latency has already been hidden.

## Registration and resource boundary

Final registration needs final source SHA/fingerprint, exact config/target/data/protocol/teacher-none hashes, fixed environment,8 actual GPU UUIDs, CPU/RAM/shared-space evidence and an unoccupied authorized-host lease. Preserve every GPU lock `/tmp/foresight_gpu_N.lock` and fresh PID/cwd/process checks. Lead dispatches only one job; real FM-transfer evaluations now occupying Tsinghua/rl-zt2 remain protected. No pressure release is delegated here.

`train_flow_policy train --registration <the new exact registration>` and its `--resume`/`--acknowledge-stop` forms are verified CLI interfaces. No executable final launch command or registration is fabricated while the source and lease are missing. The current private proposal is `/mnt/project/st-world-policy-20261016/audits/systems/flow_pipeline_profile_candidate_v1/PROPOSAL.json`, with `launch_allowed=false` and `actual_dispatches=0`.
