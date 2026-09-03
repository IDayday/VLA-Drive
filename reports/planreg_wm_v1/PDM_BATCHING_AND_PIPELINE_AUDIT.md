# Exact PDM batching and formal training pipeline audit

## What changed

The training target remains the exact DrivoR/NAVSIM PDM computation. Proposals
are detached before scoring and every returned component is collected in the
original deterministic order. No target is approximated, cached across model
updates, or moved into the scorer network.

The execution graph now submits each rank's CPU PDM jobs before the fused
current-plus-three-future EMA vision forward. CPU PDM and GPU EMA work execute
concurrently; DrivoR loss waits only for the remaining tail. A failure in EMA,
submission, or loss resolution cancels outstanding futures before propagating
the original exception.

Each scene's 64 proposals were previously split into eight tasks. The formal
global-128 layout uses two 32-proposal tasks per scene, while retaining eight
CPU scorer processes per rank. This reduces repeated LZMA/pickle metric-cache
loads and process scheduling without changing the PDM implementation.

EMA parameter updates now process only trainable Q/V LoRA and planning-adapter
tensors with grouped foreach operations. The 304,012,288 immutable frozen
vision parameters are not repeatedly multiplied and added; their EMA is
mathematically the same constant. The updated EMA subset contains 3,494,144
parameters and takes 5.68 ms per step in the final benchmark.

The frozen LLM planning path requests only its final hidden state rather than
retaining a 28-layer hidden-state tuple. LLM weights remain frozen and the
semantic-to-vision gradient boundary is unchanged.

## Existing batch scorer utility

The existing utility at
`/mnt/project/DriveVLA-M0/tools/navsim_candidate_relative_audit/score_candidates.py`
is exact and useful for fixed proposal banks. Its prior audit covered 500
scenes and 6,000 candidates with zero batch-versus-single error, deterministic
repeat output, and preserved ordering.

Training already calls NAVSIM's batched `PDMSimulator.simulate_proposals`
inside every task. The reusable part of the utility's idea is therefore task
coarsening, not precomputing training scores: proposals change every optimizer
step, so their PDM targets cannot be cached safely. The utility remains useful
for post-training all-64 Navtest evaluation and repeated checkpoint analysis.

A representative eight-scene/rank, 64-proposal, eight-worker, three-repeat
microbenchmark produced identical outputs for every partition count:

| Partitions per scene | Wall time (s) |
|---:|---:|
| 1 | 0.621231 |
| 2 | 0.613940 |
| 4 | 0.680026 |
| 8 | 0.826212 |

Two partitions were locked because they were marginally fastest in this
workload and preserve more task-level load balancing than one full-scene task.
Once overlapped with EMA vision, the difference between one and two is not a
material step-time term.

## Measured effect

The final 16-GPU, batch-8/GPU run completed 20 warmup and 300 timed optimizer
steps with the full correct-future PlanReg-WM graph:

- 27.3426 samples/s and 0.21361 optimizer steps/s;
- 4.5910 s median and 4.8809 s p90 step time;
- 69.127 GiB peak allocated and 72.752 GiB peak reserved;
- 99.12% mean GPU utilization;
- no OOM, deadlock, non-finite loss, or non-finite gradient.

The original synchronous no-checkpoint global-128 probe measured 20.9972
samples/s and 6.0435 s median step time. The optimized path is 30.22% faster
in sample throughput and reduces median step time by 24.03%. Compared with the
previous locked global-96 run, throughput rises from 20.4495 to 27.3426
samples/s (+33.71%). Estimated 27-epoch time falls from 37.884 to 28.334 hours
per initialization.

Timed losses remained finite. Across the 300 measured steps, mean total,
trajectory, scorer, raw WM, and weighted WM losses were 17.1412, 15.2105,
1.92664, 0.23444, and 0.00405. This short throughput run is a stability gate,
not a convergence or final-quality claim.

## Remaining bottlenecks

The median DataLoader wait is 5.58 ms and median explicit all-reduce is 1.20
ms, so neither input preprocessing nor TCP/NCCL synchronization is currently a
meaningful wall-time bottleneck. CPU utilization is only 17.74% because exact
PDM now fits underneath GPU work; increasing generic DataLoader workers would
mostly add memory/process pressure.

The largest exposed compute stages are fused EMA vision (1.974 s), backward
(1.087 s), frozen LLM (0.980 s), and student vision (0.505 s). Safe future
engineering candidates are static-shape/compile experiments for the EMA vision
path and an independently audited faster frozen-LLM attention backend. They
must pass forward/backward parity and a 300-step stability gate before changing
the formal lock. Reducing future horizons, future-teacher frequency, proposal
count, or caching dynamic VLM/EMA features would change the scientific method
and is not an accepted speed optimization.

The raw benchmark file retains one temporary `optimizer_step_time` key whose
hook boundaries crossed iterations. It is invalid, excluded from this audit,
and no longer emitted by the formal code.
