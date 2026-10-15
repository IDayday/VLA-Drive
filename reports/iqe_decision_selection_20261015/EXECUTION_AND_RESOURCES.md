# Execution, budgets and resource closeout

Read-only closeout time: 2026-10-15T01:36:06.310070+00:00. All nine permanently allowed hosts were verified. There were zero study training, evaluation, benchmark or continuation processes. No withdrawn host was contacted. No Base/G2 or unrelated process was signaled. The previously waiting controller correction is separately recorded; it had no scientific child process.

Parallel head training ran on training-rl-zt2 with one A800 80GB per head, cached frozen contexts and bounded CPU readers. Independent fitting/recovery and current-state preparation overlapped where capacity permitted. Native two-GPU NCCL checks ran on allowed local training-vla-zt. Final timing used the same rl-zt2 GPU6 sequentially, preserving a common device rather than changing devices to finish the comparison sooner. Existing workloads on other GPUs remained untouched.

| Run | Optimizer updates | Single-GPU elapsed seconds | Scene presentations |
|---|---:|---:|---:|
| diagnostic/M_scene | 3000 | 956.222 | 384000 |
| diagnostic/M_set | 3000 | 1098.169 | 384000 |
| refit/M_scene | 2524 | 899.695 | 323072 |
| refit/M_set | 631 | 293.157 | 80768 |
| overfit/M_scene | 300 | 86.935 | 38400 |
| overfit/M_set | 300 | 98.890 | 38400 |
| P_current | 2000 | 530.250 | see probe sampler receipt |

The registered scientific/fitting runs executed 11755 updates. Their single-GPU elapsed totals 3963.318 seconds (1.1009 GPU-hours). This includes native validation/checkpoint time but excludes context preparation, loading, post-training fit reporting, inference, source audits and separate regression/recovery checks; it is not an overall utilization integral. The two real recovery checks separately executed eight diagnostic updates each.

All data, step and model budgets stayed fixed. Both refits started from their original declared random initializations; full-D_S exposure conversion determined 2524/631 updates before external evaluation. No spare GPU was used to extend a completed endpoint, restart a held study or add another candidate. There is no pending automatic training/retry controller for this study.

Actual shared-inference benchmark elapsed wall was 668.0069720745087 seconds including loading and warmup. Per-mode steady wall/latency samples, call counts and GPU memory are in LATENCY_AND_PARAMETERS.md. Every stage exit receipt is archived in evidence; they record commands and resource routing.

The original development workspace had unrelated user changes and was left in place. This isolated worktree contains only decision-selection additions and its reports. Protected Base/adapter/expert/native trainer sources are byte-identical to audit parent1375ad2 as recorded in evidence/PROTECTED_SOURCE_COMPATIBILITY.json. Weights and data are referenced, not committed. No push, Navtest or ACTIVE update was performed.
