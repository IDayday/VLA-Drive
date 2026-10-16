# S-P3 W96: actual bounded eight-GPU result

Lead launched the registered local S-P3 profile; Systems only observed and summarized it. Registration SHA256 `27fcf3da96653724eebdcac3c0d7c475c8d62f1d96c21742c5091df87d034966`, immutable runtime `ab2184bf66894c24b3d911f0e01c9b3c986d956d`. Full evidence root: `/mnt/project/st-world-policy-20261016/profiles/sp3_pca384_8gpu_local_timing_v1`.

It completed exactly240 optimizer updates /7,680 scene presentations, saved a complete native checkpoint and exited at the registered budget boundary. Result is `PAUSED / BUDGET_PAUSED`, not an unfinished or failed training step. All240 losses and gradient norms are finite and all actual FP32-master-update checks passed. No scientific evaluation or formal initialization eligibility follows from this PROFILE.

Configuration preserves original DiT repeat8, original training precision/backend, recomputation and the20k scheduler. Eight A800 SXM4 80GB GPUs, global batch32, microbatch4, current supervision/PCA384. S-P3 has96 W queries per view rather than S-P2's48; language padded sequence length was2074 versus1930. The sampled current supervision remains96 positions per view and its per-update denominator is3,538,944; the ego denominator is32. These are checked against all240 logged steps.

The64-real-scene sequence, rank partition and presentation order are identical to the earlier eight-GPU S-P2 run across all240 updates. That is a controlled compute workset, not cold storage or full-population coverage. The model capacity differs, so no S-P2/S-P3 numerical trajectory equivalence is claimed.

| Whole-update statistic | Steps21–120 | Steps141–240 | Combined200 updates |
|---|---:|---:|---:|
| Mean seconds | 2.073389 | 2.047207 | 2.060298 |
| p50 seconds | 2.058941 | 2.036261 | 2.045717 |
| p95 seconds | 2.202463 | 2.148420 | 2.188342 |
| Scenes/s over complete window wall | 15.426030 | 15.623249 | 15.524013 |

Peak allocated memory is13,512,680,960 bytes; reserved15,619,588,096 bytes. Rank historical peak RSS sums to250,497,949,696 bytes, an upper-bound sum rather than simultaneous unique RSS/PSS. The inner trainer timer averages2.013689s and slow-rank data-fetch field0.396996s; neither replaces the whole-update figures above. GPU/CPU inclusive regions are not added to claim a causal decomposition.

Whole allocation took759.978252s, equivalent to6,079.826020 allocated GPU-seconds. Final native save took46.381613s. Its receipted payload is37,371,153,202 bytes. At save-every1000, this one save implies an estimated0.046382s/update amortization; periodic saves were not repeatedly measured. All receipt-listed files exist with exact declared byte sizes, receipt/state/source identities match, and the trainer emitted its atomic completed checksum receipt. This independent summary did not reread37GB to recompute every checksum while the next teacher job was being scheduled. A new-process S-P3 native restore is not claimed by this performance report.

S-P2's combined whole-update mean was2.045829s and15.633092scenes/s. The measured S-P3 difference is small (~0.7% throughput) relative to separate-window/run variation. This supports the feasibility and measured cost of W96 at this topology; it does not establish that adding144 language tokens is universally free.

At22:45UTC the launcher PID286276 had exited and read-only `nvidia-smi` showed all eight GPUs at0MiB/0% utilization. Lead may independently reassign them; Systems launched no replacement. The prior remote S-P3 registration remains unexecuted because another real evaluation took that host.

Detailed all-rank/window summary: `audits/systems/sp3_pca384_8gpu_timing_summary.json`. Frozen validation digest: `audits/systems/SP3_MEASUREMENTS.json`, copied into this report directory. A final formal approval must still bind the final clean code, actual complete population and teacher manifests, resource lease and independent correctness/evidence review. This report supplies measured efficiency evidence, not that approval.
