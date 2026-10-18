# Complete PROFILE240 and independent efficiency approval

The exact registered `C1_rollout_std_fulltrain_seed42_official_4096_v1` passes
the Systems efficiency review, conditional on the launcher's fresh resource and
lease admission. This approves one bounded 4096-update experiment for the Lead
to launch after all three independent reviews; it does not dispatch work, extend
the scientific budget, approve deployment, or assert scientific improvement.

Training source remains `751fcfa2b41549ae16f7b3204aef8dd18b912016` in the clean
detached runtime `/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018`.
Formal registration SHA256 is
`928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f`.
The sealed `EFFICIENCY.json` binds the exact launcher, monitor, CPU observer and
storage reservation file as well as the independent measured evidence.

## Completed measurement

The single PROFILE exited 0 after exactly 240 updates. All four ranks completed;
the launcher's timeout, torchrun, ranks and observers were absent on the read-only
19:02:10 UTC check, with no GPU compute applications remaining. `PAUSED` /
`RUNNING_CHECKPOINT` is the expected bounded diagnostic endpoint, not completion
of the later 4096-update experiment. No diagnostic weights enter formal training.

The existing `summarize_ranks` implementation measured two complete windows:
20 warmup + 100 steady updates, repeated twice. The manifest contains the original
101,592 TRAIN entries; the first 240 sampler updates consumed 7,680 unique scenes
without replaying a small loop. Global batch remained 32 (world 4, microbatch 4,
accumulation 2). The original 4096-update learning-rate clock and warmup 100 were
retained. There were 30,720 valid official scores, 61,440 repeat-8 BC presentations
and 1,536,000 Monte Carlo points for each old/current policy computation.
All recorded losses and gradient norms were finite, sampled FP32 master updates
were verified, and the frozen encoder hash was unchanged. All ranks had the same
normalization identity for every update. No monitor error, stop request, foreign
selected-GPU process or resource overrun was recorded.

| Measurement | Window 1, steps 21–120 | Window 2, steps 141–240 |
|---|---:|---:|
| Complete wall, seconds | 1534.583 | 1549.786 |
| Scenes / second | 2.08526 | 2.06480 |
| Normal updates excluding checkpoint | 100 | 99 |
| Normal mean, seconds | 15.34036 | 15.36020 |
| Normal p50, seconds | 15.34432 | 15.32684 |
| Normal p95, seconds | 16.20769 | 16.31282 |
| Allocated GPU-seconds | 6138.331 | 6199.144 |
| Live-tree sampled CPU-seconds | 8618.79 | 8671.42 |

Across 199 normal steady updates, slowest-rank wall mean is **15.35023 s**, p50
**15.33775 s**, p95 **16.28589 s**. The second throughput window includes the
periodic save; it is not silently filtered from complete-pipeline throughput.
The maximum-rank periodic checkpoint cost was **14.18211 s**. At the unchanged
formal save interval 512 this gives a planning estimate **15.37793 s/update**,
or about **17.50 hours** for 4096 updates, excluding one-time loading/final work,
invocation transitions and future workload variation. This is an estimate from
the actual bounded run, not an observed full-training duration or a guarantee.
The initial and final rank-0 checkpoint records were 6.708 and 14.166 seconds.

The full driver occupied four GPUs for 3931.423 seconds: **15,725.692 allocated
GPU-seconds (4.368 GPU-hours)**. Waited child CPU was **21,307.903 CPU-seconds
(5.919 CPU-hours)**. The live-tree sample maximum was 21,288.25 CPU-seconds;
these CPU scopes overlap and must not be summed. Window CPU boundaries were
within 0.86 seconds of an actual sample. Allocation cost is not active kernel time.

## Where time is spent

Slowest-rank instrumented region means in the two windows were respectively:

| Region | Window 1, seconds | Window 2, seconds |
|---|---:|---:|
| Official feedback | 5.670 | 5.693 |
| Current BC/FPO backward and optimizer | 3.490 | 3.482 |
| Old CFM | 2.371 | 2.374 |
| Snapshot rollout | 2.034 | 2.031 |
| Shared encoder | 1.113 | 1.115 |
| Input data | 0.616 | 0.613 |
| Advantage normalization, nested inside feedback | 0.004553 | 0.004505 |

The new normalization timer includes preparation, hashing, vote and collective
wait; it is a subset of official-feedback wall. Its p95 was 5.013 / 4.858 ms.
Do not add inclusive regions or interpret this as an isolated CUDA kernel cost.
The normalization is not the measured throughput bottleneck. Historical zt4
timings remain reference evidence; the present zt5 run is not a causal A/B test
of normalization alone. No 8-vs-4 topology search was repeated.

## Resource and storage decision

Observed maxima were 13.15 GiB PyTorch allocated, 15.66 GiB reserved, **17.68 GiB
driver GPU memory**, and **118.10 GiB summed process RSS**. RSS is an upper-bound
process sum rather than PSS. Shared free space stayed above **3278.55 GiB**.
Combined SMOKE + PROFILE output peaked at 42.49 GiB with five native directories,
within its 96 GiB / six-directory allocation. No original model/data/cache
payload was reread by this review. Physical disk bandwidth was not measured;
the measured data-region wall and storage occupancy are reported instead.

The formal controls retain these bounds:

- Only tsinghua-zt5 GPUs 0–3, four ranks, CPU affinity 0–15, and four global
  single-thread official scorers. Fresh admission rejects any compute application
  on those UUIDs, including processes still reporting zero memory.
- One total 4096-update endpoint; at most six invocations including partial
  attempts; each invocation at most 28,800 training seconds. The owned outer
  timeout sends TERM at 29,400 seconds and KILL only to its group after 180 more.
  No automatic retry, new experiment, or endpoint extension.
- At most 15 native directories, including partials, and 209 GiB own-run budget.
  Fresh storage admission requires **1529 GiB**: 209 own + 1024 shared floor +
  4 unresolved old future writes + 292 for the protected native-Qwen IL job.
  Runtime protects **1320 GiB** independently of this run's remaining writes.
- The other active job is
  `/mnt/project/qwen-sourceflow-native-20261018/runs/IL_seed42_v2` on tsinghua-zt.
  Its 292 GiB reservation covers five milestones, two rolling checkpoints and one
  atomic in-flight full snapshot at 38 GiB, subtracts already occupied output,
  and adds small future metadata. No files are deleted and no retention policy
  of that job is changed. Completed/paused artifacts already reduce filesystem
  free space and are not charged again as hypothetical future writes.
- Nominal five-second owned GPU/RSS/free/native sampling; 48 GiB per GPU and
  256 GiB RSS limits. Recursive output size is sampled first, every 300 seconds
  and on exit. Enforcement requests this run's safe optimizer-boundary stop and
  leaves other projects untouched. These are sampled guards, not instantaneous
  quotas: size detection may lag 300 seconds plus scanning and a boundary save.

The resource evidence clears the previous pending PROFILE condition. Fresh
admission, exact lease/source/control hashes, the three independent fingerprints
and the Lead's RUN_APPROVAL remain required at every invocation. Explicit native
resume is restricted to this run's own complete state; no diagnostic checkpoint,
new warmup or implicit stop acknowledgement is permitted.

## Evidence and remaining boundaries

`MEASUREMENTS.json` retains all raw record hashes, two windows, actual CPU/GPU
cost and monitor peaks. `collect_profile.py` is the read-only reproducible
collector; it was run with CUDA hidden and CPU affinity 112–113 after completion.
The mature timing implementation was reused, not replaced with selected fastest
steps. The earlier exact-control Systems test record remains 10 passed; the
separate Correctness control record is 29 passed. Neither suite was rerun for
this final measurement-only seal.

The four-step real SMOKE's independent full native checksum and 16 normalization
payload checks are inherited with exact hashes. No new GPU/ZeRO resume invocation
was run here, and no stronger recovery claim is made. The first 7,680 unique
TRAIN scenes do not measure every later input or cold full-population storage.
Scientific gain, DEV results and deployment risk remain untested by this review.
Endpoint evaluation's separate static resource appendix is not training approval.
