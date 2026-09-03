# Formal layout selection

The same immutable layout lock is used for BaseInit and Driving-VQA Init. Only
BaseInit is benchmarked because the trainable topology and tensor shapes are
identical.

Each candidate runs the complete correct-future PlanReg-WM graph: online
student vision, one fused current-plus-three-future EMA vision call, frozen LLM,
64-candidate generator/scorer, metric targets, backward, and all-reduce. After
20 warmup steps, 300 optimizer steps are timed. The original candidates were
8x2 (global 16), 8x4 (global 32), 16x2 (global 32), and 16x4 (global 64). A
total-wall-time re-audit added 16x6 (global 96), 16x8 (global 128), and an
exploratory 16x12 candidate, using the same data manifest and shared
initialization. The final 16x8 evidence also includes the audited execution-only
PDM/EMA overlap described below.

A candidate is eligible only with no OOM, deadlock, or non-finite value, peak
allocated memory below 72 GiB, peak reserved memory below 76 GiB, and a p90 to
median step-time ratio no larger than 1.35. Total epoch time is determined by
sample throughput, not step latency alone. Layouts within 95% of peak sample
throughput are treated as a practical tie; the selector then chooses the
smallest global batch among them to retain more optimizer updates and memory
headroom.

The large-batch candidates use split SDPA for read-only register attention.
Register query rows attend CLS/register/patch tokens, while legacy CLS/patch
query rows use the original CLS/patch K/V sequence with register columns
physically absent. This removes the quadratic masked probability allocation
that made the eager implementation slow and prevented larger no-checkpoint
batches. On the actual CUDA BF16 attention shape (`[1,1041,1024]`), the eager
and split paths have maximum forward and input-gradient differences of
`4.8828125e-4` and `2.44140625e-4`; all outputs and gradients are finite. The
paths are mathematically equivalent but not bitwise or `1e-5` equivalent under
different fused BF16 kernels. CPU FP32/BF16 reference tests and the realistic
CUDA forward/backward tolerance test pass.

The machine-readable metrics live under `throughput/<layout>/metrics.json`.
`formal_training_layout_lock.json` records their SHA-256 values, selected GPU
count, per-GPU/global batch, workers, scorer processes per rank, LR scale, EMA
endpoints, and exact 27-epoch step budget. Launch is blocked if the lock or any
referenced metrics file is missing or changes.

## Measured results

All rows in this table completed 20 warmup plus 300 timed optimizer steps with
zero non-finite values and no OOM or deadlock.

| Layout | Global batch | Samples/s | Steps/s | Median / p90 step (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| 8x2 | 16 | 3.8799 | 0.24249 | 4.1019 / 4.2517 | 18.756 / 24.416 |
| 8x4 | 32 | 4.0692 | 0.12716 | 7.8391 / 8.0498 | 32.486 / 43.961 |
| 16x2 | 32 | 7.2896 | 0.22780 | 4.3654 / 4.5511 | 18.756 / 24.510 |
| 16x4 | 64 | 7.6595 | 0.11968 | 8.3376 / 8.5365 | 32.486 / 43.793 |
| 16x6 split-SDPA, no GC | 96 | 20.4495 | 0.21302 | 4.6152 / 4.9789 | 53.960 / 56.586 |
| 16x8 split-SDPA, no GC, async PDM | 128 | 27.3426 | 0.21361 | 4.5910 / 4.8809 | 69.127 / 72.752 |

For the selected 16x8 run, exact detached PDM target computation takes 2.014 s
end-to-end but executes on CPU concurrently with the no-grad EMA vision pass.
The residual scorer queue wait is 0.53 ms median (21.38 ms mean), rather than
being an additive two-second phase. Mean/median-relevant phase timings are:
data wait 14.56/5.58 ms, host-to-device 27.57 ms, student vision 0.505 s,
fused EMA vision 1.974 s, frozen LLM 0.980 s, backward 1.087 s, explicit
all-reduce 4.05/1.20 ms, and EMA update 5.68 ms. Mean GPU utilization was
99.12%; mean host CPU utilization and I/O wait were 17.74% and 0.175%.
Worker decode/transform timings are overlapped by DataLoader prefetch and must
not be added to step time.

The raw 16x8 JSON contains an `optimizer_step_time` diagnostic from a temporary
hook pair whose boundaries crossed optimizer-step iterations. It is not a
valid phase duration, is excluded from every conclusion here, and the hook was
removed before formal training. All other named phase timers have matched
start/stop boundaries. The complete distributions and finite loss/gradient
evidence remain in the machine-readable metrics.

Earlier exploratory probes were not eligible lock evidence: eager 16x8 OOMed;
split 16x8 with checkpointing reached 19.7667 samples/s over 50 timed steps;
the first no-checkpoint split 16x8 probe reached 20.9972 samples/s over only 15
steps; split 16x12 had a p90/median ratio of 2.82 and only 9.7806 mean
samples/s. The final no-checkpoint 16x8 run completed all 300 measured steps,
stayed below both memory gates, and had no OOM, deadlock, or non-finite value.

An initial 16x2 attempt exposed a host-local environment mismatch: the same
nominal environment path resolved to Python 3.9/Lightning 2.6 on `vla-zt` but
Python 3.10/Lightning 2.2 on `vla-zt2`, and the first collective made no
progress. That attempt is retained under `throughput_attempts` and is not
scientifically eligible. Formal scripts now force a genuinely shared CPython
3.9.25 environment (PyTorch 2.5.1+cu124, Transformers 4.57.6, PEFT 0.17.1,
Lightning 2.6.0), shared Hugging Face code cache, and compare core/VLM source
hashes before model construction. The schema-2 node fingerprints are
byte-identical. A separate 16-rank, five-iteration 16 MiB NCCL smoke completed
in 0.803 seconds with finite output.

## Locked selection

`16x8` is selected. It is 3.751 times the sample throughput of the original
16x2 lock and 1.337 times the previous 16x6 lock, passes all 300-step gates,
and is the only eligible layout within 95% of the measured peak. Gradient
checkpointing is disabled and the attention backend is locked to `split_sdpa`.

The shared lock specifies two nodes / 16 GPUs, batch 8 per GPU, global batch
128, four DataLoader workers, eight scorer processes per rank, and two proposal
partitions per scene. For 103,288 records it gives 807 steps per epoch and
21,789 steps for 27 epochs, with eight sampler-padding slots per epoch. LR
scale before caps is `sqrt(128/32)=2.0`; actual EMA endpoints are
0.9684444339 and 0.9992002799. At measured throughput, one formal run is
approximately 1.049 hours per epoch and 28.334 hours for 27 epochs. The paired
sequential runs are approximately 56.67 hours before final evaluation, saving
about 19.10 hours versus the superseded global-96 lock and about 155.9 hours
versus the original global-32 lock.
