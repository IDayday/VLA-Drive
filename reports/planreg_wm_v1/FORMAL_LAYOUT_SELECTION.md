# Formal layout selection

The same immutable layout lock is used for BaseInit and Driving-VQA Init. Only
BaseInit is benchmarked because the trainable topology and tensor shapes are
identical.

Each candidate runs the complete correct-future PlanReg-WM graph: online
student vision, one fused current-plus-three-future EMA vision call, frozen LLM,
64-candidate generator/scorer, metric targets, backward, and all-reduce. After
20 warmup steps, 300 optimizer steps are timed. The original candidates were
8x2 (global 16), 8x4 (global 32), 16x2 (global 32), and 16x4 (global 64). A
total-wall-time re-audit added 16x6 (global 96) and exploratory 16x8/16x12
candidates, using the same data manifest and shared initialization.

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

For the selected 16x6 run, mean explicit all-reduce timing was 2.39 ms,
data wait was 15.19 ms, student vision 0.383 s, fused EMA vision 1.474 s,
frozen LLM 0.743 s, metric targets 1.050 s, scorer queue wait 0.960 s,
and backward 1.031 s. Mean GPU utilization was 98.08%; mean host CPU
utilization and I/O wait were 17.01% and 0.235%. The complete per-phase and
loss/gradient distributions remain in the machine-readable metrics.

Exploratory probes were not eligible lock evidence: eager 16x8 OOMed; split
16x8 with checkpointing reached 19.7667 samples/s over 50 timed steps; split
16x8 without checkpointing reached 20.9972 samples/s over 15 steps but reserved
72.754 GiB and left unsafe runtime headroom; split 16x12 had a p90/median ratio
of 2.82 and only 9.7806 mean samples/s. The 300-step 16x6 result is faster than
the checkpointed 16x8 probe, retains 33% more optimizer updates per epoch, and
has about 23 GiB of device-memory headroom.

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

`16x6` is selected. It is 2.805 times the sample throughput of the previous
16x2 lock, passes all 300-step gates, and is the only eligible layout within
95% of the measured peak. Gradient checkpointing is disabled and the attention
backend is locked to `split_sdpa`.

The shared lock specifies two nodes / 16 GPUs, batch 6 per GPU, global batch
96, four DataLoader workers and eight scorer processes per rank. For 103,288
records it gives 1,076 steps per epoch and 29,052 steps for 27 epochs, with
eight sampler-padding slots per epoch. LR scale before caps is
`sqrt(96/32)=1.7320508`; actual EMA endpoints are 0.9762387238 and
0.9994001500. At measured throughput, one formal run is approximately 1.403
hours per epoch and 37.884 hours for 27 epochs. The paired sequential runs are
approximately 75.77 hours (3.16 days) before final evaluation, versus about
212.6 hours under the superseded global-32 lock.
