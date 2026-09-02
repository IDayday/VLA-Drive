# Formal layout selection

The same immutable layout lock is used for BaseInit and Driving-VQA Init. Only
BaseInit is benchmarked because the trainable topology and tensor shapes are
identical.

Each candidate runs the complete correct-future PlanReg-WM graph: online
student vision, one fused current-plus-three-future EMA vision call, frozen LLM,
64-candidate generator/scorer, metric targets, backward, and all-reduce. After
20 warmup steps, 300 optimizer steps are timed. The four candidates are 8x2
(global 16), 8x4 (global 32), 16x2 (global 32), and 16x4 (global 64), using the
same data manifest and shared initialization.

A candidate is eligible only with no OOM, deadlock, or non-finite value and
peak allocated memory below 72 GiB. The selector chooses the higher-throughput
layout between 8x4 and 16x2. Global batch 64 is eligible only if it is at least
25% faster than the best global-32 layout and a separate 1,000-step stability
audit shows no trajectory/scorer loss degradation or unstable gradients.

The machine-readable metrics live under `throughput/<layout>/metrics.json`.
`formal_training_layout_lock.json` records their SHA-256 values, selected GPU
count, per-GPU/global batch, workers, scorer processes per rank, LR scale, EMA
endpoints, and exact 27-epoch step budget. Launch is blocked if the lock or any
referenced metrics file is missing or changes.

## Measured results

All rows completed 20 warmup plus 300 timed optimizer steps with zero
non-finite values and no OOM or deadlock.

| Layout | Global batch | Samples/s | Steps/s | Median / p90 step (s) | Peak allocated / reserved (GiB) |
|---|---:|---:|---:|---:|---:|
| 8x2 | 16 | 3.8799 | 0.24249 | 4.1019 / 4.2517 | 18.756 / 24.416 |
| 8x4 | 32 | 4.0692 | 0.12716 | 7.8391 / 8.0498 | 32.486 / 43.961 |
| 16x2 | 32 | 7.2896 | 0.22780 | 4.3654 / 4.5511 | 18.756 / 24.510 |
| 16x4 | 64 | 7.6595 | 0.11968 | 8.3376 / 8.5365 | 32.486 / 43.793 |

For 16x2, mean explicit all-reduce timing was 2.95 ms (p90 9.13 ms),
data wait was 12.54 ms, student vision 0.466 s, fused EMA vision 1.809 s,
frozen LLM 0.259 s, metric targets 0.508 s, scorer queue wait 0.472 s,
and backward 1.298 s. The complete per-phase and loss/gradient distributions
remain in the machine-readable metrics.

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

`16x2` is selected: it is 79.14% faster than 8x4 at the same global batch 32.
Although 16x4 is stable and 5.08% faster than 16x2, it is far below the 25%
global-64 threshold, so the required 1,000-step global-64 stability gate is not
invoked and global batch 64 is ineligible.

The shared lock therefore specifies two nodes / 16 GPUs, batch 2 per GPU,
global batch 32, four DataLoader workers and four scorer processes per rank.
For 103,288 records it gives 3,228 steps per epoch and 87,156 steps for 27
epochs, with eight sampler-padding slots per epoch. LR scale is 1.0 and actual
EMA endpoints are 0.992016 and 0.99980001. At measured throughput, one formal
run is approximately 106.3 hours; the paired sequential runs are approximately
8.86 days before final evaluation.
