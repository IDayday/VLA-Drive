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

The final measured table and selection are filled from those artifacts after
all four benchmarks complete; no layout is selected by hand.
