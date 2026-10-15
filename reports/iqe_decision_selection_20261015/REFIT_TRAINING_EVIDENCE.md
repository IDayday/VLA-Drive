# Scratch training and fixed refits

| Phase | Model | Updates | Unique scenes | Logs | Presentations | Fixed Z | GPU seconds | New parameters |
|---|---|---|---|---|---|---|---|---|
| diagnostic | M_scene | 3000 | 13139 | 169 | 384000 | 0.73173714 | 956.22 | 3299588 |
| diagnostic | M_set | 3000 | 13139 | 169 | 384000 | 0.73173714 | 1098.17 | 4090881 |
| refit | M_scene | 2524 | 16581 | 211 | 323072 | 0.74112960 | 899.69 | 3299588 |
| refit | M_set | 631 | 16581 | 211 | 80768 | 0.74112960 | 293.16 | 4090881 |
| overfit | M_scene | 300 | 64 | 42 | 38400 | 0.93657903 | 86.94 | 3299588 |
| overfit | M_set | 300 | 64 | 42 | 38400 | 0.93657903 | 98.89 | 4090881 |

Each refit INITIALIZATION hash equals its diagnostic initial random hash, not its trained best hash. Native optimizer/scheduler/RNG/sampler checkpoints are preserved. Final refits use the predeclared last endpoint; external stage_val and dev do not choose a checkpoint. Heads read frozen caches and execute zero VLM calls. Model, optimizer and losses are FP32; deployment inherits the existing Base numerical runtime. GPU-seconds above are single-GPU training elapsed time including validation/checkpoint; they are not utilization integrals. Context construction/preload and final fit reporting are additional wall time in runtime launch receipts.

M_scene: inner 2000 → refit 2524; M_set: inner 500 → refit 631. Formula: ceil(T_inner × 16581 / 13139), without an external score-dependent extension.
