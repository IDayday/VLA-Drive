# Qwen-World IL checkpoints and later RL observations

Full12146 scenes, canonical NAVSIM v1.1, seed2101, single clean-x K10 shift5 ODE. IL observations do not change the fixed IL40k RL parent. Test is historically observed.

| Snapshot | Updates | PDMS | Change vs IL40k (points) | Change vs previous same experiment (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_077000 | 77000 | 88.594568 | +0.622186 | +0.490685 |
| OPTIMIZED_IL_010000 | 10000 | 79.904274 | -8.068108 | — |
| ATTENTION_RL_050000 | 50000 | 89.905612 | +1.933230 | -0.508662 |
| MEMORY_RL_050000 | 50000 | 90.628892 | +2.656510 | -0.112418 |

Previously completed Time-LoRA50k PDMS=89.508000; reused without new inference.
Optimized-label IL10k is a fresh model trained from generic initialization. Its absolute score is an early learning observation; a comparison with old IL40k/77k at different training budgets does not isolate label quality. Both new50k RL endpoints share the original IL40k parent; new IL checkpoints do not replace that parent.

The previous-same-experiment comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/new_checkpoints_NAVTEST_20261028T2240Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
