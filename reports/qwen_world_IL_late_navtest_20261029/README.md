# Qwen-World IL：65k—83k 补充 Navtest

原始联合IL的65k/70k/75k/80k/83k，在新增评分前固定；已测77k直接复用。完整12146场景，canonical v1.1，单候选clean-x十步shift5 ODE，seed2101。Test已被反复观察，不称独立盲测；不变更既有RL的IL40k父模型。

| Checkpoint | Updates | PDMS | Change vs IL40k (points) | Change vs measured IL77k (points) |
|---|---:|---:|---:|---:|
| WORLD_IL_065000 | 65000 | 88.381806 | +0.409424 | -0.212762 |
| WORLD_IL_070000 | 70000 | 88.792910 | +0.820528 | +0.198342 |
| WORLD_IL_075000 | 75000 | 88.504936 | +0.532554 | -0.089632 |
| WORLD_IL_080000 | 80000 | 88.553076 | +0.580694 | -0.041492 |
| WORLD_IL_083000 | 83000 | 88.500809 | +0.528427 | -0.093759 |

The previous-same-experiment comparisons use full paired scenes/logs under the same protocol. A lower point estimate is an observed rollback; interpret its physical-log bootstrap interval separately. This is not multiple training-seed evidence. Different RL update budgets do not establish method superiority.

All metric components, zero-score and safety transfers, and paired95% intervals are in the per-model JSON. Raw worker costs and scene records are retained at /mnt/project/qwen-world-postcot-20261025/IL_late_checkpoints_NAVTEST_20261029T0310Z. No teacher/search/scorer chooses a deployed output; every observation is one candidate.
