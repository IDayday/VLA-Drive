# Bounded CPU parallel compact construction

Execution-only delta on the strict single-pass builder. `--workers` is explicitly
bounded to 1..4 and defaults to 1. Each spawned process owns one scene's native
feedback cache, runs all strict receipt and label checks, and emits compact rows
plus the dependency versions observed before semantic use. The parent checks
those versions again, restores original population order and seals the same
complete inventory. The queue contains at most two scenes per worker. This
changes no official scoring, simulator calls, labels, population or sample order.
No model or GPU computation occurs.

The population registration is checked once in the parent. Workers receive only
its immutable identity and the single pinned scene, avoiding serialization of
the complete population for every task. A failed worker aborts the build before
writing the compact COMPLETE datasets or validation receipts. ProcessPool shutdown
may finish the other bounded, already submitted read-only tasks.

Actual old64 evidence: `PARALLEL_BUILD_VALIDATION_64.json`.

| Builder | Strict build + both receipts (s) | Warm inventory reuse (s) |
|---|---:|---:|
| One worker, previous strict single-pass commit | 70.5358 | 0.8128 |
| Four workers | 36.6080 | 0.8057 |

These sequential runs occurred while other authorized work continued. This is a
limited execution measurement, not a long-window throughput guarantee or a
prediction of full1024 completion time. Both retained exactly the original PAIR
SHA `d20b65bef61a0d041683ec5b0b65987a51f4b5ef669e1e87955bbcd9f40a45dc`
and REPAIR SHA `198d559133160876f7e14271acc69fcbc414018f3326f99e2b0418931483f994`;
1384 pairs and all 6436 dependency files were verified. No body was re-scored.

63 targeted CPU tests passed in 17.24 seconds before removing one inadvertently
re-collected imported fixture test. A separate affected-parser plus parallel
suite then passed 31 tests in 23.29 seconds. Coverage includes real-wrapper
synthetic spawn processing, serial/parallel byte equality, worker failure on a
corrupt native record without success sealing, dependency version changes and
invalid CPU budgets. Synthetic fixtures do not count as physical research data.

Use the existing strict CLI with an explicit CPU budget:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.build_effect_pairs \
  --config configs/action_effect/shared_budget_v2.yaml --max-samples 1024 \
  --g0-root /mnt/project/action-effect-repair-20261020-budget-v2/g0/population_001024 \
  --workers 4 --resume
```

The explicit scene budget must match the registered population. A completed G0
population is required; the command cannot create or simulate missing scenes.
The original single-worker path remains available with `--workers 1`.
