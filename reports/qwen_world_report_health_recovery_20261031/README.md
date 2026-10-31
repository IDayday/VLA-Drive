# IL/RL health recovery and resource allocation — 2026-10-31 UTC

The Huber full IL run had stopped after update14,661. Its original rank6 was SIGKILLed on October29; the available evidence does **not** establish whether this was OOM or another OS/operator cause. All normal-path science remains unchanged.

## Executed recovery

- Rehashed **every native14k checkpoint shard**, matched all8 saved rank states and the original scientific source manifest. Native Adam moments, scheduler, data/bank samplers and RNG are restored strictly.
- Preserved the failed attempt, its receipts and661 uncommitted journal rows per rank. Active journals contain only the committed prefix, so recomputation cannot create duplicate effective updates. The archived report curve is preserved separately.
- Resumed the same immutable scientific source from14k on **tsinghua-zt2 GPUs0–7**, keeping seed42, global32, the registered full100k schedule and selection. Eight-rank actual optimizer progress above14k, finite nonzero gradients and zero skipped updates have been observed; see `validation.json`.
- A prelaunch maintenance helper initially encountered a historical dict/list receipt schema difference. No learner launched during that failed helper attempt; evidence is retained. The adapter now preserves either historical form. This did not change any model or optimizer computation.
- The known failed suffix consumed at least **1504.5seconds** of training-update wall time and **3.343 summed GPU-worker-hours**. These are journal-derived lower bounds; they exclude checkpoint I/O and the interrupted last batch. They are not exclusive GPU-hours. Recomputed work is not counted twice as new algorithmic data.

## Every report update

`tools.il_report.render` now runs bounded task health checks on each normal30-minute or metadata-event update. A single-owner, reviewed recovery entry may continue a verified native run. Unknown/numerical failures require review. No policy/loss/LR/data/precision changes, arbitrary process kills, indefinite retries or automatic extension of pilots are permitted. Completed, screened-out and futility-stopped arms stay stopped.

Health checks publish `maintenance/HEALTH_LATEST.json`, IL and RL state tables, the native recovery confirmation and failure alerts. A fresh owner process is not itself proof of successful training. The primary table uses actual all-rank journals.

Sixteen targeted report/maintenance tests pass: continuous journals, refused duplicates/holes, exact prefix and aborted-cost preservation, idempotence, immutable archived evidence, missing-boundary refusal, unreviewed-launch refusal, dead/reused owner handling, protected experiment states, and prior report invariants. The real8GPU strict-native continuation supplies recovery evidence beyond mock tests.

## Resources and actual dispatch

The user reauthorized **vla-zt3, tsinghua-zt2, vlawm-zt2, tsinghua-zt6 temporarily without a deadline**. Global/task grants now include these as unbounded temporary hosts. Their prior8h expiry overrides are superseded; the obsolete project-owned vlawm-zt2 CPU expiration timer was explicitly cancelled. No learner was signalled for this authorization change.

| Placement | Work |
|---|---|
| tsinghua-zt2,0–7 | Huber native14k → registered100k |
| vlawm-zt2, explained free cards | New A13k/B10k/C12k/D7k full Navtest queue |
| tsinghua-zt6, explained free cards | Same finite queue, sharing its global candidate-independent evaluation budget |
| vla-zt3 | Access blocked by SSH public-key rejection; no false claim of deployment |

vlawm-zt2 GPU6 and tsinghua-zt6 GPU2 have unaccounted allocations despite empty NVML owner lists. They are excluded without terminating any process. The new finite evaluator runs two models concurrently through the **existing qualified exporter, native parity and scorer**; each model uses single-candidate K10shift5, seed2101, canonicalv1.1, all12,146 scenes. This is not a new training matrix, best-of-N, parent selection or actor change. Final scores remain pending.

A/B/C/D, V12 and TIME_RESIDUAL journals are advancing with zero skipped updates at this audit. PPO_FULL, PPO_SAFE_COMFORT and PPO_ANCHOR40 are advancing; PPO_ADAPTIVE completed its registered26k stage and is in its existing evaluation/control flow. HARD/CISPO/NFT/BATCHSTD futility stops, the two10k pilots awaiting review, completed RAW100k and the screened-out loss arms remain intact. Registered budgets and scientists' sources are unchanged.

## Executed commands

```bash
/usr/bin/python3 -m unittest discover -s tests/il_report -v
/usr/bin/python3 -m tools.il_report.health --recover-huber --root /mnt/project/qwen-world-il-research-20261030
/usr/bin/python3 -m tools.il_report.render --once --root /mnt/project/qwen-world-il-research-20261030
```

The mature recovery owner is launched with `--arm HUBER --runtime-source /mnt/project/VLA-Drive-qwen-world-loss-from-init-20261029 --recover-owner` from the separately locked operational source. Its exact argv/checkpoint and the finite evaluation argv are indexed in `validation.json`. Re-running the maintenance command returns the existing invocation receipt; it cannot duplicate a learner. A failed repair requires an explicit new review.

Live report: `/mnt/project/qwen-world-il-research-20261030/report.html`. Recovery evidence: `/mnt/project/qwen-world-il-research-20261030/maintenance/huber_recovery_20261031/`. No images, model weights, optimizer shards or candidate caches are submitted to Git.
