# Bounded cost and storage evidence

No full feature bank, full score bank or full trainer was launched. Actual measurements use the trained 22k Base and 44 real scenes (32 fit + 4 stage_val + 4 cal + 4 dev). Resources: one A800 for encoding/private heads, two CPU official-scoring workers, 8 CPU torch threads. Another independent research job arrived after our qualification began; late head/selection timings are not isolated performance benchmarks.

| Stage | Measured wall time | Scope |
|---|---:|---|
| Feature cache | 32.46s | 44 actual image/ego inputs |
| Base candidate generation | 2.24s | 44 cached-feature scenes |
| Base official labels | 69.87s | 44 candidates, 2 workers |
| Manifest/GT audit/retrieval/plan | 101.78s | 32 fit scenes, one eligible hard original |
| New candidate generation | 4.35s | old Base reused, expert1 added |
| New official labels | 72.25s | 44 new labels, old labels reused |
| Expert update1 / resumed update2 | 21.04s / 29.09s | includes checksums, frozen audits, checkpoint and validation |
| Scorer update1 / resumed update2 | 21.15s / 20.94s | same audit/checkpoint overhead; actual resumed optimizer computation about1.12s |
| Calibration | 0.415s | 4 naturally sampled cal scenes; no beneficial switch |
| Dev / fit selection report | 12.45s / 91.36s | fixed 4 / 32 scenes |
| Bundle write | 28.34s | complete deployment weights, no optimizer |
| Official context preparation | 63.94s | independent bounded 32 stage_val scenes |

Full 103,288-scene candidate pass extrapolates to about45.6 CPU-worker-wall hours at the measured two-worker rate; this is an estimate, not a full-scale timing claim. Each new expert requires another candidate pass; old valid candidates/labels are reused. Full stage_val has5,563 scenes: a new official validation pass at this rate is roughly2.45h, so validation every200 updates can dominate private-head training. Do not hide this behind fast cached-head updates. Context preparation and per-log amortization also need capacity budgeting; scaling the measured32-scene context time linearly would be about57h, but log batching may change it materially. Workers can only be increased as an explicit resource configuration with enough free CPUs and protocol-equivalence retained.

Feature cache measured899,536 bytes for44 scenes (including receipts/locks), approximately2.11GB at103,288 scenes by simple scaling. Candidate bank measured265,651 bytes for88 candidates plus audit targets/receipts; actual full-bank metadata should be monitored. These are head-feature caches, never substitutes for image-to-trajectory latency.

Actual Base checkpoint:22,816,333,071 bytes. At22k with1000-step saves to100k:78 remaining saves,1.780TB additional weights, one22.8GB atomic-write temporary and500GiB reserve. Snapshot free space2.857TB; total required2.339TB. Capacity check READY at capture, with about518GB beyond the specified reserves. Other projects share this disk; recheck before launch and before expanding banks. No checkpoint was deleted and no cleanup policy activated. Full details in `evidence/CHECKPOINT_CAPACITY.json`.

Optimized actual Base trainer profile:32 updates from21k, global32 across8 A800, stable wall p502.7666s/p953.1418s and compute p502.6733s. Profile includes one22.8GB final checkpoint; its short-run cost must not be treated as the usual1000-step save amortization. Source and execution details are in `../iqe_performance_20261011/PERFORMANCE_AUDIT.md`.
