# Restored vla-zt3 and periodic burst IL Navtest

## Actual native continuation

vla-zt3 SSH now works and repeated inventory found all8 A800s empty. A rebuilt container lacked the existing private DDP environment. Reused the already qualified8.7GiB archive, rehashed it and restored it locally without replacing navsim/gcrlo or downloading dependencies. Torch2.5.1+cu124 / Transformers4.57.0 / DeepSpeed0.16.9 CPU imports passed.

TIME_RESIDUAL's learner was running on tsinghua-zt3 alongside V12, but its CPU owner had exited after remote Python127 during an asynchronous candidate-refresh dispatch. Reconciled the durable, verified prechild intent without restarting existing refresh workers. Requested the mature trainer's native optimizer-boundary stop file; it sealed **33,283**, exited cleanly and preserved every rank's data/bank/Adam/scheduler/RNG. No training process was killed.

Native continuation on vla-zt3 initially performed one real update then failed while decoding Chinese in a grant under the new container's ASCII locale. Retained the failure, archived the single uncommitted update and restored again from33,283. Explicit UTF8 process startup plus ASCII-escaped **semantically identical JSON** grant copies fix this engineering issue. Old grants are preserved; no permission, actor input or scientific hyperparameter changed.

All8 ranks subsequently reached **33,552**, matching native optimizer counts, finite positive gradients, zero skipped updates. Recent32-update observed wall time was **2.502s/update**, versus **4.697s/update** before migration (~1.88x). This is a live allocation observation, not an isolated benchmark, and later evaluation contention can change it. V12 stays on tsinghua-zt3; Huber stays on tsinghua-zt2; RL sources/budgets/scopes remain unchanged.

## User-requested cadence

Seven current IL lines: V12, TIME_RESIDUAL, HUBER, A/B/C/D. Completed RAW and screened-out PHYSICAL/DELTA remain stopped. Every2k eligible sealed update is queued, with at least2k separation from an already requested point. For an existing odd13k request, the next even-grid request is16k;14k would violate the user's minimum. Initial registration takes the latest available eligible checkpoint, with no unlimited historical backfill. Pinned existing requests, including nested legacy owners, prevent duplicate points.

- Full12,146-scene canonical NAVSIMv1.1, seed2101, **single** clean-x K10shift5; no scorer selection or new seed.
- Up to**32 scene workers/model**, at most2 concurrent checkpoint models, using free GPUs then verified sharing. Each model invocation remains batch1. Rank/world affect token partitioning only; scene-bound noise is independent of worker rank. Pool probes exclude invisible/unknown owners and unexplained allocations; actual workers repeat UUID/memory/lease checks.
- Both World and four-framework paths reuse their qualified exporter, native parity, deployment forward and scorer. Four's old coordinator16-worker cap is an operational limit, removed only in this burst adapter. No model math or scoring source is edited.
- Metadata-only queue wakeups every10min; **no training-journal polling**. Report cadence remains30min plus meaningful completion/failure/queue events, not every queue wakeup.
- Atomic registration/dispatch intent, one owner, adopted active bursts, no blind repeat on an ambiguous owner exit. Bounded prechild lease/headroom retries have unique receipts; actual model or scoring failures require review.
- Whole native checkpoints are hardlinked while exporting/evaluating so rolling trainer retention cannot delete a queued point. After all owned evaluator workers exit, only our redundant native staging hardlinks are retired; deployment weights, hashes, scene scores and trajectories remain. Original training checkpoints are untouched. New registration waits below500GiB free storage.

TIME30k and V12_56k are the first new periodic requests. TIME's32-way native/export parity has passed and real workers are evaluating. V12's first attempt had a proven **prechild lease conflict**, zero inference/scoring, with its original receipt preserved. A fresh attempt reuses the same sealed export and passed native/export parity. These are the same checkpoint experiment, not extra training variants or a second score population.

## Qualification and limits

23 targeted tests pass, covering the prior report/recovery invariants, dead CPU-owner detection despite fresh GPU logs, minimum2k spacing, duplicate suppression, nested registrations, safe pinning and exact disjoint8/16/32-way coverage of all12,146 tokens. World32-way real native parity has passed. Future new four-framework weights still run their own native/export check before full sharding; no unexecuted check is labelled PASS.

The cadence controller and burst operators are under `tools/il_report/`; actual argv, immutable per-burst operator snapshots, failed attempts, ownership and costs stay in the asset roots. No large model/cache/raw image is committed. Failed scorer/model jobs never become score0 or silently disappear from the evaluation denominator. All new scores are supplementary observations; registered own-parent DEV selection and IL100k/RL50k budgets are unchanged.

## Running commands

```bash
/usr/bin/python3 -m tools.il_report.navtest_cadence --interval-seconds 600
/usr/bin/python3 -m unittest discover -s tests/il_report -v
```

Use `--interval-seconds 600` (separate argument) for the actual controller. The controller already has a live owner; inspect `OWNER.json` and its shared lock before any recovery. The actual burst command is stored in each request's `LAUNCH.json`; do not start a duplicate. Live queue: `/mnt/project/qwen-world-il-research-20261030/navtest_cadence_20261031/STATE.json`. Dynamic report: `/mnt/project/qwen-world-il-research-20261030/report.html`.
