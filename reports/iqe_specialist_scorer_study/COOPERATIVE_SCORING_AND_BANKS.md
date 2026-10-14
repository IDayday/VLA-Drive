# Existing study work on released resources

The fixed six-expert/four-Scorer campaign is unchanged. The two extension
experts continue on tsinghua-zt6 GPUs 0/1 with their original optimizer,
sampler, source hash, 5,000-update schedule and validation points. No Base,
G2, MOPD or unrelated GPU process was signalled.

`iqe/internal_baseline/score_worker.py::score_item` now holds an advisory
per-response lock across identity checks, official evaluation, attempt writing
and completion receipt. A four-process fault/concurrency test runs the backend
exactly once for a common request. Changed context/trajectory hashes still fail.
The official reference scoring function and its context/seed are unchanged.

Only a new job bearing `COOPERATIVE_PROTOCOL.json` with the current job and
worker hashes may be assisted. Old jobs lacking that record are excluded.
The original coordinator remains responsible for checking every completed label.
Four CPU helpers on tsinghua-zt5 traverse their record partitions in reverse;
they neither create candidates nor mark an incomplete job complete. Existing
successful responses and previous failed attempts are skipped by helpers.
The original recorded retry policy is preserved. The initial helper launch
rejected the configured three-attempt budget because its check allowed only
two; no score was computed by that launch. The corrected four processes are
18433–18436. This failure and retry are retained in the runtime receipts.

At deployment both extension EMA validations were already complete, so the
helpers are waiting for the next new official job. **No helper speedup is
claimed yet.** Concurrency remains four; an increase requires actual completed
records/time and shared-IO evidence. Original 1,024-scene validation wall times
were raw 231.71/229.94 seconds and EMA 140.41/140.70 seconds. These different
jobs/cache states are descriptive observations, not a controlled speedup ratio.

The next deployment bank uses whole-role parallelism. Original scene order,
32-scene candidate generation batches, checkpoint identities and scoring
requests are retained within each role:

| Role | Host / GPU | Waiting PID | Native scoring workers |
|---|---|---:|---:|
| D_S | tsinghua-zt6 / 2 | 9898 | 4 |
| dev_report | tsinghua-zt6 / 3 | 9890 | 4 |
| stage_val | tsinghua-zt5 / 1 | 19362 | 4 |
| selector_cal | tsinghua-zt5 / 4 | 19345 | 4 |

These processes currently wait without a CUDA allocation. The sole existing
completion controller (PID 5452) releases their shared-storage `START.json`
records only after it freezes the selected candidate pool. Each start binds
the immutable scientific config, Base binding, pool and runtime route hash.
Mailbox dispatch avoids copying SSH credentials between training containers.
The route hash is
`8e5a61db2411bc2f7e3c399b67d078a5ad1d77787906d71b5072c240493bc830`.

`IQE_OFFICIAL_SCORE_WORKERS` can only reduce the configured CPU concurrency;
it does not modify scientific config identity, candidate count, sample order,
retry policy or score seed. All four banks must validate before the existing
controller starts any Scorer. A failed worker leaves evidence and fails the
coordinator rather than substituting data or silently retrying a new experiment.

Current regression: `python -m pytest tests/iqe -q` — **476 passed, 3 skipped**,
4 dependency deprecation warnings, 60.20 seconds. Skips remain the opt-in NCCL
tests; production study head training is single-GPU. See
`evidence/pytest_regression_v5.txt` and `evidence/COOPERATIVE_AND_ROLE_ROUTES.json`.
The frozen capture hash and active expert training source hash remain unchanged.
