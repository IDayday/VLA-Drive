# Independent finalized index controller review

Reviewed exact source commit `bf67fad2b77e443395635f8871eb757cbbb8c596` in a separate worktree. This report concerns only the new TRAIN finalization route and finite preparation wrapper. It does not authorize execution.

**Functional wiring: PASS (narrow).** Fourteen independently executed synthetic tests passed. The existing finalizer 40-case suite was not rerun. No actual payload, ledger, GPU, subprocess worker, or model checkpoint was accessed. The separate resource review confirmed an initial source-read boundary failure; this report does not override it. Actual launch is blocked for this source version.

## What the tests established

- TRAIN uses the new finalized request root and `st_future_metadata_finalization_complete_v1`. Wrong root, old schema, wrong request path, raw-log-read flag, or missing source-shard reuse are rejected. DEV's previous path and schema remain intact.
- TRAIN runtime pins must specify exact frozen runtime `3df052ea422c4ba042ef3be33dea19438c5702dc` at `/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017`; wrong schema/path/commit are rejected.
- The controller executes the hash-pinned index preparer and its real `bind_finalized_train` path. Wrong terminal state, surviving owned child, original source runtime, or preparer hash is rejected. The shared `strict_finalized_receipt` validator is an explicit mock boundary in this suite: its implementation was independently checked previously, and is not re-proven by these 14 tests.
- Index result dependency checks include the new request root, avoiding an accidental return to the failed original TRAIN build.
- Synthetic preparation returns COMPLETE only for rc0, no monitor failure, zero live descendants, permitted reads, and the expected prepared-only child status. Read overflow, surviving children, waiting status, and monitor exceptions fail closed with no automatic retry. Unknown cleanup is not reported as zero.
- The command contains exactly one CPU child, CUDA is hidden, CPU0/1 is specified for the planned actual worker, and the preparation result explicitly says no ledger write and no index build. This command construction was tested with a mocked monitor; no worker was launched.
- Existing output/control directories refuse another run. The actual bounded control-file writer and metadata inventory operate only on synthetic `/tmp` directories in these tests. Combined one-MiB overflow rejects completion.

## Measured review cost

`TEST.log` is the complete output. Tests: **14 passed, 0 failures, 0 errors**. Wall: **1.412568 s**; RSS: **28,360,704 bytes**; total logical reads: **2,288,126 bytes** (startup 967,527; tests 1,320,599); test writes: **1,059,695 bytes**. CPU affinity was `[6, 7]`; Torch was not imported. An external 110-second timeout and 1-GiB address-space cap bounded the review process. These figures are synthetic audit costs, not full-population production cost claims. The source diff and targeted reports were also read; no recursive search or broad regression was performed.

## Runtime and source pins

`SOURCE_PINS.json` records exact controller, wrapper, preparer, and TRAIN runtime-source hashes. The controller reuses frozen core `4b2f0fccfd4bf3d655198f61179c88f6b2a51840731465c720b8addb6169b1a2` and supervisor `d829cf48764b23b5fce08cdd6ea45c6d3e61e633763ad2cb876e5ab8d96d8902`. No changes to those implementations were made by this review.

## Boundaries that remain

The functional PASS is scoped to this source and the tested branches. The resource reviewer confirmed a temporary FIFO blocks in `read_source` until its independent 200-ms alarm, while the first `load(ADAPTER)` occurs before the wrapper installs its own alarm. Evidence commit: `3be2757489ff272bed02c862bf2f90949d7de723`. This resource failure must be fixed and independently rechecked before calling the wrapper bounded or launch-ready. This suite does not cover that counterexample and does not grant resource admission.

Actual index preparation/consumption must still bind new finalizer completion and its observed terminal, preserve the failed original build lineage, verify the complete TRAIN/DEV populations and original selection order, and produce bounded resource and cleanup evidence. The previously documented full-population harness tail assertion remains unresolved by these synthetic tests. No claim of full population order validation is made here.

Run the review (no production launch):

```bash
timeout --signal=TERM --kill-after=5s 110s taskset -c 6,7 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261017/finalized_index_controller_correctness_v1/run_review.py
```
