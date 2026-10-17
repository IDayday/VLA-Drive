# Independent history64 pilot review

Reviewed Lead commit `80cc3e34e09c58efaef1b63caeb6ea9c801afbf3`, implementation `77181fc` + `a673d7c`. Reviewer `/root/st_history_pilot_independent_v2` did not implement or alter the reviewed writer, monitor, ledger or scientific code. This review changes only this report directory.

**Result: FAIL_FINITE_WRAPPER_BOUNDARY. Do not dispatch the production pilot from this version.** The causal data path is acceptable; three narrow resource/receipt defects remain. They do not invalidate existing model checkpoints or require a model redesign.

## Findings and minimum repairs

1. **H64-1, P1: the launch is not bounded end to end.** `run_pilot_once.py:97` calls `core.publish_guarded` before `core.monitor`. That enters `metadata_writing` and the original `StorageBudget._locked` blocking `flock` at `cache_storage_budget.py:226`. Neither the 900-second work timer nor the owned lifetime signal scope has begun. An already active TRAIN metadata writer therefore causes an unbounded wait instead of a prompt refusal. The result publication at line 107 can also block after monitor exit. The production wrapper does not acquire a nonblocking supervision lock across this interval. Its read-only START file descriptor is an ownership token, not an acquired exclusion lock.

   The independent fixture uses the **actual** `_locked` and `publish_guarded` with an isolated temporary ledger. While another fixture owner holds the lock, the publication process remains alive without a receipt; it proceeds only after the lock is released. An AST check establishes that the actual adapter invokes this publication before monitor creation. The test lasts 0.3 seconds, not 940 seconds: it demonstrates the blocking call and absence of a timer rather than pretending to measure a 940-second timeout.

   Keep the single real ledger and fail safely on contention. A narrow fix should bound the entire wrapper invocation, include pre/post publication in that deadline, and retain an independently acquired nonblocking supervisory lock. A momentary lock probe followed by the same unbounded acquisition is not sufficient because another writer can enter between them. Do not bypass storage checks or modify the active TRAIN runtime.

2. **H64-2, P2: setup reads are omitted.** `controls` reads the admission and all controller sources and imports helpers before it loads `run_build_once.py` at line 57. That module records `BOOT_IO` only when loaded. Later `charge_after_reap` uses this late baseline. The independent counterexample performs a 1 MiB pre-import control read and demonstrates it is excluded. Capture the IO/time baseline at wrapper entry and thread the immutable baseline into the existing monitor/terminal accounting. Child lifetime IO and the repaired Linux wait-boundary formula are correct in the tested normal path; do not reintroduce double counting.

3. **H64-3, P2: an exceptional monitor exit has no terminal failure receipt.** An exception at line 103 bypasses all result publication. Fault injection into that exact call leaves only the START marker. Preserve a structured bounded failure, cleanup evidence and conservative accounting for exceptions. The failure writer itself must respect H64-1's absolute deadline; an exception handler that blocks on the same ledger would not resolve the problem.

## Checks that passed

- The fixed small control was actually read and its SHA matched: 64 unique train scenes from 64 logs. No reselection based on losses or scores occurs. The proposal's full train/dev identities are separate, and no full-scale launch is granted.
- The wrapper only permits registered `target_times=[0]` and delegates the existing causal history computation: negative offsets, same log, true missing masks, current timestamp/view binding, and rejection of future sensor timestamps. Original GT tensors are hashed opaquely for identity, not decoded as model inputs. The raw log pickle contains annotation fields, but those values are not consumed by the history feature constructor. Poisoning label/future geometry fields in synthetic fixtures leaves history outputs unchanged.
- No teacher, visual model, optimizer, training step or new teacher tensor bank is invoked. Current-only teacher request metadata remains separate from history input records.
- The actual wrapper passes `(6,7)` into the reused `monitor`, whose `taskset` construction consumes the parameter. Thus the earlier C4 hard-coded `(0,1)` issue is absent. This review's real tiny subprocess uses authorized CPU8/9 and verifies the same parameterized path; it is not presented as an actual CPU6/7 history-data pilot.
- The pinned checkpoint receipt really contains 2303 GiB remaining obligations and a 1 TiB free floor. Code enforces its hash, amount, lease and fresh free space; the original storage context also validates the receipt schema/current interval. The receipt is time limited and must still cover the eventual actual launch, so this check does not authorize using an expired receipt.
- The original campaign ledger serializes mutations and retains claims/failed partials. It cannot run two writes concurrently. That safety property does **not** prove finite lock acquisition.
- A real tiny child exited normally with no owned process remaining. Terminal IO counting includes the child once. Actual-start lease shortage is rejected before `Popen`. Missing admission and widened current-only scope are rejected.

## Actual test evidence and limits

`TEST_LOG.txt`: **26 passed, 2 deselected**, pytest 3.93 seconds; outer execution 5.42 seconds. Peak observed test-child RSS 644,788,224 bytes; parent-after-reap total logical reads 141,430,371 bytes. All fixtures remained within CPU8/9, 120 seconds, 4 GiB RSS, 256 MiB reads and 16 MiB output. The two deselections avoid the owner's real CPU6/7 fixture and an unnecessary torch collation test outside this review's narrow CPU budget. A torch import emitted a deprecation warning; CUDA devices were hidden and no GPU workload was run.

The tests named `test_reproduce_*` intentionally pass when the existing defect is reproduced. Their green pytest status is **not** a green resource qualification. Full command and counters are in `TEST_RUN.json`; reviewed code/control hashes and per-check verdicts are in `REVIEW.json`.

No raw scene/log/image, teacher/model/checkpoint tensor was opened. No actual metadata admission, claim, ledger write, pilot, SSH dispatch, process signal to a live campaign job, GPU job or full cache was executed. Real 64-scene cost remains **NOT_RUN**. Full-history scale and its unchanged 20,000-entry/O(N²) census boundary remain **NOT_ADMITTED**. Repair and independently recheck only the affected wrapper boundaries; retain this negative evidence.
