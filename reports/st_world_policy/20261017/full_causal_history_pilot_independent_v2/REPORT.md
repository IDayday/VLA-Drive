# Independent H64 boundary re-review

Reviewed exact implementation `3b5869cabfccaceca3562b1842abb6e326d2561a` (Lead integration reference `b7bdbe47`). Reviewer `/root/st_history_pilot_independent_v2` changed only this new report directory. The prior negative review `b20c9b5b` remains intact.

**PASS_CODE_CONTROLLER_BOUNDARY. Actual launch remains NOT_ADMITTED.** This is approval of the three local wrapper repairs, not evidence that real history64 processing, full history production, actual resource reservation or scientific training has completed.

## H64-1: finite lock and invocation boundary

The adapter now applies `LOCK_NB` to the **actual** original `fcntl.flock` acquisition inside a scoped context. It does not probe and then perform a second blocking acquisition. START/RESULT publication and child metadata admission use that adapter; the original ledger/state validation remains in place. The real original `_locked` fixture rejects a held synthetic lock immediately, creates no artifact START, and emits the separately bounded failure receipt. It succeeds after release, and the patch restores the original flock function on exception.

The wrapper captures entry monotonic time, encloses controls/publications/execution in a real SIGALRM deadline, subtracts elapsed preparation and a five-second terminal reserve before the child monitor, and preserves the existing forty-second cleanup limit. The independent real child timeout fixture terminated only its owned child and recorded zero remaining owned processes. No campaign process was signalled.

A real nonblocking supervisory lock is held in the control directory throughout parent work and terminal handling. A competing supervisor is refused. Failure handling does not reacquire the original ledger. The existing marker, immutable output and no-retry semantics remain.

## H64-2: complete entry baseline

`ENTRY_IO` is captured at wrapper entry before admission/source/helper reads and assigned to the reused core's `BOOT_IO`; the same applies to monotonic start. The final successful charge is recomputed after the ledger RESULT publication. An independent 1 MiB control-read example is included by this baseline. The existing repaired wait-boundary calculation still counts the reaped child's reads only once.

The tiny terminal receipt's own serialization is bounded separately by the control cap. Measurements remain logical `/proc` reads, not physical storage bandwidth. No full-history or real-image IO rate is inferred.

## H64-3: honest terminal failure evidence

START contention, monitor exceptions and RESULT contention produce immutable `FAILED_NO_AUTO_RETRY` evidence without waiting on the original ledger. Missing complete accounting is conservatively charged at the full 16 GiB allowance. Before child launch cleanup is `NOT_STARTED`; on an exceptional monitor exit it remains `UNKNOWN_MONITOR_EXCEPTION`. Only an actually returned monitor cleanup record can supply a known cleanup outcome. Existing evidence cannot be overwritten.

## Separate control reservation: code checks versus actual admission

The admission must name the exact control root and exactly 1 MiB; missing or altered values are rejected. The artifact registration is limited to 31 MiB, preserving the combined 32 MiB cap. Runtime census permits only the small declared control filenames, rejects unexpected files, symlinks, hardlinks and excess allocation, and includes temporary files. Writes are exclusive and immutable.

**MISSING_ACTUAL_CONTROL_RESERVATION:** this repair does not itself create or validate an already existing campaign allocation for the new control directory. There is currently no real Lead-issued control reservation to approve. Before dispatch, Lead must wait for the active original-ledger writer to become quiet, explicitly reserve the 1 MiB control allocation and at most 31 MiB artifact allocation in the existing campaign budget, and pin that actual evidence together with the fresh admission. A report assertion is not a resource ledger entry. If the existing budget contract requires a small additional reservation-source binding, review that delta before dispatch; this does not reopen unrelated scientific reviews.

The actual checkpoint commitment/lease, fixed scene list, host, immutable runtime and free-floor checks must still succeed at launch. This review neither renews the existing checkpoint lease nor grants a real pilot approval.

## Unchanged behavior and actual test cost

The causal writer, original history math, future metadata helper, original ledger implementation, history schema and reused owned monitor are byte-identical to the earlier reviewed versions. The only changed file among the eight v1 source pins is the history pilot wrapper. Its current SHA256 is recorded in `REVIEW.json`. Prior fixed64/current-only/causality and explicit production CPU6/7 routing evidence are inherited; no scientific computation changed.

Actual rerun: **32 passed, 1 deselected**, pytest 3.71 seconds; measured outer duration 5.03 seconds. The deselected owner's tiny child uses CPU6/7; this review's actual child runs only on CPU8/9. CUDA was hidden.

- Peak complete observed test process tree RSS: **642,301,952 bytes**.
- Parent-after-reap cumulative logical reads: **141,403,052 bytes**.
- Peak observed descendant process count: **2**.
- No 120-second, 4 GiB RSS, 256 MiB read or 16 MiB output limit was reached.

`TEST_RUN.json` records the exact command and accounting scope. A torch import produced the retained pynvml deprecation warning; it did not launch GPU work. Tests use only synthetic temporary files/processes. No real scene/log/image, checkpoint tensor, campaign ledger, claim, GPU or SSH task was accessed or modified. The actual history64 cost is **NOT_RUN** and full-history scale remains **NOT_ADMITTED**.
