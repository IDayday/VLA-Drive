# Count terminal parent/child reads once across wait

This change starts from `1ee0660c901547df23cc6f852834954ca112bd38` in the new isolated worktree `/mnt/project/VLA-Drive-st-metadata-final-accounting-20261017`. The active controller worktree is untouched. Its source still hashes to `a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`. No active TRAIN process was signalled, restarted or reconfigured; no actual builder, model, cache or ledger was accessed by these tests.

## Confirmed defect and preserved history

Independent commit `c14f0aa255d038feeb4f78cce212c33417c19302` established that Linux parent `/proc/io` incorporates waited-for child IO. The old controller read the parent after `owner.finish()`/wait, then added the same child's terminal `rchar` again. This was an erroneous double count, not an intended resource multiplier.

The already completed dev attempt2 recorded 8,595,964,797 bytes: child 4,273,867,611 plus post-reap parent 4,322,097,186. The latter includes the child; net parent beyond it was 48,229,575. Those quantities are diagnostic arithmetic from the independent small receipt. The historical receipt, existing cumulative 10,798,774,978-byte dev charge and earlier 2,202,810,181-byte failed-attempt charge remain unchanged. This patch does not rewrite historical spent balances or imply a real TRAIN completion.

## Minimal accounting change

Active monitoring remains parent baseline delta plus the unreaped owned child's current absolute count. PID/session/start identity checks, bounded process reads and the existing pidfd cleanup are unchanged.

Immediately before reap, the new controller saves the parent's counters alongside the verified child's terminal counters. Immediately after successful reap, it saves the parent's counters again. It checks monotonicity and verifies that post-reap parent reads include at least the pre-reap parent plus the terminal child. The final aggregate is then:

`post_reap_parent.rchar - parent_baseline.rchar`

The child is not added again. Parent activity between the two samples is explicitly retained as `post_reap_parent - pre_reap_parent - child`. Later parent reads before RESULT construction are added only as `current_parent - post_reap_parent`. The receipt retains all samples and uses `st_wait_boundary_read_accounting_v1` to describe this calculation.

Missing or inconsistent parent/child terminal evidence still produces a failed result with the entire remaining read allowance charged conservatively. Exact integer checks reject bool, negative and fractional counters. A nonmonotonic later parent sample is rejected. No process cleanup is repeated after successful reap.

Unchanged: every role budget, prior-charge constant, no-auto-retry rule, original builder/PLAN/source pins, storage policy, checkpoint commitments, root capacity, GPU prohibition and sidecar-free observation. This version applies only to a later independently admitted invocation. No index adapter or new framework is added.

## Current evidence

- 61 CPU tests PASS: original 44 and 17 new wait-boundary tests.
- Two new real owned child cases read a 4 MiB fixture each; the parent reads its own 2 MiB fixture before reap and again after it. Normal exit and exit 3 both demonstrate once-counted child IO, retained parent IO, and zero owned residue.
- New fixture payload totals 12 MiB across the two cases; no image/raw-log/model/native checkpoint fixture is used.
- Synthetic arithmetic covers nonzero baselines, post-wait parent tail, missing/inconsistent samples and the exact historical dev double-addend numbers.
- The full regression also retains seven existing tiny owned CPU lifecycle/resource tests.
- Bounded rerun: CPU affinity 0/1, 2 GiB address-space ceiling, 180-second deadline. Tests passed in 2.82 seconds, complete wrapper 3.193 seconds, peak RSS 69,353,472 bytes. Exact output and limits are in `TERMINAL_ACCOUNTING_TEST_OUTPUT.txt` and `TERMINAL_ACCOUNTING_TEST_RESOURCES.json`.

New controller SHA256: `a775a87532dab70a376cb64a7c0c19403e21e2f3a3b8040152fbefb8cebf10b9`.

Independent review is pending. This owner has not signed a TRAIN success, dispatched a real build, changed the ledger or revised any prior result.
