# Terminal read-accounting addendum

The frozen `a866ae42…` controller double-counts the completed child at final accounting. This is an accounting defect, not evidence that the active train has exhausted its 512 GiB allowance. The controller, running task, budget, prior charges, and ledger were not changed.

## Evidence

Linux `/proc/PID/io` includes the process and its waited-for children. `rchar` measures successful read-like syscall bytes; physical `read_bytes` is a different measure. [Linux man-pages: proc_pid_io(5)](https://man7.org/linux/man-pages/man5/proc_pid_io.5.html).

In the pinned controller, lines214–218 capture the child's terminal counters before reap. Line239 calls `OwnedSession.finish`; the pinned original helper calls `child.wait()` at line271. Line299 then adds the terminal child `rchar` to the parent's delta **after** that wait, although the parent's counter already includes the child. The active-monitor check at lines228–230 occurs before that reap and does not have this same terminal duplication.

The completed dev attempt2 supplies exact arithmetic:

| Quantity | Bytes |
|---|---:|
| Recorded final charge | 8,595,964,797 |
| Terminal child lifetime rchar | 4,273,867,611 |
| Derived parent delta after reap | 4,322,097,186 |
| Parent delta beyond the same child | 48,229,575 |
| Duplicate child addend | 4,273,867,611 |

The child's terminal value agrees with its final stdout counter up to131bytes of final activity. The parent-after-reap delta is the reconstructed once-counted process-tree contribution for this invocation, not a rewritten receipt or a new approved spent balance. The existing cumulative dev charge remains10,798,774,978bytes. The earlier failed dev has the same approximate doubling pattern, but its sampled child counter is less exact; no exact correction is asserted for it.

## Declared budget and consequence

The actual train admission expressly says `512GiB logical parent+child read budget`; it does not declare deliberate double weighting of child reads. Final accounting implements a different, overly conservative measure. This can reject a completed builder at receipt finalization even though each process's logical reads counted once remain below the limit. Repeated legitimate reads still count normally; this finding does not imply unique-file-byte accounting.

The preceding two live snapshots remain sealed in commit `57a0a6e9`. Their preliminary full-run projection used the dev final/child ratio, so it reflects the current duplicate-charging implementation, not proof of real budget exhaustion. Keep the active512GiB limit and controller unchanged. No stop, restart, automatic retry, or budget expansion is authorized by this addendum.

## Minimal next change, separately reviewed

For a future version, define one consistent process-tree accounting rule across the wait boundary. For this single owned child, the parent's after-wait delta already includes the child; do not add that child again. Preserve pre-reap child accounting for live enforcement/failure fallback, avoid silently losing descendant counters, and retain existing conservative behavior when evidence is unavailable. Add a tiny real-child read/wait test covering live and terminal counters, nonzero exit and cleanup; retain old receipt charges as historical facts. Do not hot-patch the active process or amend pinned admissions.

This review used only the small pinned sources and existing dev/admission JSON. No image, raw log, tensor, model, large index, active process control, or ledger write was used. Exact paths and hashes are in `EVIDENCE.json`.
