# Existing monitor with an outer bootstrap deadline

Two independent synthetic fixtures pass. They do not build an index, modify a ledger, read real payload, or signal any existing task.

1. The actual `run_index_once.py` module executes with only its first pinned-core read redirected to a temporary FIFO. `timeout --signal=TERM --kill-after=40s 0.3s ...` interrupts the blocking import in 0.305 seconds, returns 124, and leaves no fixture process.
2. The actual pinned `core.monitor`, ownership/session/pidfd helper, and lifetime signal handler run a synthetic child and grandchild. Outer TERM at one second reaches the existing monitor, which records `SUPERVISOR_SIGNAL_15` and finishes after 1.355 seconds with two verified owned processes and zero live remainder. Both recorded fixture PIDs are absent. Only the monitor's memory-provider function is a fixture-local substitute; ownership and cleanup are the existing real implementation.

The test process used CPU8/9, 1.689 seconds wall time, 17,788,928-byte harness VmHWM, a sampled 42,373,120-byte fixture-tree RSS peak, and 5,837,583 logical read bytes including reaped children once. Files and payload are small synthetic controls. No Torch/model import was needed.

For actual builds the exact default GNU timeout prefix is `--signal=TERM --kill-after=40s 900s` for DEV and `--signal=TERM --kill-after=40s 3600s` for TRAIN. Do not add `--foreground` or `--preserve-status`. Its 40-second escalation is the same cleanup envelope already allocated to the core, not another 40 seconds. A timeout exit, failed cleanup or missing receipt stays a failed attempt without automatic retry.

This bounded composition can cover the controller's otherwise unguarded initial source read without introducing another monitor or changing model/data code. Source pins, runtime checks, whole-process read/RSS/storage checks, ownership checks and the core's own absolute deadline still apply. An immutable pathname/hash alone is insufficient; the admitted executable prefix supplies the startup deadline.

This report does not authorize the actual build or claim full-population throughput. Actual role admissions still require the completed preparation, additive migration receipt, fresh checkpoint reservation and exact command including this prefix. The original source-boundary FAIL and all failed historical real attempts remain unchanged.
