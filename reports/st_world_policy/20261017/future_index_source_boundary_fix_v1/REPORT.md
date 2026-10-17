# Index preparation bootstrap boundary correction

Scope: `future_index_sequential_control_v1/prepare_once.py`, based on lead
`af4461eee927a288211c96f91af55f55a981bb1c` and owner implementation
`bf67fad2b77e443395635f8871eb757cbbb8c596`.

The independent resource review at
`3be2757489ff272bed02c862bf2f90949d7de723` remains FAIL in its original tree.
Its FIFO counterexample showed that the first pinned adapter read could block
before the invocation alarm was installed. No previous receipt is changed.

The narrow fix rejects non-regular files, opens with `O_NONBLOCK|O_NOFOLLOW`,
checks descriptor and path identities, reads at most 1 MiB plus one overflow
detection byte, and checks the final checksum. The regular-file check is
repeated on the descriptor so replacing the file with a FIFO between `lstat`
and `open` cannot introduce a blocking read. Changes to inode, size, mode,
mtime or ctime reject the source.

A small bootstrap alarm now covers argument handling and the first adapter
load. It uses the original absolute `START+120s` deadline, then hands off to
the existing pinned finite adapter with that same deadline. It rejects an
existing caller timer and restores its previous signal handler. The existing
monitor, accounting, 120+40 seconds, 384 MiB read cap, 1 GiB RSS cap, CPU
placement, one-MiB output allowance, and one-shot behavior are unchanged.

Actual local regression command (no workload launch):

```bash
taskset -c 10,11 timeout 120s /root/miniconda3/envs/ddp/bin/python -B \
  reports/st_world_policy/20261017/future_index_source_boundary_fix_v1/test_boundary.py \
  > reports/st_world_policy/20261017/future_index_source_boundary_fix_v1/TEST.log 2>&1
```

14 tests passed. They cover the original FIFO, FIFO/symlink replacement races,
bounded growth, source replacement, existing and expired alarms, stalled
initial loading, exact deadline handoff, and unchanged monitor source pins.
Measured total script wall time was 0.099611 seconds, peak RSS 25,976,832 bytes,
once-counted process logical reads 1,671,872 bytes, and kernel reads 143,360
bytes (reported separately). Fixtures used at most 2 MiB logical file extent;
no payload, checkpoint, huge manifest, ledger, GPU, or actual preparation was
accessed. All test processes exited normally.

New `prepare_once.py` SHA256:
`49b21405d1dacee17405598c1a7165727db9855af4de66c2feff70ab30b93ce5`.
Existing launch/source-pin documents must be rebound to this exact source
only after independent review. This author regression is not launch approval.
