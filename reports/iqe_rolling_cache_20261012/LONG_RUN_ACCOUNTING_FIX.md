# Long-run accounting and observed worker failure

The original rolling producer counted residency using a glob over every file
in every historical generation, calling `is_file()` before filtering suffixes.
At95 windows this cost0.2185s per call on average; linear extrapolation to3342
windows suggested7.69s, an estimate rather than a full-scale measured result.
The monitor had a related lifetime-growth problem: archived JSON receipts would
exceed its8192-entry parsed-JSON LRU and eventually its per-observation budgets.

There was also an observed failure, retained in `producer_rate96_0.log`: worker0
exited on an ENOENT for a `.partial` path in window96. The file had disappeared
between type checking and the separate stat in residency accounting. The existing
worker pool still finished the requested window and Base continued. This is a
producer infrastructure failure, not a bad trajectory, missing valid label or
reason to reset training.

## Narrow producer fix

`Pool.resident_owned_bytes()` now uses `OwnedPayloadInventory`: windows and
generation directories are tracked by device/inode/size/mtime/ctime; live payload
and partial files are stat'ed each time, including zero-byte growing partials.
Only empty, committed, unchanged generations skip repeated directory walks.
New windows, new generations on rewind and changed directories are discovered.
Symlinks are refused. Atomic rename/GC disappearance is handled without treating
the transient name as a worker failure. This is occupancy accounting; payload
record SHA256 and immutable receipt validation remain in the existing reader and
writer. Capacity/reserve thresholds and the original sampler are unchanged.

The consumer hashes its execution modules for exact resume. Therefore this fix
was committed in a separate producer worktree, branch
`perf/iqe-rolling-producer-index-8fe9f61`, commit
`27f3b5e6a1d3896c67e1ac1a8419113d893e077b`, based on8fe9f61. The running consumer
still uses its original8fe9f61 source files. The frontend/codec/shard schema is
identical, so old and new workers can cooperatively fill the same windows.

Real read-only accounting at114 windows measured original0.380/0.333/0.269s and
new0.050s cold,0.0115/0.0102s warm. All measured byte totals happened to match;
concurrent writes are allowed, so stable-fixture tests provide the exact count
comparison. These timings measure accounting, not an additional end-to-end
training speedup.32 producer/core tests passed, including a real filesystem
atomic-rename race, partial growth, old-window rebuild, symlink rejection and
no repeated walk of archived receipt files.

Producer processes are replaced one host at a time after verifying PID, start
ticks, full argv and cwd. Only those frozen producer processes receive SIGTERM;
the current shard can commit, or an interrupted temporary shard can be recovered
by the existing atomic writer. No Base process receives a signal. One absent
worker is explicitly accounted for by its preserved FAILED log. A `/proc` exit
race interrupted the first administrative wait; inspection confirmed the old
processes had exited, then the same replacement was resumed without duplicates.
Launch receipts preserve the old workers and original24h deadline. Existing
complete cache shards are reused.

## Monitor fix

The read-only observer now maintains compact immutable receipt counters instead
of reparsing the full historical tree. Every historical receipt is still stat
checked, and a changed identity forces comparison against its original checksum.
Edits, replacement, deletion and added archived receipts are rejected. Active
generations are rescanned; a checkpoint rewind adds its new generation once.
Late cold initialization proceeds across multiple observations within the same
128MiB budget. It reports NOT_VERIFIED until complete and never invents a rate.

20 tests passed. The3342-window worst-case fixture (512 misses and43 shard pairs
per window) exercises294096 checks under the500000-entry ceiling and performs
no historical JSON re-read. This is a simulated full-size metadata fixture,
separate from the real temporary-file and real-pool tests. A real initial scan
of101 generations used15.44MB metadata and21490 entries. The monitor is separate
from the training execution identity and can be replaced without changing any
model or optimizer state.

The first measured200-update window remains1.2353s/update with8.086% cache wait.
The long-run fixes remove an observed failure and growing metadata overhead;
they are not presented as a new200-step throughput result. Mild periodic supply
waits remain a monitored limitation.
