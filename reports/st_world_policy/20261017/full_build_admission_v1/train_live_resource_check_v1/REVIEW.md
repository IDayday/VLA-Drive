# Bounded live train metadata resource check

**RUNNING_HEALTHY_WITH_BUDGET_WATCH; keep the current invocation unchanged.**
Two read-only snapshots were taken at Unix 1792231135.0591934 and
1792231199.502745 (09:58:55 and 09:59:59 UTC). The requested roughly-one-minute
interval was actually 64.44 seconds including collection/tool overhead; there
were no further snapshots. Source reads by the bounded snapshot script totalled
3,556,207 bytes; preliminary PLAN/header inspection added under 0.3 MiB. This
review stayed below 16 MiB, used CPU0 and a 512 MiB address-space ceiling, and
read only small controls, /proc, file stats and the small trailing selected-row
metadata of completed shards. No payload/image/raw-log/tensor was opened.

The exact approved parent PID480124/start_ticks1366936766 and child
PID480140/start_ticks1366936849 were unchanged across both snapshots. Cmdlines
bind the admitted controller and train token, cwd is the pinned train runtime.
The parent waits in ep_poll. Child main-thread futex waiting is consistent with
its single worker performing work; open image paths advanced to another log.
No result or COMPLETE file exists yet. No processes were signalled or dispatched.

| Measure | First snapshot | Second snapshot |
|---|---:|---:|
| Time since START | 380.27s | 444.72s |
| Completed log shards | 17/1176 | 21/1176 |
| Scenes in completed shard manifests | 3470/101592 | 3968/101592 |
| Parent+child sampled RSS | 1,021,157,376B | 1,021,161,472B |
| Root inclusive allocated/logical charge | 33,931,264B | 38,825,984B |
| Running sum of absolute proc rchar | 9,585,276,115B | 11,250,028,908B |

The second snapshot is about 3.91% of scenes. It added 498 scenes and four logs
in 64.44 seconds: 7.73 completed scenes/s recently, 8.92 scenes/s since START.
A straight scene-rate estimate gives roughly 3.0–3.5 hours remaining, inside the
original four-hour invocation window at this early point. It is not an ETA
promise. Log sizes differ greatly; extrapolating only log count is misleading.
The longest observed commit interval was about 89s for an 824-scene log, then
62s for 543 scenes and 47s for 450 scenes. These are adjacent-file-mtime intervals,
not instrumented per-log timing.

The three largest raw logs are still ahead: 171,848,564, 147,814,233 and
137,629,626 bytes. All fit the explicit 192 MiB raw-input cap, but that does not
prove their decoded RSS. Current RSS has substantial margin under 4 GiB; no
worker or resource budget was changed.

CPU consumption across this interval was 28.28 CPU-seconds, or 0.439 core
 equivalents / about22% of the two-core allocation. Child logical rchar grew
1.654 GB; physical read_bytes grew15.202 GB. This is consistent with substantial
I/O/pacing waiting, but these snapshots cannot uniquely separate image hashing,
filesystem read-ahead, deliberate pacing and other waits. The 64 MiB/s pacing is
not a physical Lustre bandwidth ceiling. There is no evidence of a stopped or
stuck process, and no new performance changes were attempted.

## Budget risks worth watching

Current root use is only about37 MiB of the2 GiB limit. Extrapolating current
shard bytes per completed scene and the actual dev final/shard ratio gives
about1.82 GB final metadata. This is a rough projection, not reserved-size proof;
full finalization, partials and large later logs stay under the existing cap.

Running absolute rchar sum currently leaves about538.5 GB of the549.76 GB
logical read limit. It is not the final receipt charge. In the actual completed
dev run, final parent+child charge was2.011 times the owned child's lifetime
rchar. The parent final count was close to the child count plus overhead. At
this early train per-scene cost and that actual multiplier, final charge projects
to about574.2 GB, slightly above512 GiB (549.76 GB). This is an **early accounting
budget risk**, not an observed overrun. It must not be hidden by presenting the
live sum as an accurate forecast of final charge. Source accounting remains
unchanged for the active invocation; a potential duplicate-accounting question
can be assessed separately from changing the running budget.

No stop, retry, new claim, budget adjustment, worker increase, teacher tensor
cache or GPU work was performed. The original complete scene population remains
intact. Lead retains sole monitoring/dispatch ownership.
