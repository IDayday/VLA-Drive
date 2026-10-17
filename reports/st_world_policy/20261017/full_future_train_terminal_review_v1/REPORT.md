# TRAIN metadata terminal accounting: bounded independent snapshot

Reviewer: `/root/st_owned_descendants_review_v2`, independent of the builder and controller implementation. This review made no task signals, GPU calls, dispatch, cache mutation, or ledger writes.

**Result: SNAPSHOT_ONLY_TERMINAL_NOT_OBSERVED.** Between 2026-10-17 12:58:23 and 13:07:56 UTC, parent `480124/start1366936766` and child `480140/start1366936849` remained present with matching identities. Neither `COMPLETE.json` nor `CONTROL_BUILD_RESULT.json` appeared. The last legal 30-second sample was 573.088 seconds after the first observation; another such sample would exceed the 600-second window. No later target inspection is included.

This does not establish successful completion, an actual terminal duplicate charge, a resource PASS, or permission to retry. It does not alter an original FAILED result. In particular, all shards existing is different from final metadata publication succeeding.

## Current task identity and preserved shards

The 13:06:55 UTC additional inspection was explicitly requested by the lead before a possible controlled stop. Both processes retained the original controller argv and the same admission hash `e7cd20b754b4f8fab6b3045d1ab7f70e79b1e4299966ce80a0096761e5cc7389`. The child has `--child`; both cwd values are `/mnt/project/VLA-Drive-st-metadata-build-train-v1-20261017`, whose frozen runtime is `838dd21f7a16a79086a44bc63be056d6a750570e`.

Only directory entries and stat values were inspected. `log_shards` contained exactly 1176 regular `.json` files, no symlinks, no non-JSON/partial files, and 984,972,623 logical bytes. No shard contents, raw logs, teachers, images, or large requests/source index were read by this reviewer. This is not checksum/coverage verification of those shards.

At the last observation:

| Process | rchar bytes | VmRSS KiB | Identity |
|---|---:|---:|---|
| Parent 480124 | 1,271,369,845 | 64,692 | Matched before and after read |
| Child 480140 | 300,687,765,796 | 1,053,964 | Matched before and after read |

These are absolute live `/proc` counters, not the final budget charge or a retrospective peak claim. `read_bytes` is a separate kernel physical-IO counter and is not substituted for the registered `rchar` budget.

## Original terminal double-count path

The untouched original controller SHA256 is `a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`:

- `monitor`, line 239, calls `owner.finish(...)`, which waits/reaps the owned child.
- Lines 249–256 retain the maximum child lifetime `rchar` from last-active, terminal, and child-final records.
- `run`, line 299, then calculates `charged_child + current_parent_rchar - BOOT_IO_rchar`.

The parent counter after wait includes the waited child's accounting. Thus this source contains the known duplicate-child terminal path. The source-level issue is established; **its terminal result for this invocation was not available within this review window**.

For a later unchanged original result, retain all original fields and status. If `T` is its recorded `charged_parent_child_rchar_bytes` and `C` is its recorded `resources.charged_owned_child_rchar_bytes`, `T-C` exposes exactly the parent-after-wait-minus-baseline term that the old formula added to `C`. Validating that as a one-charge bound also requires the actual child terminal/final counters and verified wait/cleanup outcome. This report does not supply fabricated terminal values or turn that calculation into BUILD_COMPLETE. A controlled stop remains a failed/incomplete attempt even if corrected accounting fits the allowance.

## Lead-requested stop-path review

The admission's actual `supervise_once.py` was independently hashed as `a29527dfbe23a4563a66266532f44644f5f184f4a8f59842883524632a63f2d6` in the frozen spatial-probe runtime-v2 directory.

`lifetime_scope` replaces SIGTERM/SIGINT handling with a stop flag instead of interrupting cleanup or publication. The original metadata monitor sees the flag, records `SUPERVISOR_SIGNAL_15`, signals registered live owned pidfds, escalates only if needed after 30 seconds, and retains a 40-second cleanup deadline. `OwnedSession` checks PID/start/SID, uses WNOWAIT until closure, and only reaps its verified owned/adopted children. It does not reconstruct authority from a historical PID receipt.

Following cleanup, the original controller attempts to publish `FAILED_NO_AUTO_RETRY` because a stop is a failure condition. Publication is still subject to the original metadata guard and can itself fail. This old helper covers the original spawn session; it must not be described as the later cross-SID descendant helper. The current read-only review has not observed cleanup, sent a stop, or certified that no residual process exists after a future stop.

## Observer resource evidence and limitation

Command:

```bash
taskset -c 8,9 python -u reports/st_world_policy/20261017/full_future_train_terminal_review_v1/observe.py
```

The collector made 17 bounded observations, read 74,998 target/proc payload bytes, and wrote 32,445 bytes of observation JSON. The initial tool observation and later explicitly requested identity/stat census are separate preserved artifacts. Administrative source/admission reads were small; no exact aggregate rchar measurement covering every tool invocation is claimed.

**Observer resource status: FAIL_REPORTED_RSS_OVER_CAP.** Its own retained `getrusage(RUSAGE_SELF).ru_maxrss` is 1,569,472,512 bytes, exceeding the 256 MiB cap, despite setting `RLIMIT_AS=256 MiB` after imports. No independent live `/proc` RSS of that observer was retained, so the cause cannot be resolved from this run. Two later small interpreter checks do not substitute for the original measurement and are not used to claim a PASS. The raw abnormal value is preserved in `OBSERVATION.json` and `REVIEW.json`; no full re-observation was launched to overwrite it.

`RLIMIT_FSIZE`, cumulative output checks, a bounded read helper, CPU affinity 8/9, and the absolute observation deadline were enforced by the collector. The resource-accounting limitation does not transform the recorded target PID/start/io values into a completion result.

## Evidence and remaining boundary

- `FIRST_TOOL_SNAPSHOT.json`: fresh initial PID/start/io observation.
- `OBSERVATION.json`: all 17 before/after identity checks and small receipt probes.
- `CONTROLLED_STOP_PREFLIGHT.json`: extra lead-authorized identity/argv/cwd and shard stat census.
- `REVIEW.json`: source hashes, scope, state and evidence hashes.
- `observe.py`: exact read-only collector used.

Remaining work belongs to a separately bounded terminal receipt inspection after the actual process exits. No actual cache completeness, dataset coverage, scientific result, future full-run approval, or historical peak-RSS approval is granted here.
