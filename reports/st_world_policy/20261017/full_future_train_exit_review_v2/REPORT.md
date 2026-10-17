# TRAIN metadata: independent controlled-exit verification

Reviewer: `/root/st_owned_descendants_review_v2`. Separate authorization: one read-only inspection, CPU8/9, at most90 seconds,8 MiB reads and1 MiB reports. No signals, task launch, payload/raw-log reads, or ledger mutation.

**CONTROLLED_STOP_CONFIRMED_ORIGINAL_FAILURE_RETAINED.** Fresh observations at13:14:09 and13:14:47 UTC on2026-10-17 find both exact former process identities absent: parent480124/start1366936766 and child480140/start1366936849. The retained stop receipt's actual timestamp is13:13:27Z; the approximate13:16 time in the task handoff is not used as evidence.

The original `CONTROL_BUILD_RESULT.json` SHA256 is `0a29abe2f9f57ba3a70b923abb9dccb839aa8c65a3cdf374d4f4d402cc75fca6`. Its status remains `FAILED_NO_AUTO_RETRY`, failure `SUPERVISOR_SIGNAL_15`, child return code `-15`, verified owned processes1 and remaining owned processes0. The child supplied no final accounting/CLI artifact. `COMPLETE.json` is absent. This reviewer neither changed these fields nor authorized a retry.

## Retained data and local lock observation

Directory/stat inspection finds1176 regular `.json` files in `log_shards`, totaling984,972,623 bytes, with no symlinks or non-JSON/partial files. These counts match the preceding pre-stop stat snapshot. No shard payload/checksum or semantic coverage was inspected, so this is preservation evidence rather than a completed-cache certificate.

`BUILD.lock` remains a zero-byte regular file, device major/minor/inode `941:925238:144115927531689814`. `/proc/locks` contains no matching record, and its former owned processes have exited. This confirms no local kernel lock record is visible at inspection time. No exclusive/shared lock was acquired by the reviewer, and no claim about an unknown remote host's lock state is made.

## Terminal duplicate charge actually occurred

The exact original source was checked in the preceding report: it waits/reaps the child before computing `T = C + parent_after_wait_rchar - BOOT_IO_rchar`. The resulting record contains:

| Quantity | Bytes |
|---|---:|
| Original recorded total `T` |624,063,660,824|
| Added child lifetime charge `C` |311,382,658,089|
| Actual child terminal `/proc` rchar |311,382,658,089|
| Once-charged parent-after-wait term `T-C` |312,681,002,735|
| Parent's own increment inferred as `T-2C` |1,298,344,646|
| Registered read cap |549,755,813,888|

The child charge exactly matches the captured terminal child counter. The original formula adds it to the post-wait parent term, which has already absorbed the waited child. The redundant child addition caused the recorded total to exceed the cap by74,307,846,936 bytes. The once-charged recorded term is291.206877 GiB,237,074,811,153 bytes below the cap. This subtraction uses the original retained counters, not estimated dataset sizes or elapsed-time extrapolation.

The once-charged term ends at the original controller's terminal accounting read; subsequent result-publication administrative reads are outside that recorded term. Its numerical correction does **not** mean BUILD_COMPLETE: controlled stop, returncode-15, missing child completion/accounting and absent COMPLETE independently remain failures. The original failure receipt is preserved byte-for-byte. The separate read-amplification finding motivating the stop is only referenced from the pinned stop receipt; this review did not reread its9.55 MB source index or independently recompute that separate analysis.

## Scope and resources of this additional inspection

The bounded Python `-S` inspection completed its target reads in0.348 seconds onCPU8/9, between13:14:09 and13:14:47 UTC including the first probe. Its explicit bounded content reader charged10,935 bytes; `/proc/self/status` reported VmRSS/VmHWM15,308 KiB, agreeing with `getrusage`15,675,392 bytes. It used256 MiB address-space and1 MiB output limits and did not load Torch, CUDA or model/data payloads. Directory/stat operations and interpreter startup are not mislabeled as an independently measured global rchar total.

The previous long observer's anomalous1,569,472,512-byte `ru_maxrss` and `FAIL_REPORTED_RSS_OVER_CAP` remain intact in `../full_future_train_terminal_review_v1/`. This short task is a separately authorized exit inspection, not a rerun or replacement of that resource result. No historical GPU/metadata peak qualification is granted.

Evidence:

- `ORIGINAL_RESULT.json`: exact original2961-byte failed result.
- `STOP_RECEIPT.json`: exact2547-byte receipt, matching requested SHA256 `14e75d9e34f5c5f4274cd3a3b10432ff5249037843b8c8ffe9d7f87924d51f63`; includes the two earlier rejected-before-signal attempts.
- `OBSERVATION.json`: fresh process absence, result/stop hashes, stat census, local lock observation and observer resources.
- `REVIEW.json`: machine-readable accounting derivation and scope.

The only inherited operational conclusion is that the identified invocation has stopped and its committed shards remain available. Completing finalization, changing source/ledger, or starting another attempt requires its own separately reviewed execution plan; this report does not dispatch one.
