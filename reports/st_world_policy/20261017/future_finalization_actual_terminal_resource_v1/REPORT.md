# Independent actual finalizer terminal review

Status: **PASS_ACTUAL_TERMINAL_RESOURCE_REVIEW_ONLY**. This is a bounded read-only review of the executed controller receipts, matching completion identities, storage registration, and process exit. It is not a fresh continuous measurement of the historical process peak and is not training/index-build approval.

The new finalizer completed 101,592 scenes across 1,176 logs. The original failed source remains `FAILED_NO_AUTO_RETRY` with its original SHA256. No teacher extraction or tensor-on-disk output was reported; the new metadata root is separate.

| Executed scope | Measured receipt value | Bound |
|---|---:|---:|
| Whole invocation | 283.738 s | 3600 + 40 s |
| Logical reads, terminal | 2,867,654,447 B | 16 GiB |
| Aggregate RSS peak | 952,291,328 B | 4 GiB |
| New metadata current allocated charge | 773,234,688 B | 2 GiB |
| Separate controls allocated charge | 20,480 B | 1 MiB |
| Compressed scene cache | 165,783,978 B | 512 MiB |

The original verified parent and child have exited. Cleanup reports zero remaining processes. Post-wait accounting absorbs the child exactly once; the remaining 1,112,482 bytes are later parent publication reads, not a second child charge. The root's intermediate return marker is correctly distinguished from the separate completed terminal receipt.

The audit read only exact small JSON files (55,473 bytes in the scripted review; the earlier four-file inspection read another 38,146 bytes). It statted a bounded root listing without opening the large requests file or shards. CPU 8/9, 0.0294 s scripted wall, actual process VmHWM 15,052,800 bytes. No live ledger read or mutation, no signals, no dispatch, no payload hashing, no GPU access.

`REVIEW.json` records exact pins and assertions. The generic `image_or_other` hash-cost bucket does not establish image decoding; metadata-only scope follows the reviewed implementation and bound receipts. Next index preparation requires its own current controls and whole-process budget, including lazy Torch import and validator reads. Full index build requires a separate admission after preparation/migration.
