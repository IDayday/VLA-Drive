# Bounded cache: real metadata qualification

The user declines multi-terabyte cache construction. The chosen route reuses
the existing current teacher tensors and computes missing future FP32 targets
on demand with a bounded RAM LRU. It does not compress targets, alter labels,
change the sampler, or remove tasks. No new teacher tensor disk was allocated
by this qualification.

The real 32-scene, two-log future preparation and independent original-code
reference both completed. All 480 time/view slots and 3,532 leaf fields matched,
including timestamps, original image sizes, complete image SHA values and raw
log provenance. The original `build_target_scene`/`image_reference` code objects
were reused for the reference with budgeted I/O only; the new selector was not
used as its own oracle. This particular subset has no missing future view;
missing/tie/error cases are covered separately by the independent CPU tests.

Combined controlled invocation time was 31.261 s. Actual counted reads were
441,607,148 bytes, peak aggregate parent/child RSS 756,162,560 bytes. All output,
control receipts and directory allocations occupied 659,456 bytes. These are
metadata preparation measurements, **not teacher or training throughput**.
The requested limits were 8 GiB of reads, 900 s and a 32 MiB output root.

`FUTURE32_RESULT.json` binds the exact artifacts and implementation runtime
`1162b955`. There were no previous failed real invocations under this identity;
historical synthetic failures and previous PLAN versions are preserved. Each
stage is one-shot, with its marker checked and written under the same campaign
lock. Unknown failed read work consumes its full reserved half; no automatic
retry resets the budget.

The reviewed additive policy migration preserves the original single 768 GiB
campaign envelope. The metadata package is at most 4 GiB including retained
pilots, indexes, staging and controls. Currently declared additions total
1,072 MiB: 16 MiB retained pilots, 1,024 MiB current index, 32 MiB qualification.
These are reservations, not measured newly written bytes. The checkpoint
commitment is separate: a conservative 1,983 GiB for the nine registered runs,
plus the separate 1 TiB free-space floor. Existing tensor caches remain intact.

Actual future32 build and reference commands use the archived
`run_bounded_metadata_v1.py` and the exact resource approval in
`../metadata_actual_admission_v1/RESOURCE_APPROVAL.json`. The controller captures
bounded output, checks aggregate process RSS and elapsed time, and may signal
only the new child process group that it owns. No running training process was
signalled. The source archive is byte-identical to the executed private helper;
historical approvals expire and are not reusable launch permissions.

The full-current metadata index build also completed in a separate CPU-only
invocation. It covers all 101,592 original scenes and 304,776 image records.
The database is 381,902,848 bytes (364.21 MiB); elapsed outer wall time was
218.157 s and peak aggregate RSS 673,263,616 bytes (642.07 MiB). The guarded
disk peak, including transient SQLite files, was 536,460,904 bytes. No image or
teacher tensor payload was opened and no teacher tensor was written. Metadata
was read and hashed: its total read bytes were not instrumented by this
invocation, so zero payload opens must not be described as zero I/O.
`CURRENT_INDEX_RESULT.json` binds its completion and controller receipts.

Full-future population preparation and long-window on-demand training remain
unexecuted. These results validate bounded metadata preparation, not long-run
cache hit rate, training throughput, or formal temporal training. Existing
current feature tensors remain in place; the index itself does not replace
their storage. Missing future tensors will use the separately qualified FP32
on-demand path with a bounded RAM LRU, rather than a new multi-terabyte disk
population. The user disk constraint does not authorize deleting old evidence
or reducing target precision.
