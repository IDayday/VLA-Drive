# Independent finalized-TRAIN index preparation review

**PASS_NARROW_BINDING_ONLY_WITH_HARNESS_LIMITATION** for exact commit `99ff3af1b255cf3b37b13954850e8b657c6b493f`, helper SHA `360a52ab7b8cf94bb1e8b1c7d2d6068a7ffe1acd1eec3f7761834a519d5a04d0`. This is source/control correctness only, not actual index completion or launch admission. Reviewer `/root/st_calibration_independent_correctness_v2`; isolated worktree and report-only changes.

## What was verified

- TRAIN explicitly consumes the new finalization completion and separate terminal, including successful owned cleanup, complete read accounting, exact artifact, admission, registration, command and original failed-source lineage. A synthetic full-population fixture exercised `bind_finalized_train` through the **actual unmocked shared validator** in frozen runtime `3df052ea`. Mutated failed-source evidence and self-consistent wrong target times were rejected.
- DEV retains its original completion/attempt2 and prior-failure pins. New TRAIN requests derive from the new completion parent, not the old incomplete request path.
- Original PLAN code stays bound to its original source tree; new index callback sources are checked separately. All five supplied runtime pins were hash-checked; the original callback scanner verified its three actual dependencies and rejected a wrong dependency SHA.
- Current static policy plus both pending index claims yields **7701 MiB of 8192 MiB** exactly once. Baseline mutation and oversized plans reject. No live ledger was read or changed.
- Missing new-runtime inputs and waiting completions produce no proposal directory. Failed/pending terminal, wrong cleanup/returncode/accounting, auto-retry, scientific-result flag and self-reviewer cases reject before shared validation.

## Actual tests and limitations

18 unique cases were attempted: **17 completely passed; one review-harness tail assertion remained incomplete**. Nine author adapter cases were actually rerun, rather than inheriting their result. Additional cases exercised pins, rejection behavior, and the real shared validator on synthetic metadata.

The initial setup incorrectly imposed the 2 GiB RSS allowance as virtual address space, preventing Torch shared-library mapping before tests started. That log is retained. With a real RSS watchdog, the first suite completed 14 cases and four errors caused by a reviewer class-stored lambda binding `self`. After correcting this fixture, only those four cases were rerun with Lead's explicit extra import allowance: three passed. The remaining case successfully returned from the actual strict validator and verified the original registration, then failed in an extra reviewer order assertion because full-population PLAN deliberately has `selected_rows=None`. No implementation changed and no third run was made.

This report **does not claim all tests passed** or new order evidence from that failed tail assertion. Original order semantics were already covered in the earlier independent finalizer review and are unchanged here. The separate complete adapter-to-validator positive and source/time negative tests passed, supporting this narrow binding conclusion. No new product-code failure was found.

Two measured processes used CPU6/7: 6.88284120708704 seconds combined, peak RSS 626851840 bytes, logical reads 277857699 bytes (273912393 startup; 3945306 test window), writes 143870 bytes during test windows. The extra <=145 MiB import/read run was separately authorized. Initial immediate import failure did not capture IO/RSS; its log and tool-measured short failure are retained without inventing those counters. CUDA remained uninitialized.

Only synthetic small metadata was generated under `/tmp`; no real payload, teacher tensors, images, actual ledger, or live workload was accessed. A stat-only check found the actual original TRAIN PLAN is285070 bytes, within the original4 MiB small-control limit; its contents were not read. No real101592-scene or index performance result is claimed.

## Execution boundary

Actual preparation still requires successful finalized TRAIN outputs, fresh policy/ledger and source pins, separately approved process read/RSS allowance covering Torch imports, and Lead dispatch. The helper's own `metadata_read_bytes` explicitly excludes validator/import IO and must not be presented as total process reads. Existing failed builds and prior observer failures remain preserved. This review does not authorize a build, mutation or PROFILE launch.

`FIRST_TEST.log`, `RETRY_TEST.log`, both measured result JSONs and the initial import log preserve the complete outcomes. The prior40 finalizer tests were not rerun.
