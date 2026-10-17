# Independent actual fixed64 packed-history review

**PASS_BOUNDED_ACTUAL_METADATA_PILOT** for runtime
`b847f5a99dd44ee5256362d380d9d96efb47d35b` and actual output
`/mnt/project/st-world-policy-20261016/audits/history_data/causal_history_cost64_v2`.
Reviewer: `/root/st_calibration_independent_correctness_v2`.
This is a completed metadata-only 64-scene pilot, **not full-scale qualification**,
not a teacher tensor cache, training result, or automatic follow-up permission.

The exact signed admission SHA256 is
`abe66a14cbfc9e617f876b145088b4c66b9df4b1f934e6d7d8dce37db4e1e9e8`.
Lead LAUNCH, controller INVOCATION, metadata admission, registered builder,
COMPLETE and final control receipts are cross-bound in `REVIEW.json`. Launch
was within the admission window. Parent PID533059 is absent; final cleanup
reports zero owned processes and one verified process.

## Actual artifacts

- The fixed original64 token/log sequence is unchanged; there are64 unique
  scenes and64 unique logs. The original selection SHA256 is
  `dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01`.
- Runtime selection adds current/ego file hashes to each original token/log;
  resolved config and manifest consistently retain those pins. This enrichment
  does not change the population or ordering.
- There are64 `packed_log_v1` shards. Every referenced offset/length covers
  its actual record, and all64 record byte hashes match the history manifest.
  COMPLETE pins the complete history and teacher-request manifest byte hashes.
- History contains128 frames at requested offsets[-1,-0.5] seconds, all causal,
  within declared timestamp tolerance;384 valid camera views, no masked views.
  Three-camera order is CAM_F0/CAM_L0/CAM_R0.
- Teacher requests contain only target time0, one current frame per scene.
  No future target time, tensor payload, GT or score data is among the output
  files. The builder's recorded `original_gt_or_scores_read=false` is preserved.
  This reviewer did not re-read original data to independently reconstruct it.
- Actual root+control file content totals1,156,349 bytes; allocated file blocks
  total1,298,432 bytes. Both are far below combined32MiB. Ledger retains the
  full31MiB artifact reservation, observed peak1,290,240 bytes; separate control
  cap remains1MiB. Reservations are not confused with actual written size.

## Resource and terminal interpretation

| Recorded measurement | Actual value |
|---|---:|
| Wrapper invocation wall |81.623069895 seconds|
| Supervised child wall |77.531918438 seconds|
| Builder final cost wall |73.190216310 seconds|
| Process-tree peak aggregate RSS |1,653,936,128 bytes|
| Child peak RSS |1,005,670,400 bytes|
| Final wrapper charged logical reads |1,991,446,716 bytes|
| Intermediate charged logical reads |1,990,343,395 bytes|
| Builder source-byte meter |1,668,979,101 bytes|
| Child kernel block-read counter |3,376,922,624 bytes|

Logical `rchar` and kernel block reads measure different things. Admission
limits logical reads to16GiB; the larger block-read counter is separately
reported rather than relabeled. Parent post-reap accounting equals
parent-pre-reap delta + child-terminal bytes + parent activity during wait,
so the child is not added again. Final charged reads conservatively include
additional terminal evidence work. RSS is below4GiB and wall below900s.

`CONTROL_HISTORY_RESULT.json` is intentionally
`BUILD_RETURNED_PENDING_TERMINAL_ACCOUNTING`; the wrapper subsequently performs
terminal resource/cleanup checks and writes the separate control
`RESULT.json` as **COMPLETE**. The latter is present, finite and bound to that
intermediate receipt. Child return code0, monitor error null, failure null and
cleanup0 support actual completion. The intermediate status alone would not
have sufficed.

`RESOURCE_COST.json` is a snapshot before writing itself. Its wall and written
bytes differ slightly from the later artifact cost snapshot (415 additional
bytes); both are retained, not silently selected to improve a number.

## Independent-review cost and limitations

Final inspection: CPU6/7,0.14311 seconds,1,223,988 explicitly read bytes,
1,224,085 process logical-read bytes and peakRSS20,381,696 bytes.
Earlier schema inspection and two stopped verifier attempts read only small
metadata and remain within the8MiB total scope. Their logs are preserved:

1. The first incorrectly required byte-level equality between the original
   selection and its legitimate hash-enriched runtime copy. Token/log order
   and every original field are now checked, plus added pins across artifacts.
2. The second conflated `current_record_identity` (canonical JSON hash) with
   `current_record_sha256` (original file byte hash). The reviewed source at
   `st_world_history.py:170` distinguishes these. The review no longer asserts
   the false equality. Original current files were deliberately not opened;
   their semantic identities are not independently recomputed here.

These were verifier assumptions, not failed builder executions or permission
to rebuild. Only one real pilot ran. No source change, GPU call, tensor/image/raw
read, test-suite rerun, ledger mutation, launch or signal occurred in this
review. Cross-record source geometry is not independently reconstructed.
CPU/noGPU execution inherits the exact reviewed controller; no new GPU sampler
was run. Prior TRAIN `FAILED_NO_AUTO_RETRY` and unrelated resource failures are
unchanged and not overridden by this pilot's PASS.
