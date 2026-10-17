# TRAIN64 W extraction: terminal failure retained

Status: **TERMINAL_FAILURE_PRESERVED_METADATA_CONSISTENT**. OriginalRESULT remains `STOPPED_NO_AUTO_RETRY`, reason `OWNED_DESCENDANTS_AFTER_LEADER_EXIT`; this review does not relabel it COMPLETE.

The actual child exited0 and its CLI/committed cache marker both report64/64COMPLETE with the original train-plan signature. All64 small sidecars agree with that identity;64tensor files exist at the expected size. Their contents, shapes and finite values were not read. The original childPID544166/start1368591802 is absent. The monitor reports6verified owned processes and zero remaining; these small receipts do not expose the other5identities, so no independent full-process census is claimed.

| Actual recorded resource | Value |
|---|---:|
| Supervisor wall |466.2916s|
| Sampled aggregateRSS peak |26,804,613,120B|
| GPU allocated / reserved peak |12,837,610,496 /13,287,555,072B|
| Extraction read charge |153,889,914,639B|
| Prior TRAIN plan charge |37,575,105,055B|
| Cumulative role charge |191,465,019,694B|
| Remaining role allowance before this review |83,412,887,250B|
| Current cache allocated bytes |76,099,584B|

Read charge follows the original rule `max(hash_admitted + nominal_torch_load, proc_rchar)`, without adding these alternate counters together. All declared resource limits are satisfied, but successful resources do not erase the supervisor failure.

The prior DEV artifact-acceptance pattern can be used: independently validate already committed tensors while keeping the failed execution receipt. Its existing implementation is pinned specifically to DEV16, its failed SHA, and consumer `train_plan`; it cannot directly authorize TRAIN64 orfit. A separate finite64-record original-validator/hash/shape/finite review and explicitly bound TRAIN reconciliation are needed. Do not re-extract or bypass the original fit prerequisite.

This read-only audit used96,166metadata bytes,0.156s and actual selfHWM15.3MB on CPUs8/9. Earlier inspections read additional small receipts/source, with total below2MiB. No tensor payload, live ledger, GPU query/allocation, signal or dispatch occurred. The original supervisor result remains unchanged.
