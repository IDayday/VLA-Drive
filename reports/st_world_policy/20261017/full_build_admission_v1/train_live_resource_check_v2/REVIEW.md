# Bounded live TRAIN metadata progress update

The existing metadata TRAIN build remains active and progressing. Both snapshots revalidated parent480124/start1366936766 and child480140/start1366936849, exact cmdline/cwd/admission, and frozen controller SHA `a866ae42…`. No process, budget, code, ledger or source data changed. No terminal success is asserted.

| Observation |10:35:21.736 UTC|10:36:14.075 UTC|
|---|---:|---:|
| Complete log shards |124|130|
| Complete scenes |23,906|24,606|
| Root-inclusive allocated/logical charge |231,776,256B|238,505,984B|
| Parent+child RSS |1,094,787,072B|1,094,795,264B|
| Live parent+child absolute rchar |65,787,953,609B|67,170,008,705B|
| Observed partial logical bytes |0|0|

The interval is52.339seconds. Counts come from complete immutable log-shard `selected` tails; no scene duplication or population subsetting is introduced. Final coverage is24,606/101,592scenes(24.22%),130/1176logs. `COMPLETE.json` and `CONTROL_BUILD_RESULT.json` were absent at both observations. Directory observations are non-atomic: one partial disappeared during the first stat scan, consistent with commit/rename. Zero observed partial bytes does not guarantee none existed between samples.

During the interval six logs and700scenes completed. Recent throughput13.37scenes/s exceeds the cumulative9.39scenes/s. CPU consumption22.25CPU-seconds corresponds to0.425cores, about21.3% of the two allocated CPUs. Child syscall read bytes increased1,375,515,350B, while block-layer `read_bytes` increased11,593,719,808B. This remains consistent with IO/pacing waits rather than CPU saturation, but these snapshots do not isolate Lustre latency, readahead or source pacing as a unique cause. No rate/parallelism changes were made. Physical read bytes are not the registered logical `rchar` budget.

## Forecasts, not guarantees

The scene-rate estimates leave1.60–2.28hours of row work, versus3.27hours remaining in the4hour invocation limit. Log populations are uneven (11.05%oflogs but24.22%ofscenes), and finalization/long remaining logs may differ; this is not a promised completion time. The longest observed gap between successive completed-log mtimes is89seconds for824scenes. Those gaps include all intervening work and are not dedicated per-log timers.

Applying the previously observed dev final/shard multiplier1.84498937 to current complete-shard bytes per scene gives an estimated final footprint1,812,736,616B(about1.69GiB). The2GiB cap would leave about334,747,032B(319MiB). Current root has1,908,977,664B remaining. This is a data-size extrapolation, not measured final peak/staging usage; retained partials and later record sizes may alter it. There is no observed capacity breach.

Live parent/child absolute rchar is67,170,008,705B against549,755,813,888B. It is a conservative live observation that includes parent startup, not a rewritten formal accounting receipt. The old post-wait double-count bug remains in this active source. Applying its dev-derived multiplier to current child cost per scene projects554,064,204,596B, about0.78%over the old cap. This is only a possible terminal-accounting risk, not actual exhausted reads or a reason to mutate the active budget. The existing invocation continues under its unchanged cap; final pre/post/child evidence must be examined after real termination before any reconciliation or success conclusion.

## Scope and evidence

The reviewer read17,925,562bytes through the bounded snapshot reader, within512MiB; it used only exact process controls, PLAN, completed metadata tails and file stats. No image/raw-log/model/teacher-tensor contents were read. Open-data-path strings in process receipts came from `/proc/PID/fd` symlinks only. CPU affinity0/1 and2GiB address-space cap were applied. Exactly two snapshots, separated by52.34seconds, were taken; no extra observer or task was dispatched.

`OBSERVATIONS.json` SHA256: `74bbbfce6caa54a45437898225e4a1bba2da99050e067ae8b4ba50c89ceac4c4`.

Recommendation: keep the original finite build running unchanged. Monitor its actual final directory size/time and independently audit terminal accounting when available. No terminal PASS, renewed permission, retries or additional storage are granted by this read-only update.
