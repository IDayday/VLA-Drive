# SP2/SP3 continuation storage reconciliation

This independent, read-only reconciliation preserves the original 4,143 GiB checkpoint allocation and all existing files. It distinguishes completed bytes already occupying the filesystem from the remaining reservation. It does not authorize a launch, alter a scientific identity, or release an unused commitment.

The frozen snapshot validates 846 unique payload `(device, inode)` identities using COMPLETE receipts and file statistics. Credit is the smaller of the recorded logical size and allocated bytes. No tensor payload was read or rehashed. The original 20 SP 5k checkpoint credit was revalidated. Additional credit covers the 14 complete SP2/SP3 continuation checkpoints and completed C/PCA/T1/T3/full-population profile checkpoints demonstrably included in the original allocation chain. Cross-run aliases cannot receive duplicate credit.

| Accounting item | Bytes |
| --- | ---: |
| Unchanged gross checkpoint allocation | 4,448,512,376,832 |
| Previously credited and revalidated | 747,345,660,520 |
| New SP2/SP3 complete checkpoint credit | 523,159,425,948 |
| Other originally allocated complete checkpoint credit | 911,981,386,632 |
| Total already-on-disk credit | 2,182,486,473,100 |
| Remaining checkpoint commitment | 2,266,025,903,732 |
| Observed filesystem free | 4,310,776,377,344 |
| Remaining checkpoints + full 768 GiB cache + 10 GiB metadata + 1 TiB floor | 4,200,908,670,580 |
| Observed headroom | 109,867,706,764 |

One additional 80 GiB T0 capacity run would leave 23,968,360,844 bytes of headroom at this observation. That is a proposal calculation, not a reservation or GPU admission. A fresh filesystem and device check is required before either continuation or capacity dispatch. Quota was not independently available; this report does not equate filesystem free space with a verified per-project quota.

SP2 paused normally at update 11,762 and SP3 at 11,582 when their eight-hour invocation budgets expired. Their latest COMPLETE receipts and payload inventory sizes are valid. Training performs the existing full payload integrity verification during an actual resume; this stat-only review does not replace it. The original 20k optimizer/scheduler timeline, data cursor, run identity, registration and resource lease must remain unchanged.

The native cache guard was called read-only with the actual ledger and the new remaining checkpoint commitment. It passed without changing its conservative legacy-cache accounting. The stronger full-768-GiB reserve also fits. No cache credit, cleanup, retention-policy change, ledger edit, checkpoint rewrite or process signal was performed.

`reconcile.py` records the reproducible metadata-only procedure. The two JSON files are the frozen evidence and resource receipt; do not rerun the script merely to refresh timestamps. Future changes require a new version.
