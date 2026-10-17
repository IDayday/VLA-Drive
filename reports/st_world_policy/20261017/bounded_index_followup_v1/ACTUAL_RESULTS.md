# Actual bounded index builds

Both registered one-shot builds completed under the independent 81dad039
review, with the original campaign ledger and checkpoint commitments.

| Index | Population / records | Database | Wall | Peak process-tree RSS |
|---|---:|---:|---:|---:|
| Future request only | 32 / 294 | 249,856 B | 5.749 s | 650,620,928 B |
| Original PCA fit-cache metadata | 64 / 957 | 1,155,072 B | 5.834 s | 652,521,472 B |

Final combined allocated storage, including controls/directories, is 1,470,464 B
(1.402 MiB). Each namespace retains its conservative 32 MiB reservation,
including failed-partial/staging allowance. The original full-current index and
all old claims are preserved. No new teacher tensor or image was read or written
by these index builds; no GPU, optimizer or training step was run. PCA remains
bound to the original `fae6ed77…` cache and has not been refitted.

The two writers reported respectively 10/18 guard callbacks and 7/15 publication
callbacks; outer monitoring includes the parent and child. Exact identities,
receipt checksums, total wall and storage measurements are in ACTUAL_RESULTS.json.
These times describe metadata construction, not model training throughput.

This closes the missing-index dependency only. The new cached/on-demand
composition still needs its independent receipt-validator fixes and actual
FP32 image/tensor qualification. No synthetic test or metadata hit rate is being
reported as real numerical/training equivalence. The W probe controller is also
still under independent review; no W extraction or temporal formal training has
been launched by these two builds.
