# SP3 step12000 incremental storage accounting

Actual current20k run: `/mnt/project/st-world-policy-20261016/formal/spatial/runs/SP3_fulltrain_seed42_official_20k_v1`. The specified12k COMPLETE receipt is valid as small metadata; its exact nine-payload inventory and file sizes were checked using stat only. All nine inode identities are new against the846 identities recorded in immutable V3. No old payload was restatted; no tensor was read or hashed.

New already-occupied credit: **37,371,024,994 bytes**. Original gross stays **4,448,512,376,832 bytes (4143GiB)**. Total occupied credit becomes **2,219,857,498,094 bytes** and future remaining becomes **2,228,654,878,738 bytes**. V3 and the active history attempt3 V3 pin remain unchanged. Native payload checksum verification remains the native resume requirement; this review does not claim to have rerun it.

At 1792335293.827 Unix, free=4,195,920,404,480. Required existing free, including full768GiB cache +10GiB metadata +1TiB floor +4GiB evaluation, is 4,167,832,612,882; remaining headroom=28,087,791,598. The unchanged T0 timing40GiB hold would require 4,210,782,285,842; headroom=-14,861,881,362. **BLOCKED_STORAGE**. C1/C4 completion does not release their4GiB reservation here.

Recording a new completed checkpoint prevents double-counting its physical occupancy. It does not create disk space, and no new capacity is attributed to this reconciliation. No GPU admission, device scan, launch, signal, cleanup or ledger mutation was performed. Running history retains its static V3 guard and can still conservatively count this new checkpoint twice.

Small report output is limited to this directory and below1MiB; it fits within existing10GiB metadata accounting. No new cache or checkpoint allowance was created.
