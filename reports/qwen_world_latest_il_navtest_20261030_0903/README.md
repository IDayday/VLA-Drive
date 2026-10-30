# Latest complete IL Navtest

User-requested optimized-label and time-residual IL snapshots, pinned before scoring. Full12146, single-candidate clean-xK10shift5, canonicalv1.1, seed2101. Previously evaluated checkpoints reused, vla-zt3 excluded; no training/parent selection changes.

CPU exports prepare immediately; GPU inference waits for the four short native/export checks so that the sole currently large-memory device is not held by a long shard. This is scheduling only; all native inference and scorer code is unchanged.
