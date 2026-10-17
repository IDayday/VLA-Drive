# Bounded-storage index completion

The user requested that the new training path avoid multi-terabyte feature generation. The existing decision retains current teacher banks, computes requested future FP32 targets on demand, and uses a bounded in-memory LRU. Student training follows the verified official BF16 execution / FP32 master-and-Adam policy; teacher precision is unchanged.

| Role | Scenes | Records | Database bytes | Wall seconds | Peak aggregate RSS bytes | Logical read bytes |
|---|---:|---:|---:|---:|---:|---:|
| dev | 1696 | 9948 | 8585216 | 9.527 | 741601280 | 228842607 |
| train | 101592 | 603750 | 515739648 | 282.778 | 1329537024 | 19832531041 |

Both existing builders exited successfully with no remaining owned process. Neither opened image/tensor files nor wrote teacher features. These are compressed metadata indexes, not visual feature caches. The combined database is 524,324,864 bytes (about 500 MiB). All temporary staging remains within the independently bounded metadata allocation.

The original DEV admission expired before dispatch during context handoff. Its lead guard rejected before any subprocess or index write; the immutable renewal retains the same single actual attempt and budget. Historical failed extraction and metadata evidence is preserved.

This establishes index construction, not full-population training parity or throughput. The original provider still needs its ordered population checks; the bounded two-rank training PROFILE remains pending. Existing C1–C4 jobs and old model/checkpoint identities were not changed. No new Navtest, formal training, ACTIVE update, cache deletion or push occurred here.
