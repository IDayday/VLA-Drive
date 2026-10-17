# Official execution and bounded storage: current stage

This is a timestamped execution snapshot, not a completed research result. The
adjacent JSON binds actual receipts and independent reviews. The older
`20261016/IMPLEMENTATION_STATUS.json` remains an initial implementation snapshot;
its zero-update statement is no longer the current state.

The user's official DriveDreamer-Policy execution choice is implemented:
BF16 execution weights, Torch AdamW handed to DeepSpeed ZeRO2 with FP32 masters
and Adam moments, FlashAttention2, no offload, and the official path's effective
absence of language activation checkpointing. The common ST research schedules
remain 20,000 W updates and 4,096 C updates, explicitly adapted from the official
recipe. Canonical deployment is reconstructed from FP32 masters and uses SDPA
with one ten-step Flow trajectory. Training and deployment precision are not
conflated, and production FA2 is not claimed to give cross-run bitwise identity.

## Cache and checkpoint capacity

All three current training target caches are complete and have full-record
verification for 101,592 scenes / 304,776 images each:

| Target cache | Actual bytes | Purpose |
|---|---:|---|
| legacy C1 grid | 29,978,529,544 | SP0 |
| native 384 grid | 539,427,538,104 | SP2/SP3, shared |
| pooled 384 grid | 59,939,229,456 | SP1 |

Their total is **629,345,297,104 bytes (586.123 GiB)**. These are frozen DINO
teacher targets, not a multi-terabyte Qwen visual cache. All registered caches,
development/diagnostic data, metadata and staging remain within the **768 GiB**
cache cap, with a **1,024 GiB filesystem free-space floor**. The exact current
reservation is recorded in JSON. The floor is a safety admission rule, not a
verified user quota or a new disk allocation.

No extra future times, layers or full visual caches are automatically generated.
Later temporal stages must first fit a bounded storage plan or use explicitly
qualified on-demand computation. No dtype, target, preprocessing or sampling
change is hidden as a storage optimization. Existing immutable writers were
prebounded; they are not retroactively described as using the newer per-shard
guard. No user's cache or checkpoint has been deleted.

Native checkpoints are budgeted separately: 240 GiB per W run for this four-arm
5,000-update stage. Saving space in caches does not remove that requirement.

## Current execution

SP0–SP3 each have actual, independent Evidence, Correctness and Efficiency PASS
receipts bound to their registration. All four have been dispatched on locked,
freshly checked GPU allocations. Each starts from the same bound original S0,
uses global batch 32, and pauses after 5,000 updates on the unchanged 20,000-update
LR clock. PROFILE weights are not used for initialization. SP3's W96 extension
is an explicit architecture comparison; its initial function is not claimed to
equal W48.

The consequence path's fixed-reference metadata issue has been corrected and
independently reviewed. Its new 240-update profiles are running separately;
formal C0–C4 are not yet launched. The old profile that failed after 162 updates
is retained as a failure, not relabelled complete. Full training context hashes
are being prepared from the existing legal metric cache without creating a new
large cache.

Actual native W and C smoke checkpoints have been exported through the real
image-to-single-trajectory path and scored on the same 16 diagnostic development
scenes. These qualify engineering paths only. They are not formal planning gains
or an untouched development evaluation.

The next scientific evidence is the registered common endpoint evaluation;
lower auxiliary loss, successful checkpoint loading and running GPU jobs do
not establish planning improvement. No Navtest, ACTIVE update, held IQE/MOPD
experiment restart or automatic push occurred.
