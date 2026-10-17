# Storage-constrained teacher execution: current evidence

Recorded 2026-10-17 04:52 UTC. This update supersedes the pending-status statements
in earlier cache progress reports without deleting their historical evidence.

The user's instruction rules out a multi-terabyte feature prebuild. The chosen
path reuses existing current-frame targets and computes missing future DINO
targets on demand into bounded RAM. It does not cache trainable student features
or change the registered targets, masks, sampling, precision or loss. The teacher
is FP32; student BF16 execution with FP32 master/Adam state is a separate policy.
This is not a change of the entire model to FP32.

The aggregate campaign cache cap remains **768 GiB**, including the existing
**586.123 GiB** of current training target files. Existing reservations total
645.318 GiB, not an additional allocation on top of the cap. Checkpoints retain
separate reservations and a 1 TiB filesystem free-space floor. A fresh filesystem
stat observed 8,428,692,357,120 available bytes; this does not enlarge the cache
budget. No old cache/checkpoint deletion or new full future tensor build occurred.

## Measured evidence and its limits

- Real64 teacher qualification compares all output tensors against the original
  cache and across the fixed8 execution layouts. Both comparisons passed exactly;
  see `FIXED_TEACHER_REAL64_RESULT.md`. Newly written target tensor bytes: **0**.
- The real student diagnostic completed two updates, plus a separate pause1 and
  resume1 run. Independent review confirms functional native recovery, consumed
  IDs and all saved RNG streams. Both paths write zero new target tensor bytes.
  See `correctness/fixed_provider_native2/README.md` and the updated
  `FIXED_PROVIDER_TRAINING_DIAGNOSTIC.md`.
- Independent runs showed gradient variation before any resume. The installed
  FA2 backward mechanism was tested separately. This is recorded explicitly;
  current full-model cross-run bitwise identity is **not** claimed. The production
  backend remains unchanged.
- The owner has implemented bounded SQLite metadata indices in commit
  `9c7743f5323cbc0e85c3aaa4f9e13685ac4ae234`. Its current256/future64 metadata pilot
  produced 1,622,016 bytes of SQLite data, with no image or tensor reads. Owner
  tests: 73 passed. Independent review is pending at this snapshot; this statement
  is not training or large-population approval.

## Remaining work

1. Review and integrate the bounded metadata adapter, retaining immutable source
   hashes, exact row order and corruption rejection.
2. Prepare genuine full-future **metadata**, not teacher tensors. The existing
   full-population request has only time0 and cannot stand in for future coverage.
   Preparation must keep the locked population/order and use bounded per-log work
   and atomic resumable outputs. A raw-log provenance ledger avoids embedding a
   population-sized header. The small cost/parity check precedes a full scan.
3. Register the indexed provider/PCA application identity and run the necessary
   bounded efficiency qualification before allowing formal temporal training.

Sustained temporal training throughput remains unmeasured. The small qualification
does not establish that on-demand computation is faster than a complete cache.
Its verified benefit is avoiding the large target storage requirement while
preserving the measured target values and functional training path.

Existing W/C runs use their immutable source snapshots and continue unchanged.
No held IQE expert/Scorer or MOPD job was restarted, no navtest was launched, and
no formal ACTIVE was updated. No push was performed.
