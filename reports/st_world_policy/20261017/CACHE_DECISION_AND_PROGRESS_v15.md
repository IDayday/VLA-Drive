# Disk-bounded teacher supply: integrated full-index qualification path

Recorded 2026-10-17 10:54 UTC. This is a progress report, not formal temporal
training approval or a full-population throughput result.

The user decision remains: do not create several terabytes of new cache data.
Reuse the existing immutable current-frame teacher bank, compute missing future
targets with the same frozen FP32 teacher, and retain them only in the bounded
RAM LRU. New future teacher tensor bytes on disk remain zero. The student uses
the independently verified official execution recipe (BF16 execution, FP32
master weights and Adam state, ZeRO2); this does not change teacher precision.
No existing cache, checkpoint, input population or supervision was removed.

## Actual storage and work in progress

- The completed development metadata covers 1,696 scenes / 16 logs and occupies
  30,265,344 allocated bytes. Existing immutable receipts are recorded in v13.
- The original training metadata build is still running, without restart or
  hot patch. At 10:47:11 UTC, 153 complete log shards represented 30,220 of the
  fixed 101,592 scenes. Those shard files alone occupied 292,720,640 allocated
  bytes; this is not a complete-root measurement or a final cache size.
- Its actual COMPLETE and final resource result are still pending. Neither
  full future request index has been built. No missing identity is fabricated.
- Metadata reservations remain 5,620 MiB, including staging; the proposed
  separate 1 MiB index-preparation allocation is still pending. The existing
  campaign cache ceiling, checkpoint commitments and free-space floor remain
  separate controls. Free space is not permission for a multi-TB prebuild.

## Integrated code and independent evidence

Commit `00ae8d46df07cd2194cb7d3f149117af034a7a2a` integrates the narrow
full-population indexed PROFILE path. Its teacher_provider.py SHA256 is
`c36cb97a7b2d7acf3ee143338fc9922ffc11e3b3bc054714e89acfdea65314f2`.

It streams the selected population against both current and future indexes,
requires their real completed provenance, and retains the original trainer,
sampler, 20k learning-rate timeline and warmup1000. The separate PROFILE
identity permits at most256 cumulative optimizer updates across initial
execution and all resumes. It does not grant a new budget on resume or
formal-training permission. The proposed two-rank global2 diagnostic is not
the formal global32 topology.

- Independent correctness: **141 passed, zero failures/skips**, including
  23 independent negative/cumulative-sampling tests. Report:
  `full_index_profile_independent_review_v1/REVIEW.md`.
- Independent resource boundaries: **68 passed**. Synthetic258-scene streaming
  validation retained at most two row objects concurrently. Report:
  `full_index_profile_resource_review_v1/REVIEW.md`.
- These test sets overlap and must not be summed into a unique test count.
  Their synthetic metadata is not real-image or full-population qualification.
- Fifteen mathematical, reader and PCA files match the prior real32 runtime.
  The inherited trainer additionally validates the training-manifest pin;
  the entire trainer is therefore not described as byte-identical to real32.
- The original dataset still loads full source/selection metadata, verifies
  current/ego records per rank at startup, and constructs the original sampler.
  Startup IO/RSS, steady-state throughput and checkpoint costs must be measured
  separately; the SQLite/LRU limits are not a whole-process memory claim.

The owner tree at `bccebb00319512049e93df58d18cb5ead49575f0` is retained clean
for possible immutable runtime use, avoiding another unnecessary checkout.
No GPU PROFILE, actual index write, storage-ledger mutation or formal temporal
launch follows merely from these CPU reviews.

## Next actual execution

After the existing metadata builder finishes, validate its real terminal
receipt, then build the bounded indexes under the reviewed single-writer
protocol. Preserve any failed attempt and reconcile only evidenced accounting
errors; do not rerun or relabel a failure automatically. The full-population
PROFILE then measures actual initialization, two-rank supply, failure behavior
and native continuation under its own finite resource budget.

The registered C-line training and fixed development evaluations continue
independently. C2/C3@1024 have complete official scores; their fuller executed
motion diagnostics are being prepared using the existing frozen evaluator.
C4@1024 remains in development inference. No new navtest, held IQE/MOPD
training, formal ACTIVE update or push was started.
