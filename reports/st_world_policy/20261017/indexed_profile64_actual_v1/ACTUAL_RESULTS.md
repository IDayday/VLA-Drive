# Actual64-update result — completed, bounded exact32

The actual PROFILE invocation exited0 and paused at its registered64-update
budget, with128 exposures. All32 pinned scenes appeared exactly4 times. The
20k training timeline/warmup1000, original objectives, BF16 execution/FP32 master
state, original PCA and FP32 fixed8 teacher were unchanged. This diagnostic's
weights are not admitted for formal training. It is not a scientific result.

Source: `2d5e9483a8c8bc80b71348211756f3cab6aa21b1`; actual host
`training-vla-zt-worker-0`, physicalGPU0, world1/micro1/global2. Base and source
lineage are recorded in the actual registration, whose checksum appears in
`RESULT.json`. The collector read161,421 bytes of small evidence, plus a separate
small observer-completion receipt; no model/cache payload was read.

## Complete post-update wall intervals

All status steps0–64 were observed: no missing interval was reconstructed.
The original trainer `status.updated_unix` values, not poll times, give:

| Window | Intervals | Mean | p50 | p95 |
|---|---:|---:|---:|---:|
| All normal updates1–64 |64|2.9380s|2.7192s|4.5229s|
| Updates9–64, first8 excluded |56|2.7476s|2.6777s|3.7735s|

The latter window processes0.7279 scenes/s at global batch2. This measures the
same32 scenes reused over epochs and existing current-cache/index inputs;
it does **not** establish full-population or cold shared-storage throughput.
The8-update exclusion is the explicitly recorded analysis convention, not the
separate240-update timing protocol. Both full and trimmed results are shown.

Cold launch→status0 was455.6922s. The trainer's own allocation wall was712.0289s;
that counter starts inside the child, so it does not have exactly the same origin
as the outer launch timestamp. Native save cost was73.1697s. At this diagnostic's
save_every64,64 complete normal intervals plus the single save average4.0813s
per update, excluding cold startup. The observed same-step pre/post-save status
delta73.2113s is separately retained, not counted as a normal update.

The old internal timer reported mean2.8917s and data mean1.7159s. Those are
diagnostics inside an iteration and are not used as throughput denominators.
No speed comparison with the old two-update smoke is made.

## Supply and memory

-128 provider batches;1,338 immutable cache hits;6 bounded RAM hits;
  576 image targets computed on demand;0 failures.
-128 fixed8 calls,1,024 executed image slots,576 requested and448 padding slots.
  Padding is a real cost of the already qualified fixed8 execution, not a new
  input population or extra supervision. It remains unchanged in this run.
-Teacher tensor LRU peak14,159,232 bytes, equal to its declared bound;
  metadata cache budget33,554,432 bytes across2 connections;
  `metadata_population_materialized=false`; new teacher disk tensors0 bytes.
-Observed rank RSS peak8,565,043,200 bytes (7.9768GiB). Median RSS over updates1–8
  and57–64 was8,561,520,640 bytes in both windows; measured median growth0 bytes.
  This supports bounded stability in this short window, not long-run leak freedom.
-Student per-step CUDA peak allocated56,284,111,360 bytes (52.4187GiB), reserved
  60,131,639,296 bytes (56.0020GiB). The existing reset occurs after data/teacher
  preparation; these are not claimed to be entire-invocation teacher peaks.
  RSS is the training rank, not whole-process-tree memory.

## Objectives and actual updates

All64 updates had finite component losses and gradient norms and verified
sampledFP32 master changes. Mean current loss2.620687, future loss2.621269 and
ego-flow loss0.00403805. Gradient norm ranged3.45226–12.38521. No auxiliary target
was removed to obtain these timings.

Each update used current/future-a/future-b denominators221,184 and2 ego scenes.
Across64 updates, each feature stream accumulated14,155,776 valid elements and
ego128 scene exposures. The actual engine check reports BF16 execution and
gradient accumulation, FP32 master2,552,673,668 elements and FP32 Adam moment
states5,105,347,336 elements. Sampled changed master elements were positive on
every update (2,667–2,728); this is update evidence, not a full-weight bitwise test.

## Native lifecycle

Exactly one native checkpoint committed:
`step_000064_1792226247328387018`.
Native tensor files total37,365,927,466 bytes; directory including COMPLETE
receipt totals37,365,928,667 bytes (34.7997GiB). The model/optimizer/RNG inventories
and file sizes match the complete receipt. Payload checksums are the actual
writer's existing evidence, not recomputed by this observer. The subsequent
budget pause reused this exact boundary; no duplicate final checkpoint exists.

This proves one real indexed-source64-update finite execution and native save.
It does not claim exact resume, full-model bitwise continuity, planning gains,
full-population supply, or formal profile approval. There was no retry, no extra
GPU task and no change to another active training job. A different<=256-scene
population would require its own fixed manifest, resource review and actual run.
