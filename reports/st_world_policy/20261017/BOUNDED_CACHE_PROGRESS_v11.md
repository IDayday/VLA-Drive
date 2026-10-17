# Small-disk execution update

2026-10-17 09:24 UTC. The user's constraint is retained: no new multi-terabyte
teacher feature bank. Existing data and checkpoints are not deleted. This
update concerns the active ST DINO-target pipeline, not restarting the held
IQE expert/Scorer or old Query Base visual-cache campaigns.

## Actual evidence

- The original FP32 fixed-batch teacher and lossless cached references remain
  unchanged. Existing current-feature tensors are reused; missing future targets
  are computed on demand, with a bounded RAM LRU. No future teacher tensor file
  is written by this path.
- The completed real 64-update PROFILE used 32 scenes, global batch 2 on one GPU.
  All 64 updates were finite; new teacher tensor disk bytes were **0**. Final
  LRU occupancy was 14,159,232 bytes. After eight warmup updates, 56 updates
  averaged 2.7476 seconds (p50 2.6777, p95 3.7735). This repeats a small population
  and is not full-population, multi-GPU throughput evidence. See
  `indexed_profile64_actual_v1/ACTUAL_RESULTS.md`.
- CPU-only full-population PLAN actually completed for **101,592 training** and
  **1,696 development** scenes, without reading image/raw-log payloads or models.
  The two runs used 9.79 and 4.38 seconds, respectively; no workers remained.
  See `FULL_METADATA_PLAN_ACTUAL.md` for hashes and exact entry points.
- Metadata preparation controls now retain all nine original PLAN/supervision
  files. The sixteen-pin ceiling changes only how many precisely checksummed
  control files can be adopted, not any byte/model/data limit. Independent
  negative tests passed (19 tests); no old PLAN or evidence was rewritten.

## Storage decision

Naive extrapolation from the measured 32-scene metadata build estimates roughly
2.64 GiB for full future metadata plus indexes. This remains an estimate, not
completed storage usage or a guarantee. The explicitly versioned combined
metadata envelope is at most **8 GiB**. The complete proposed reservations are
**5,620 MiB**, including old claims, temporary files, indexes and separately
accounted administrative controls. The extra 4 MiB preparation-control root
is additive; it is not silently borrowed from the previously bound 128 MiB
administrative roots.

The first proposal adds only actual dev/train preparation claims (128 MiB and
2 GiB). Index claims wait for their real completed request identities. All
old cache claims, typed W records and checkpoint commitments remain protected.
The overall cache ledger's existing allocation is not a request to create
that amount again. New future teacher tensor allocation remains zero.

Builds are bounded, serial CPU tasks with one worker and aggregate 64 MiB/s
source IO limits. The full training population, five target times and three
camera identities are unchanged. At this timestamp the PLAN and controls are
complete; full preparation/index builds and full-population training admission
are **not yet complete**. No existing training was paused for this work.

## Other campaign work, kept separate

SP0's previously computed 16 W records were independently validated. An additive
reconciliation receipt preserves the old supervisor failure instead of editing
it into success. The next original train64 plan completed with no extraction;
its RESULT SHA is
`001952cd9d5f608ba1761056d41159135ec3e61853d9aff5d3da90f1590aef15`.
Train extraction and readout fitting remain separate pending stages.

C4@512 and SP3@5000 fixed full-development evaluations completed: 1,696 scenes,
16 logs, no invalid scores. Registered paired differences were +0.000227 and
+0.229014 points /100, respectively; both log-bootstrap intervals include zero.
SP3's mean gain accompanies new zero/safety failures. These do not select a
winner or terminate C training. See
`../20261016/evaluation/FIXED_PAIRS_C4_512_SP3_5000_v1/REPORT.md`.
No Navtest, held IQE/MOPD run, ACTIVE change or push was initiated.
