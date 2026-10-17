# Disk-bounded cache implementation progress

Observed 2026-10-17 10:29 UTC. This continues the active ST campaign under the
user's instruction not to generate several terabytes of new cache data.

The selected path remains unchanged: reuse the immutable current-frame DINO
targets, compute missing future targets with the same frozen FP32 teacher,
and retain computed targets in a bounded RAM LRU. There is no new full future
teacher tensor bank. Student BF16 execution, FP32 master/Adam state, ZeRO2,
training targets and sample populations retain their registered definitions.

## Actual storage and remaining work

- Development metadata: COMPLETE, 1,696 scenes / 16 logs; final root allocation
  30,265,344 bytes. Teacher extractions and new teacher tensor bytes are zero.
  The actual receipt bindings remain in `CACHE_DECISION_AND_PROGRESS_v13.md`.
- Training metadata: the original 101,592 scenes / 1,176 logs remain fixed.
  At 10:29:47, 104 completed log-shard files occupied 202,264,576 allocated
  bytes. This is a partial shard observation, not full-root usage, a scene
  completion percentage, or a COMPLETE result.
- Existing metadata reservations total 5,620 MiB within the 8 GiB ceiling,
  including temporary space. A proposed separate 1 MiB index-preparation
  administrative allocation would make this 5,621 MiB; it has **not** been
  admitted or written into the live policy by this report.
- Neither full future request index has been built. Both actual completed
  request identities, the separate administrative allocation and fresh bounded
  runtime admissions are required before those index writes.
- Existing caches, checkpoint commitments and the free-space floor remain
  separate protected resources. No cache or checkpoint was deleted.

## Corrected index execution path

The original request-only indexer and storage ledger are reused. Independent
review found two defects in its new bounded execution adapter: a stale process
snapshot could report a normal leader exit as a remaining descendant, and a
completion file growing after its initial stat could exceed the read bound.
Both original failures are retained in the review history.

The fixes refresh the owned-process snapshot after confirmed leader exit and
read COMPLETE through a validated, nonblocking, no-follow file descriptor with
an explicit 1 MiB plus one-byte rejection sentinel. Real descendants, changed
files and excess data still fail. The two exact sources are:

- Adapter SHA256: `eb9713666a84e8ffba7a55c75f656bb6133218dc626d772200189adad6d41a9a`.
- Monitor SHA256: `8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644`.

Independent correctness: **143 passed**, 4.28 seconds, in
`correctness/full_future_index_execution_review_v2/`.
Independent resource-boundary addendum: **43 passed**, in
`full_future_index_adapter_resource_addendum_v2/`.
These include small CPU fault fixtures, not real full-population indexing or
multi-GPU throughput. The shared filesystem's timestamp granularity made an
initial 1 ms mutation fixture ineffective; that failure and the measured
timestamp behavior are preserved. A visible 2-second change was rejected by
the unchanged production version checks.

The active training-metadata controller remains byte-for-byte unchanged at
SHA256 `a866ae4261cad36312355028cdfba3b24f52f64b174433635d0473420ddc84bf`.
It has not been restarted or hot-patched. Its eventual result must be assessed
from its real terminal evidence, including the previously documented legacy
post-wait I/O accounting issue if that issue actually affects the result.

## Qualification boundary

The prior 32-scene equality test and 64-update PROFILE remain small-path
evidence only. Inspection identified that the existing PROFILE admission caps
the indexed population at 256 scenes. A full-population reader qualification
therefore needs a narrow, explicit bounded-PROFILE adaptation; replacing an
index path alone is not valid. That adaptation must retain update/exposure
limits, the original sampler and learning-rate timeline, and must not grant
formal training permission. It must not fabricate identities for unfinished
indexes.

No full-population supply-rate, multi-rank or formal temporal-training result
is claimed here. The no-terabyte-cache decision is implemented; its complete
production qualification remains in progress.

## Parallel registered evaluation

C2 and C3 at update 1024 each completed the fixed 1,696-scene development
evaluation with zero invalid or unresolved scores. Their PDMS means are
91.0182376744 and 91.0092363180 on the explicitly stated 0–100 display scale;
both have 50 zero-score scenes. These are intermediate diagnostic endpoints,
not evidence for stopping their registered 4096-update experiments.

- C2 official COMPLETE SHA256:
  `3eeaa46c74d0ab6c9fbe4493848896250ea4811ba267aa2e92f3e24b043f3f03`.
- C3 official COMPLETE SHA256:
  `8076856d4c88d80bb711285bbf00cb5ebba36e9af95a6699affe0fa95afb5b2c`.

C4 at update 1024 remains in registered development inference. No navtest,
held IQE experiment, formal ACTIVE change or push was started.
