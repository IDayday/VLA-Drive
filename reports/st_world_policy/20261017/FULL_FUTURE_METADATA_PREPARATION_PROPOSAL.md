# Full future metadata: bounded preparation proposal

2026-10-17 UTC. Source inspected: `9c7743f5323cbc0e85c3aaa4f9e13685ac4ae234`.
This document proposes preparation, not a completed build or training approval.
Only small receipts/configuration and source were read. No full image scan,
teacher extraction, GPU work, tensor write, dataset change, or live runtime edit.
Full temporal population: **WAITING_INPUTS**.

## Exact population and boundary

Use the existing original population, in its original order. Do not substitute
current-only coverage for `[0, 0.5, 1, 2, 4]` future requests.

| Binding | Value |
|---|---|
| Original student root | `/mnt/project/ddp-foresight-artifacts/20260928/student_train_v1` |
| Population | 101,592 scenes / 1,176 logs, train |
| Student identity | `f5fbbb2ddbc1d0ec67f9db2209b7953eadd3130e656365796c03282bbbbc0967` |
| Index semantic identity | `1045890740e1a4ec8c11b802dede2b6a42d0c0b7131d7f581081e6da1f3b7b7d` |
| Index file SHA256 | `53e6175193f69f394f94dcc0c715222042c5abcbbf217f83fb6ffacb4bcf1179` |
| Partition SHA256 | `ab956543a4f6a96c64e5211ca5fce61e74580901f2335c772e29473ba2cc56d2` |
| Ego identity file SHA256 | `a8b6416c68b79e174d88dd0625594accda390b866b42436d94035a68f5974206` |
| Ordered scene IDs hash | `9cd301e72bf2a9e46f5f9199dc0322c9fa7db92071e3aea4bed1558c77b89c3a` |
| Existing selected-manifest SHA256 | `58dffa070d434d15cb9adf6382010520aff57f4aa624aa081412b70e6f28b7ea` |

These values were read from
`/mnt/project/st-world-policy-20261016/audits/history_data/full_current_population_v1/train/`.
`REGISTRATION.json` SHA256 is
`f7bad022b4aae85c5925b0a1afa143077aa0764efc1849d80f622d092beebbec`;
`COMPLETE.json` SHA256 is
`0c1c913e72733191fc47cf1969b2fb8462b1081ae8c893b656b7aba5d022ad3f`.
Their status refers to **current_only_metadata**, not futures. This proposal did
not rehash all original image content.

The prior future64 configuration declares raw logs at
`/mnt/navsim/trainval_navsim_logs/trainval`, primary images at
`/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval`, and fallback images at
`/mnt/navsim/trainval_sensor_blobs/trainval`. Its file SHA is
`8a6e9312d001821ebc82443b20344b613cd7c9737388050f2229ed45e3a7bc97`.
These are discovered configuration values, not a current mount/access claim.
The eventual Lead-registered plan must resolve authorized host mounts explicitly.

## Minimal reusable interfaces and required changes

Reuse `prepare_current_teacher.source_population` for exact source/index,
partition, role, log isolation and ego identities; retain the existing selected
manifest and current/ego SHA pins. Reuse the bounded worker queue, per-log
registration/shard pattern and shared read-budget concept. Do not copy its
finalizer unchanged: it currently accumulates all scenes and selected rows in
Python dictionaries before serializing the complete population.

Reuse the **semantics** of `prepare_history.build_target_scene`: exactly the
five requested times, <=50 ms tolerance, strictly increasing raw timestamps,
unique closest frame, ambiguity rejection, current frame required, and original
`None` behavior for truly absent future frames/views. It currently rebuilds the
whole-log token map/timestamp vector and hashes overlapping images for each scene.
A future-only helper should instead:

1. Load and verify one raw log once per bounded worker; construct one token map
   and timestamp vector. Keep the same nearest-frame algorithm, including ties.
2. Reuse SHA/header metadata for the same resolved, unchanged image within that
   log, with a bounded memo. First access hashes the actual original bytes;
   filename equality alone does not admit reuse. A mutated file fails the log;
   it does not turn into a missing target. Current/ego hashes remain verified;
   no GT values enter target image selection.
3. Write one immutable complete shard per log. Pin registration, exact token
   order, original log path/SHA, source records, all five frame rows and genuine
   missing views. Do not emit causal-history files unless separately required.
4. Finalize in original index order using a bounded log reader, rather than
   completion order or a global scene dictionary. Stream the canonical semantic
   hash and final sorted `atomic_json`-compatible file. Keep all raw image SHA
   values and exact frame/view order. No new sampler or training population.

The original source index is only 9,549,650 bytes on disk and may remain the one
bounded control population. Do not copy several complete expanded target objects.
A per-log raw pickle may itself be large; qualify actual maximum resident size.
If two loaded logs exceed the 4 GiB bound, use one worker or stop with a resource
failure. Do not omit scenes or change time sampling to fit memory.

### Required provenance format change

`prepare_history` embeds `source_logs` in every request header. With 1,176 logs
this may exceed the new index's explicit 64 KiB non-scene header bound. The 9c
current256/future64 parity result is valid, but does **not** demonstrate unchanged
old-style full-history headers will be admitted.

The approved direction is a separate immutable raw-log provenance ledger. New
future requests explicitly bind its path, raw SHA256, semantic identity and
schema; do not expand the header guard, modify historical requests, or silently
reuse an old request identity. The ledger records the exact registered log set,
original-order log list, raw paths and each raw log SHA, plus source/partition
and registration identities. Admission must open and validate the ledger itself,
check its semantic and raw hashes, and verify exact scene-log coverage. Merely
recording a pathname/hash string is insufficient.

This needs a narrow optional ledger verifier in the metadata index before new
requests can be admitted; it is **not implemented in 9c**. Legacy manifests keep
their existing path and identity. The new provider/PCA execution identity must
bind the new request/index identities; the frozen PCA basis is not refit. No
shared trainer/factory or scope gate is changed by preparation.

## Budgets and atomic recovery

Proposed hard maxima: 2 CPU cores, at most 2 concurrent log workers, aggregate
source-read rate <=64 MiB/s, builder process RSS <=4 GiB, preparation metadata
<=2 GiB including shards, final files, temporary files and retained interruption
evidence. No target tensor files, image copies or teacher execution. Deadline is
explicit per invocation; a resumed build reuses only verified complete shards.
Reader metadata remains separately bounded to <=256 MiB per rank.

The index sidecar also retains its independently reviewed <=2 GiB aggregate
campaign budget. Thus request preparation plus indices need a **proposed** <=4
GiB combined metadata reservation, not a claim that the existing 2 GiB index
budget includes another 2 GiB preparation. Lead must register both totals and
check actual free space; no reservation or full build was performed here.

Observed small request size: future64 `TEACHER_REQUESTS.json` is 502,318 bytes.
Linear scaling is 797,367,035 bytes for 101,592 scenes before changes to headers,
deduplication or JSON formatting. Keeping both shards and final file is roughly
1.49 GiB before ledger/receipts/temporary overhead. This is an **estimate**, not
proof of fitting 2 GiB. Check projected sizes in the 256-scene qualification and
stop before writes exceed the registered budget. Retained partial evidence also
counts; do not automatically delete it. History64 files total 936,677 bytes;
causal-history generation would be a separate material expense and is not part
of this future-only proposal.

Registration is immutable and pins source, selected manifest, time/view policy,
implementation hashes and resource limits. A shard is published only after its
complete declared log subset is validated, fsynced and atomically renamed. The
per-log receipt pins bytes and content identity. On reuse, validate original
raw-log SHA/source record pins and image file versions; any conflict requires an
explicit new registration. Unfinished logs can be recomputed; never silently
trust partial JSON or change their missing-view masks.

Finalization streams into an unpublished file, recomputes coverage and ordered
identity, fsyncs and renames before writing COMPLETE. COMPLETE binds requests,
ledger, selection and all shard receipts. A failure between file and receipt
publication can verify/reuse the immutable final bytes. An unrelated/conflicting
final file is retained and rejected. Full COMPLETE requires all 101,592 scenes
and all 1,176 registered logs; a qualification subset has a different scope and
cannot be promoted to full coverage.

## Proposed <=256-scene qualification; not executed

Lead first records a PLAN with a fixed original-order subset of at most 256 unique
scenes before reading any new labels/results. Include the existing 64 future
scenes as an anchor if their pins match, plus at most 192 additional scenes from
fixed contiguous source windows across several log lengths. Group windows permit
measurement of real repeated-image reuse; the earlier one-scene-per-log selection
does not estimate that reuse well. Preserve the original full-index relative
order. Missing-frame coverage can use known metadata; synthetic negative cases
cover cases not naturally present. Never enlarge or resample the real subset to
obtain convenient results.

Compare the old selection/reference implementation and the bounded implementation
on that same subset: every frame/token/time/view/path/SHA/shape/missing record,
current/ego source pin, scene order and ledger log coverage. A new provenance
header necessarily has a new request identity; compare exact scene bytes and
explicitly normalized common semantics, not falsely require identical whole-file
identity. No teacher or tensor numerical claim follows from this check.

Inject interruption after one shard, during finalization and before COMPLETE;
resume must reproduce request and provenance identities. Reject changed images,
raw logs, source/partition hashes, duplicate tokens, timestamp ambiguity, invalid
current frames, unregistered logs, exceeded IO/disk/RSS budgets and stale partial
outputs. All source reads by both reference and optimized checks count separately.
A first bounded qualification may use 15 minutes and <=8 GiB actual source reads;
if insufficient, record PARTIAL and its cost rather than changing the population.
That small scope does not prove full-population coverage or raw-log memory bounds.

## Measure cost before registering the full build

Record per log: raw-log bytes/hash wall time, unpickle time and RSS, lookup build,
unique image files/bytes, memo hits, actual SHA/header read bytes/time, throttle
wait, current/ego pin checks, metadata serialization/write/fsync, finalizer time,
peak temporary disk and total wall time. Separate CPU time, IO wait and overlapping
workers; do not sum inclusive times into a fictitious critical path.

The historical current-only COMPLETE reports 64,204,201,738 source bytes read in
1,698.646 s. It is useful context only: future raw-log decoding, frame reuse and
new IO differ. There are at most 1,523,880 scene/time/view references before true
missing records and deduplication; that is **not** the count of unique images or
an additional tensor-cache allocation.

Estimate full cost from actual fixed-window unique-file statistics, measured log
sizes and original population group sizes; report a range and its sampling limits.
The IO lower bound is total bytes to hash divided by the registered aggregate
rate. Include restart revalidation, serialization, target metadata writes and
shared-storage contention separately. A four-hour invocation cap is a stop budget,
not a promised full completion time. Full preparation can resume verified shards.

## Follow-up permission and deliverable boundary

The next implementation may add the narrow future preparation module, its
synthetic behavior tests, the optional ledger admission check and a metadata-only
PLAN generator. A real <=256-scene scan, full preparation, teacher extraction,
student PROFILE/FORMAL and any source/PCA/runtime migration are separate Lead
registration/dispatch decisions. This report selects no temporal cell or spatial
winner and grants no training permission.
