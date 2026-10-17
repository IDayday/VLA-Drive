# Independent metadata resource review

2026-10-17, reviewer `/root/st_metadata_resource_review`. Review source:
`4e3e36a0784f685e77cb853d678b4d28e0d62c40`. This agent did not implement the
metadata index, run a builder, inspect images/tensors, or dispatch any workload.
The owned evidence is `READ_ONLY_INVENTORY.json`; private copy is
`/mnt/project/st-world-policy-20261016/audits/metadata_resource_review_v1/`.

**Decision: the current-only index has a bounded CPU execution design and fits
the proposed storage envelope, but launch is BLOCKED_PENDING_LEDGER_AND_GUARD.
The future preparation proposal is RESOURCE_PLAN_ACCEPTABLE, IMPLEMENTATION_NOT_RUN.
No temporal PROFILE, formal training, or full-future preparation is approved.**

## Actual evidence and scope

Read AGENTS and the original ST research/execution contracts. Read index,
index CLI, storage guard, current/future preparation code, owner metadata parity
evidence, and independent Correctness v2. Source index/CLI hashes match that
review. Correctness v2 SHA is
`635b14c1ebf5a065aebdcfbf38cfce13eb55fa2691b4d6e2bc4c963f27202bf7`.
Its limited result is inherited, not rerun or promoted to full training proof.

For the full-current source, this review read only 2 KiB head and 4 KiB tail
from each of the following three files, plus stat metadata. No full parse/hash:

`/mnt/project/st-world-policy-20261016/audits/history_data/full_current_teacher_train_stage_plan_v3/cache/train/native384/`

| Source | Bytes | Observation |
|---|---:|---|
| requests.json | 170,157,016 | sorted atomic_json, current-only times `[0.0]` |
| identity.json | 145,979,873 | FP32 native18x24x1024, records array, 9,525 shards |
| COMPLETE.json | 1,189,892 | COMPLETE, 101,592 scenes, 304,776 images |

The ends of these files match the parser's expected layout; this is not a
proof every middle row is valid. The builder must check the complete metadata.
Request identity is `2c107e58146482d3f3985d7803d52ecf64d2e3ce9939435d8fce86e24fcf6d6f`;
cache identity is `d92dcced1f92106e0e10a361bc65ea67348be299f07b78e38b75ba973079aa56`.
The COMPLETE header gives request SHA
`9f47d8f5eeb552364211f0066b9b1af5e4e0f58ffb13f53a72162f0505512431`
and identity-file SHA `bf9e434cfae3a54391e1edd1508235869272843b44cd4daf27a9fd5fc6a74c64`.
These were observed, not independently rehashed here.

No full `source_logs` object was observed in this current-only header. Future
legacy manifests can exceed the index's 64 KiB non-array header limit; the
proposed separate provenance ledger still requires implementation and review.

## Storage arithmetic: one envelope, not another four GiB on top

At 05:02:23 UTC the exact v4 policy SHA was
`5fcc4421789cda3ed23b366dd78cf9fa7a867c3e7734aa3f1ed6bdaf1836618a`, ledger SHA
`da4a5b89427a7aacf41265b0c3babe528b6a1999557b3e34dc709c13ccab743c`.

| Item | Bytes | GiB |
|---|---:|---:|
| Existing legacy + claims + ledger reservations | 692,905,198,438 | 645.318253 |
| Proposed metadata maximum, **including existing pilots** | 4,294,967,296 | 4.000000 |
| Maximum combined reservation | 697,200,165,734 | 649.318253 |
| Campaign cap | 824,633,720,832 | 768.000000 |
| Headroom after the entire metadata envelope | 127,433,555,098 | 118.681747 |

The historical 586.123 GiB is measured current tensor storage, whereas 645.318
GiB is conservative reservation accounting. Do not add both together. The
1,099,511,627,776-byte free-space floor is separate. So are the actual remaining
checkpoint commitments; a fresh admission must include their current receipts.
Filesystem available bytes were 8,414,056,316,928 at this observation, not quota
permission and not a guarantee at launch. No quota claim is inferred from df.

Stat-only inventory of six known pilot/review roots found 62 files,
12,913,208 logical bytes and **13,045,760 allocated bytes**. These retained
versions include failures/old prototypes and are not deleted. They are outside
v4's allowed roots and absent from the ledger. Round each root up to MiB:
3+3+2+3+2+2 = **15 MiB**; preserve the precise root mapping in the inventory.
This 15 MiB comes from the two-GiB index allocation, not an extra allowance.
Directories, migration snapshots, reports/logs and atomic publication overhead
must also be counted in their specified metadata/ledger bounds.

Suggested maximum partition of the **same four-GiB package**, registered in
phases as dependencies become real:

- Existing immutable pilots: 15 MiB.
- Full-current index root: 1,024 MiB, including metadata, partials and journal.
- Future preparation: 2,048 MiB, including shards, final requests, provenance,
  temporary files, qualification output and retained interruptions.
- Remaining future index/metadata allowance: 1,009 MiB.

The last allowance is not presently shown sufficient. See the SQLite limit
below. Claim identities must bind an actual immutable build registration and
inputs; do not invent a future request/index identity before it exists. An
unregistered allowance does not authorize a write. Aggregate new reservations
and adopted pilots must never exceed four GiB as later claims are admitted.

## A: full-current index cost and runtime bounds

The implementation streams one JSON row at a time, limits rows/headers to
64 KiB, validates scenes individually, uses SQLite indexes for sorted record
joins, and does not assemble the population in Python. SQLite page cache is
8 MiB, mmap disabled; per-row decompression is bounded. The source import
includes Torch, accounting for much of the observed small-test ~0.6 GiB RSS.

Full build reads each source metadata file three times: initial raw checksum,
semantic parse and final checksum. Those files sum to 317,326,781 bytes, so
source-file reads total about **951,980,343 bytes**, before SQLite IO and its
final hash. Verify reads the source again and hashes/iterates the new DB. This
is metadata IO, not a scan of 539 GB of source tensors. Every new rank's reader
initialization rehashes source metadata plus its DB; amortized startup cost is
real and must be measured when training is later qualified.

Owner current256 DB size is 983,040 B. Linear projection is 390,113,280 B
(0.363321 GiB), **not** a measurement or a storage guarantee. A single build
invocation on two CPU cores, RSS<=4 GiB, elapsed<=1,200 s and one-GiB total root
is a reasonable finite measurement. There is no evidence to promise completion
within 1,200 s. Run at most one build/verify process, sequentially; no GPU.

The current code checks peak RSS and root bytes every 128 row operations. It
also limits SQLite pages to **half** `max_disk_bytes` to reserve rollback space.
Use a 1,008-MiB builder limit inside the one-GiB reservation, keeping 16 MiB for
admission/log/receipt overhead. This gives a 504-MiB database limit, above the
current size projection. A limit error is a retained PARTIAL/FAILED attempt,
not permission to enlarge storage or delete evidence. RSS checks are sampled
process checks, not a hard cgroup cap: use a process-tree memory cap or an owned
process monitor at admission and preserve any failure evidence. Do not set a
four-GiB virtual-address limit around Torch and call that an RSS limit.

**A launch cannot use the raw CLI alone.** `index_teacher_metadata.py` has no
StorageBudget call, no shared free-space check, and no campaign precharge. Its
per-directory limit does not protect the campaign from simultaneous builders.
The small changes needed are listed in MIGRATION_PLAN.md. They affect resource
admission, not teacher precision or target semantics.

## B: future metadata preparation boundaries

The proposal reuses exact original source/partition order, per-log work,
unchanged timestamp/tie/missing-view semantics, bounded source hashing and a
streaming finalizer. This is a suitable resource direction, but the per-log
memo, streaming finalizer and provenance-ledger admission are not implemented.
No source image scan or real <=256 qualification was run by this reviewer.

Required initial qualification is <=256 fixed real scenes, two CPU cores,
aggregate RSS<=4 GiB (including children), no GPU, <=15 minutes, <=8 GiB source
bytes and <=64 MiB/s aggregate source IO. The old reference path's reads count
as well. All raw-log reads/unpickle passes, hash reads, PIL header reads,
current/ego pin reads and restart revalidation must pass the same budget;
`prepare_current_teacher.current_image_reference` currently throttles the SHA
read but not its separate PIL header read. Do not simply reuse that helper and
claim all future preparation IO is bounded. Prefer one worker first if the raw
log memory envelope is unknown; two workers are permitted only under the same
aggregate RSS limit.

Future64 requests scale to ~797,367,035 bytes for 101,592 scenes. Retained log
shards plus one streaming final file project to ~1.485 GiB before overhead.
The two-GiB preparation cap is plausible but unproven. A partially written
final file must be counted, including old orphan attempts. Full future request
coverage is still WAITING_INPUTS. Four hours is an invocation stop budget,
not a projected completion time. Do not turn max-scenes into downsampling.

Future64 DB scales to 1,014,294,528 bytes (~0.945 GiB) before possible full-log
image deduplication. Together with the current DB this projects to ~1.308 GiB
of published DBs. However the current builder's half-budget page limit means
the remaining 1,009-MiB **directory** budget permits only ~504.5 MiB of DB.
Therefore it is incorrect to sign full-future index readiness from the sum of
final DB sizes. The grouped <=256 qualification must measure true unique-image
coverage and DB size. If the remaining bound is insufficient, keep this stage
BLOCKED and review a bounded index/write-budget refinement or a documented
within-four-GiB reallocation using actual earlier sizes. No automatic cap
increase, unreviewed journal changes, eviction or precision change is approved.

Full future preparation, even if metadata fits, does not approve indexed
teacher/student execution. Factory/PCA/source bindings, real indexed output
parity, recovery and complete-update resource qualification remain separate.

## What this review did not do

No implementation tests were rerun; no actual full-population build, metadata
rehash, image or tensor read, teacher invocation, GPU dispatch, cache mutation,
policy/ledger edit, deletion, training, or source runtime change. Existing
Correctness results are attributed to that reviewer, not counted again.
Only the owned reports and bounded inventory evidence were written.
