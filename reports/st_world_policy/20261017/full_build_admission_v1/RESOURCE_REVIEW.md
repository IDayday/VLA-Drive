# Full future metadata: actual preparation resource review

Independent reviewer: `/root/st_metadata_resource_review`, 2026-10-17.
No dispatch, ledger migration, GPU/model work, image access, raw-log payload read
or checkpoint hashing is performed in this review. Model/source correctness
reviews already completed are not repeated.

## Exact prepared package

The actual `PREPARATION.json` under
`/mnt/project/st-world-policy-20261016/audits/full_future_metadata_preparation_v1/prep_phase_v1`
has SHA `dce5b37f391f37c3076a232510e554f382c3a61ad8d983ec381d63c59970855c`.
Its new policy SHA is
`59371867175a65dd032cc27c88a050227d8ac4ba34a681c2e31b250c58a0978d`.
The live ledger inspected is still
`a4c1fb15f4799187b03035343b8582484d8feb02fd23c7b632b5bc968662789d`.

All eight old declarations are preserved byte-for-field. All old legacy
reservations, allowed roots, campaign/state directory, typed-W package,
768 GiB common cap, ledger overhead and 1 TiB free floor are unchanged.
The sole envelope transition is verified metadata v1 4 GiB to v2 8 GiB, retaining
the exact original package and its digest. New declarations are exactly the
128 MiB dev preparation and 2 GiB train preparation roots. The proposed future
index allocation is **WAITING_REQUEST_ID**, not two invented current claims.

Both real PLANs are COMPLETE in their prior supervisor records. Dev is exactly
1,696 scenes / 16 logs; train 101,592 / 1,176. Each root contains exactly nine
preexisting pinned files. Their complete sets and all hashes were independently
checked, including the preserved PLAN supervisor evidence. Root-inclusive
allocation at review is 49,152 bytes dev and 327,680 bytes train.

New immutable build runtimes are clean:

- dev: `/mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017`, commit
  `1265b2376864ee929d95513f4f91e58d0910d147`;
- train: `/mnt/project/VLA-Drive-st-metadata-build-train-v1-20261017`, commit
  `838dd21f7a16a79086a44bc63be056d6a750570e`.

Every one of the seven implementation hashes recorded in each original PLAN
matches its build runtime. The four declared callback source pins also match.
The narrowly expanded preexisting-pin count accepts the real nine-file roots;
it does not remove a file or weaken byte accounting. Separate independent
correctness review `ec7455c5` reports 19 targeted tests passing. This resource
review does not duplicate those source tests or claim native-model coverage.

`EXACT_PREPARATION_REVIEW.json` records the verified actual pins and limits.

## Disk budget and checkpoint commitments

Retain all original 128 MiB administrative reservations. Add a separate explicit
**4 MiB** allowance for the entire
`audits/full_future_metadata_preparation_v1` control tree. Lead accepted this
correction; its files are outside the two cache roots and must not be silently
charged against a previously named report root.

Current prepared metadata allowance is 3,444 MiB: old 1,264 + two preparation
roots 2,176 + new controls 4. The later two request indexes need another 2,176
MiB, giving **5,620 MiB / 8 GiB** and **2,572 MiB uncommitted**. Earlier preplan
64 MiB is absorbed by the two actual preparation roots, never charged twice.
No tensor cache is added. No old data or checkpoint is deleted.

Fresh `CHECKPOINT_COMMITMENTS_2183G_V2.json` retains the entire original nine-run
1,983 GiB, previous smoke 120 GiB and PROFILE 80 GiB obligations. It does not
deduct already written checkpoints or pretend future allocations are consumed.
It is bound to the prior receipt and actual metadata preparation. Fresh available
space is **7,280,315,932,672 bytes** at Unix 1792228852.226822. The receipt expires
at **1792250452.226822 (2026-10-17 15:20:52 UTC)**, covering dev's 30 minutes and
train's four hours, plus startup/cleanup allowance. The independent 1 TiB free
floor remains additional. Existing cache reservations are also checked by the
same storage guard before and throughout writing. Quota remains **NOT VERIFIED**;
the receipt is a conservative obligation record, not a filesystem quota.

## Actual execution bounds required

Use the existing `metadata_storage.run_registered_metadata` only, through a
thin fixed two-role owner controller. Do not use the old hardcoded
`run_bounded_metadata_v1.py` unchanged: it names runtime `1162b955`, old stage
names and old registration paths. No new trainer or general retry framework is
needed. The actual controller still needs its final independent source pin.

| Role | Worker/CPUs | Process-tree RSS | Running wall | Source read budget | Inclusive root |
|---|---:|---:|---:|---:|---:|
| dev | 1 / 2 | 4 GiB | 1,800 s | 16 GiB | 128 MiB |
| train | 1 / 2 | 4 GiB | 14,400 s | 512 GiB | 2 GiB |

The existing builder paces aggregate reads and metadata writes at 64 MiB/s,
checks time/RSS/directory bounds, and invokes the same ledger guard before
publication. The outer controller must cover imports and noncooperative calls,
own only its new child/session, enforce bounded stdout/stderr, and preserve a
one-shot START/RESULT. Store START/RESULT and captured output in each cache root
as registered `CONTROL_*` artifacts; retain the original `control/` files
unchanged. Their bytes and atomic serialization overlap belong to the existing
claim. Reserve bytes before writing, using the existing JSON write-bound helper.

The actual 32-scene completion's per-log result row is about 300 serialized
bytes; 1,176 rows should fit a 1 MiB stdout cap, but this is an estimate, not a
license for unbounded output. A capped stderr stream and explicit serialized
receipt bound are still necessary. Exceeding a cap is a preserved failure.

Build is metadata-only but **does read** raw logs, image headers and image bytes
for identity hashing, and opaque original ego files for checksums. It never
computes or stores teacher tensors. Do not reuse the PLAN-only claim of zero
image/raw-log I/O. The one-worker 192 MiB train log ceiling is bounded by actual
4 GiB RSS monitoring; it is not evidence that every deserialized log fits.

The storage guard holds the same global ledger lock while a metadata build runs.
Run dev and train serially. Read-only model training, checkpoint saving and W
planning do not become cache writers merely because they inspect cache metadata.
Other actual cache writers must be scheduled by Lead at clear stage boundaries;
do not quietly start competing writers and confuse lock waiting with progress.

## Quiescence and admission boundary

Before same-ledger migration, Lead supplies a fresh, pinned small process census
covering the eight currently allowed hosts and the exact old claim/legacy roots,
with no active or unknown cache/teacher/PCA/metadata/W-extraction writers, plus
an explicit disabled-queue statement. A W `train_plan` only reads metadata and
weights; it is not W extraction and must not be incorrectly classified as a
cache writer. No real/unknown training or evaluation is stopped.

The quiescence receipt must bind old/new policy, exact old ledger, empty new
pilot-adoption inventory and this fresh checkpoint receipt, have a valid short
window, and use the existing `migrate` API and shared lock. Old pending/snapshot
records remain immutable. Lead alone may execute migration and the later
one-shot build. This review of actual prepared inputs is **PASS**. The fresh census below now
permits the exact migration only; actual build still awaits its controller pin. It does
not authorize indexes before real completed request identities, teacher tensor
extraction, further GPU tests or formal temporal training.


## Exact migration admitted after fresh census

`MIGRATION_RESOURCE_ADMISSION_V1.json` authorizes one exact same-ledger migration,
with explicit command argv, runtime and source hashes. Its SHA is
`664c64646626081cbdb53dc4c24d73f9c233552c54637a2f1442c826d96eeed0`.
The latest start is 2026-10-17 09:50:12 UTC and quiescence expires at 09:52:12 UTC.
This is separate from build admission. No migration was executed by this reviewer.

The real eight-host observation SHA is
`d73308cad68ec178a7e90da82c39f004ca39ea4d1f81bff06ad94a3c0c4829ca`.
All hosts were reachable; no unreadable/oversize process records were present.
Four matches were precisely the existing read-only training-observer program
(SHA `67cb7eb1d0ab95512132e69b6f2eaaaa23f3841d9b6022dbd1245b95be08c5ff`).
Its source reads training process state and writes its own observation log, not
cache data or ledger state. Lead separately attested no automatic queued cache
writers. The quiescence SHA is
`6c8ec3807e5646b04102612b229f839e41ba04d3c1f88ac2a49fbf2493ab9ae6`;
all five exact hash bindings and the full set of 28 old claim/legacy roots match.
The live ledger still matched `a4c1fb15...` when admission was produced.

The 4 MiB new administrative reservation is explicitly included; old 128 MiB
reservations are untouched. The existing storage policy's authoritative common
cap remains 768 GiB. The new 8 GiB envelope is a versioned metadata allocation,
not a new teacher-tensor bank and not a second ledger. Quota remains unverified.
