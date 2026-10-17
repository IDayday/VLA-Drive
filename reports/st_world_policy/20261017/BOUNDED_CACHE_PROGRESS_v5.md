# Bounded cache implementation progress

Recorded 2026-10-17 UTC. This updates v4 without replacing its evidence.

The user's storage constraint remains unchanged: no multi-TB future-feature
prebuild. Existing current tensors are reused. New future targets remain FP32,
computed on demand with a bounded RAM LRU and zero target-tensor disk output.
Official BF16 training execution and FP32 optimizer/master state are separate
from this teacher-label numerical contract.

## Storage accounting

Independent resource review is integrated at `db615170` (reviewer commit
`3821ac6c`). Existing current tensors occupy about586.123GiB; the conservative
campaign reservation is645.318GiB. These are different accounting quantities,
not two amounts to add. Existing pilots, future metadata, SQLite indices,
journals and interrupted files must share one new package of at most4GiB,
inside the unchanged768GiB campaign cap. The1TiB free-space floor and checkpoint
commitments are separate. No policy or live ledger migration has been applied.

The prospective full-current metadata index is limited to oneGiB and two CPU
cores; a bare CLI currently lacks campaign admission and is not launched.
Explicit additive migration and writer checks are being implemented against
the existing ledger and lock, with independent review still required.

## Future metadata and a real failure boundary

Implementation `69ee4eb6` is integrated as `8c5a93e0`: bounded per-log future
references, original-order finalization, provenance-ledger validation and
optional indexed-reader dependency. The owner's53 CPU tests passed. No real
raw-log/image build, teacher forward or future-tensor write occurred.

Independent review `d9f9081b`, integrated as `8c4520f2`, found substantive
counterexamples: symlink/hardlink output escape, insufficient validation of a
reused COMPLETE receipt, and an unbounded provenance read after a size check.
Its25 passing and8 failing cases are preserved. Status is **CHANGES_REQUESTED**;
the owner's successful tests are not an approval. Fixes are in progress.

The fixed qualification still uses the same32 scenes from two source logs.
PLAN v4 reduced workers from2 to1 and metadata from2GiB to32MiB. It contains no
new observation sampling or result-based selection. Its cache-root PLAN occupies
16,384 allocated bytes and must be included in the reservation. Code fixes will
invalidate its code pins; preserve it and bind a new final plan after integration.
No actual qualification or full-population build is yet admitted.

## Independent training and evaluation continue

Read-only status at05:12:52UTC showed eight progressing registered runs:
SP0/SP1/SP2/SP3 at4033/3623/3085/2751 and C0/C1/C2/C3 at3749/508/389/99.
This is a progress observation, not a new numerical-correctness or quality gate.
Training configurations, processes and immutable runtime sources were untouched.

C1 subsequently saved the complete512-step native checkpoint
`step_000512_1792214036659339678`. Its receipt identity is
`2742bbfe2c45ba2dbcb4248ddd9b6b2dc2a9ff4c8d05edab42e24009add2ba5d`.
Metadata identity and native file sizes were checked; payload hashes are left
to the actual exporter before loading. The fixed1696-scene development job is
WAITING_ALLOWED_GPU, not completed and not a new Navtest task.

The reviewed v3 development helpers were deployed byte-for-byte after99 Lead
metadata/lock checks. They bind exact per-job host/GPU/checkpoint routes and
share four export slots and two scoring slots across the three prepared hosts.
No export or scoring was started during this deployment. Existing v1/v2 files
and completed C0 evidence remain intact.

Private receipts:

- `audits/evaluation/C3_ROUTE_HELPER_v1/LEAD_DEPLOYMENT.json`
- `audits/evaluation/READY_C1_512_v1.json`
- `audits/storage_writer_quiescence_v1/OBSERVATION_1792213801021439355.json`
- `audits/bounded_future_metadata_plan32_v4/`

The writer observation is a read-only process census, **not** the final
quiescence/ledger-migration approval. No GPU process was signalled, no previous
cache/checkpoint was removed, no held experiment was resumed, and nothing was
pushed or marked formally complete.
