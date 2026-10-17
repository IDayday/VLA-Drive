# Actual 32-scene future metadata qualification: independent receipt review

**PASS_REAL_METADATA_QUALIFICATION_32_ONLY**.
This is the new limited real-data conclusion; it does not repeat or replace the
earlier synthetic code reviews.

Reviewed artifact root:
`/mnt/project/st-world-policy-20261016/caches/future_metadata_qualification32_v2/`

Execution runtime: `1162b955b3bf9abb6728823a84a50c38e1a19bb8`, clean at review.
The actual runtime implementation hashes match the independently reviewed
builder, guard and legacy comparator. All source pins in the actual PLAN and
storage registration were checked against the runtime files.

## Real result and exact population

The real reference result is **REFERENCE_MATCH**:

- 32 scenes from the same two original logs, in the same order as the prior v4
  PLAN. The selected rows, original source identity and original selection
  dependencies are unchanged; no resampling occurred.
- Five requested times `[0, 0.5, 1, 2, 4]`, three original camera views,
  **480 slots / 3532 checked leaf fields / 0 mismatches**.
- This particular real set has **0 missing-view slots**. It does not add a new
  real-data missing-view or ambiguous-tie qualification; those remain the scope
  of the earlier synthetic tests.
- Actual request and raw-log-ledger file hashes, semantic identities, complete
  receipt, shard receipts, both shard contents, ordered scene/log coverage and
  the reference command's pinned dependencies all agree.
- No teacher extraction, tensor output or training admission. Ego/GT tensors
  were not decoded by the qualified preparation/reference paths.

The matching original-reference result was produced by the previously reviewed
comparator. This reviewer inspected its complete linked artifacts and receipts;
the real raw logs/images were deliberately **not rerun or rehashed**.

## Recorded bounded execution

| Stage | Outer invocation wall | Charged reads | Outer peak RSS |
|---|---:|---:|---:|
| Guarded future preparation | 15.843630731 s | 203,803,324 B | 756,162,560 B |
| Legacy reference comparison | 15.417619962 s | 237,803,824 B | 748,396,544 B |
| Combined | **31.261250693 s** | **441,607,148 B** | 756,162,560 B maximum |

Both outer receipts report PASS, exit 0, no automatic retry, CUDA hidden and CPU
affinity `[0,1]`. The reference's prior-charge fields exactly match preparation's
completed cost; the invocations are sequential. Each has the registered
4 GiB/450-second half, within the combined 8 GiB/900-second limit. The outer
times above include process overhead and are the relevant combined wall
accounting; the comparator's internal 21.6793-second sum is not substituted for
them.

The guarded preparation recorded **32 checks, 30 writer callbacks and 14
publication callbacks**. Its actual admission ties to the registered root and
the exact policy/registration hashes. The existing registered allocation is
32 MiB, not an added package. The root currently occupies **659,456 allocated
bytes**, including control records. The reference records zero cache writes.

These are cross-checked execution receipts for this bounded run, not an
independent resource approval or a sustained-throughput benchmark.

## Immutable result identities

| Artifact | SHA256 |
|---|---|
| PLAN | `32bcf1d6edd7204a4b21d275fc521c66e817809cad127860f70bfddcdc781794` |
| COMPLETE | `597903c13d346751ee60a54c7f2ca385d5760452e241690bebacd6f3e155d25e` |
| TEACHER_REQUESTS | `edcee46d6227fcf5fbd2a16284c31bffe94fdca76e442da0e1fa1cfaabf1cd99` |
| RAW_LOG_PROVENANCE | `16ba94b5981a4e5c7557a5e0679db0bdc617739cf0385a32d3ed7703f69e6a83` |

`EVIDENCE_HASHES.json` records all metadata/code files inspected, byte sizes and
their actual SHA256. `audit_receipts.py` contains the assertions;
`receipt_audit.log` is its successful execution output, with the machine-readable
conclusion in `REVIEW.json`. No private image or raw-log bytes are committed.

## Review boundary

The independent inspection read **898,840 bytes** of metadata/code including the
initial receipt inspection, below its 2 MiB limit. It performed zero raw-log,
image or tensor reads, zero new qualification invocations, zero GPU work and
zero campaign ledger mutations. It changed only this review directory.

The completed qualification supports equivalence and guarded preparation for
this exact 32-scene population. It does **not** qualify full-population metadata
preparation, large index creation, sustained IO, teacher extraction, PCA/provider
training integration, formal training or scientific benefit. Those gates remain
separate; this reviewer has not signed them.
