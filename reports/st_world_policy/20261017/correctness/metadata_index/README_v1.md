# Independent metadata index review — fixed 9c7743f5

Decision: **CHANGES_REQUESTED**, limited to the CLI contract pin boundary. The metadata index and adapter checks below passed; this is not formal/provider/profile admission.

Reviewed source: `9c7743f5323cbc0e85c3aaa4f9e13685ac4ae234`, isolated branch `agent/st-correctness-metadata-index-20261017`. Reviewer: `/root/st_correctness`. No production implementation was edited.

## Blocking finding META-F01

`tools/st_world_policy/index_teacher_metadata.py:42` hashes a contract receipt; line44 then separately reads it for JSON parsing. An atomic replacement between these reads permits a different valid teacher contract to be indexed under the original approved receipt hash.

The independent test replaces the path immediately after its first read returns. On fixed9c, the CLI publishes `COMPLETE` with `teacher_identity=9*64`; the pinned bytes described `teacher_identity=a*64`. The test fails on the actual published contract. The earlier alias-hook reproduction and the later implementation-independent read-hook reproduction are both preserved, and count as one finding.

Fix requested: a single bounded byte read, SHA validation of those bytes, and parsing the same bytes. Existing size and schema checks must remain. Concurrent mutation may be rejected, or the approved snapshot may be used consistently; the replacement contract must never be published. The reviewer has not implemented the fix.

## Independent checks

- 13 metadata boundary tests passed in3.73s; one distinct contract-pin counterexample failed (latest3.47s). CPU affinity0–1, thread caps2, each command bounded to120s. No skips. One existing pynvml deprecation warning.
- Real current256 metadata:256 scenes,768 records,983040-byte SQLite. Every scene, record, source receipt and original shard/row ordinal matched the original reader metadata.
- Real future64 metadata:64 scenes,957 records,638976-byte SQLite. All four future times plus current, reversed/repeated access, masks and identities matched. Membership was192 current hits,765 future misses,3 absent-view references.
- The independent actual check took1.859279s excluding imports; process peakRSS627769344B. Two reader allowances total33554432B. This measures metadata work, not training or shared payload throughput.
- Cold state restoration verified metadata identity and empty resident state; idempotent rebuild succeeded. It did not execute teacher inference or a student optimizer.
- The actual check guarded payload APIs and file opens:0 image/tensor files,0 teacher calls,0 tensor bytes written. Only bounded original metadata was read; full population was not scanned.

The new index streams canonical rows, checks raw source SHA and semantic identity, preserves original cache record order, validates per-row checksums and detects source/index/receipt changes. Indexed adapters reuse the original tensor/provider path; this review does not claim to have rerun that path.

## Scope limits

The owner-reported full legacy `source_logs` header may exceed the fixed64KiB header bound. Fixed9c safely rejects it. The present256/64 result does not establish full-population compatibility. The proposed future provenance ledger and later preparation/factory wiring are outside this review.

Full future metadata remains `WAITING_INPUTS`. No real indexed teacher/student qualification, formal registration, profile admission, GPU operation, large-weight read or new scoring was performed. Previously reviewed numerical and native resume evidence is neither replaced nor expanded.

The JSON review binds exact source and evidence SHA256 values. Private evidence is under `/mnt/project/st-world-policy-20261016/audits/correctness/metadata_index_review_v1/`.

## Reproduction

```bash
cd /mnt/project/VLA-Drive-st-correctness-metadata-index-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_review_metadata_index.py::test_cli_contract_must_be_the_same_bytes_that_passed_pin -q
```

This must fail on the reviewed9c implementation; the fixed implementation must satisfy the same semantics. Preserve this failed evidence. After the owner fix, rerun only the affected CLI/identity boundaries in a new pinned checkout.
