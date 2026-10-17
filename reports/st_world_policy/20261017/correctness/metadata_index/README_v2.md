# Independent metadata contract pin follow-up

Decision: **PASS_METADATA_IMPLEMENTATION_LIMITED** for `400a5607a318b8cc3a35a1d777534e898fd4661a`. Reviewer `/root/st_correctness` did not implement production code. This does not authorize formal or profile execution.

META-F01 is fixed. The CLI now reads at most1MiB+1 bytes once, verifies file version/length, checks SHA256 of those bytes, and parses the same buffer. The independent atomic-replacement counterexample now rejects without publishing `COMPLETE`. A normal approved contract is opened exactly once; invalid SHA, oversized and empty receipts reject. Schema and contract checks remain intact.

The bounded follow-up ran **9 passed,9 deselected,0 failed,0 skipped in3.43s** with CPU affinity0–1,2-thread caps and120s command timeout. The tests use metadata-only fixtures and nonexistent image paths. Owner test counts and prior test counts are not added to this run.

The index and adapter SHA256 values are byte-identical to9c. The prior13 passed metadata boundary tests and independent actual current256/future64 parity therefore remain applicable. No second metadata pilot was run. That actual check verified256 current scenes/768 records and64 future scenes/957 records, complete ordinal and receipt parity, cold metadata identity restoration, and idempotence; its1.859279s excluding imports and627769344B peakRSS are metadata measurements only.

The prior `CHANGES_REQUESTED` report and both failure logs remain unchanged. This report closes only the contract-pin finding. The new full-future preparation proposal, optional provenance ledger, factory/trainer/PCA wiring, real indexed teacher/student execution, throughput and formal admission remain outside the reviewed scope. Full future metadata is still `WAITING_INPUTS`; the existing64KiB header guard safely rejects an oversized full legacy source-log header.

No GPU, teacher, raw image, tensor, large-weight read, full population scan, new training or scoring occurred in this follow-up.

Reproduction from the isolated follow-up checkout:

```bash
cd /mnt/project/VLA-Drive-st-correctness-metadata-pin-20261017
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_review_metadata_index.py \
  -k 'contract or lineage or actual_cli' -q
```

Exact artifact hashes and implementation hashes are in `REVIEW_v2.json`. Private equivalent: `/mnt/project/st-world-policy-20261016/audits/correctness/BOUNDED_METADATA_INDEX_REVIEW_v2.json`.
