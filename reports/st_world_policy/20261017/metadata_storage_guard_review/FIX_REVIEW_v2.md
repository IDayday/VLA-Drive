# Independent metadata guard fix review

Reviewed 2026-10-17 05:30 UTC. **PASS for the bounded source/CPU scope.**
No remaining blocking finding was reproduced in the three requested fixes.
This supersedes the three findings' open status in `REVIEW.md`; the original
counterexamples and failing logs remain unchanged as historical evidence.

## Version and independence

- Original implementation: `6e20c8b6d9b883a21653b1f79dd39c132ca9bf2f`.
- Original independent counterexamples: `2e2757723a635776d3ee418d491a0f1c9bc2a22d`.
- Fix under review: `2ef904200095e5d0b3534342f297794e331aede4`.
- Independently applied fix: `fcd6b3ee` in isolated review worktree
  `/mnt/project/VLA-Drive-st-metadata-guard-review-20261017`.
- Reviewed `metadata_storage.py` SHA256:
  `698b5425dbb6af8aa534682ac14a1bca13253e819e310b28e1bda08a0b77bae1`.
- Reviewer changed only independent tests/reports; no production implementation.

## Findings closed

| Finding | Previous failure | Current evidence |
| --- | --- | --- |
| MG-F01 | Pending migration bypassed receipt expiry | Original counterexample passes. Fresh quiescence is required inside the same global lock, before state publication and before completion. Explicit renewal preserves the original migration ID, pending bytes and snapshot. |
| MG-F02 | Nonexistent parent directories bypassed prewrite rules | Original counterexample passes. Every planned ancestor and maximum depth is checked before creation. Existing parent path/filesystem checks remain. |
| MG-F03 | Unwired CLI could publish COMPLETE before hook-count rejection | Original counterexample passes. Exact registered entry/dependency pins and explicit callback capability are checked before parser, writer admission and builder. Final actual hook-count checks remain. |

## Additional independent boundaries

Four additional test cases were added to `tests/st_world/test_review_metadata_guard.py`:

1. Quiescence expires after snapshot: no state or COMPLETE publication; explicitly
   renewed receipt resumes the identical pending migration.
2. Quiescence expires after state: state remains recoverable, no false COMPLETE;
   renewed receipt completes the same migration without rewriting pending bytes.
3. Process evidence changes after a renewed receipt is pinned: checksum failure,
   unchanged ledger and no completion.
4. A registered dependency changes while the entry is being imported: second pin
   validation rejects before parser, root creation or new ledger charge.

The owner tests independently rerun here additionally cover all four interruption
windows, conflicting renewal identity, active writers in renewed evidence,
implicit replacement of the original receipt, planned nested depth, missing pins,
missing capability, changed source, missing dependency and wrong entry path.
Existing accounting/concurrency tests cover one shared ledger lock, preservation
of original claims and reservation, package cap, partial precharge, free floor,
checkpoint commitments, deadline/RSS and sticky callback errors.

## Commands and actual results

```bash
cd /mnt/project/VLA-Drive-st-metadata-guard-review-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  timeout 180s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_cache_storage_budget.py \
  tests/st_world/test_metadata_storage_guard.py \
  tests/st_world/test_review_metadata_guard.py
```

Python 3.10.20. First rerun before adding independent boundaries: **93 passed in
5.82s** (`INDEPENDENT_FIX_RETEST.log`). Final suite: **97 passed in 5.88s**
(`INDEPENDENT_FIX_WITH_BOUNDARIES.log`). One existing pynvml deprecation warning;
no GPU execution. `git diff --check` passed.

All policy/ledger/cache artifacts created by tests are temporary fixture data.
No real campaign ledger migration, source manifest scan, teacher tensor/image
open, GPU work or cache build was performed by this review.

## Exact remaining boundary

This is an implementation review, separate from the earlier resource proposal
review in `metadata_resource_review/`. It does **not** approve real migration or
current/future full builds. Source-hook owner integration is not included in
`2ef90420` and must be reviewed separately. Markers alone do not prove all write,
hash and read loops actually call the hook with the required bounded interval.

Before a real invocation, Lead still needs fresh original-writer/queue quiescence,
exact pilot inventory, pinned policy/registration/checkpoint commitments, actual
capacity and a bounded process monitor. Use a fresh standalone interpreter from
the registered immutable source tree; this review does not establish hot reload
or arbitrarily preloaded Python dependency equivalence. The ≤4GiB aggregate
metadata package includes pilots and partials. The previously proposed current
index ≤1GiB root and future ≤256 qualification limits are unchanged. No formal
temporal training qualification, GPU dispatch or full future preparation is granted.
