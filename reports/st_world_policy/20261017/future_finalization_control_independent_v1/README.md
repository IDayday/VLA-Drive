# Independent finalizer control wiring review

**PASS_NARROW_CONTROL_WIRING_ONLY**, for exact owner commit `7d698b6d6f04661833f03f73ce4f0e92f23820cb` and the pins in `REVIEW.json`. Reviewer: `/root/st_calibration_independent_correctness_v2`. This is a correctness review, not resource admission or actual-run success.

The actual proposed `FINALIZE.lock` allowlist matches the frozen writer. The pinned worker command dispatches `finalize-existing` through the reviewed guarded runner in immutable runtime `3df052ea422c4ba042ef3be33dea19438c5702dc`. Finalizer, metadata writer, and provider bytes match the earlier independently reviewed implementation. Original source `FAILED_NO_AUTO_RETRY` lineage is pinned and remains a failure. Policy comparison preserves all 11 old claims and adds the exact 2 GiB finalization claim; the separate 1 MiB terminal directory remains explicitly admitted. No live ledger was read or modified.

Ten owner control tests were actually rerun, plus 13 independent actual-draft, binding and rejection cases. First execution: **21 pass, 2 errors**. Both errors were caused by this review harness placing a SIGALRM in the same process: the implementation correctly refuses to overwrite an unrelated existing caller alarm when publishing failure evidence. The first runner, logs and result are preserved. Only those two cases were rerun under an external 20-second watchdog, without an in-process caller alarm: **2 pass**. No implementation change, guard relaxation, or rerun of the previously reviewed 40 finalizer cases occurred. Production main establishes its own marked deadline handler, which the bounded terminal-publication path explicitly permits.

Independent tests cover actual draft hash/command/source pins, fixed failed lineage and original 1176-shard metadata declaration, exact runtime inheritance, old-claim retention, FINALIZE.lock and publication allowlist, changed command/worker/row limit/resource limits/source dependency rejection, checksum corruption, short CP lease rejection, no retry on busy ledger, monitor failure retaining UNKNOWN cleanup, artifact mismatch, and single-count terminal accounting. Approved-shape admissions existed only in memory inside tests; they were never written as actual authorization.

Measured test-process total: 0.2203204706311226 seconds, maximum RSS 29,728,768 bytes, total logical reads 3,970,540 bytes (including 1,777,814 startup bytes), test-window logical writes 20,163 bytes. Both processes used CPU 6/7. No Torch import, CUDA, real source payload, images, weights, teacher files, or actual ledger access. Additional manual review only read the small source/control files; no broad scan.

## Boundaries

- Actual finalization has not been executed by this reviewer. Real output/order/coverage/cost and actual terminal cleanup must be checked after the separately admitted invocation.
- Live migration, CP lease, free-space, process quiescence and launch authority belong to the separate resource reviewer and Lead; this report does not attest their current state.
- `teacher_provider` already has the independently tested strict new receipt branch and is byte-identical here. The **new TRAIN index-preparation branch is still deferred**: this patch contains only `INDEX_PREPARATION_ADAPTATION.txt`, not that implementation. The old helper cannot yet be claimed to consume the new finalization receipt. A later narrow patch and review are required before index preparation.
- Successful metadata finalization will not retroactively make the original interrupted build or prior observer failures successful.

## Reproduction

The original runner with its unrelated in-process alarm is retained only to reproduce the harness conflict. The successful targeted rerun is:

```bash
cd /mnt/project/VLA-Drive-st-future-control-correctness-20261017
timeout --signal=TERM --kill-after=5s 20s taskset -c 6,7 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/miniconda3/envs/ddp/bin/python reports/st_world_policy/20261017/future_finalization_control_independent_v1/run_retry.py
```

This command runs only two synthetic control cases, not a finalizer or any real workload. Full first-run and targeted-rerun logs, costs and exact pins are committed beside this report.
