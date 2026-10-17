# Independent review: bounded PLAN receipt retention

2026-10-17. **PASS for the exact source change and CPU behavior below.** This
review does not dispatch or approve a real build, change a live registration,
or qualify teacher values, training, throughput, or scientific performance.

Reviewed source: `838dd21f7a16a79086a44bc63be056d6a750570e`.
`tools/st_world_policy/metadata_storage.py` SHA256:
`46233cf232a61e73e7e24f967b1cf316298473b017f49f4e96432d7e1b1a3591`.
The only executable change is the maximum count of preexisting exact file pins,
from 8 to 16. Comments explain the bounded PLAN control records. No checksum,
path, coverage, whole-root accounting, package, or claim limit changes.

## Evidence

**19 passed in 0.52s**, including twelve independently authored cases and seven
focused existing cases. Full output is `TEST.log`; independent fixtures and the
executable-diff assertion are in `test_independent.py`.

The positive fixture reproduces the actual nested layout: `PLAN.json`, empty
`PLAN.lock`, and `control/{ADMITTED.json,CHILD.json,CHILD.log,CLI_RESULT.json,`
`CONTROL_WRITE.lock,COUNTERS.json,RESULT.json}`. The second lock is also empty.
Nine and sixteen files are admitted with complete exact pins and the explicit
`control` directory/file patterns. All bytes survive unchanged, the whole-root
claim remains 3 MiB, and repeat admission succeeds. Empty locks have the SHA256
of their actual empty bytes; they are not parsed as JSON or exempted from pins.

Negative cases reject seventeen files, a wrong hash, an omitted empty lock,
duplicate pins, path traversal, symlink, hardlink, an unregistered control
directory, and files larger than the unchanged whole-root claim. Each fails
before admission publication and leaves the ledger byte-identical. Increasing
the count ceiling alone does not authorize a control directory or new suffix.

Command (CPU affinity 0,1, no CUDA devices, thread limits 2, 120-second timeout):

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 timeout 120s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_metadata_storage_guard.py::test_bounded_plan_receipts_retained_without_expanding_byte_budget \
  tests/st_world/test_metadata_storage_guard.py::test_nine_plan_receipts_still_require_exact_complete_pins \
  tests/st_world/test_metadata_storage_guard.py::test_preexisting_empty_lock_has_exact_raw_pin \
  reports/st_world_policy/20261017/correctness/metadata_plan_receipts_review/test_independent.py
```

## Actual small metadata binding

Read-only stat confirmed the same nine-file layout and two zero-byte locks in
both existing `full_future_metadata_{dev,train}_v1` PLAN roots. The two PLANs
were read only to verify their SHA256 and seven implementation pins against the
new immutable runtimes. Every pin matched. Exact evidence is in
`RUNTIME_BINDING.json`.

| Role | Build runtime commit | PLAN SHA256 |
|---|---|---|
| dev | `1265b2376864ee929d95513f4f91e58d0910d147` | `58fb4439094bf488becd0e4b9a1944be23899c96d603213e06dbde7939082781` |
| train | `838dd21f7a16a79086a44bc63be056d6a750570e` | `bffbad1f6378aa9e2a766bc0cd9445a511af6459f0a3b43dc18f969fcdfa2ef0` |

Both runtimes contain the exact reviewed metadata guard bytes. Dev retains its
original future-preparation implementation; train retains the separately
reviewed explicit 192 MiB raw-log size admission. No PLAN was rewritten.
Lead's `FULL_METADATA_PLAN_ACTUAL.md` records successful planning; this review
did not repeat those planning operations or independently rerun their resource
measurement. No raw logs, images, model weights, teacher tensors, or GPU were
read or executed.

## Decision boundary

The count-limit patch correctly permits preservation of the nine existing PLAN
files **under an exact, separately admitted registration**. There is no source
correctness blocker in this change. Resource admission, actual registration
pins, and subsequent build supervision remain Lead/resource-review duties.
This PASS does not reserve additional bytes or authorize a full build itself.
