# Independent legacy future-reference comparator review

**PASS_COMPARATOR_CODE_LIMITED** on Lead integration
`8ffeeda6814732f9416d5a7f1afd8fa8d72591d8`, integrating implementation `60afeba4`.
Independent reviewer: `/root/st_future_metadata_correctness`, 2026-10-17 UTC.

Reviewed file: `tools/st_world_policy/compare_future_metadata.py`.
Actual SHA256:
`3ddc07679451d6843ff3d0f75790564127f20d37f2ea83fff64ccc84a70f6949`.

## Independence and comparison coverage

The reference uses the original `prepare_history.build_target_scene` code object,
defaults and closure, with a private copy of its global dictionary. Only the
image-reference binding changes. That binding also reuses the original
`st_world_history.image_reference` code object; only full-file SHA reading and
PIL header opening are replaced by budgeted IO. Module globals, nearest-frame
selection, tie handling, tolerance, missing-view handling and camera order are
not rewritten. The new `TargetLogLookup.scene` can be disabled and the comparator
still succeeds on the fixture. It is not comparing the new selector with itself.

Every scene/frame/view leaf is compared recursively and type-sensitively, with
extra/missing dictionary keys and list-length changes included. The comparator
also reconstructs raw-log SHA, size/mtime, ordered token coverage and complete
ledger identity from the reference inputs. Header source/partition/registration
bindings are compared. The ledger-reference object is first strictly validated
against its actual body/hash/coverage, rather than trusting its strings.

Independent mutations of requested time, frame token, frame timestamp, missing
view, extra field and integer-to-float conversion all return mismatch after
updating outer artifact hashes. The owner tests independently rerun here also
cover image SHA, dimensions, size, mtime, changed raw bytes and ambiguous ties.
No GT tensor is decoded: opaque ego bytes are hashed only. Monkeypatched tensor
load, image decode, write helpers and the new selector are not called.

## Budget and read-only boundary

The comparator requires a qualification PLAN with <=32 scenes, one worker,
build limits <=4 GiB reads/450 seconds and <=32 MiB metadata. The reused
registration verifier validates all existing hard limits, including <=64 MiB/s,
<=4 GiB RSS and <=128 MiB raw logs, as well as source hashes and target policy.
The completed build's recorded bytes and time must also fit its half.

The reference gets a separate <=4 GiB/450-second budget. Raw hashing,
deserialization, repeated image hashing and bounded headers use prepaid readers;
source-control and reused metadata validators are explicitly prepaid. A
metadata/stat-only conservative remaining-read estimate is checked before raw
or image reads. Header reads are limited cumulatively to one MiB per original
image-reference call. Reference images are hashed in full without pixel decode.

The result reports both halves and their observed sum; a valid pair is bounded
by 8 GiB/900 seconds. **This alone does not count interpreter startup, failed
attempts or previous retries.** The tool explicitly returns
`prior_failed_invocations_accounted=false` and
`outer_cumulative_invocation_ledger_required=true`. Lead's outer finite
supervisor/receipt must account all invocations and remaining joint allowances;
retries cannot silently receive fresh halves. Blocked native operations also
require the outer wall/RSS monitor. These are retained explicit boundaries, not
new resource approvals from this review.

The comparator does not write a cache, PLAN, ledger, receipt or result file. Its
main entry emits <=1 MiB JSON to stdout and exits 0 for MATCH or 2 for
MISMATCH/FAILED. The synthetic subtree is byte-identical before and after the
comparison; real `python -m` execution is included in the independently rerun
tests. Lead must capture stdout within the existing registered control allowance.

## Actual tests

Fixed clean detached runtime:
`/mnt/project/VLA-Drive-st-reference-comparator-review-runtime-8ffeeda6`.
Exact HEAD, clean status and imported source path/hash were asserted.

| Execution | Result | Full evidence |
|---|---|---|
| Implementation's 12 tests, executed independently | **12 passed, 5.32 s** | `owner_tests_independent.log` |
| New independent field/budget/read-only tests | **10 passed, 5.11 s** | `independent_tests.log` |

No skip/xfail. Only the existing pynvml warning. CPU affinity 0,1; two
OMP/MKL/OpenBLAS threads; CUDA hidden; bytecode disabled;
`/root/miniconda3/envs/ddp/bin/python` (Python 3.10).

Commands, from the fixed runtime with its directory first in PYTHONPATH:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_future_metadata_reference.py -q

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  --import-mode=importlib \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/reports/st_world_policy/20261017/correctness/future_reference_comparator_v1/test_reference_boundaries.py -q
```

Recorded runs used an equivalent pytest harness with the identity assertions.
All source logs/images were tiny generated fixtures. **No actual raw log/image,
real 32-scene PLAN/build/comparison, teacher/tensor/GPU or campaign ledger access
was executed.** The real qualification remains Lead-owned and untested by this
report. No all-population preparation, training or storage admission is granted.
