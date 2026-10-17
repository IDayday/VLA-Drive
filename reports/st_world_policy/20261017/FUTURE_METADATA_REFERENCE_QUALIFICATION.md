# Read-only reference qualification, maximum 32 scenes

Implementation parent: `8c9efe0073d3da3b9fd795c789e11002f27eb871`.
Lead reference: `7db26ae4`. The relevant original reference, future builder and
metadata index files have no byte differences from that Lead revision. This
change adds only a new comparator, synthetic tests and evidence; it does not
alter the builder, selection, guard, trainer or existing PLAN.

## Reference evidence and exact reuse

The reference is the original
`tools/st_world_policy/prepare_history.py::build_target_scene`, not the new
`prepare_future_metadata.TargetLogLookup`. The original function implements
same-log nearest timestamp selection at `[0, .5, 1, 2, 4]`, tolerance `.05`,
ambiguous-nearest rejection, current-frame availability, ordered
`CAM_F0/CAM_L0/CAM_R0`, and missing-future masks. It calls
`starVLA/dataloader/st_world_history.py::image_reference` for path/SHA/file
size/mtime and original width/height.

The adapter uses the **same immutable Python code objects and defaults**, with
private copies of their global dictionaries:

- `build_target_scene`: replace only its `image_reference` binding.
- `image_reference`: replace only `file_sha256` with a budgeted SHA reader and
  `Image.open` with a context manager over a budgeted header stream.
- No module globals change. No copied/reimplemented timestamp selector is used.
- The SHA still covers all image bytes. Header reads are capped at one MiB per
  image-reference call, including repeated reads/seeks. A larger header fails;
  it is not replaced with a guessed dimension. No decode/resize/load is called.
- Repeated image occurrences are intentionally rehashed by the original
  reference, so this comparator remains independent of the new per-log memo.

The source/function-text/AST hashes and exact override list are saved in
`evidence/future_metadata_reference_v1/REFERENCE_SOURCE_EVIDENCE.json`.
Tests check code-object identity and unchanged original globals, plus equality
with an unadapted original call and the same ambiguous-timestamp exception.

## What is compared

For every scene in the fixed original order, every one of five times and all
three views is compared recursively and type-sensitively, including missing
entries, frame tokens/timestamps, image paths, SHA256, bytes, mtime and width/
height. The full original raw-log ledger is reconstructed from actual raw SHA,
size/mtime, selected token order and per-log coverage and compared. Header
source/partition/registration/selection identities are checked as well.

Only one raw log is loaded at a time; at most 32 small scene records are retained.
The comparator verifies exact PLAN/registration/COMPLETE pins, existing shard
coverage/hashes, original source population/current records and opaque ego-byte
SHA. It does not decode ego/GT tensors or invoke a teacher. The successful output
is `REFERENCE_MATCH`; differences are `REFERENCE_MISMATCH`, not a tolerated
numeric deviation. A malformed/missing input is `REFERENCE_FAILED`. Failure
returns exit code 2. None of these states grants training admission or a
scientific result.

## One combined budget, not two full budgets

The final registered preparation PLAN must declare at most:

- Build: **4 GiB source/control reads, 450 seconds, one worker**, 32 MiB metadata.
- Reference: **4 GiB source/control reads, 450 seconds**, two allowed CPU cores,
  four GiB RSS, at most 32 scenes, one log resident, 128 MiB maximum raw log.
- Combined: **8 GiB and 900 seconds**, with the outer finite supervisor counting
  startup/shutdown and every invocation. No GPU or teacher tensor storage.

The comparator rejects the old v4 8 GiB/900-second build PLAN. It also rejects a
completed build whose recorded invocation cost already exceeds its half. All
reference reads, including control/code SHA verification, full source-control
checks, raw-log hash+deserialization, repeated image hashes and PIL header IO,
are prepaid/counted by the existing budgeted readers. Noninterruptible native
operations still require the outer wall/RSS supervisor and fail closed on return.

**Before the first raw/image read**, the comparator derives a conservative
remaining-read estimate from verified metadata and file stat:

```text
2 × sum(registered raw-log bytes)
+ sum(image bytes for every valid time/view occurrence)
+ valid image occurrences × 1 MiB header cap
+ source-control reads (index/partition/source pins, with read multiplicities)
+ selected current JSON and opaque ego bytes
+ 8 × generated control/shard metadata bytes
```

It refuses to proceed when this plus already counted reads exceeds four GiB.
The estimate assumes unchanged pinned sources; actual counted reads remain the
hard bound, and changed source provenance cannot pass comparison. No real cost
estimate or throughput measurement is claimed before the actual final inputs
exist. The result reports both per-arm and combined observed costs.

COMPLETE records only its successful build invocation. It cannot retrospectively
prove the cost of earlier failed attempts. Therefore output explicitly says
`prior_failed_invocations_accounted=false` and requires the outer cumulative
invocation ledger. Retrying either arm must consume the remaining shared
allowance or require a separately approved qualification; it cannot reset the
900-second/eight-GiB qualification budget silently.

## Prepared commands; not executed on real data

Lead must first fix and independently review the discovered real `python -m`
storage-guard module-state issue, integrate final source and create one final
same-32-scene/two-log PLAN. Do not reuse or rewrite v4. Final registration must
include the exact log/result paths and all their bytes under the same 32 MiB
qualification control allowance. The comparator adds no writer kind or ledger.

The build uses the existing registered guard CLI with an **outer 450-second**
wall bound and the final PLAN's four-GiB source bound. Then, with actual pinned
values from that invocation (environment names below are inputs, not invented
paths), the reference command is:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout --signal=TERM 450s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.compare_future_metadata \
  --plan "$ST_FINAL_PLAN" --plan-sha256 "$ST_FINAL_PLAN_SHA256" \
  --registration "$ST_PREP_REGISTRATION" \
  --registration-sha256 "$ST_PREP_REGISTRATION_SHA256" \
  --complete-sha256 "$ST_COMPLETE_SHA256"
```

Output is stdout only, bounded to one MiB. Lead captures it in the registered
qualification invocation; the tool does not create an output directory, write
inside the cache root, or generate a second TEACHER_REQUESTS/cache. `--help`
output is saved in `evidence/future_metadata_reference_v1/cli_help.log`.

## Actual tests and remaining boundary

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  timeout 120s taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_future_metadata_reference.py -q
```

**12 passed, 8.81s**, one existing pynvml deprecation warning. Full log:
`evidence/future_metadata_reference_v1/focused_tests.log`.

Tests cover all fields/ledger, missing future, ambiguous timestamps, changed
image SHA/dimensions/mtime/size, wrong pins, stale raw provenance, forbidden
double-full budgets, header limits, no new files and a real `python -m`
subprocess invocation on tiny fixtures. The new selector is deliberately
disabled in a reference test to ensure it is not reused as its own oracle.

No real raw log/image, real metadata build, new PLAN or GPU was run. No actual
32-scene equality, production IO rate or formal approval is claimed yet.
