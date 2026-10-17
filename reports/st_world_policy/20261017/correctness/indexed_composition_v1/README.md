# Independent indexed composition review — admission defects

**FAIL_RECEIPT_VALIDATION. Do not execute the real qualification or admit its
descriptor until the narrow defects below have been corrected and independently
retested.** No implementation was changed by this reviewer.

Tested commit: `fd9cb5056c76be29b6bb3cbe83af9bab3b267239` in detached clean
runtime `/mnt/project/VLA-Drive-st-indexed-composition-review-runtime-fd9cb505`.
Original implementation report and 30 tests were read before the independent
checks. No production request/source DB, image, raw log, native checkpoint,
teacher weight, GPU, actual PLAN/build, ledger or dispatch was used.

## Tests and precise failure interpretation

- The implementation's **30 new tests independently passed in 29.29 seconds**.
- The corrected independent 12-case run has **5 failures and 7 passes**; exact
  duration and all assertion traces are retained in `independent_tests.log`.
- `independent_tests_initial.log` is preserved. Its apparent sixth failure was
  a reviewer test mistake: `sign(report)` rejected NaN before the consumer was
  called. Canonical hashing already rejects NaN. The corrected test verifies
  rejection by the actual consumer and passes. **NaN is not a demonstrated
  implementation vulnerability.**
- All runs used two CPU cores/threads and `CUDA_VISIBLE_DEVICES=''`, generated
  tiny synthetic metadata/tensors/images and a fake executor. They do not
  establish real cached/online DINO numerical parity or resource feasibility.

## F1: incomplete or contradictory all-element evidence is accepted

Location: `tools/st_world_policy/qualify_indexed_provider.py:325–340`,
`validate_receipt`.

The consumer checks the key/scene/slot lists and each comparison's
`bitwise_equal` / `max_abs`, but not its `elements` nor the reported aggregate
summaries. Starting with a successful synthetic actual-adapter result and
re-signing it:

1. Setting every comparison's `elements=0` is accepted.
2. Setting every comparison's `elements=1` is accepted.
3. Setting `legacy_reference_summary.bitwise_equal=false` and `max_abs=1`,
   while leaving the top-level success flag true, is accepted.

Repro: `test_resigned_incomplete_measurement_cannot_admit` parameters
`zero_elements`, `one_element`, `contradictory_summary`.
All fail with **DID NOT RAISE** in the real `validate_receipt` implementation.

This is a strict evidence-consumption defect. It does not claim that the actual
`qualify_layouts` routine compares only one element: the independent last-element
test confirms it detects a one-subnormal-value change in the final FP32 element.
Nonconstant per-key miss features also retain correct multi-scene/time/view
alignment. The problem is that the receipt gate accepts incomplete or internally
contradictory evidence claiming full qualification.

Required narrow correction: derive the expected full feature element count from
`prod(contract.output_grid_hw) * contract.channels` (442368 in the fixture/native
contract), require that count for every relevant comparison, verify finite,
nonnegative mathematical statistics and recompute/check the complete fixed and
legacy summaries from their bound row lists. Keep exact bitwise/no-tolerance
gates. The additional per-slot online comparisons should not contradict the
composed/reference result. No new target or numerical policy is needed.

## F2: malformed logical-input and compute counters are accepted

Location: `qualify_indexed_provider.py:354–361`, `validate_receipt`.

Starting from the same valid result and re-signing it:

1. `requested_images=-1` and `padding_image_slots=8*calls+1` still satisfy the
   sum check and are accepted, despite impossible measured compute accounting.
2. `process_rchar_bytes=true` is accepted because bool participates in numeric
   comparisons/arithmetic in Python.

Repro parameters: `negative_requested`, `bool_read`, both **DID NOT RAISE**.

Required narrow correction: exact nonnegative integer types for byte counts,
calls/image slots and per-layout counts; positive requested images for actual
compute; physically valid requested/padded ranges and consistent totals.
Validate measured time/RSS/GPU values with finite/range checks. Canonical NaN
rejection already exists and should remain intact. Do not weaken budgets or
change the qualification population to make these tests pass.

## Verified behavior retained

The reviewed runner creates an independent fresh fixed8 reference for all new
keys, compares regrouping, reads original cached values and compares those
separately, then exercises the real indexed provider. Cached values are not
replaced by the online reference to hide a legacy difference. A mutation of the
last tensor element is detected. Key-dependent, nonzero misses across shared
multi-scene keys preserve ordering and missing masks.

The old qualification remains only body/execution evidence. Owner tests confirm
that it cannot directly admit the new request population. Exact request/index/
source/teacher/provider pins are retained; a new receipt is SMOKE-only with
at most 32 scenes and the existing at-most-four-update configuration boundary.

Independent CPU guard tests reject exceeded 900-second, 8-GiB RSS, 16-GiB CUDA
allocated-peak and 64-GiB logical-input limits. The guard is cooperative between
bounded calls; it is not a hard preemptive limit or a measured production GPU
resource result. GPU reserved/device-wide usage is not the allocated-peak
metric; Lead still needs the owning resource supervisor and actual free-resource
admission. No physical I/O throughput claim follows from logical-read counters.

The real 32-scene layout must actually exercise both mixed immutable/compute and
partial-LRU/compute reads. Repeating a scene is not by itself proof of partial
hits; its key order and LRU capacity determine this. Existing tests use an
appropriate synthetic layout. This review did not read real indexes to claim the
new population satisfies that condition, and it is not an additional proven
defect in this report.

## Follow-up and reproduction

Owner should change only the strict receipt validation and add focused negative
tests. Preserve this failure evidence. Subsequent independent retest should cover
the five failures and adjacent valid receipt/layout paths; no full old-suite
rerun or real teacher task is needed to locate these issues.

Run the tests from the fixed runtime with the repository and `tests/st_world`
on `sys.path`, using `/root/miniconda3/envs/ddp/bin/python`, two CPU threads and
GPU visibility disabled. The implementation test target is
`tests/st_world/test_indexed_composition.py`; the independent target is the
absolute path to `test_composition_independent.py` in this report directory.
Full commands and source hashes are captured in `RESULT.json`.
