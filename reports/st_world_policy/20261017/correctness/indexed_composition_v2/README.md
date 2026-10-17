# Indexed composition consumer: independent narrow retest

**PASS_CODE_CPU_FIXTURES_ONLY.** The five original receipt-consumption failures
are closed. From a correctness-code perspective, Lead may proceed with the
original bounded **32-scene** real composition qualification, after the existing
plan/source-pin, metadata allocation and resource admission checks. This is
permission to test the finite composition, not evidence that real cached/online
teacher values are equivalent or that training is qualified.

## Exact source and retained evidence

- Tested fix: `42854f40d0dab551463d63abae80a1045b64722f`.
- Detached clean runtime:
  `/mnt/project/VLA-Drive-st-indexed-composition-review-runtime-42854f40`.
- Consumer SHA256:
  `d072f46ac3b1da4fffb42dec2439f35382e793b4a9de1cd4317eb60872a46bbb`.
- `teacher_provider.py` SHA256:
  `83410555d7e879407b09039f6a094fa6daac1259d0df7379cd6a220c2954b6bf`.
- The nine relevant source files match Lead integration
  `31c3dca2fbc697b419bf343ce05d10be0a4fe13f` byte-for-byte.
- Original FAIL commit remains intact:
  `21cac2d3ded5f046fd532b550bbd25109e687d9e`.

[CODE_SCOPE.json](CODE_SCOPE.json) records the source comparison. AST comparison
also verifies that the original plan, layout, run, resource guard and remaining
pre-existing functions are unchanged; only receipt validation and three small
validation helpers changed. The reviewer modified tests/reports only.

## Actual independent result and review budget

**18 passed, 0 failed, 0 skipped, 22.40 seconds**; one existing pynvml warning.
The bounded child completed in **24.23 seconds**, with observed `VmHWM`
**848,433,152 bytes**, below the 2 GiB review ceiling. CPU affinity `[0,1]`,
two CPU math threads, `CUDA_VISIBLE_DEVICES=''`. The harness monitored this
owned child against 2 GiB / 600 seconds and verified fixed hashes/clean source.

The tests comprise:

- All original 12 independent cases, including the five previous failures.
- Three new checks: self-consistent extra padded batch counts, impossible cold
  hit/miss reassignment, and boolean tensor-disk bytes.
- One focused owner test containing 24 adjacent invalid mutations.
- Two valid end-to-end synthetic layout/receipt/factory cases, including
  multiple scenes, shared keys, reorder and distinct missing-view masks.

The old 99-test suite was not redundantly rerun. Full commands, resource
observation and output are in [RUN.json](RUN.json),
[run_bounded_review.py](run_bounded_review.py) and [retest.log](retest.log).
The synthetic fixtures use tiny generated metadata, image files and tensors
with a fake executor. They do not read production images, raw logs, teacher
weights/tensors, native checkpoints or production indexes.

## Closure of F1: full comparison evidence and summaries

`_full_difference` now derives the required element count from the contract's
grid and channel dimensions and checks every comparison, including per-slot
online comparisons. Zero/one-element claims are rejected. The exact statistic
schema, finite nonnegative scalars and zero-error consistency are enforced.

`_checked_summary` reconstructs the fixed and legacy summaries from their actual
bound comparison lists. A false legacy summary with a true top-level success
flag is rejected; truncated counts, boolean statistics and contradictory
squared error also fail. Key order, scene coverage, slot coverage, sampler
digest shape and reference digest format are checked without weakening the
bitwise threshold.

The valid tests still compare all elements: a change in only the final FP32
element is detected. Nonconstant, key-specific miss features preserve the
correct multi-scene/time/view layout. Existing cached values remain separately
compared against fresh fixed8 outputs rather than replaced by the reference.

## Closure of F2: compute and resource accounting

Exact integer checks now reject negative requested images and bool-as-byte
values. Cold/restored counters must match the actual source hit/miss key sets;
per-scene unique-key coverage and accumulated provider totals must agree.

Expected calls and requested images are reconstructed from both independent
reference passes plus each provider layout's computed misses. A fabricated
additional fully padded batch is rejected even if calls, image slots, padding
and RNG-check totals are modified to remain arithmetically consistent.

RSS/GPU allocated bytes, logical read/mmap bytes, tensor-disk bytes and provider
resident accounting use strict nonnegative integers. Elapsed time remains
finite and bounded. NaN was already rejected by canonical hashing in the original
version; it is not presented as a newly fixed defect. The original failure log
and corrected-test explanation remain available in the v1 report.

## Finite real qualification boundary

Code review now permits the unchanged finite request population/source pair to
be tested under the registered limits: at most **32 scenes**, fixed8 FP32,
**900 seconds**, **8 GiB process RSS**, **16 GiB CUDA allocated peak**,
**64 GiB logical input**, **32 MiB metadata**, zero teacher-tensor disk output.
The final PLAN must bind the actual integrated source hashes; an old PLAN with
different code pins may not be silently reused. No new population or widened
budget is approved here.

The guard remains cooperative between bounded calls, with Lead's outer owning
supervisor responsible for deadline/resource enforcement. The allocated-GPU
limit is not a claim about reserved or device-wide memory. Current free resources
and the metadata allocation/ledger are outside this review and were not changed.

Owner's separately authorized metadata simulation reports 7 partial-LRU+compute
and 111 immutable-cache+compute reads for the exact real 32 scenes. This review
read that committed report but did not repeat the real index scan or claim its
own numerical evidence. The actual teacher run must still reproduce those
signals and compare complete tensors and masks. A real numerical difference
must remain a failed qualification; no tolerance may be introduced afterwards.

Formal/full-population training, real numerical equivalence, long-window
throughput, PCA/student smoke and any dispatch remain **not approved or measured
by this CPU review**. The separate existing at-most-four-update SMOKE boundary
is retained. No ledger, live job, checkpoint or cache identity was modified.
