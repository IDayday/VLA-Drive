# Indexed provider/PCA narrow independent retest

**PASS_CODE_CPU_FIXTURES_ONLY.** Both independently reproduced v1 defects are
closed on the fixed source below. This does not approve a real provider factory
qualification, a larger population, formal training, or long-window throughput.

## Identities and prior evidence

- Tested implementation: `ac6b93a43de6d41585a1c01e3370f5a6db41f456`.
- Detached runtime: `/mnt/project/VLA-Drive-st-indexed-provider-review-runtime-ac6b93a4`.
- Prior implementation: `5e0e2ec3ff236b73d3cf2f850f1e6f17dc71967d`.
- Original independent failure evidence commit:
  `697d6b19ab1f759b9c5dc8ff941697930fdb3440`.
- [Original v1 evidence](../indexed_provider_factory_v1/README.md) and its
  `8 passed / 2 failed` log remain unchanged. The implementer's original 28
  tests passed independently on that old implementation; they did not cover
  the two counterexamples and are not substituted for this review.

Reviewed file SHA256:

| File | SHA256 |
| --- | --- |
| `tools/st_world_policy/teacher_provider.py` | `e9bde7c61e1e6f5e01d75195d3888722bfb6f4d0a32641240275fa3a5ff6c512` |
| `tools/st_world_policy/provider_pca.py` | `06b80899a4e5559bf3ca01571d7dae33ccae2f5c9be7cbb86c54db8929ddb223` |

The teacher factory file is byte-identical to the initially reviewed version.
The PCA changes are limited to schema separation and reader-budget admission.
Source hashes and runtime cleanliness were asserted before and after tests.
The reviewer changed only tests and reports, never implementation.

## Actual results

On 2026-10-17 at 06:17 UTC: **19 passed, 0 failed, 0 skipped, 17.55 seconds**.
Python 3.10.20, pytest 8.4.2, CPU affinity `[0,1]`, two CPU threads,
`CUDA_VISIBLE_DEVICES=''`. One existing pynvml deprecation warning.
The complete output is [retest.log](retest.log).

- Original 10 independent cases: all pass, including both formerly failing
  cases, fixed-execution/RNG gates, executor-failure cleanup, and exact raw
  request-byte identity.
- Five additional boundary cases: registration and runtime budget admission,
  exact-limit acceptance, and early legacy/indexed schema rejection.
- Four adjacent checks: valid legacy scope, unchanged indexed PCA basis and
  channel provenance, raw synthetic output/mask/key/RNG equivalence, and actual
  `python -m tools.st_world_policy.teacher_provider` CLI success and rejection.

These fixtures generate tiny synthetic metadata indexes, generated toy image
files, synthetic basis/features and fake executors in test temporary directories.
No production metadata DB, raw log, real image, teacher weight, teacher execution,
GPU, PLAN, build, qualification or training runtime was accessed or launched.

## Defect closure

### F1: legacy scope rebinding to an indexed provider

The old `validate_provider_scope` legacy branch admitted a correctly re-signed
legacy scope whose provider had been changed to the indexed provider. That
bypassed the explicit fitted-index scope and could create redundant readers.
It did not demonstrate that different PCA basis weights were applied.

Fixed source `provider_pca.py:69–73` rejects unknown schemas and requires both
the actual provider and its pinned descriptor to be legacy before entering the
legacy branch. New sentinel tests separately cover indexed target and indexed
descriptor cases. They fail if either the population-sized fit metadata is read
or a provider preview is created before rejection. Both now pass. Valid old-old
and indexed-indexed paths also pass, with identical original basis bytes and
transform outputs.

### F2: combined reader budget checked too late

Old registration opened a 16 MiB fit reader plus 128 MiB request and 128 MiB
source readers before rejecting their declared 272 MiB allowance. The v1
counterexample tracked concurrently open declared allowances, **not measured
RSS**. The analogous old runtime ordering was at `provider_pca.py:125–128`.

Fixed source adds `_check_indexed_preview_budget` at `:15–32`. Registration
calls it at `:101`, before opening any index at `:102`, and retains the check
against the constructed provider at `:112`. Runtime validation calls it at
`:150`, before opening the fit reader at `:151`.

New runtime tests show that a 240 MiB fit binding with an existing 32 MiB
provider is rejected with **zero new reader constructions**, while the existing
provider's readers remain open and usable. A 224 MiB fit binding plus 32 MiB
provider is accepted at exactly 256 MiB, closes only its temporary fit reader,
and preserves existing ownership. Registration similarly accepts exactly
120 + 120 + 16 = 256 MiB and closes all three previews. The original over-budget
registration test now passes and leaves no scope artifact.

## Preserved boundaries and remaining qualification

The narrow fixes do not alter teacher execution, target selection, numerical
precision, metadata source identities, PCA fitting, basis weights, training
recipe or existing qualified request keys. The earlier audit of exact 64-scene /
4-step smoke limits, original qualified source/runtime, fixed batch 8, RNG,
full-source SHA pins and compact-header provenance is retained. The selected
regressions verify the affected ownership and legacy boundaries without
needlessly rerunning the whole 103-test implementation suite.

Dataset-side eager metadata work remains acknowledged. A passing tiny factory
fixture does not establish full-population memory, end-to-end data admission or
training throughput. Any actual bounded factory/PCA qualification must be
separately registered by Lead with current assets, resource checks and budgets.
There is no real-data or resource PASS in this report.

## Reproduction

Run from the fixed detached runtime, with this report checkout available:

```bash
cd /mnt/project/VLA-Drive-st-indexed-provider-review-runtime-ac6b93a4
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/reports/st_world_policy/20261017/correctness/indexed_provider_factory_v2/run_review.py
```

The harness asserts the exact implementation commit, clean runtime and both
source-file hashes, and lists all 19 tests in the captured output.
