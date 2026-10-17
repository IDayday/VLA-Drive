# Independent acceptance of the actual 32-scene composition receipt

Outcome: **PASS for the exact bounded 32-scene indexed composition qualification**.
This is a read-only acceptance of newly produced real execution evidence, using
the actual production receipt consumer. It is not another teacher run, a new
population qualification, a trainer test, a throughput result or scientific gain.

## Code and evidence identity

- Runtime: `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`, detached worktree
  `/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017`, clean before
  and after review.
- Previously independently tested consumer fix:
  `42854f40d0dab551463d63abae80a1045b64722f`.
- Consumer source SHA256:
  `d072f46ac3b1da4fffb42dec2439f35382e793b4a9de1cd4317eb60872a46bbb`.
- `teacher_provider.py` SHA256:
  `83410555d7e879407b09039f6a094fa6daac1259d0df7379cd6a220c2954b6bf`.
- All nine actual PLAN source pins match the active runtime. These are the same
  nine implementations independently checked in `indexed_composition_v2`.
- Original plan and resolved run plan are equal, both SHA256
  `f96e5612c068aa4000172f7076ea37031b8fe09d03b2f60706d57300e540f0eb`.
- Real `RESULT.json` SHA256:
  `a4f4c86f0b5c09d18a58d02e8919dd73e659b48a53ed0d2a25c7d68e497b7a2f`.
- PLAN identity:
  `df38f4997f888c868a7de3991768be065646907d3d30480a720147131683688a`.
- Receipt identity:
  `ab47303d09323f10f1fcf5b4a13a32ab857bed7856a1a4abec4555a09233372d`.

Evidence is under
`/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/`.
The reviewer did not alter that directory, its PLAN directory, any cache/index,
resource reservation, ledger, source implementation or runtime artifact.

## Actual consumer, without replaying data

`review_receipt.py` imports the fixed runtime's
`tools.st_world_policy.qualify_indexed_provider.validate_receipt` and calls it
with the real parsed plan/result. Its descriptor argument is an in-memory view
of the exact plan dependencies required by this function, not a newly admitted
or published provider descriptor. No mock replaces validation.

The call verifies the new request/source/teacher/provider identity against that
view, all source pins, the independently qualified prior fixed-8 body and its
original population evidence, both signed result identities, full numerical
statistics, scene/key coverage, summaries, layout counters, actual fixed-8
requests/padding, resource bounds, sampler/RNG and frozen-state records.

It does **not** rebuild a plan, construct a provider, open SQLite, load a teacher,
read image/tensor/weight payloads or dispatch work. An audit hook rejects project
writes, non-metadata extensions and files larger than 32 MiB. The opened small
metadata/source file inventory is in this directory's `RESULT.json`. Prior
fixed-8 metadata is read to verify that body evidence, not used as a substitute
for the new population's comparisons. CUDA remains uninitialized.

## Complete numerical coverage

| Check | Actual receipt evidence |
|---|---:|
| Exact scenes / slots / missing slots | 32 / 480 / 0 |
| Unique image keys | 294 |
| Existing-cache keys / computed-miss keys | 189 / 105 |
| Full feature elements per comparison | 442,368 |
| Fixed-policy comparisons | 2,214 = 294 regrouped references + 4 × 480 scene slots |
| Fixed-policy compared elements | 979,402,752 |
| Legacy cached-value versus fresh reference comparisons | 189 |
| Legacy compared elements | 83,607,552 |
| Additional per-layout online comparisons | 1,920 |
| All checked difference records / element comparisons | 4,323 / 1,912,356,864 |

Every record contains the expected full tensor element count and complete
finite statistics. Both summaries recompute exactly; all recorded differences
are bitwise equal with zero maximum absolute difference. The final total counts
repeated layout/reference checks, not that many unique features or scenes.

All four layouts contain 32 scenes and 480 slot comparisons: cold, partial LRU,
cold restored and reversed scene order. The partial-LRU layout actually has
**7** scene calls combining RAM hits and computation. Immutable cache plus
computation occurs in **111** calls across layouts (28 + 27 + 28 + 28), rather
than merely being asserted by a Boolean. There are 29 total RAM hits, 1,344
immutable hits and 547 computed images across the provider layouts.

Masks, sampled identities across layouts, construction RNG, 234 forward RNG
checks and frozen initial/final versions all pass. The exact real population
has no missing-view slot; absent-view behavior remains covered by earlier CPU
negative/behavior tests, not by an invented missing view in this real result.

## Measured finite execution and storage bounds

| Quantity | Actual | Bound |
|---|---:|---:|
| Elapsed seconds | 170.356904849 | 900 |
| Fixed-8 calls | 234 | 367 |
| Compute slots | 1,872 | 2,936 |
| Requested images / padding | 1,135 / 737 | Exact accounting: 1,135 + 737 = 1,872 |
| RSS peak, bytes | 3,158,171,648 | 8,589,934,592 |
| CUDA allocated peak, bytes | 1,416,995,328 | 17,179,869,184 |
| CUDA reserved peak, bytes | 1,533,018,112 | Reported separately, not substituted for allocated |
| Process rchar + explicit tensor mmap logical bytes | 18,935,222,456 | 68,719,476,736 |
| New teacher tensor disk bytes | 0 | 0 |
| Provider payload resident peak, bytes | 14,159,232 | 14,159,232 |

Calls derive independently as 37 original-order reference + 74 regrouped
reference + 123 provider calls. These numbers include padding and repeats and
are not a long-run throughput estimate. RSS includes more than the provider's
small payload LRU, whose limit explicitly excludes teacher/activation/metadata
and caller storage.

The administrative report reservation has SHA256
`1ab0e8a8673b354ec81d99311444c68a502ac8d4d0f86e66e103373f90c0744e`.
It binds this runtime and PLAN and reserves 32 MiB for each of the plan/run
roots, total 64 MiB, within the existing 4 GiB metadata budget. This is **not a
cache-ledger claim** and does not change that ledger.

The read-only root census includes directories, files, progress/log files and
allocated blocks; it rejects symlinks/nonregular files/hardlinked regular files:

- Plan root: 225,280 allocated bytes / 217,119 apparent bytes.
- Run root: 2,199,552 allocated bytes / 2,192,792 apparent bytes.
- Combined: **2,424,832 allocated bytes**, below 64 MiB and each individual cap.

The reservation's historical `RESERVED_NOT_EXECUTED` and the last periodic
`PROGRESS.json` value `RUNNING` are preserved. Neither overrides the terminal,
signed `RESULT.json` status `INDEXED_COMPOSITION_BITWISE_QUALIFIED`; neither is
evidence that a job remains active. No process-state inference is made here.

## Review execution and limits

The independent consumer invocation exited 0 using CPU affinity 0,1, hidden
CUDA, the existing `ddp` Python environment and an external 180-second timeout.
Review peak RSS was **631,558,144 bytes**, below 2 GiB; receipt checking after
imports took 0.219103 seconds. `review.log` contains only the existing pynvml
deprecation warning. No actual teacher execution or existing test suite was
repeated.

Reproduction from the fixed runtime, stdout goes to the caller:

```bash
cd /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  timeout --signal=TERM --kill-after=5s 180s taskset -c 0,1 \
  /root/miniconda3/envs/ddp/bin/python \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/reports/st_world_policy/20261017/correctness/indexed_composition_real32_v1/review_receipt.py
```

This closes the **actual finite composition numerical qualification** for this
exact request population, source cache, fixed-8 FP32 implementation and runtime.
It does not authorize formal/full-population training, prove trainer/PCA
integration or resume, establish production throughput, or demonstrate scientific
gain. Existing SMOKE admission remains bounded to the exact population and no
more than four updates, subject to its separate factory/trainer/resource gates.
No descriptor, trainer, deployment or ledger was advanced by this review.

Original failure evidence in `indexed_composition_v1` and the independent repair
verification in `indexed_composition_v2` remain unchanged.
