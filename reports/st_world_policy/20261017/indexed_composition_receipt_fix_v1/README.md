# Indexed composition: narrow receipt-validator correction

This fixes the independent receipt-consumer findings against `fd9cb505`.
It does not change the teacher, numerical threshold, request population, LRU
algorithm, old qualifier, PCA, student, trainer or training admission scope.
Real teacher qualification remains **NOT_RUN**.

Independent original evidence is retained at commit
`21cac2d3ded5f046fd532b550bbd25109e687d9e`,
`reports/st_world_policy/20261017/correctness/indexed_composition_v1/`.
Its final 12-case run had **5 FAIL / 7 PASS**. The earlier NaN test initially
failed in its own signing step; the corrected test confirms existing canonical
NaN rejection. NaN is not reported as a newly fixed implementation bug.

## F1: partial or contradictory full-tensor evidence

Original consumer accepted re-signed reports with `elements=0` or `elements=1`
and accepted an explicitly failing legacy summary while the top-level flag
remained true. The runner itself compared real complete tensors; the consumer
failed to enforce completeness and internal consistency.

The consumer now derives expected elements from grid and channels (442,368
for this native contract) for every independent, cached, composed and per-slot
online comparison. It checks the exact statistic field set, positive integer
element count, finite nonnegative scalars, boolean equality and zero error
statistics. It recomputes both summaries from their exact row lists, requiring
matching schema, counts, elements and values. Exact scene/key/slot order and
patch/pair sample identities remain bound.

No tolerance is introduced or relaxed. The original separation between cached
legacy FP32 compatibility and fixed-8 execution repeatability is retained.

## F2: impossible counters or Python boolean-as-integer values

Original consumer accepted `requested_images=-1` balanced by excessive padding,
and accepted `process_rchar_bytes=true` as a number.

All per-layout counters, byte measurements and compute/RNG counters now require
exact nonnegative integer types (and positive values where actual calls/requests
are required). Cold/restored counts must match the exact source hit/miss sets.
Unique-key coverage, provider totals and resident-byte bounds are checked.
Fixed-8 call and requested-image totals are reconstructed from the two online
reference passes and each layout's computed misses. Slots and padding must
match those totals and the original budget. Measured elapsed time must be a
finite, non-boolean, in-range number; RSS and allocated-GPU bytes are integers.

The 16 GiB GPU guard remains an **allocated peak** limit. This patch does not
quietly reinterpret it as a reserved/device-wide limit. Actual resource
admission and the outer owning supervisor remain Lead responsibilities.

## Real metadata check of partial-LRU triggerability

Lead separately authorized one read-only CPU metadata simulation, capped at
180 s, two CPUs, 2 GiB RSS and 1 GiB logical reads. The request and full-current
indexes were each opened once with their real SHA/dependency checks. No target
tensor, image, model or raw-log payload was opened; no GPU discovery/execution,
cache claim, source cache or active job was changed.

Actual identities and rows are in `METADATA_LRU_AUDIT.json`; the executed helper
is `metadata_lru_audit.py`. It replays the provider's real key access sequence,
including collecting all misses before delayed compute delivery, with eight
resident records. It is a metadata simulation, not a numerical qualification.

- 32 scenes, 294 unique keys, 189 source hits, 105 misses.
- Source miss counts include four scenes above eight misses (9 or 12).
- Seven partial-layout reads naturally have both RAM hits and compute misses.
- 111 reads across the four layouts have both immutable hits and compute misses.
- 701,545,747 rchar bytes; 625,164,288 bytes peak RSS; 1.4822 s measured audit wall.

This resolves the layout feasibility question for this exact population without
changing the case schedule, selecting easier scenes or manufacturing counts.
The actual teacher run must still produce these signals and pass all numerical
checks. The metadata timing is not GPU/teacher/training throughput.

## Verification scope

Owner reruns the original independent counterexamples plus the existing focused
factory/qualifier regressions. A new adjacent negative test checks 24 mutations
using one valid synthetic receipt: comparison/summary schema and quantities,
online comparison coverage, invalid scalar types, read and compute accounting,
per-layout/provider totals, samples and record ordering. Synthetic CPU receipts
remain explicitly fixtures and are never written as real qualification evidence.

`RESULT.json` and `tests.log` record actual commands and final counts. The original
independent reviewer must retest the submitted fixed source before real dispatch;
owner test PASS is not an independent approval or formal launch permission.
