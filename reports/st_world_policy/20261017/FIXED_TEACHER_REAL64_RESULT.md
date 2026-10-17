# Bounded RAM teacher: real 64-scene result

On 2026-10-17 the Lead executed one bounded qualification on an otherwise idle
A800 80GB on the authorized rl-zt2 host. This exercised actual source images,
the existing frozen FP32 DINO weights and immutable reference labels. It did
not run a student optimizer or create a tensor cache on disk.

Both separate gates passed: **fixed-policy repeatability** and **legacy reference
compatibility**. The previous variable-batch failure remains preserved under
`audits/ondemand_teacher64_run_v1`; it was not rewritten or given a wider
tolerance. The correction fixes teacher compute shape to eight actual/padded
images and discards padding outputs, without adding supervision samples.

| Measurement | Observed result |
|---|---:|
| Fixed train scenes / logs | 64 / 64 |
| Distinct source images | 957 |
| Physically missing future views | 3, original false masks retained |
| Fixed-policy full-tensor comparisons | 3,448; 2,629,435,392 elements, exact |
| Legacy full-tensor comparisons | 4,021; 1,778,761,728 elements, exact |
| Teacher calls | 729 |
| Requested image presentations | 3,701 |
| Actual compute slots, including padding | 5,832 |
| Discarded padding slots | 2,131 |
| External RNG guards | 729 passed; constructor also preserved RNG |
| Real RAM hits / evictions | 320 / 1,527 |
| Partial-hit scenes | 64/64 exercised hits and misses together |
| Provider tensor payload peak | 14,159,232 bytes |
| Process peak RSS | 3,318,284,288 bytes (3.09 GiB) |
| CUDA peak allocated / reserved | 1,416,995,328 / 1,530,920,960 bytes |
| Qualification elapsed | 477.571 seconds |
| New tensor disk bytes | **0** |
| Operational errors / exit code | 0 / 0 |

The timing includes full tensor comparisons, hashes, reference reads and RNG
audits. It is neither student training throughput nor production teacher
throughput. Frozen tensor object/version fingerprints matched before and after;
this is not described as a separate post-run byte scan of all teacher weights.
The fixed runtime uses deterministic algorithms, FP32 teacher execution and
storage, disabled TF32/AMP, and the explicitly pinned existing operator sources.

Executed source: `1cb97e5d5b1e0f32eed961c38729b3140f4a90a5`, immutable runtime
`/mnt/project/VLA-Drive-st-fixed-label-runtime-v2-20261017`.

Private evidence root: `/mnt/project/st-world-policy-20261016/`:

- `audits/fixed_teacher_batch64_runtime_plan_v1/PLAN.json`, identity
  `3b85a3b8e9dc42f76e694e449a1346c95856a8ad2937721c2747a273e6dce9dd`.
- `audits/fixed_teacher_batch64_runtime_run_v1/RESULT.json`, SHA256
  `22d93138fdc856dfd34db3116f9d72dbc0a8d9ca17eaed2066698d816eed67ca`.
- `resources/fixed_teacher64_smoke_v1/LAUNCH.json` and `EXIT.json` bind the
  actual resource admission, command and process exit. A later read-only check
  found both owned PIDs gone and the GPU at 0 MiB / 0%.
- Independent Correctness: `audits/correctness/FIXED_TEACHER_BATCH64_REVIEW_v1.json`,
  SHA256 `9852e4675098d05114625a75a354cad5b4567eb12fa904a51b0b47e1199eca73`.
  Its 19 focused CPU tests passed in 3.68 seconds; they are part of the earlier
  implementation suite, not 19 extra unique tests to add to its count.
- Independent Evidence: `audits/evidence_fixed_teacher_batch_v1/SOURCE_REVIEW.json`,
  SHA256 `1633b60eac4f7e0e32dc3470aed64b52d1f5b520b2f909942b1f9dedcc5856fe`.

The old full-cache training branch remains unchanged. Optional provider wiring
has a separate identity and independent limited review; fixed-policy integration
into that wiring is the next change. Student backward, complete native recovery,
same-runtime loss/update comparison and throughput are **NOT_RUN** for the new
fixed provider. No formal future/history experiment is approved by this result,
and no spatial winner is selected. Scientific gain remains **UNTESTED**.

The storage decision remains: reuse existing current caches, no multi-terabyte
future prebuild, aggregate campaign cache hard cap 768 GiB, separate 1 TiB free
space floor and independent checkpoint reservation. No existing data was deleted
or quantized. The next diagnostic inherits the official DriveDreamer-Policy
training backend, while this frozen teacher retains its own FP32 contract.
