# Bounded on-demand teacher: real64 numerical evidence

This is a completed finite **SMOKE**, not a formal temporal-training approval.
It implements the user's storage constraint without constructing a full future
cache. New tensor disk writes were **0 bytes**; no existing cache/checkpoint was
removed. The aggregate cache cap remains768 GiB including existing reservations,
with a separate1 TiB free-space floor and separate checkpoint reservations.

Immutable executed source: `ead79be656c8eba5ad83b1c3b5f6ed1657611214`,
`/mnt/project/VLA-Drive-st-bounded-label-runtime-v1-20261017`.
Actual launch used an idle A80080GB on the authorized rl-zt2 host, one GPU,
two CPU threads, at most900 seconds and8 GiB process RSS. Existing training and
the independent development evaluation were not stopped or modified.

Evidence root: `/mnt/project/st-world-policy-20261016/`.

- Plan: `audits/ondemand_teacher64_plan_v1/PLAN.json`, identity
  `555f1e414a3b8e5ed4488d15677c446de028920f962e2252f92019a771ef3034`.
- Actual launch/exit: `resources/ondemand_teacher64_smoke_v1/`.
- Full per-record comparisons: `audits/ondemand_teacher64_run_v1/RESULT.json`.
- Independent interpretation is recorded separately by Correctness; this
  implementation/Lead report does not grant itself a numerical PASS.

The fixed population contains64 legal training scenes from64 logs,957 unique
images,192 current immutable-cache hits and765 computed future images. Three
genuinely absent future views retain their original masks. DINOv3 ViT-L/16,
384x288 input,18x24x1024 dense targets,FP32 compute/storage,TF32 off and all four
future times are unchanged. No training update, student inference, metric call
or new scientific result was produced by this test.

| Case | Actual comparisons | Result against existing tensors |
|---|---:|---|
| Original production batch8 and original shard tails |957 images /423346176 FP32 elements |Bitwise identical |
| Reordered batch8 current images |192 |Bitwise identical |
| Current images at batch1 |192 |Not bitwise: maximum absolute difference0.0001503453, global L2 relative2.249584e-6 |
| Current batch8 plus duplicate tail1 |193 |Not bitwise: maximum absolute difference0.00001077773 |
| Provider cold |64 scenes,765 new image forwards |Not bitwise to stored future targets: max absolute0.00008267164, global L2 relative1.156597e-6 |
| Provider fully warm |64 scenes,0 new image forwards |Exact target bytes versus first provider cold pass |
| Provider cleared and same-order replay |64 scenes,765 new image forwards |Exact target bytes versus first provider cold pass |

All masks and tested patch/time-pair sample IDs matched. All617 guarded forward
calls preserved external Python/NumPy/CPU/CUDA RNG; construction also preserved
RNG. Frozen tensor object/version checks matched before/after. This is not an
independent rehash of every teacher parameter after execution.

The result is **NUMERICAL_REVIEW_REQUIRED**, exit2, with no exception. The gate
was declared before execution: any nonzero floating difference cannot become a
bitwise equivalence PASS. Small relative error is not an automatic acceptance
criterion. Near-zero feature elements produce large elementwise relative errors;
the full report retains these rather than hiding them with a global average.

Measured total qualification time was287.672 seconds, with3064 image forwards
in617 calls. Process peak RSS was5243781120 bytes; CUDA peak allocated was
1391321600 bytes and reserved1503657984 bytes. The RAM provider held at most
1693798128 payload bytes under its2 GiB limit. Metadata is additional and bounded.
Cold/warm/replayed read-and-comparison passes took75.686/21.052/65.878 seconds.
These include full CPU tensor comparisons, hashes and RNG checks; they are
**not** measured training update times or isolated teacher throughput.

## What remains unsafe to claim

The provider batches only misses. A partially populated LRU can therefore change
a record from an8-image computation to a1-image computation; a cold resume clears
the LRU. The observed real batch-size sensitivity means target bytes can depend
on residency. The tested fully-cold/fully-warm/same-order replay cases do not
cover this boundary and do not establish exact training resume.

Formal use remains blocked on stable target execution grouping and the affected
numerical/gradient/resume review. A synchronous, no-RAM-reuse diagnostic wiring
can establish a fixed per-scene miss grouping without allocating tensor disk,
but is not yet a production efficiency conclusion. No precision change,
quantization, reduced future supervision, target replacement, larger cache cap,
or tolerance widening has been authorized as a workaround.

Active spatial and consequence runs still use their original immutable runtimes
and approved inputs. The next temporal experiment has not been selected or
launched; its teacher execution and storage identity must be registered explicitly.
