# Fixed-eight teacher provider adapter

Engineering source: safely merged Lead `1cb97e5d` into the isolated provider
branch as `54571941`, preserving all earlier commits. Only the existing
`teacher_provider.py`, focused tests and this report are changed by this adapter.
No active runtime, dataset, trainer, PCA implementation or cache is edited.

The adapter accepts the distinct `st_fixed_teacher_qualification_v1` /
`st_fixed_teacher_qualification_plan_v1` / `st_fixed8_dino_execution_v1` family.
It dispatches to the existing `FixedBatchDINOExecutor`; legacy dependencies
continue to use `FrozenDINOExecutor`. It never rewrites legacy metadata into the
new schema and never relaxes the legacy no-RAM-reuse rule.

Admission checks include:

- Report self-hash, execution hash, immutable file pins, request population and
  actual current COMPLETE cache; qualified teacher/backend/source identities.
- All six recorded cases and expected record counts. Regrouped and cold-resume
  tensor comparisons must be bitwise equal, not merely summarized as success.
- Actual partial-LRU exercise, unchanged masks and sampling, complete RNG call
  accounting and frozen-state hashes. Padded compute slots are counted, including
  repeated inputs that never become returned supervision records.
- `fixed_policy_repeatability=True` and its exact zero-error aggregate. Legacy
  compatibility has an independent boolean/summary/status and is never inferred
  from fixed-policy repeatability.
- Strict deterministic FP32 execution, warn-only off, fixed-eight padding policy,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and matching actual archived source files.

Only then may the new identity use a RAM LRU. Its initial capacity cannot exceed
the **actually qualified** eight-record resident budget; scene and batch limits
also cannot exceed the qualification plan. This does not approve multi-GiB or
full-population reuse. The existing SMOKE-only, at most64 train scenes,
`loader_workers=0` restrictions remain. Formal gates are unchanged and closed.

The existing factory, dataset and native checkpoint path are reused. The new
provider identity and PCA provider scope bind the distinct execution; the PCA
basis is not refitted. Runtime metadata includes requested images, all compute
slots and padding slots separately. Sampler/checkpoint progress contains no
cache hit/miss counters.

## CPU checks

`FIXED_PROVIDER_ADAPTER_TEST.txt` covers the new metadata adapter, the earlier
provider wiring, and existing fixed executor/qualification-plan tests. Negative
fixtures reject unfinished or incomplete cases, a false repeatability flag,
partial-LRU omission, changed cold or regrouped tensors, RNG/frozen mismatch,
hidden padding costs, forged legacy compatibility, backend/schema/hash mismatch,
and an unqualified larger LRU. Failed preparation writes no immutable descriptor.
Fixtures exercise rejection logic; they are not real numerical evidence.

No teacher weights are loaded and no GPU/training job is launched by these tests.
Actual focused run: **52 passed**, one pre-existing pynvml deprecation warning,
5.31 seconds, with CUDA hidden and Torch/BLAS thread budget two. This includes
explicit official-backend preservation and unchanged PCA-byte tests. Command:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout 90 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_fixed_provider_training.py \
  tests/st_world/test_provider_training.py \
  tests/st_world/test_fixed_batch_teacher.py \
  tests/st_world/test_qualify_fixed_batch_teacher.py
```

Existing real qualification is owned and dispatched by Lead in the immutable
runtime `/mnt/project/VLA-Drive-st-fixed-label-runtime-v2-20261017`.
Its archived output is
`/mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1`.

## Actual qualification and prepared diagnostic

Read-only evidence from Lead's completed real run:

- RESULT SHA256 `22d93138fdc856dfd34db3116f9d72dbc0a8d9ca17eaed2066698d816eed67ca`:
  `FIXED_POLICY_AND_LEGACY_BITWISE_QUALIFIED`, all cases complete, no errors.
- Fixed-policy repeatability and independent legacy-reference compatibility are
  both true. Their whole-tensor summaries have max absolute error zero. All 64
  scene partial-LRU/cold-resume cases, masks, sampling and RNG checks passed.
- The real run took 477.571 seconds on one A800: 729 calls / 5,832 actual compute
  slots, including 3,701 requested images and 2,131 padding slots. Peak RSS was
  3,318,284,288 bytes; peak CUDA allocation 1,416,995,328 bytes. No tensor disk
  writes. These are teacher qualification measurements, **not training speed**.
- Independent raw-target correctness review:
  `audits/correctness/FIXED_TEACHER_BATCH64_REVIEW_v1.json`, SHA256
  `9852e4675098d05114625a75a354cad5b4567eb12fa904a51b0b47e1199eca73`.

The actual helper accepted this report and produced a new namespace:
`/mnt/project/st-world-policy-20261016/audits/fixed_provider_wiring_v1`.
It contains only 28,494 bytes of newly written JSON; no tensors or weights were
generated. Existing PCA statistics were read and verified, not refitted.

| Artifact | SHA256 or identity |
|---|---|
| `configured/training_config.json` SHA256 | `0737e43dcf4403d7a8e33da31359dd50d86748b7a660eaf35fda23e9c7cabc78` |
| `PROVIDER.json` SHA256 | `84cdf37e816e64622fb36334b1473e06d630bdba5b743877cfaefb7d78f49c75` |
| Runtime provider identity | `5e537de60be495fa86414ee8fe0f90d7683e4b0246b0a32dcd624c4728519458` |
| New PCA provider scope identity | `7280f6b39f0754772d5bd81093607447f69b97497966924c1c43277737286349` |
| Unchanged PCA basis identity | `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46` |
| `REGISTRATION_DEPENDENCIES.json` SHA256 | `bcc0428f5033578e935d3f262591a5d941297ac4b48701d516a880361cb1da49` |

This T2 diagnostic inherits the actual
`smoke/world_official_runtime_real_v1/CONFIG.json` recipe. Its explicit
`runtime.official_execution` resolves to BF16 execution, FlashAttention2,
Torch AdamW, ZeRO2 and FP32 master/moments, without offload. Effective gradient
checkpointing is false. Deterministic execution is true, matching the qualified
teacher. The 20,000-update cosine horizon / 1,000-update warmup remains unchanged.
One GPU, microbatch 1 / global batch 2 and `stop_after=2` are declared diagnostic
overrides; runtime wall budget is 900 seconds, loader workers zero, and the
required Torch/OMP CPU thread count is two. Teacher compute remains FP32.

The 64-scene registration dependency is the existing, fully pinned
`audits/history_data/smoke_train64_pinned_v2/SELECTED_SCENES.json`; its ordered
tokens/logs and student/index identities were checked against TEACHER_REQUESTS.
The selected native384/PCA384 family is an engineering fixture, **not selection
of the formal spatial winner**. RAM LRU is limited to the qualified 14,159,232
bytes (eight raw records). The source remains the real COMPLETE 256-scene current
cache, supplying 192 current records in this population; future targets use the
qualified fixed-eight executor. True absent future frames retain original masks.

Metadata preparation can be reproduced without GPU or model loading from the
committed implementation (the input configuration is already pinned):

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.teacher_provider \
  --requests /mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json \
  --source-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384 \
  --teacher-config /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json \
  --qualification-result /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_runtime_run_v1/RESULT.json \
  --max-cached-bytes 14159232 --max-scene-mib 64 --max-batch-images 8 \
  --output /mnt/project/st-world-policy-20261016/audits/fixed_provider_wiring_v1/PROVIDER.json \
  --training-config /mnt/project/st-world-policy-20261016/audits/fixed_provider_wiring_v1/SELECTED_OFFICIAL_NUMERIC_SMOKE_INPUT.json \
  --pca-fit-cache-root /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1 \
  --configured-output /mnt/project/st-world-policy-20261016/audits/fixed_provider_wiring_v1/configured
```

The existing `tools.st_world_policy.train_world` remains the sole trainer.
Actual run registration, independent reviews and launch are Lead-owned. No
registration or approval is fabricated here. Native student forward/backward,
two-update uninterrupted versus one-plus-one recovery, projected target/RNG
equivalence, peak training memory and training overhead are still **NOT_RUN**
for this provider. Teacher raw-target qualification does not establish those
results or grant formal launch. Legacy failed-numerics reports remain unchanged.
