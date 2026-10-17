# Fixed teacher batch candidate: evidence and boundaries

Design ID `ST-TEACHER-FIXED8-001`; recorded 2026-10-17 before implementation.
Source start `4410b4849ddaabf4d234f8238e7332c77e0a32b6`.
Status: bounded execution candidate; no formal approval or training dispatch.

The completed real 64-scene run at
`/mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_run_v1/RESULT.json`
 (SHA256 `05312ea01878ee6cd5d3bca24984f95580681365b026b45a3dd21f8efee7bf04`)
reproduced all 957 existing targets exactly when replaying original batch-8 and
shard boundaries. Batch-1 differed (maximum absolute error 0.0001503453), and
the provider's scene-local 8+4 batches differed. Warm/full-cold replay was exact
for that particular access schedule, but it did not exercise partial LRU hits.
Partial hits can change compute batch shapes after resume; that is the problem
being corrected, rather than a request to relax the tolerance.

Primary implementation evidence, inspected in the installed timm 1.0.20:

- [timm Eva, tag v1.0.20](https://github.com/huggingface/pytorch-image-models/blob/234907eca5d801a369b536e001ec0b45d8ec9d60/timm/models/eva.py):
  `vit_large_patch16_dinov3`, `Eva.forward_features`, `EvaBlock.forward`, and
  `EvaAttention.forward` preserve the image batch dimension. Attention uses
  `[B, heads, tokens, channels]` and token attention; normalization is LayerNorm,
  with no cross-image batch normalization. Eval sets attention dropout to zero.
- [timm position implementation](https://github.com/huggingface/pytorch-image-models/blob/234907eca5d801a369b536e001ec0b45d8ec9d60/timm/layers/pos_embed_sincos.py):
  `RotaryEmbeddingDinoV3._augment_coords` bypasses random coordinate augmentation
  in eval. The actual wrapper uses strict local weights and excludes 5 prefix
  tokens, retaining all native 18x24 patches.
- [PyTorch 2.5.1 numerical accuracy, batched computations](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/numerical_accuracy.rst):
  mathematically identical batching is not a guarantee of bitwise identity.
  A fixed shape is a testable mitigation, not a general proof of kernel parity.

Installed source SHA256: Eva
`52e00af9628a8ca785378dad256fd3d5931d61846fc08bbec80deb172d8844be`;
RoPE `265ca0ce9070d5d5ed4c182330f8c072a66072bcabd66633b726aaa446194dc8`;
patch embedding `78f2d6afba3bf71a0b8a51cfb06e034b08af0708f4a49509511e4b419e448cda`.
These are local implementation evidence, not a claim that every future timm
version has identical behavior. The execution identity additionally pins loaded
module source files.
The three installed files above were downloaded from the pinned official commit
and compared byte-for-byte on 2026-10-17; all three matched exactly. No local
weight/model was instantiated for this source comparison.

The new adaptation pads 1..7 actual preprocessed FP32 images to a compute batch
of exactly 8 by repeating the last real input tensor, then discards padded
outputs. It does not fabricate records, labels, masks, missing frames or student
samples. All 8 compute slots count toward the qualification budget. Existing
executor/default behavior and active runtimes remain untouched. Teacher FP32,
TF32-disabled, AMP-disabled, frozen/eval, external RNG preservation and active
formal `deterministic_algorithms=True` remain explicit execution requirements.

Two distinct gates are required: (1) the fixed execution policy must reproduce
its own targets across requested batch 1/4/8, shuffle, tail, partial eviction and
cold resume; (2) compatibility with immutable legacy cache values must be
reported separately, including its historical short final batch. Passing (1)
does not grant (2). Any nonzero difference is reported without widening a
tolerance. Neither gate alone grants formal training approval, student-gradient
parity, native distributed resume or end-to-end throughput qualification.

Implementation adds only `fixed_batch_teacher.py`,
`qualify_fixed_batch_teacher.py`, their tests and these reports. The inherited
`FrozenDINOExecutor`, provider, existing qualifier, dataset, trainer and PCA code
are unchanged. New execution IDs include the padding policy, the legacy and new
executor source hashes, installed operator sources/settings, hardware type,
deterministic mode and cuBLAS workspace configuration. Changing those dependencies
invalidates the corresponding qualification; this is not an alias for the old
execution identity. The matching-qualification helper checks the completed
result's checksum and exact execution identity, and deliberately does not grant
legacy reuse or formal approval.

CPU command, actually executed with CUDA hidden and two CPU cores:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_fixed_batch_teacher.py \
  tests/st_world/test_qualify_fixed_batch_teacher.py \
  tests/st_world/test_ondemand_teacher.py \
  tests/st_world/test_qualify_ondemand_teacher.py
```

Result: **58 passed in 10.24 s**, one pre-existing pynvml deprecation warning.
This includes the existing two-process CPU/Gloo failure path. New tests use an
explicitly synthetic batch-sensitive encoder to reproduce the legacy partial-LRU
failure, then verify fixed-8 target/RNG identity across partial hits and cold
restore. Other checks cover padding accounting, 1/4/8 request sizes, ordering,
repeat requests, backend drift, external RNG restoration on exceptions, mutated
buffers, FP32/deterministic guards, wrong identity, worker rejection and tampered
qualification plans. These fixtures do not establish real DINO equivalence.

The real-image plan was generated on CPU without loading teacher weights, target
tensors or CUDA. Path:
`/mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_owner_plan_v1/PLAN.json`.
SHA256 `af1774f366e91f5dd4a450fe71f61530a0289e2cb0e0941a03d069b0b591f70d`.
Identity `1ae062a3da6bc37577bb640d030ef852b5a78a2d34076e20d76ae80ef10f5949`.
It pins 64 scenes/64 logs, 957 images, 3 truly missing future views, at most
793 teacher calls/6344 compute-image slots including padding, 900 seconds,
8 GiB peak process RSS, 2 GiB reference tensors, 8-record LRU (14,159,232 bytes),
32 MiB metadata including atomic staging, and zero new tensor disk bytes.

Real fixed-policy GPU result: **NOT_RUN by implementation agent**.
Student/PCA integration, distributed optimizer resume, end-to-end throughput and
independent approval remain outside this patch. Regenerate the plan from the
final integrated immutable worktree before Lead dispatch; its source paths and
hashes must refer to the actual code being executed.
