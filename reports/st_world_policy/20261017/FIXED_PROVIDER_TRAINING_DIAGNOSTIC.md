# Bounded teacher provider: integrated student diagnostic

This continues the user's no-multi-TB-cache decision. Existing current-frame
targets are reused; missing future labels are computed by the frozen FP32 DINO
executor into bounded RAM. No new teacher tensors are written to disk. This is
not a change to the student execution precision, supervision, or sample order.

The integrated implementation is `8c1e3f86268ef5ed79f46d69abf33d5c9ca74bdf`.
Its immutable runtime is
`/mnt/project/VLA-Drive-st-fixed-provider-runtime-v1-20261017`.
The raw-teacher real64 result is described in `FIXED_TEACHER_REAL64_RESULT.md`;
both the fixed-policy and original-cache whole-tensor comparisons passed exactly.
That result alone does not prove student training or native restoration.

Independent adapter review:

- `audits/correctness/FIXED_PROVIDER_ADAPTER_REVIEW_v1.json` under the campaign
  artifact root, SHA256
  `9d153086cefe5ca9ecf003713d7a928ed967ec4ae9e230a30df9786e97c65d12`.
- 33 focused CPU tests passed; the actual 64-scene dataset/PCA metadata and
  five rejection cases passed. The review did not run another teacher extraction.
- The existing trainer, dataset, PCA basis and fixed numerical executor remain
  unchanged. Formal and PROFILE provider entry points remain closed.

The Lead admitted an isolated continuous two-update SMOKE on the actually idle
rl-zt2 GPU1. The comparison is a separate one-update pause plus one-update native
resume from the same initialization. Each identity retains the original
20,000-update cosine horizon and 1,000-update warmup; neither is a scientific
training result or an allowed formal initializer. Effective global batch2,
microbatch1, world1, native save every update, and 900-second invocation limits
are explicit diagnostic overrides.

The actual student backend is BF16 execution, FlashAttention2, Torch AdamW and
DeepSpeed ZeRO2 with FP32 masters/moments, no offload, and effective language
checkpointing disabled. The separate label teacher remains deterministic FP32,
TF32/autocast disabled, fixed compute batch8, with a 14,159,232-byte RAM LRU.
Padding compute slots never become extra observations or supervision records.

Actual continuous registration:

```
/mnt/project/st-world-policy-20261016/smoke/world_fixed_provider_native2_v1/continuous/REGISTRATION.json
SHA256 72ca0de30170a2bd1dda23d247af0ff19dcff8fa7ec3992d76b9083a3fe69477
CONFIG SHA256 a1fbb86793a141a869f18c1438e39626b398d42dc142da1f9b13fb877fd91d17
```

The private Lead controller is
`resources/qualify_fixed_provider_training_v1.py`, SHA256
`8d7049bd2e7557121c78d02674590004ab923a8129dd9c30ffd3cbc7e8ef8925`.
It checks current authorization, the clean source, input hashes, actual host/GPU,
exclusive task locks, and lease lifetime. Its `prepare`, `run`, `request-stop`
and `resume` actions are explicit; no automatic retries or unrelated process
signals are installed. Invocation metadata is preserved before live status/result
files are updated by resume.

Native checkpoint storage is separately budgeted at **160 GiB for both arms
combined**, including partial writes, with an additional **1 TiB free-space
floor**. This is an admission reservation, not preallocation or a filesystem
quota. It is not part of a new feature cache. No existing files are deleted.

Status at this report's first publication: continuous SMOKE launched and loading;
split/resume and independent full-state comparison **NOT_RUN**. Do not quote this
as a student numerical PASS, multi-rank qualification, or training throughput.
The later completed receipts and an independent comparison are required before
updating that status. Existing formal W/C runs use their unchanged immutable
runtimes and are unaffected.
