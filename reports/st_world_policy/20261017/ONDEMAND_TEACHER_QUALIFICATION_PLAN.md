# Foreground teacher numerical qualification runner

Implementation start: `214cf43bcf41cc24c6f742d83b1968b74738cda8`.
This adds an executable **plan/run** smoke runner. It does not dispatch a job,
select a spatial winner, load teacher weights in this engineering task, modify
training, or authorize temporal training. No GPU numerical result is claimed.

## Actual metadata discovered and checked

- Locked request file: `/mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json`.
- Complete numerical reference: `/mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1`.
  It contains **957 images**, not just 192 current images. Its COMPLETE receipt
  records 1,693,808,448 tensor bytes. Its actual native FP32 contract and original
  build batch size eight are checked by the runner.
- Complete reuse source: `/mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384`.
  Among its 768 current records, exactly **192** are current images of this
  locked64 population. It contains **zero** of this population's765 future
  images. This smoke therefore exercises192 current reuse and765 future misses.
- The historical441/324 match/miss count refers to the *full101k current bank*.
  It is not this smoke's executed reuse fraction. No full101k JSON is parsed or
  copied by this runner; no shadow reader, synthetic COMPLETE or subset cache is
  created. Both sources use the real `TeacherCacheReader`.
- The teacher path is taken only from the reference's actual
  `resolved_config_build_rank0.json`:
  `/mnt/project/ddp-foresight-artifacts/20260928/generic/dinov3_vitl16_lvd1689m_timm`.
  The runner checks that it is a build config for this exact reference output,
  FP32 storage and production batch8. It never guesses a model path.

A CPU metadata-only call to `make_plan` completed on these actual artifacts:
64 scenes /64 logs;192 current,765 future,957 unique;three genuinely missing
future views. No record tensor, image, teacher weight or CUDA API was read by
that plan. The actual `--help` command was also run successfully.

## Fixed protocol and gates

1. Recompute all957 reference records in their original shard/key order,
   production batch8 and original tail. Compare **all elements** with the
   corresponding immutable FP32 tensors and all masks.
2. Recompute the same192 current images individually, in reversed batch8 order,
   and in batch8 with one duplicate to create a final tail of one. Compare with
   both immutable targets and the newly computed production8 outputs.
3. Execute the actual provider over all64 scenes, with the current-only source,
   first cold, then warm, then after clearing state for cold restoration. Each
   cold pass computes765 future images. Warm must generate zero. The provider
   retains the original all-four-future semantics; no request-dependent skipping.
4. Compare all current/future tensor bytes, valid masks and sampled patch/pair
   IDs across warm and restored passes. Only true absent source views are false.
5. Snapshot Python/NumPy/CPU/CUDA RNG around the **actual model constructor** and
   every real forward, plus each provider read. Preserve frozen parameter and
   buffer versions. Teacher eval/frozen FP32 and no TF32 are enforced by the
   existing executor. No student, loss or optimizer is loaded by this runner.

Stored/warm/restored targets, masks and RNG must be bitwise equal. No absolute
or relative tolerance is introduced to manufacture PASS. If generated targets
have any finite nonzero difference from their immutable reference, the complete
report is `NUMERICAL_REVIEW_REQUIRED`, exit2. It reports absolute max/mean/RMS,
L2 relative error, elementwise relative maxima with fixed1e-12 denominator floor,
and whole-record digests. The observed online cross-batch envelope is descriptive
only: it is not a threshold fitted afterward to approve caching. Invalid tensors,
masks, external RNG changes or failed cold/warm identity are failures, not zeros.

The original cache metadata did not identify every historical CUDA/backend flag.
The runner explicitly records its own FP32/TF32-disabled/SDP/determinism settings;
a production-size batch is not a promise of identical historical kernels. These
limits remain visible even if all measured tensors happen to match.

## Resource bounds and incomplete results

- One explicitly visible authorized GPU; CPU threads2.
- At most64 scenes,957 unique images,**3064 total image forwards** across the
  fixed cases, at most2048 calls as a second hard admission bound.
- Runtime budget900 seconds, checked between bounded calls/scenes. An external
  foreground `timeout` remains necessary for a hung device operation. A single
  operation can cross the cooperative deadline; no hard real-time guarantee.
- Peak RSS8 GiB check;2 GiB online comparison tensors and2 GiB provider resident
  payload cap. These are measured at run time. Estimated resident raw payload is
  about3.2 GiB for two957-image copies, plus teacher/load/preprocess/runtime
  overhead. This is a planning estimate, **not a measured memory result**.
- Metadata including replacement staging ≤32 MiB; no tensor disk writes.
  The small immutable banks total roughly3.1 GB of reference tensor storage and
  are reused. Their existing shard checksums remain enforced.
- Runtime/budget exhaustion writes a `PARTIAL_BUDGET` result and exits2; errors
  write `FAILED` and exit1. Fresh output directories are mandatory. Prior
  results are not overwritten or promoted. No retries/launches are automatic.

`run` regenerates the metadata plan and compares it exactly before accessing
CUDA. Editing a plan and recomputing its JSON identity cannot enlarge RAM,
forward count, masks or the protocol. Every input and source file is pinned.
Use a freshly generated plan from the final immutable execution worktree.

## Commands for Lead dispatch only

The output paths below are new qualification destinations, not claims that a
GPU run already exists. Run from the final integrated immutable worktree.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_ondemand_teacher plan \
  --requests /mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json \
  --reference-root /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1 \
  --current-cache-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384 \
  --teacher-config /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json \
  --output /mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_plan_v1 \
  --max-seconds 900
```

After Lead checks the authorized GPU is free, the foreground command on that
host (GPU ordinal0 below must be the actual allocated device):

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout --signal=TERM --kill-after=30s 930s \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_ondemand_teacher run \
  --plan /mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_plan_v1/PLAN.json \
  --output /mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_run_v1
```

The timeout controls only this foreground qualification process, not another
training job. `PROGRESS.json` is retained if externally interrupted; such an
interruption is not COMPLETE. No background/service process is created.

## Engineering verification and remaining chain

Focused CPU tests cover metadata-only planning without CUDA/teacher reads,
actual build-config provenance, oversized metadata, tampered plan/input rejection
before CUDA, all-element difference summaries, one-ULP non-PASS behavior, atomic
metadata capacity, forward budget and CLI argument checks. See
`ONDEMAND_TEACHER_QUALIFICATION_RUNNER_TEST.txt` for the actual final run.
The earlier provider tests remain separate evidence.

Still NOT RUN: this real teacher qualification, student forward/backward parity,
PCA provider-scope integration, native trainer resume, sustained training cost
and independent formal reviews. Even a future bitwise teacher PASS is not a
formal temporal training approval or evidence of planning benefit.
