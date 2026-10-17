# Finite fixed-8 teacher qualification

These are callable commands, not a dispatch authorization. The Lead must first
verify current host/GPU authorization, an idle admitted GPU, process/lease locks,
CPU/RAM headroom and a clean immutable integrated worktree. No existing training
or cache is modified. No optimizer, student, PCA fit, loss or tensor writer runs.
The prior completed result is retained unchanged.

CPU-only owner plan already exists at
`/mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_owner_plan_v1/PLAN.json`.
Run from `/mnt/project/VLA-Drive-st-fixed-teacher-batch-20261017` for that exact plan;
after integration, use a new plan/output namespace and regenerate it from the
final immutable runtime. Do not edit plan fields or overwrite older results.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_fixed_batch_teacher plan \
  --requests /mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json \
  --reference-root /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1 \
  --current-cache-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384 \
  --teacher-config /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json \
  --output /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_integrated_plan_v1 \
  --max-seconds 900
```

After Lead allocation exposes exactly one authorized GPU via
`CUDA_VISIBLE_DEVICES`, the finite foreground command is:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
timeout --signal=TERM --kill-after=30s 930s \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.qualify_fixed_batch_teacher run \
  --plan /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_integrated_plan_v1/PLAN.json \
  --output /mnt/project/st-world-policy-20261016/audits/fixed_teacher_batch64_integrated_run_v1
```

Do not set a physical GPU number from this document: the allocator must provide
the currently leased device. The outer timeout signals only its own child. The
runner also checks the 900-second budget, 8 GiB RSS and compute limits between
bounded calls; a GPU kernel is not preempted by those cooperative checks. An
external timeout can leave `PROGRESS.json` without `RESULT.json`, which is an
incomplete attempt, never a qualification. Preserve that directory and rerun only
under a new explicit budget/output if admitted.

The reference caches are already-existing small stores; the runner refuses
metadata above 32 MiB and populations above 64 train scenes. It does not load the
full 101k manifest or create a multi-terabyte cache. Source and input checks run
before CUDA or weight loading. The CUDA run configures its own fresh process to
FP32 teacher, no TF32/AMP, `deterministic_algorithms=True`, cuDNN benchmark off,
cuDNN deterministic flag false (matching the current formal setup), and the
unchanged allowed SDPA backends. Importing or constructing the executor does not
silently configure an active trainer's flags.

`RESULT.json` has two separate gates:

- `fixed_policy_repeatability`: full tensors agree across requested 1/4/8,
  reversal, repeated tail and actual partial eviction/cold restore. Fixed policy
  labels are not licensed merely because Python tensors have the same shape.
- `legacy_reference_compatibility`: all compared computed values exactly match
  immutable legacy values, including historical short final batches. The old
  cache had a different, incompletely recorded numerical execution context.

Exit 0 requires both gates. Exit 2 means numerical review or an exhausted budget;
`FIXED_POLICY_REPEATABLE_LEGACY_REVIEW_REQUIRED` is explicitly not a compatibility
PASS. Exit 1 is an invalid input/runtime/contract or other failure. No tolerance
flag, automatic approval, GPU worker, cache builder or training launch exists.
All padding compute is included in `image_forwards` and `compute_stats`; actual
requested images and discarded padding slots are separately counted.

Keep original missing-source masks, paired temporal requests and sampled patch
IDs unchanged. The provider reads all declared future times before sampling.
The 8-record LRU intentionally causes partial hits: simply comparing fully cold
and fully warm access would miss the recovery defect under investigation.
