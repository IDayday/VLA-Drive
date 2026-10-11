# Runtime v2 commands and migration boundaries

Worktree `/mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06`; authorized isolated worker `training-vlawm-zt3`. Python `/root/miniconda3/envs/ddp/bin/python` (3.10.20), PyTorch2.5.1+cu124, Transformers4.57.0, NCCL2.21.5,8×A800-SXM4-80GB. Commands below are manual; this task has not stopped/replaced the active Base, started a full Base, launched navtest or updated ACTIVE.

Every runtime uses the actual original101592-scene ordered population in the parent checkpoint and keeps the original100k LR schedule. `--updates` counts **additional** optimizer updates; it does not truncate/restart cosine. Global batch32, local batch4,8 ranks and all four objectives are mandatory. New output identities are required on resume, with a parent checkpoint link. Runtime/source/environment mismatch is refused. No implicit world-size adaptation or epoch reset occurs.

CLI/help and CPU regression:

```bash
cd /mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06
/root/miniconda3/envs/ddp/bin/python -m iqe.performance.run --help
/root/miniconda3/envs/ddp/bin/python -m pytest -q tests/iqe
```

Repeat a bounded200-update measurement from the actual complete21k parent (choose a new output directory each attempt). Run only on an empty policy-authorized8-GPU worker:

```bash
cd /mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.performance.run --mode bounded \
  --config configs/iqe/performance_v2/source.yaml \
  --runtime configs/iqe/performance_v2/combined_visual_bf16.yaml \
  --checkpoint /mnt/project/DriveVLA-M0-iqe-20261009/outputs/iqe/main/query_base/step_021000.pt \
  --output /mnt/project/iqe-perf-v2-runtime-3855e06/manual_bf16_200 \
  --updates 200 --warmup 10 --save
```

Use `combined_fp32.yaml` to retain the original numerical precisions; use `p0.yaml` for the already-existing vision-ahead baseline. Runtime options are strict: no unknown keys or silently ignored precision/checkpointing/semantics values. Failed near-equivalence screens (`p2_visual_bf16`, `p6_bf16_master`) are not certified replacements. Native repeat1 is a different stochastic estimator and gave no worthwhile speed benefit here.

For a trace (never use its time as throughput), substitute `-m scripts.iqe.performance_v2.detailed_trace`, `--updates 7 --warmup 3 --trace`, and a fresh output path. This adds image open/decode/resize, tokenizer, DeepStack injection, individual loss reductions and separate finite-check/clip scopes to the normal instrumented regions. CUDA-event intervals across streams/nested scopes are not additive.

Random IO pilot over the existing64-scene local visual cache:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m scripts.iqe.performance_v2.cache_io_pilot \
  --cache /var/tmp/iqe-perf-v2/vision_pilot64 \
  --output /mnt/project/iqe-perf-v2-runtime-3855e06/manual_cache_io.json
```

This entrypoint refuses anything other than64 unique committed pilot scenes. It evicts only its own16 shard files with advisory DONTNEED, performs seeded randomized8-rank reads, and separately reports checksum/mmap/combine/H2D. It does not evaluate a full model or claim training throughput. Full visual caching is not approved: approximately5.76TB is needed before deduplication, exceeding current free capacity. No existing checkpoint/cache was deleted.

Manual long-job migration is conditional on a matching PASSED runtime certificate and a checkpoint-safe pause authorized by the user. **Never signal a historical PID.** Reinspect the live job; ask it to save at an optimizer boundary through its existing handler only when replacement is authorized. Validate that the process exits and its final complete checkpoint receipt exists. The supplied script does not send any signal: it scans live PIDs/cwd/cmdline on the original host and refuses to duplicate an active Base. It then verifies the newest complete checkpoint checksum, without falling back silently.

After that pause, on the tested empty `training-vlawm-zt3` worker:

```bash
cd /mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06
bash scripts/iqe/performance_v2/migrate_after_pause.sh \
  configs/iqe/performance_v2/combined_visual_bf16.yaml \
  reports/iqe_perf_v2_3855e06/BEST_RUNTIME_QUALIFICATION.json \
  1000
```

The1000-update window writes a new `/mnt/project/iqe-formal-runtime-v2/continuation_TIMESTAMP` identity and retains the original100k schedule/optimizer state. It is not a1000-step cosine schedule or a new model initialization. Checkpoint frequency remains1000; the output lock prevents duplicate launches. Storage planning reserves all new checkpoints plus one atomic-write temporary and the existing500GiB safety margin; no automatic retention/deletion is implemented. Extending to all remaining100k updates requires enough actual space; do not bypass this check or count a bounded runtime checkpoint as a new qualified incremental Base.

The active job's historical nondeterministic backend to this deterministic backend is an explicit execution migration. Visual BF16 is a further T1 numerical change. All weights/moments/scheduler/RNG/data cursor are restored, but cross-backend stepwise bitwise equivalence is not asserted. Within a fixed qualified runtime/environment, actual interrupted-resume evidence is required and bound by the certificate. The richer environment/identity checks were finalized after initial screening/P0 long measurement; old diagnostic checkpoint identities remain immutable and are not silently rewritten to match final code. Use the final combined-runtime checkpoints for final-code resume.

The separately locked step022000 remains the incremental E2/resampling_only start. Its weights, input/normalizer, FP32 frozen-visual execution, feature/candidate identity and serving bundle are unchanged. Any future choice of a new Base/shared numerical path needs its own Base/feature/candidate identity; this runtime certificate is not a Base-quality or deployment qualification.

Actual storage snapshot: each complete Base checkpoint is22,816,333,071bytes (21.25GiB). A1000-update continuation with the existing500GiB safety reserve fits; retaining every1000-step checkpoint through100k currently reportsBLOCKED_STORAGE_CAPACITY. See evidence/STORAGE_PLAN.json for exact required/free bytes. The bounded continuation command is usable without promising that disk can retain the whole remaining run. No checkpoint was deleted; choose capacity/retention explicitly before extending the horizon.

`source.yaml` supplies the original model/data/optimizer recipe to this dedicated runner. The performance-only execution choices are the strict `--runtime` file; this runner always uses the already-qualified one-step vision-ahead/prefetch pipeline. It does not invoke the old generic trainer or select a short-run LR schedule. The source config retains the old base_runtime metadata for compatibility with its existing strict schema; it is not a second performance-options namespace.

Actual fastest-runtime interrupted-resume example (the checkpoint below was produced and successfully restored in this campaign, not a placeholder). Use a fresh output path for each attempt and the same tested worker/environment:

```bash
cd /mnt/project/DriveVLA-M0-iqe-perf-v2-3855e06
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.performance.run --mode bounded \
  --config configs/iqe/performance_v2/source.yaml \
  --runtime configs/iqe/performance_v2/combined_visual_bf16.yaml \
  --checkpoint /mnt/project/iqe-perf-v2-runtime-3855e06/combined_visual_bf16_pause1/step_021201.pt \
  --output /mnt/project/iqe-perf-v2-runtime-3855e06/manual_bf16_resume_from_21201 \
  --updates 1 --warmup 0 --fingerprint
```

The actual proof used `combined_visual_bf16_resume1` as output and compared it against `combined_visual_bf16_continuous2`. All8 ranks passed complete parameter/optimizer/RNG/sample/loss/gradient/scheduler comparison at21202. The one-update resume diagnostic is not used to estimate throughput. Resume restores saved RNG after model/DDP/optimizer construction; the recipe still ends at100k and does not restart warmup. Exact continuation uses this dedicated runner and the identical runtime/source/environment identity.

Regenerate summary tables from preserved artifacts only (no training/test runs):

```bash
/root/miniconda3/envs/ddp/bin/python -m scripts.iqe.performance_v2.summarize_campaign \
  --root /mnt/project/iqe-perf-v2-runtime-3855e06 \
  --output reports/iqe_perf_v2_3855e06
```

Detailed trace analysis is likewise read-only:

```bash
/root/miniconda3/envs/ddp/bin/python -m iqe.performance.analyze_trace \
  /mnt/project/iqe-perf-v2-runtime-3855e06/best_detailed_trace --rank 3
```

Large traces, weights and image-derived payloads stay outside git. Committed evidence contains metrics and hashes, not raw image tensors or checkpoints. Full run commands are manual; the bounded queue used in this task never dispatches continuation/full, navtest or ACTIVE updates.

The conservative precision path has its own passed numerical/200-update/eight-rank resume certificate. After the same authorized source-job pause, its manual migration command is:

```bash
bash scripts/iqe/performance_v2/migrate_after_pause.sh \
  configs/iqe/performance_v2/combined_fp32.yaml \
  reports/iqe_perf_v2_3855e06/FP32_RUNTIME_QUALIFICATION.json \
  1000
```

Its actual interrupted-resume checkpoint is `/mnt/project/iqe-perf-v2-runtime-3855e06/combined_fp32_pause1/step_021201.pt`. A complete bounded reproduction command is:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -u \
  -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m iqe.performance.run --mode bounded \
  --config configs/iqe/performance_v2/source.yaml \
  --runtime configs/iqe/performance_v2/combined_fp32.yaml \
  --checkpoint /mnt/project/iqe-perf-v2-runtime-3855e06/combined_fp32_pause1/step_021201.pt \
  --output /mnt/project/iqe-perf-v2-runtime-3855e06/manual_fp32_resume_from_21201 \
  --updates 1 --warmup 0 --fingerprint
```

The actual `combined_fp32_resume1` run passed every complete-state check against `combined_fp32_continuous2`; see `FP32_EXACT_RESUME.json`. Do not reuse either output directory: completed identities are immutable and a new parent-linked attempt is required.
