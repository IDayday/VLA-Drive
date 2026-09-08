# PlanReg-WM-V2 validation

Fixed base: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`.
Isolated branch: `feature/planreg-wm-v2-integrated-20260908`.
No full training, full Navtest, push, PR, merge, reset, old-checkpoint overwrite or original-worktree cleanup was performed.

## Evidence classification

- **Source findings:** see `SOURCE_PROVENANCE.md` and `IMPLEMENTATION_MATRIX.md`.
- **Unit / numerical validation:** the expanded regression command below completed with **71 passed**, including two-process Gloo unequal-valid-count/all-invalid reductions, actual CPU GradScaler skipped optimizer/EMA steps, FP32 sub-ULP EMA accumulation, causal TF/RO, K=1/8/64, strict V1 scorer parity, and a CUDA BF16 read-only-attention forward/backward test.
- **Real-model development validation:** a real InternVL3-2B / NAVSIM 32-step smoke completed on 32 unique training scenes from eight logs. No mock PDM labels were used. A genuine two-VLM shared-initialization audit found all 585 trainable tensors bitwise identical. An actual deterministic 4-step versus 2+2-step resume audit matched model, optimizer moments, scheduler, sampler progress and RNG exactly.
- **Final integrated replay:** the current EMA activation-boundary, low-frequency diagnostic and whole-optimizer-batch validity-count changes are being revalidated. Final measurements will replace this paragraph in a follow-up evidence commit; prior runs are not represented as reruns of these final changes.
- **Performance:** NOT_EVALUATED. Historical epoch27/33 scores are old reported results, not V2 results. No claim of exceeding 91.3 or achieving 94 is made.

Development artifacts are outside the repository at `/mnt/project/DriveVLA-M0-stage2/planreg_v2_validation_20260908/`. Weights, raw data, input caches and credentials are not committed.

## Reproducible regression command

Use Python `/mnt/project/DriveVLA-M0-stage2/reproduction_diagnostics/envs/navsim_py39_exact/bin/python`, `PYTHONNOUSERSITE=1`, `PYTHONPATH=$PWD:/mnt/project/DriveVLA-M0-env/lib/python3.9/site-packages`, and one-thread OMP/OpenBLAS/MKL.

```bash
python -m compileall navsim/agents/EpisodeDrive
python -m pytest -q \
  tests/test_planreg_v2_data.py tests/test_planreg_v2_architecture.py \
  tests/test_planreg_v2_world_model.py tests/test_planreg_v2_contracts.py \
  tests/test_planreg_v2_distributed.py tests/test_drivor_scorer_parity.py \
  tests/test_legacy_forward_parity.py tests/test_student_checkpoint_export.py \
  tests/test_read_only_register_attention.py tests/test_register_patch_parity.py \
  tests/test_internvl_planning_registers.py tests/test_vision_qv_lora.py \
  tests/test_future_image_paths.py
python scripts/audit_drivor_scorer_parity.py
```

The one V1 test fixture change sets the pre-existing `compute_dtype` attribute when that test deliberately bypasses backbone construction; V1 production forward was not changed.

## Development failures, not hidden passes

1. The original exact loss mutates NC/DDC label views in place. Passing shared FP32 labels into all source BCE terms caused a real backward version-counter error. The V2 call boundary now supplies FP64 labels so the original per-head dtype conversion allocates independent FP32 labels; the source loss remains unchanged.
2. A too-strict endpoint tolerance rejected real `.4997/1.4988/3.9988` timestamps and produced zero WM supervision. The run was stopped; it is **not** counted as successful full-WM smoke. Explicit 20 ms timestamp tolerance now preserves actual times and still cannot relabel 3 seconds as 4 seconds.
3. An initial real resume comparison diverged due to CUDA nondeterminism even before interruption. Explicit deterministic algorithms plus cuBLAS workspace fixed the subsequent actual comparison. The failed artifacts were preserved.
4. Gradient accumulation now normalizes trajectory and WM targets with the whole optimizer-batch global validity counts. It does not average unequal microbatch validity means. Original scorer loss/reduction semantics are retained.

## Not run / blocked formal prerequisites

- Complete 103,288-scene statistics/input-cache artifact: not generated in this bounded development run. The measured 32-scene statistics are **smoke-only**. Formal launch rejects statistics whose unique count, split or token hash differs from the full final-fit manifest.
- Actual GB128 distributed/accumulated throughput and peak-memory profile: not run. No formal layout lock is fabricated; V1's microbatch evidence is not accepted.
- Full old V1 checkpoint → new V2 model migration replay: not run. The output-head conversion is numerically tested and a strict explicit warm-start artifact/consumer exists; this is not whole-model parity or lossless resume.
- Producer-side logged-vector coordinate provenance must accompany each new dataset. The builder explicitly records the selected source frame and never infers it from the pose `in_global_frame` flag.
- Full Base/VQA 27-epoch training, final-fit comparison and Navtest: intentionally not run under this task's authorization. Provided launchers require explicit launch flags.

The executable K-dimensional kinematics/rollout interface is implemented. No undefined multi-candidate consequence labels or new consequence heads were invented; this is not a validated multi-candidate consequence model.
