# PlanReg-WM-V2 validation

Fixed base: `d9ca73f3d61f059285fcbf12a5bc81177ee350d7`.
Isolated branch: `feature/planreg-wm-v2-integrated-20260908`.
No full training, full Navtest, push, PR, merge, reset, old-checkpoint overwrite or original-worktree cleanup was performed.

## Evidence classification

- **Source findings:** see `SOURCE_PROVENANCE.md` and `IMPLEMENTATION_MATRIX.md`.
- **Unit / numerical validation:** the final expanded regression command below completed with **76 passed, 0 failures, 0 skips** (50.93 s), including two-process Gloo unequal-valid-count/all-invalid reductions, actual CPU GradScaler skipped optimizer/EMA steps, FP32 sub-ULP EMA accumulation, causal TF/RO, K=1/8/64, strict V1 scorer parity, scene/candidate permutation, and a CUDA BF16 read-only-attention forward/backward test.
- **Real-model development validation:** a real InternVL3-2B / NAVSIM 32-step smoke completed on 32 unique training scenes from eight logs. No mock PDM labels were used. A genuine two-VLM shared-initialization audit found all 585 trainable tensors bitwise identical. An actual deterministic 4-step versus 2+2-step resume audit matched model, optimizer moments, scheduler, sampler progress and RNG exactly.
- **Integrated real replay:** `final_smoke32` completed 32 optimizer steps with the integrated EMA activation path, TF+RO, real future images, original simulator/PDM labels and whole-optimizer-batch validity counts. `ddp2_accum2_smoke4` completed an additional real two-GPU, accumulation-two run. `final_resume_accum2` matched uninterrupted 4-step and 2+2-step accumulated training bitwise. Student export replay had `max_abs_diff=0.0`, with no teacher or predictor instantiated.
- **Performance:** NOT_EVALUATED. Historical epoch27/33 scores are old reported results, not V2 results. No claim of exceeding 91.3 or achieving 94 is made.

Development artifacts are outside the repository at `/mnt/project/DriveVLA-M0-stage2/planreg_v2_validation_20260908/`. Weights, raw data, input caches and credentials are not committed.

Small measured evidence is published under `reports/planreg_wm_v2/`. The main real smoke used code stage `90c80d4`; the accumulated resume used `88b9849`. Later changes add stricter invalid-long handling, metadata, migration guards and evaluation/profiling reporting; the final full regression covers those changes. This is not a claim that every earlier run was repeated after every report-only change.

## Real numerical and resource measurements

Environment: Python 3.9.25, PyTorch 2.5.1+cu124, Transformers 4.57.6, PEFT 0.17.1; local NVIDIA A800 80 GB GPUs. The 32 unique scenes come from eight log **segments of one recorded drive**, not eight independent driving domains. The data is suitable for bounded functionality checks, not broad convergence or generalization claims.

| Run | Layout | Optimizer steps | Peak allocated / reserved GiB | Median / p90 step, excluding first |
|---|---|---:|---:|---:|
| `final_smoke32` | 1 GPU × microbatch 1 × accumulation 1 | 32 | 12.076 / 14.783 | 2.966 / 3.322 s |
| `ddp2_accum2_smoke4` | 2 GPUs × microbatch 1 × accumulation 2; GB4 | 4 | 12.272 / 17.385 (rank-zero measurement) | 5.939 / 6.023 s |

The first run's training-loop duration was 147.43 s, including first-worker startup, metadata and checkpoint I/O, but excluding model construction and export replay; overall loop throughput was 0.217 samples/s, steady median approximately 0.337 samples/s. The two-GPU run was a DDP/accumulation correctness check, not a throughput optimization campaign. Subsequent profiling code takes the **maximum over ranks**, not only rank zero. These numbers must not be extrapolated to GB128 or compared as an optimized replacement for previous large-batch training.

- All **585 / 585** trainable tensors changed across the real 32 steps; total trainable parameters **26,414,111**. Frozen parameter count including the internal teacher: **2,396,420,864**.
- Actual checkpoint: **101 FP32 EMA masters**, **1,170 FP32 AdamW moment tensors**, **96 FP32 visual LoRA A/B matrices** in 24 blocks, **224 FP32 language LoRA A/B matrices** over 112 q/k/v/o projections in 28 layers. Optimizer updates = EMA updates = 32.
- TF+RO loss was nonzero from the first step (`lambda=0.01`): first **2.284405**, last **1.172678**. These are different training scenes in a tiny run, not proof of convergence or improved planning.
- Same-batch actual audit confirmed nonzero WM gradients to visual LoRA, registers/readout, language LoRA and semantic queries. It did not alter `.grad` or choose a new lambda.
- Fixed-source scorer parity: all six logit differences **0**, aggregate-score difference **0**, selected indices identical, proposal gradient absent, scene gradient nonzero, TTC invalid-mask test passed.
- V1 disabled-forward/config regressions passed. This is not a claim that a new V2 architecture reproduces a full V1 checkpoint's policy.
- Base/VQA architecture and tokenizer IDs matched exactly; shared trainable initialization was bitwise identical. Smoke-only shared artifact SHA: `16cbf90f49a77783f39916ad8754bac3c9f7b9a1e7db646fd0d07ada5fbad9fd`.

Logged-motion read-only audit on 232 moving intervals: interpreting raw vectors in ego axes yielded velocity-vector RMSE **0.1669 m/s** against interval displacement; treating them as global vectors yielded **7.4217 m/s**. This supports the explicit ego-frame contract for these logs. The difference calculation is diagnostic only; WM still uses the actual logged vx/vy/ax/ay. New datasets still require their own source contract.

Complete accumulated resume comparison: model tensors and metadata, optimizer moments, scheduler, RNG and sampler progress all equal; EMA masters are part of the compared model state. Export replay used real current RGB/tokens and no future keys, with exact final trajectory agreement.

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

Real entry commands, with VLM/shared-init/normalizer environment set as in `docs/planreg_wm_v2.md`:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_planreg_v2.py \
  --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --manifest /mnt/project/DriveVLA-M0-stage2/planreg_v2_validation_20260908/input_cache_32/manifest.json \
  --output /new/smoke32 --smoke-steps 32 --microbatch 1 --accumulate 1 --workers 2 --replay-export
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node 2 \
  scripts/train_planreg_v2.py --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --manifest /mnt/project/DriveVLA-M0-stage2/planreg_v2_validation_20260908/input_cache_32/manifest.json \
  --output /new/ddp_smoke --smoke-steps 4 --microbatch 1 --accumulate 2 --workers 2
CUDA_VISIBLE_DEVICES=0 python scripts/validate_planreg_v2_resume.py \
  --config navsim/planning/script/config/common/agent/planreg_wm_v2_base.yaml \
  --manifest /mnt/project/DriveVLA-M0-stage2/planreg_v2_validation_20260908/input_cache_32/manifest.json \
  --output /new/resume_audit --accumulate 2
```

Also executed: standalone VLM pair audit, real shared-init creation and Base/VQA shared-state audit, serialized precision audit, logged-motion audit, `bash -n` for all launchers, and official NAVSIM Hydra `--cfg job --resolve` using the student factory. The last command resolves configuration only; it does **not** evaluate Navtest.

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
- Producer-side logged-vector coordinate provenance must accompany each new dataset. The actual bounded frame-consistency audit is reported above; it does not establish the frame of every uninspected future dataset.
- Full Base/VQA 27-epoch training, final-fit comparison and Navtest: intentionally not run under this task's authorization. Provided launchers require explicit launch flags.

The executable K-dimensional kinematics/rollout interface is implemented. No undefined multi-candidate consequence labels or new consequence heads were invented; this is not a validated multi-candidate consequence model.

Official selected Navtest and the preserved training-PDM candidate diagnostic have separate entrypoints and explicit protocol names. `navtest.sh` uses the existing official `pdm_score` path; `evaluate.sh` reports `official_navtest_result=false`. No generic training-score scalar is relabeled as a new official Navtest result.
