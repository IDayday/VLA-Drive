# Independent temporal calibration correctness review

**PASS_CODE_CPU_ONLY** for `baa6184ab7e4b3919a09adf7593dc82904c5f0ba` and the exact real4 diagnostic configuration pinned in `REVIEW.json`. This is not approval of actual resource feasibility, real model gradients, proposed numerical weights, or formal training. The reviewer did not implement the audited modules and changed no production source.

## Findings and checked call chain

1. `calibrate_temporal_scale.py:165–255` constructs the existing `STWorldTrainingDataset` and original `SpaceTimeStudent`, samples one four-scene request table, fixes full-batch valid counts, then slices that table and the same targets into microbatches. The four unweighted terms come directly from `SpaceTimeStudent.forward`; the calibration tool does not reconstruct a different MSE or FM objective. Original `losses.py` retains current channel-wise valid normalization, the half-sum of independently normalized future endpoints, and raw endpoint difference at jointly valid positions. FM retains repeat8 and original-scene denominator. The independent test calls the actual forward with analytic small components, checks repeated action arguments, mixed masks, actual per-task autograd gradients, and unchanged requests. It does not instantiate the real DiT or Qwen.

2. `calibration_measurement.py:140–207` forwards once per microbatch and uses `autograd.grad` for all four terms on that one graph. Dropout, noise, time and patch realization are shared between those task measurements. Contributions already carry complete-batch denominators and are summed without another microbatch division. No optimizer, scheduler, training cursor, `.backward()`, or model `.grad` accumulation is constructed. Per-task accumulators use parameter dtype; FP64 dot products are reporting only. Floating-point per-task backward/accumulation is not claimed bitwise identical to one combined DeepSpeed backward, and this tool intentionally rejects distributed use.

3. `preserved_rng` restores Python, NumPy, CPU/CUDA and named generator states in `finally`. Both normal and injected-failure tests pass. Full parameter bytes, persistent/nonpersistent buffers and module modes are checked before/after measurement. Preexisting `.grad` is rejected rather than cleared. Mutation aborts the diagnostic; it does not pretend to repair a mutated copy. The standalone loader uses a separate model, so no active training optimizer/RNG/cursor is touched.

4. `shared_groups` selects every parameter of the actual last language block and `foresight_queries`. Exact names/dtypes/shapes/hashes are reported. Norms/cosines are for this declared subset, not the entire shared VLM. A missing individual path is recorded; a zero required auxiliary gradient group, zero valid count, zero required loss, or nonfinite required term does not yield guessed weights. An independent test verifies that nonzero union norm cannot conceal a zero W-group gradient.

5. The coefficient rule is explicitly a local hypothesis, `min(current_loss/task_loss, current_subset_norm/task_subset_norm)`. It is not GradNorm training or an optimality claim. The original contract §6.3 permits this one-shot fixed-training-batch scale measurement. The current diagnostic does not apply any proposed weights, modify temporal trainer configs, select by dev scores, or launch formal work. Scientific use still requires the selected spatial setup and common reference/batch to be locked before measurement.

6. Common S0 uses `read_binding(..., verify_files=True)` and `load_student`, reconstructing FP32 masters and strict state loading before the existing registered ST/history initialization. This is an allowed common learned initialization. Native ST loading also uses strict reconstruction and rejects architecture differences and learned SMOKE/PROFILE sources. This review checked the code and small binding checksum only; it did not read or load actual weights. No `strict=False`, arbitrary checkpoint loader, or external model teacher was added.

7. The exact diagnostic controls bind four unique training tokens and indices, separate patch/horizon/noise/time/dropout seeds, original input/teacher contracts, existing immutable targets, microbatch1/global4, original20k/warm1000 reference, official BF16/FA2 runtime and no recomputation. No optimizer schedule is executed. The measurement needs a separate exact-source admission and an external owner enforcing its resource limits. A controller token is an ownership marker, not independent proof of qualification.

## Actual tests and resource evidence

Command (from the isolated review worktree):

```bash
taskset -c 10,11 env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python \
  reports/st_world_policy/20261017/temporal_scale_independent_correctness_v1/run_cpu_review.py
```

Current result: **26 passed**, 3.42 seconds of pytest time; monitored invocation 4.723 seconds, peak RSS647,626,752 bytes, logical reads140,168,030 bytes. Tests consist of the original23 plus3 independent behavior/negative tests. GPU hidden, CPUs10/11, one BLAS thread. PyTorch emitted its existing pynvml deprecation warning.

The first invocation failed during collection because this review's script entrypoint omitted the repository from `sys.path`. Its complete log and costs remain as `ATTEMPT1_*`. Only the review harness was corrected. Across both invocations: logical reads159,656,528 bytes, monitored wall6.834 seconds, peak RSS647,626,752 bytes; all remain within the assigned120s/4GiB/256MiB/16MiB scope. No model/image/teacher payloads, GPU, optimizer updates, SSH, claim, ledger or dispatch were used. Small pinned controls were separately checksum-verified without reading referenced payloads.

## Boundaries

- Real Qwen/DiT/teacher input, FA2 multiple backward, actual last-layer gradients, full-model hash cost and runtime memory are **NOT_RUN** here. The independent resource review and admitted real diagnostic must resolve them.
- Canonical strict-load and BF16 execution conversion are inspected call paths, not new measured integration evidence.
- The new tests are analytic CPU tests using real loss/forward code with small components. They are not presented as real data qualification.
- No weight proposal or scientific gain exists yet. Zero or disconnected real signals must retain UNCALIBRATABLE/failure evidence rather than be replaced by arbitrary coefficients.
- This review is tied to the exact source/config hashes in `REVIEW.json`; changes in those relevant paths/configurations require impact review. Documentation-only integration does not require a whole-repository re-audit.
