# Execution status and limits

## PASS — new executions

| Check | Evidence / result |
|---|---|
| Complete repository tests | `pytest -q tests`, see final count in `TEST_RESULTS.json` |
| Static checks | `compileall navsim/agents/EpisodeDrive`, `git diff --check`, launcher `bash -n` |
| Exact scorer | Six logits and log-PDM max_abs_diff=0; indices equal; proposal.grad=None; scene gradient norm≈18.20 |
| Legacy real model | Epoch33, four old bank scenes, FP32: all proposal differences=0, selected indices equal |
| Shared new initialization | Actual Base and Driving-VQA InternVL3-2B, 465 effective trainable tensors bitwise equal |
| Actual real updates | 150 repeated-two-scene numerical updates; all 465 trainable tensors changed; all 24 LoRA blocks have nonzero gradients |
| Precision | 23,926,559 trainable FP32 values, 47,853,118 actual FP32 Adam moments; 121 FP32 EMA master tensors |
| Online teacher drift | Real future targets differ from initialization (RMS≈.232); master/forward-copy values match exactly |
| Deployment | Real student-only export and reload; no EMA/predictor construction; current-only trajectory max_abs_diff=0 |
| Same-batch audit | Real model after 150 updates; no optimizer .grad pollution; CPU two-rank DDP pending-backward/RNG test |
| Input cache | Two real input-only records rebuilt in a separate root with single-front prompt hashes |
| Scorer learnability | 16 train-log scenes, frozen upstream, 150 additional exact scorer updates; no Navtest labels |
| Config pair | Real resolved primary configs differ only on six allowed VLM/experiment/output identity paths |

The 150-update numerical run uses **microbatch two**, retains the proposed GB64
LR/schedule reference, and repeats the same two train-log scenes. It is not a
formal GB64 training run, a representative convergence comparison, or a valid
distributed throughput estimate. Total loss decreased 29.0117→18.3754; unweighted
WM loss changed 0.1110→0.1372. Teacher targets move, and the joint objectives need
not decrease together. Do not claim monotonic WM improvement.

At its last same-batch audit, current lambda≈0.01308. WM/planning gradient norm
ratios were ≈0.382% for vision LoRA and ≈0.380% for registers, with cosines≈0.664
and≈0.249. These are pre-clipping, same-batch, same-parameter comparisons—not
loss-value fractions. At lambda=.10 the same fixed gradients would give ≈2.92%
and≈2.91%; that rescaling is arithmetic, not a newly trained .10 experiment.

Frozen predictor control cosine losses: correct≈.0837, action-only≈.8576,
shuffle-current≈.1066, copy-current≈.1056; target variance≈.0535. These controls
on two seen scenes are not proof of general world-model quality or PDMS benefit.
The final-path peak allocated GPU memory was ≈18.70 GiB. No speedup claim is made
against the initial uncheckpointed diagnostic because its memory path differs.

## NOT_RUN / NOT_BUILT

- Full V1.1 103,288-record prompt-versioned input cache: only the two-record cache
  smoke was built; the full copy/consolidation command is supplied separately.
- GB64 8×8 vs 16×4 and GB128 full-chain throughput benchmarks: not run. They need
  that new full cache and a representative sampling manifest. No old V1 throughput
  metrics are relabelled as V1.1, and no new layout lock has been issued.
- Equal-exposure GB64/128 train-only convergence comparison and representative
  log-split WM-weight pilot: not run. Two-scene numerical tests cannot substitute.
- Full 16-GPU real-model DDP/resume/throughput: not run; CPU two-rank hook coverage
  and real single-GPU parameter/checkpoint coverage are reported separately.
- V1.1 full Base/VQA 27-epoch training, multi-seed pairing and full Navtest PDMS:
  not run; this code task must not implicitly launch all multi-day experiments.
- New formal replay sampler or auxiliary-head training: not enabled. Standalone
  train-only bank tooling/probe and opt-in light head supervision exist, but no
  train-only generalization evidence yet promotes either change.

New formal launchers intentionally reject the old V1 layout/cache and require a
new train-only-pilot-backed lock. They are not ready to claim a promoted formal
training configuration merely because the unit and numerical tests pass.

## Artifacts and scientific boundaries

External raw artifacts are under
`/mnt/project/DriveVLA-M0-formal-runs/v1p1_audit_20260905`; full weight files and
frozen candidate features stay outside Git. Lightweight source-fingerprinted JSON
reports are committed here. Failed attempt logs are retained and described in
`PRECISION_AUDIT.md`. No original worktree, checkpoint, bank or image was replaced.

Old reported results, new offline bank calculations, source fixes, synthetic
tests, and new real-model experiments are deliberately labelled separately.
The NAVSIM evaluation skill was used to preserve the scorer/official-PDMS/oracle
boundary, immutable banks and train-only probe provenance. No full benchmark
promotion is inferred from these diagnostics.

Multi-trajectory consequence modeling remains **not implemented**. Both formal
initialization variants retain WM enabled, K=1 GT-only future supervision and
current-only deployment.
