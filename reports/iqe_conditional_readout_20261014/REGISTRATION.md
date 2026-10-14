# Candidate-conditioned visual reading: bounded follow-up

The original S00–S11 and R00–R11 matrices are complete and remain immutable.
The active user goal requests further evidence-based work toward a reliable
internally trained Scorer. This is a separately registered follow-up under that
goal, not an unreported extension of either matrix. No Base/expert training,
navtest, external Scorer loading, precision change or ACTIVE update is allowed.

The immutable registration was written before optimizer execution at
2026-10-14T21:22:47.406966+00:00:
`/mnt/project/iqe-conditional-readout-20261014/v1/EXPERIMENT_BINDING.json`.
It binds the source Base70k campaign, full D_S and stage_val banks, fixed
four-candidate pool, old S11 recipe and random initialization. The committed
training source and per-file checksums are recorded separately in RUN_IDENTITY.
Previously observed development outcomes informed this research direction;
the development set is not claimed to be untouched or independent of the
research process. This follow-up's checkpoint selection uses stage_val only.

## Evidence and intervention

R00 is only +0.1658 PDMS points on 1696 development scenes, with a log-bootstrap
interval crossing zero and 11 high-to-zero replacements. All four R variants
overestimate genuine zero-scoring development candidates. Existing rich-bank
analysis already identified close geometric pairs with different safety outcomes;
there is no basis to assume its only negatives are obvious bad trajectories.

The completed S11 visual reader changed its weights, but zeroing or mismatching
V changed none of the selections in the recorded 16-fit/16-validation diagnostic.
This is limited evidence of weak use of that channel, not proof that images are
uninformative. The new reader changes one factor: its private visual queries
also receive the actual candidate's time-token embeddings. Visual projection
is still once per scene; per-candidate attention and its added cost are explicit.
No calibrated ground-to-image correspondence is claimed by this construction.

The unchanged control is StudyScorer S11. The new module has exactly the same
parameter names, shapes, count (4,898,311), seed and initial hash:
`7d484e0c72a87bf16f86fd8fa803c612c75a13e2f2fd5de91ab725f022a8dd57`.
The trained S11 weights are never loaded. Both initializations were checked
against real source contracts and cached W/A/V on 2026-10-14.

The additional read-only R10 gradient audit used 128 natural D_S scenes and
zero optimizer steps. Shared safety-event versus positive-value gradient cosine
was -0.0373 at initialization, +0.4309 at the selected 1000-step checkpoint and
+0.0813 at 6000. This does not establish persistent destructive interference;
we therefore do not introduce independent metric networks as the next change.
Evidence: `mean_risk_v1/qualification/pointwise_gradient_conflict/REPORT.json`.

Trajectory-aware feature interaction is motivated by a primary-source example,
[SparseDriveV2](https://arxiv.org/html/2603.29163v1). We do not reproduce its
trajectory vocabulary, architecture, weights or published results; the present
experiment is the narrow private-query change described above.

## Fixed budget and interpretation

* Two 300-update diagnostics: original global reading and conditioned reading,
  on the same fixed 64 non-tie real D_S scenes. Each uses the existing rich bank,
  global batch 128, microbatch 16, random seed 20261014 and warmup 100. Neither
  diagnostic checkpoint is eligible to initialize or replace the formal model.
* At most one new 6000-update formal conditioned Scorer, after the diagnostics
  establish valid finite training and input use. It starts again from the same
  random initialization; it is compared with the completed S11 control.
* Formal recipe remains S11: first 3000 natural-rich updates; last 3000 deploy
  natural/pair updates, unchanged SmoothL1 + 0.2 components + 0.2 ranking,
  AdamW 1e-4, weight decay 0.01, warmup100 then constant, batch128, clip1.
* Validation at 3000/4000/5000/6000; maximum raw-argmax PDMS, then fewer new
  NC/DAC failures, then earlier step. Record both fixed 6000 and selected endpoint.
  Original raw argmax and optional independently calibrated margin stay separate.
* No further architecture, objective, candidate-pool or hyperparameter search is
  admitted by this registration. A failed diagnostic or heldout result is retained.

Same-parameter count does not mean same FLOPs: the conditioned reader attends
separately for each candidate. Cached-head timings are not image deployment
latencies. All comparisons retain a single frozen visual/language execution.

## Commands

The completed diagnostics now have a negative budget-expansion decision (see
`RESULT.md`). The formal train command below is retained for interface/history
documentation and deliberately exits nonzero for this experiment identity.
Do not remove or overwrite the decision to rerun an unregistered trial.

Run from `/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014` on a currently
permitted host and verified free GPU; never use protected vla-zt2. The runtime
resource checks are repeated by the CLI. No temporary-host lease is needed.

```bash
PY=/root/miniconda3/envs/ddp/bin/python
CFG=configs/iqe/conditional_readout/followup_v1.yaml
$PY -m iqe.conditional_readout.cli bind --config "$CFG"
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.cli diagnostic --config "$CFG" --mode global --device cuda:0
CUDA_VISIBLE_DEVICES=1 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.cli diagnostic --config "$CFG" --mode conditioned --device cuda:0
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.cli train --config "$CFG" --device cuda:0
# Only after an actual interruption; full sampler, optimizer and RNG are restored.
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.cli train --config "$CFG" --device cuda:0 --resume auto
# After the formal fixed budget completes; lock on stage_val, calibrate, then dev.
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.evaluation evaluate --config "$CFG" --device cuda:0
$PY -m iqe.conditional_readout.bundle --config "$CFG"
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.qualification online --config "$CFG" --device cuda:0
# Run as a separate CLI process after online writes its fixed reference.
CUDA_VISIBLE_DEVICES=0 IQE_SPATIAL_READER_MODE=mmap_v1 OMP_NUM_THREADS=2 $PY -m iqe.conditional_readout.qualification roundtrip --config "$CFG" --device cuda:0
```

## First real diagnostic and deterministic implementation repair

The global control completed all 300 updates on rl-zt2 GPU0. On its fixed
64 training scenes, deployment-pool Base was 93.5053, Oracle 96.9317 and the
diagnostic selected score 94.1814. Wrong-scene V changed zero choices. This is
training-set evidence only and the model is not eligible for deployment.

The first conditioned attempt failed before any optimizer update because CUDA
`upsample_linear1d_backward` has no deterministic implementation. Its records
are retained in `failed_attempts/conditioned_cuda_interpolate`. After the global
control exited, the operator was replaced with the same fixed half-pixel linear
interpolation matrix and GEMM backward. The deterministic runtime was retained.
CPU float64 value/derivative tests cover 8-to-16 and other time dimensions at
1e-12 tolerance; the failure does not justify claiming GPU backward passed.
The next real diagnostic separately verifies that. No weights or sample cursor
were discarded, because this failed attempt completed zero updates.

The strict roundtrip test compares identical eval/requires_grad flags on both
sides. An initial comparison with mismatched flags was not bitwise equal;
the corrected test aligns those execution conditions and passes without
relaxing tolerance. It does not establish bitwise equality between arbitrary
attention execution paths.

The repaired model has completed real GPU optimizer updates under the original
deterministic runtime; all 86 trainable parameter tensors had finite gradients
at its first update. The focused regression run is 23 passed / 1 skipped
(the inherited opt-in two-GPU terminal-resume test was not run again).
Final diagnostic results and the decision about formal training remain separate
from those engineering checks. New online/export entrypoints are implemented
and unit checked; their trained real-image evidence requires the formal endpoint.
