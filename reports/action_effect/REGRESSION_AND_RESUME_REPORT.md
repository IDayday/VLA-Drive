# Regression and actual interrupted-training evidence

This report consolidates completed evidence; it does not request or execute another
recovery experiment. Exact artifact hashes and original command logs are indexed
in [EVIDENCE_INDEX.json](../effect_repair/EVIDENCE_INDEX.json).

## Current CPU source

At source **b1a5438f**, the Lead ran the complete `tests/action_effect` suite:
**374 passed,0 failed,0 skipped,1 warning,44.01s**. The full original log is copied
without editing to
[REGRESSION_CURRENT_B1A5438F.log](../effect_repair/current_regression_b1a5438f/REGRESSION_CURRENT_B1A5438F.log).
The sole warning concerns deprecated pynvml packaging, not a failed numerical
check. Independent new-control scope is separately16 passed/4.36s, retained in
[INDEPENDENT_CONTROL_CPU.log](../effect_repair/current_regression_b1a5438f/INDEPENDENT_CONTROL_CPU.log).

These groups overlap earlier253-test and focused reviews. Counts are not summed.
Earlier source091d617a had253 passed/24.61s; later2ba24432 additions had19 passed/
4.60s. Their original logs remain individually pinned. Test progression is
historical source coverage, not scientific effect evidence or a new result
inherited by an arbitrary later revision.

## Actual CUDA same-runtime resume

The completed experiment source is **57dbe933776c86935e60368e9c493d86a844c8a4**.
It loaded real S0 current-context caches and actual native TRAIN64 effect pairs,
not tensor substitutes. Three independent CLI processes ran continuous16,
split8, then resumed8. The full1000-step scheduler horizon and warmup remained
unchanged; only the segment limit bounded new updates. Total scheduled updates
were32, not an additional1000-step trial.

The independent CPU audit re-read five complete checkpoints and their provenance.
It reconstructed next-update9's scene IDs and pair IDs from both consumed sampler
states. It compared typed exact bytes of private model parameters, complete Adam,
scheduler, scene sampler, pair RNG, Python/NumPy/CPU/CUDA RNG, consumed counters,
losses, gradient norms and learning rate. The first resumed update9 and final16
were exactly equal on the same A80080GB/CUDA12.4/PyTorch2.5.1 deterministic FP32
backend. The independent audit added zero GPU calls or optimizer updates and did
not overwrite the author's original result.

Actual root:
`/mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1`.
Primary result is `RESULT.json`; independent audit is
`INDEPENDENT_CPU_AUDIT_v2.json`, content identity
`0adce9f805d94a1c7b9f2d324f1bea3f886cafa0c1770c3e6788152511648f79`.
The committed complete audit is
[G1_REAL_RESUME_INDEPENDENT_AUDIT.json](../effect_repair/G1_REAL_RESUME_INDEPENDENT_AUDIT.json).

This supports same-runtime G1 exact recovery at those boundaries. It does not
prove cross-GPU/hardware bitwise equivalence, learned effect accuracy, successful
policy repair, every later ablation's exact state or a generic distributed trainer.

For an optional *CPU-only* repeat audit of existing evidence, use the audited
entry in an immutable source checkout and a new output receipt path:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.action_effect.audit_effect_resume \
  --qualification-root /mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1 \
  --expected-source 57dbe933776c86935e60368e9c493d86a844c8a4 \
  --config /mnt/project/VLA-Drive-action-effect-repair-20261020/configs/action_effect/main.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020/g0/population_000064/PAIR_DATASET.json \
  --arm delta_feedback --seed 42 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/g1_real_resume16_v1/INDEPENDENT_CPU_AUDIT_RECHECK.json
```

This is a reproducibility command, not a newly executed audit. The old original
config/context/pair paths deliberately belong to the actual qualification; do not
substitute the newer1024 population and call it the same experiment. Resume tools
now also reject foreign COMPLETE reuse and a different actual training identity.

## Actual NCCL and action-head portability

The existing two-rank NCCL qualification used real context/pair labels and
compared all-valid and empty-label-rank reductions against one-device global
scene loss.62 gradient tensors per case satisfied preregistered
`atol=2e-6, rtol=2e-4`. Largest absolute discrepancy was4.8243e-7. The reported
relative maxima58/116 arise near cancellation/zero tensors and are not hidden;
bitwise NCCL equality is not claimed. This performed zero optimizer updates and
qualifies the loss/gradient collective component only.

`qualification/nccl_real64_v1/RESULT.json` and the exact original two-process
CLI command are pinned by the index and `resources/QUALIFICATION_COMMAND_0.json`.
No repeated NCCL/GPU trial was dispatched by this documentation work.

Two real independent-process action-head roundtrips on the ts2/ts5 hosts each
strict-loaded the same original S0 head and one real cached condition/noise.
The raw and physical outputs exactly matched the cached original S0. Each used
zero Qwen calls, zero simulation and zero updates. Their COMPLETE receipts are
`qualification/head_roundtrip_ts2_v1/COMPLETE.json` and the corresponding ts5
path. This is cached-head portability, not real-image latency or a learned policy
improvement. Final A/B image-to-final deployment still needs the actual bundles.
