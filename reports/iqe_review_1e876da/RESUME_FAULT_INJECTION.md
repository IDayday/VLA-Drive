# Resume selection and exact continuation

`iqe.training.resume.resolve_resume` is the sole resolver used by CLI, workflow, Base and private-module trainers. It enumerates receipts and orphan step files, verifies bytes/checksum, actual state schema, step/filename/receipt consistency, dependency/run identity and resolved trainer config. Maximum compatible optimizer step wins. `result.json` is diagnostic evidence and cannot override newer valid state.

| Case | Reproduction / result |
|---|---|
| A | result1000 + completed2000 →2000, stale result reported, both Base and private schemas |
| B | stop→resume→new checkpoint→second interruption → newest complete progress; next IDs/loss/gradients/parameters/RNG equal |
| C | partial/corrupt latest rejected; no fallback unless `resume.allow_fallback: true`; no valid checkpoint never starts fresh |
| D | continuous vs optimizer-boundary interrupt/resume match; world-size change rejected |
| Explicit old | higher-step original outputs preserved; FORK_RUN_REQUIRED; new folder records lineage and uses historical best snapshot from that checkpoint |

Tests: `test_readiness_repairs.py`, `test_training_resume_ddp.py`. Historical best weights now have immutable step-specific snapshots; a legacy checkpoint lacking a required historical-best snapshot cannot be claimed an exact fork. This does not rewrite historical checkpoints.

Real diagnostic (`scripts/iqe/real_resume_probe.py`): actual image FeatureBundles plus audited GT, 2 continuous updates vs1+1; **next IDs, loss, gradient norm, actual gradient tensors, model parameters, Python/CPU RNG all equal**. See [result](evidence/REAL_EXACT_RESUME.json) and [resolution](evidence/REAL_RESUME_RESOLUTION.json). The actual interrupted checkpoint is `/mnt/project/iqe-review-runtime-1e876da/real_resume/interrupted/step_000001.pt`; the same folder now also holds completed step2, so explicitly resuming step1 there correctly requires a fork.

Real standalone `train-scorer --round3 --resume auto` resolves completed step2 and reuses the verified result. Its next IDs are null because there is no remaining update; this is not substituted for the real1+1 backward test. Both Gloo and NCCL two-process tests verify optimizer-boundary resume and unequal valid denominators. No accumulation-midstep guarantee is claimed; existing eight-GPU Base was not interrupted to perform a resume experiment.
