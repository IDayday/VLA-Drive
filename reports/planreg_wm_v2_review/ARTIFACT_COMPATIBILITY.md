# Artifact compatibility, not silent resume

Tested source: `1e01bcbd9e0d2baf668a103ea969ae5e5d980c64`. Existing V1, original V2, old input-cache roots, shared init and checkpoints are preserved. Old execution remains in its original worktree; this review does not pretend incompatible V2.1 artifacts are V2.2 resumes.

| Contract | V2.2 identity | Rejection / migration rule |
|---|---|---|
| Architecture | `planreg_wm_v2.2` | Explicit construction only |
| Long target | `long2_progressive_not_a_knot_actual_time_v2` | Recompute from actual poses; reject uniform/linear labels even if manifest is relabeled |
| Input cache | `planreg_v2_input_0138_progressive_long_v2` | Version on manifest, per-scene item and target; no dynamic visual/EMA features |
| Shared init | `planreg_v2_shared_fp32_std02_v2` | Validate architecture, register std, recipe, names/shapes/dtypes and tensor hashes; reject old std1e-6 artifact despite matching shapes |
| Raw-GT statistics | `planreg_v2_raw_gt_statistics_v2` | Token set + actual original GT content + heading/statistics contract hashes; long never enters original-GT statistics |
| LR recipe | `planreg_v2.2_gb32_sqrt_capped_v2` | Fixed-LR optimizer state cannot silently resume; use an explicitly declared warm start |
| LR schedule | `warmup05_cosine10_fixed_budget_v2` | Effective batch includes accumulation; stop limit does not replace schedule length |
| Training checkpoint | `planreg_v2_training_v2` | Restore model, optimizer, scheduler, EMA/master, normalizers, RNG, sampler and run identity |
| Deployment checkpoint | `planreg_v2_student_v2` | No teacher/predictor construction or training I/O; normalizer retained |

## New bounded artifacts

Root: `/mnt/project/DriveVLA-M0-stage2/planreg_v2_review_20260908`.

- `input_train48`: 48 unique scenes, 12 recorded drives; `input_probe8`: eight scenes, two other drives. All 56 labels were regenerated from real raw logs using the final source and compared bitwise with cached targets; raw-GT statistics also matched (`CACHE_REVALIDATION.json`). Old caches were not overwritten.
- Training token SHA-256: `279736bd5f0478dc265b8ac61eebbf43a941fbc80e9cdcdea25cd3c2a6071cec`.
- Original-GT content SHA-256: `098a65354ba8d0ab1ac556d51afbafcc80cfa12a458ba50d66d6dd08d47301e5`.
- `shared_init_std02_seed0.pt` SHA-256: `2864c09016be1b9be63d738fc1c194eb3676dcee3188dc27d933d2d89593b047`.
- Shared bank has 585 trainable tensors. No-WM selects its 546 common tensors without a predictor; TF-only and compact-memory each load all 585. Actual Base/VQA constructions match bitwise. A same-seed claim alone was not used.
- Actual statistics std floor is 0.001, triggering fraction zero in this 48-scene set. The artifact is **smoke-only**, not full 103k statistics. Old raw-GT statistics are not automatically approved; reuse requires the above contract proof for the exact new final-fit token set.
- Base and dense driving-VQA paths, fingerprints, tokenizer IDs and architecture audit are in `VLM_PAIR_AUDIT.json`; no VQA training run was performed.

## Explicit V1 warm start

The real source is the old epoch27 checkpoint under `formal_dual_init_gb128_asyncpdm_20260903`. Migration coverage and its SHA are in `MIGRATION_COVERAGE.json`. All declared compatible modules must be accounted for, not merely accepted by a target strict load. Generator decoder and final head4 are mapped; the physical output layer and ego input columns are transformed consistently. New semantic/memory parameters stay new. Teacher is rebuilt from the loaded student. Legacy BF16 EMA lost increments are unrecoverable; old optimizer/scheduler state is not resumed.

This migration is for explicitly labeled diagnostics/warm start, never a VLM-only formal main result and never whole-agent V1/V2 parity.

## What the deployment artifact does and does not contain

Student keeps frozen VLM weights, visual and language LoRA, soft queries, rich memory, the shared generator head, scorer and trajectory normalizer. It drops internal teacher, FP32 EMA master and WM predictor plus optimizer/scheduler training state. The constructor does not instantiate the removed modules.

Real AgentInput replay succeeds with teacher/predictor construction, PDM, future trajectory and all image-file opens trapped (input contains an already decoded current image). The exported weights are not completely installation-independent: the declared InternVL trust_remote_code structure/config and tokenizer files remain necessary. This is recorded instead of claiming a fully self-contained executable.

## Formal readiness remains conditional

No formal run may load the 32-step smoke checkpoint. Generate the full progressive-long cache and matching 103,288-scene statistics, re-audit named initialization against that resolved architecture, and verify full-data tile lengths fit the measured layout. The preprocessor can admit up to 12 crops plus thumbnail; the measured profile covers nine tiles, not an unmeasured 13-tile worst case. A source-bound GB128 lock is evidence for its recorded envelope, not permission to extrapolate it.

Neither a GB1/GB4 smoke nor the intentionally aborted one-GPU attempt is a layout pass. No new long training, no-WM/TF-only/compact training, full Navtest or new performance result is included.
