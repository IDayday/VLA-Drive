# DDP SimWAM longitudinal-anchor follow-up

Status (2026-10-15): CPU contract, source-form anchor arithmetic and real eight-GPU interruption/resume qualification PASS. Formal training started from fresh F-SFT on `training-vlawm-tsinghua-zt`, immutable source `368aba3b3b00b877b7329b15f72645c41fcb6804`. No new policy improvement is claimed before evaluation.

This is the user-approved follow-up to the completed R_SOURCE/R_CANONICAL comparison. One new R_CANONICAL_X1 run starts from the same F-SFT, LoRA initialization and hard population. The only scientific configuration change is native anchor weights `[0.2,1,0.5,0.5]` to `[1,1,0.5,0.5]`. The source weighted-average denominator remains active: longitudinal effective weight is multiplied by 11/3, lateral and heading-pair weights by 11/15. This is not a pure longitudinal coefficient change at fixed other effective coefficients.

The original two recipes remain under their full immutable configuration fingerprint. The new loader validates X1 against the entire original canonical recipe plus exactly the registered change. Learning-rate, seed, budget, clipping and distributed mutations are rejected. All historical results and running worktrees remain unchanged.

See [registration](registration.json), [CPU evidence](cpu_contract.json), and the parent [audit](../ddp_simwam_audit_recovery_20261015/README.md). The prior control remains fixed; there is no new seed, best-of-N, new reward or optimizer matrix.

Budget: 20,000 AdamW updates, 5,000 fresh global rollouts, 40,000 scene exposures, 320,000 candidates. Seed 42. Fixed 5k/10k/15k/20k DEV and TRAIN diagnostic readouts use seed 2101; original uniform ODE and shift5 ODE are reported separately. Final Navtest uses the fixed 20k endpoint after weight/code lock, with the same two grids and seed; reuse existing matched SFT and canonical controls.

The user-requested [multi-source Flow design review and completed TRAIN sampler probes](flow_design_review/README.md) are read-only diagnostics, separate from this formal control. They scored 3072 matched sampler outputs plus 1024 saved-chain final-innovation counterfactuals; no additional training algorithm was launched.

The [detailed Chinese report assembled on 2026-10-16](flow_design_review/multi_source_analysis_20261016.md) consolidates source versions, sampling and loss semantics, LoRA and action-scale checks, completed 20k results, diagnostic evidence, costs, limitations, and the next single-factor hypothesis. Its X1 progress is explicitly timestamped: training has reached its registered 20k endpoint; final evaluation is pending. The [evaluation recovery record](evaluation_recovery_20261016/README.md) explains the source-hash comparison failure, preserved completed results, and the qualified controller restart.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.simwam_recipe.qualify_x_anchor --output /path/to/new_cpu_contract.json
python -m scripts.analysis.simwam_recipe.qualify_recovery_run --name eight_gpu_x1 --root /mnt/project/simwam-x-anchor-20261015 --config configs/flow_rl/simwam_canonical_x1_ddp.yaml --interrupted-after 2
```

The second command must run on an authorized eight-GPU host from the frozen source snapshot. Qualification weights never initialize the formal run. No claimed final driving benefit follows from contract or recovery tests.

The qualification interrupted after inner 2 and resumed the remaining inner updates. At update 8, all eight ranks matched uninterrupted adapters, Adam/scheduler states, RNG, next fresh chain, old probability, advantages, losses and gradient norms exactly. A merged update-8 qualification export also ran from two real TRAIN observations, with native ODE identical to the original deployment entry; these weights do not initialize formal training.

The finite milestone controller uses Tsinghua zt3 for DEV and zt4 for the fixed TRAIN diagnostic cohort, eight workers per host. Its final controller waits for all four registered milestones and the complete 20k checkpoint before locking the two-grid, seed2101 Navtest. Existing SFT and R_CANONICAL results are reused as matched controls. Controller changes do not alter actor, sampler, scoring or the running source snapshot.

```bash
python -m scripts.analysis.simwam_recipe.watch_evaluations --snapshot /path/to/frozen/controller/source --administrative-local --followup-x1
python -m scripts.analysis.simwam_recipe.final_navtest serve --snapshot /path/to/frozen/controller/source --administrative-local --followup-x1
```
