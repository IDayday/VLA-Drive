# DDP SimWAM longitudinal-anchor follow-up

Completed (2026-10-16): registered20k training, all four DEV/TRAIN stages and the fixed-endpoint full Navtest are complete. Native single-ODE PDMS is **89.3817**, +0.4701 versus SFT and +0.4300 versus the canonical control. NC/TTC still regress relative to SFT; the92 target is not reached. See the [full endpoint report](final_navtest_20261016/README.md) and [results/costs](final_navtest_20261016/results.json). All recorded X1 evaluation workers and controllers have exited; do not relaunch completed work.

This is the user-approved follow-up to the completed R_SOURCE/R_CANONICAL comparison. One new R_CANONICAL_X1 run starts from the same F-SFT, LoRA initialization and hard population. The only scientific configuration change is native anchor weights `[0.2,1,0.5,0.5]` to `[1,1,0.5,0.5]`. The source weighted-average denominator remains active: longitudinal effective weight is multiplied by 11/3, lateral and heading-pair weights by 11/15. This is not a pure longitudinal coefficient change at fixed other effective coefficients.

The original two recipes remain under their full immutable configuration fingerprint. The new loader validates X1 against the entire original canonical recipe plus exactly the registered change. Learning-rate, seed, budget, clipping and distributed mutations are rejected. All historical results and running worktrees remain unchanged.

See [registration](registration.json), [CPU evidence](cpu_contract.json), and the parent [audit](../ddp_simwam_audit_recovery_20261015/README.md). The prior control remains fixed; there is no new seed, best-of-N, new reward or optimizer matrix.

Budget: 20,000 AdamW updates, 5,000 fresh global rollouts, 40,000 scene exposures, 320,000 candidates. Seed 42. Fixed 5k/10k/15k/20k DEV and TRAIN diagnostic readouts use seed 2101; original uniform ODE and shift5 ODE are reported separately. Final Navtest uses the fixed 20k endpoint after weight/code lock, with the same two grids and seed; reuse existing matched SFT and canonical controls.

The user-requested [multi-source Flow design review and completed TRAIN sampler probes](flow_design_review/README.md) are read-only diagnostics, separate from this formal control. They scored 3072 matched sampler outputs plus 1024 saved-chain final-innovation counterfactuals; no additional training algorithm was launched.

The [detailed Chinese report assembled on 2026-10-16](flow_design_review/multi_source_analysis_20261016.md) consolidates source versions, sampling and loss semantics, LoRA and action-scale checks, completed20k results, diagnostics, costs and limitations. Its dated updates now include full Navtest. The [evaluation recovery record](evaluation_recovery_20261016/README.md) preserves the earlier source-hash comparison failure and qualified controller restart; it is historical, not a current fault.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.simwam_recipe.qualify_x_anchor --output /path/to/new_cpu_contract.json
python -m scripts.analysis.simwam_recipe.qualify_recovery_run --name eight_gpu_x1 --root /mnt/project/simwam-x-anchor-20261015 --config configs/flow_rl/simwam_canonical_x1_ddp.yaml --interrupted-after 2
```

The second command must run on an authorized eight-GPU host from the frozen source snapshot. Qualification weights never initialize the formal run. No claimed final driving benefit follows from contract or recovery tests.

The qualification interrupted after inner 2 and resumed the remaining inner updates. At update 8, all eight ranks matched uninterrupted adapters, Adam/scheduler states, RNG, next fresh chain, old probability, advantages, losses and gradient norms exactly. A merged update-8 qualification export also ran from two real TRAIN observations, with native ODE identical to the original deployment entry; these weights do not initialize formal training.

The original finite controller registration used Tsinghua zt3/zt4. Its qualified recovery used currently authorized zt/zt4 and completed the remaining stages and fixed20k, two-grid, seed2101 Navtest. Existing SFT and R_CANONICAL results were reused as matched controls. Controller changes did not alter actor, sampler, scoring or the immutable training source. The commands below are historical interfaces, not instructions to restart this completed run.

```bash
python -m scripts.analysis.simwam_recipe.watch_evaluations --snapshot /path/to/frozen/controller/source --administrative-local --followup-x1
python -m scripts.analysis.simwam_recipe.final_navtest serve --snapshot /path/to/frozen/controller/source --administrative-local --followup-x1
```
