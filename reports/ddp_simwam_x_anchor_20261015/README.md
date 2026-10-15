# DDP SimWAM longitudinal-anchor follow-up

Status: CPU contract and source-form anchor arithmetic PASS; real eight-GPU qualification and formal training pending. No new policy result is claimed.

This is the user-approved follow-up to the completed R_SOURCE/R_CANONICAL comparison. One new R_CANONICAL_X1 run starts from the same F-SFT, LoRA initialization and hard population. The only scientific configuration change is native anchor weights `[0.2,1,0.5,0.5]` to `[1,1,0.5,0.5]`. The source weighted-average denominator remains active: longitudinal effective weight is multiplied by 11/3, lateral and heading-pair weights by 11/15. This is not a pure longitudinal coefficient change at fixed other effective coefficients.

The original two recipes remain under their full immutable configuration fingerprint. The new loader validates X1 against the entire original canonical recipe plus exactly the registered change. Learning-rate, seed, budget, clipping and distributed mutations are rejected. All historical results and running worktrees remain unchanged.

See [registration](registration.json), [CPU evidence](cpu_contract.json), and the parent [audit](../ddp_simwam_audit_recovery_20261015/README.md). The prior control remains fixed; there is no new seed, best-of-N, new reward or optimizer matrix.

Budget: 20,000 AdamW updates, 5,000 fresh global rollouts, 40,000 scene exposures, 320,000 candidates. Seed 42. Fixed 5k/10k/15k/20k DEV and TRAIN diagnostic readouts use seed 2101; original uniform ODE and shift5 ODE are reported separately. Final Navtest uses the fixed 20k endpoint after weight/code lock, with the same two grids and seed; reuse existing matched SFT and canonical controls.

```bash
CUDA_VISIBLE_DEVICES='' python -m scripts.analysis.simwam_recipe.qualify_x_anchor --output /path/to/new_cpu_contract.json
python -m scripts.analysis.simwam_recipe.qualify_recovery_run --name eight_gpu_x1 --root /mnt/project/simwam-x-anchor-20261015 --config configs/flow_rl/simwam_canonical_x1_ddp.yaml --interrupted-after 2
```

The second command must run on an authorized eight-GPU host from the frozen source snapshot. Qualification weights never initialize the formal run. No claimed final driving benefit follows from contract or recovery tests.
