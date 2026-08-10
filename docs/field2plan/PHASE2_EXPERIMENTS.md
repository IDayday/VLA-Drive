# Field2Plan Phase 2 Formal Experiment Matrix

Date: 2026-08-08 UTC

## Fixed contract

Each script launches exactly one experiment in one non-interactive DLC
container. All runs use one node, 16 accelerator processes, per-device batch
2, gradient accumulation 1, and effective batch 32. The frozen proposal is
the 193514 checkpoint cache with seed 20260808 and 10 flow steps. Default
training length is 100,000 optimizer steps, with checkpoints every 10,000
steps. Cache manifests are checksum-pinned into the saved run config.

The first round uses seed 42 only, so it occupies at most eight containers.
Seeds 43 and 44 are deferred until navtest/probe results identify the best
two or three configurations.

## Eight first-round groups

| Group | Script | Supervision | Planner access | Teacher/control | Scientific question |
|---|---|---:|---:|---|---|
| P2-00 | `train_p2_00_nosup_noaccess.sh` | no | no | equal-capacity head | Can extra parameters or an unconditional refiner explain gains? |
| P2-10 | `train_p2_10_sup_noaccess_da3.sh` | DA3 | no | real, aligned | Does auxiliary geometry learn without being exposed to the planner? |
| P2-01 | `train_p2_01_nosup_access.sh` | no | yes | equal-capacity | Can planning loss alone learn a useful visual field? |
| P2-11-DA3 | `train_p2_11_sup_access_da3.sh` | DA3 | yes | real, aligned | Main metric-depth Field2Plan effect. |
| P2-11-VGGT | `train_p2_11_sup_access_vggt.sh` | VGGT structure + DA3 scale | yes | real, aligned | Does VGGT multiview structure improve over plain DA3? |
| P2-Random | `train_p2_random_access_da3.sh` | synthetic random depth | yes | token-seeded | Is improvement merely an auxiliary-loss/regularization effect? |
| P2-Shuffled | `train_p2_shuffled_access_da3.sh` | DA3 from another batch item | yes | deterministic shuffled | Does scene-aligned geometry matter? |
| P2-StateMLP | `train_p2_state_mlp_access.sh` | no external teacher | yes | current-state-only MLP | Can current ego state and added capacity explain the result? |

The state MLP reads only the existing normalized current state `[B,1,4]`.
No control receives GT future action, demonstrated future images, evaluator
outcomes, or candidate-specific counterfactual labels.

## One command per container

Run each command in a separate one-node/16-card DLC container:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_00_nosup_noaccess.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_10_sup_noaccess_da3.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_01_nosup_access.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_11_sup_access_da3.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_11_sup_access_vggt.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_random_access_da3.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_shuffled_access_da3.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p2_state_mlp_access.sh
```

The default output paths are
`navsim_exp/field2plan-<experiment>-steps100000-seed42`. A `.field2plan_started`
guard prevents a DLC retry from silently restarting an incomplete run from
step zero, and `.field2plan_complete` prevents duplicate completed runs.

## Promotion gate

Do not select a winner from training loss alone. Each completed run needs:

1. geometry probe metrics and real-vs-random/shuffled comparison;
2. fixed-seed navtest inference for draft and final trajectories;
3. NAVSIM-v2 PDMS plus component metrics;
4. paired per-scene bootstrap versus the same frozen 193514 draft;
5. delta-norm, field-validity, safety-regression, latency, and memory checks.

Only the best two or three configurations are repeated with seeds 43 and 44.
Phase 3 remains gated on a positive aligned-teacher/access result rather than
on total PDMS alone.
