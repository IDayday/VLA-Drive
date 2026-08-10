# Field2Plan Phase 3 Formal Experiment Matrix

Date: 2026-08-09 UTC

## Fixed scientific contract

Phase 3 keeps the frozen 193514 proposal and the Phase 2 geometry path fixed.
The only new source is an **offline** V-JEPA 2.1 latent cache. Training never
imports or executes V-JEPA and the action-free writer receives only the current
visual geometry field plus frames 0--3 historical ego transforms. Frames 4--11
and their images are used only to construct detached supervision targets.

All formal jobs use one node, 16 accelerator processes, per-device batch 2,
gradient accumulation 1, and effective batch 32. Checkpoints are emitted every
10,000 optimizer steps. Each experiment has a thin one-container launcher so
that DLC retries cannot silently change the scientific arm.

## First-round groups

| Group | Script | Geometry S/A | Dynamics S/A | Control | Question |
|---|---|---:|---:|---|---|
| P3-D00 | `train_p3_dyn_nosup_noaccess.sh` | 0/0 | 0/0 | equal capacity | Can added dynamics/refiner capacity alone explain a gain? |
| P3-D11 | `train_p3_dyn_only_real.sh` | 0/0 | 1/1 | aligned V-JEPA | Does an action-free dynamics prior help without geometry supervision/access? |
| P3-GD11 | `train_p3_geo_dyn_real.sh` | 1/1 | 1/1 | aligned DA3 + V-JEPA | Main factorized geometry+dynamics model. |
| P3-D10 | `train_p3_dyn_sup_noaccess.sh` | 0/0 | 1/0 | aligned V-JEPA | Is an auxiliary dynamics objective sufficient without planner access? |
| P3-D01 | `train_p3_dyn_access_nosup.sh` | 0/0 | 0/1 | equal capacity | Can planning loss alone train the dynamics field/reader? |
| P3-GD-TShuffle | `train_p3_geo_dyn_temporal_shuffle.sh` | 1/1 | 1/1 | future time shuffled | Does correct future time alignment matter? |
| P3-GD-BShuffle | `train_p3_geo_dyn_batch_shuffle.sh` | 1/1 | 1/1 | scene shuffled | Does scene-aligned future information matter? |

Here `S/A` means supervision/access. A shuffled demonstrated future is used
only as a negative/control supervision target; it is never treated as the
counterfactual outcome of a candidate trajectory.

## Cache command (one 16-card DLC container)

Run this once before any Phase 3 training job:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && \
bash scripts/field2plan/11_cache_dynamics_vjepa.sh
```

Expected output:

```text
/mnt/zhangt_workspace/project/DriveDreamer-Policy/field2plan_cache/
  dynamics_vjepa2_1_vitl384_c96_16_v1/
```

The launcher hashes the 5.15 GB checkpoint before allocating work, pins the
external repository commit, writes one atomic NPZ per token, consolidates a
manifest only after all 16 independent shards finish, and then validates every
entry with the runtime reader.

## One command per training container

After the cache command reports `complete`, run any arm in a separate 16-card
DLC container:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_dyn_nosup_noaccess.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_dyn_only_real.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_geo_dyn_real.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_dyn_sup_noaccess.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_dyn_access_nosup.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_geo_dyn_temporal_shuffle.sh
```

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && FIELD2PLAN_RUN_SEED=42 bash scripts/field2plan/train_p3_geo_dyn_batch_shuffle.sh
```

## Incremental Phase 3 evaluation

Once Phase 3 checkpoints begin to appear, one separate 16-card DLC container
can watch and evaluate all seven arms. Results are written checkpoint by
checkpoint, so the summary remains readable before the full queue finishes:

```bash
cd /mnt/zhangt_workspace/project/DriveDreamer-Policy && \
bash scripts/field2plan/15_eval_phase3_ckpts_16gpu.sh
```

The live NAVSIM v1.1 PDMS summary is:

```text
/mnt/zhangt_workspace/project/DriveDreamer-Policy/navsim_exp/
  field2plan_phase3_eval_16gpu_live/
  navsim_v1_1_pdms_ws2_seed20260808-distenvfix-v1/summary/summary.md
```

This launcher intentionally does not claim navhard support. The checkout has
raw `navhard_two_stage` assets but no validated processed metadata/datalist
matching the current inference dataset contract. A navhard job must fail fast
until those assets are generated and checked; navtest scores alone do not
satisfy Phase 3 scientific acceptance.

## Promotion and early-stop rule

Training loss alone is not a stop criterion. A stop recommendation requires at
least two matched checkpoints (normally 10k and 20k), complete fixed-seed
navtest predictions, valid NAVSIM v1.1 PDMS/component results, and the dynamics
probe metrics from the same run.

- Keep `P3-D11` and `P3-GD11` running unless they are technically invalid
  (non-finite gradients/losses, corrupt cache) or both 10k and 20k are clearly
  worse than their matched Phase 2 reference with no positive trend.
- A capacity-only or shuffled control may stop after 20k when it is dominated
  at both checkpoints and its paired per-scene 95% bootstrap upper bound cannot
  reach the real-teacher arm. This is evidence of futility, not a failed run.
- Do not stop an arm from one aggregate score, one checkpoint, or auxiliary
  loss magnitude. Do not promote a model unless aligned-teacher probe similarity
  beats temporal- and scene-shuffled controls and gains also appear on navhard.
- Repeat only the best two or three arms with seeds 43/44 after the seed-42
  screening round.

The current Phase 2 evidence and paired intervals are recorded in
`docs/field2plan/PHASE2_INTERIM_EVIDENCE.md`. Several valid local 10k/20k
results now exist, but none supports stopping an unfinished arm; the first DLC
live evaluator remains invalid because it inherited distributed environment
variables.
