# Requested interim full-navtest evaluation

Status at launch: **RUNNING; no RL score available yet**. This records an actual
job, not a claimed completed evaluation or a change in the training recipe.

The checkpoint is the latest COMPLETE seen at the initial request snapshot,
2026-09-20 00:34:56 UTC: `train/checkpoints/update_008700`. It was fixed before
viewing any new navtest result. Training on training-vla-zt2 continues untouched.

Evaluation uses all12146 official navtest tokens, original single-candidate
ten-step ODE and token-derived seed42, without proposal selection or smoothing.
Eight independent log-group shards run on training-rl-zt4 GPUs0–3 and
training-rl-zt2 GPUs0–3. Already occupied GPUs and unrelated jobs are untouched.
Original inference precision/processor/resolution/source hashes match the locked
SFT protocol. The EpisodeDrive64-proposal scorer workflow is not applicable to
this DDP policy evaluation.

Official v2 one-stage EPDMS is produced by the existing original evaluator.
Official v1.1 PDMS is scored from those same saved physical trajectories on16
CPU workers, reusing the genuine v1 metric cache. Neither reward nor metric
definitions are changed. The v1 scorer runs after inference on CPUs64–95 of
training-rl-zt4, with nested library thread counts1.

The same-seed complete SFT baseline is reused: PDMS0.8886436998476597 and
EPDMS0.8823780219885501. The comparison checks input identities, inference source
hashes and finite complete token sets, and produces paired means, log-cluster
bootstrap95% intervals, win/loss/tie counts, zero-score recovery and regressions.
Training-only checkpointing/ZeRO metadata do not redefine original ODE inference;
actual inference numerical settings remain required to match.

This is an interim **single-seed** monitoring evaluation. It does not replace the
registered final five-seed result and does not select learning rates, noise,
training seeds or checkpoints from navtest outcomes.

Actual entry point (the exclusive lock rejects duplicate controllers):

```bash
cd /mnt/project/DriveDreamer-Policy-action-batch
export PYTHONPATH="$PWD/navsim:$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/root/miniconda3/envs/ddp/bin/python -m scripts.analysis.intermediate_navtest \
  --spec runs/action_batch/qualified_w8_single/intermediate_step8700_seed42_spec.json
```

Live phase and final result:

- `runs/action_batch/qualified_w8_single/intermediate_step8700_seed42/progress.json`
- `runs/action_batch/qualified_w8_single/intermediate_step8700_seed42/result.json`
- Per-scene paired CSVs: `PDMS_v1_paired.csv`, `EPDMS_v2_paired.csv` in that directory.
- Launch log: `runs/action_batch/qualified_w8_single/intermediate_step8700_seed42_launcher.log`.

The new helper's seven CPU tests passed (exit0). They validate the actual pairing
and identity checks; they make no GPU/model-quality claim. No native training,
acceptance, model or original evaluation code was edited for this request.
