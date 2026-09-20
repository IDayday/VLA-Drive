# Requested interim full-navtest evaluation

Status: **COMPLETE**. All12146 scenes from136 logs were scored, all eight GPU
workers exited0, and the official CPU scorer exited0. Elapsed evaluation time
was1872seconds (31.2minutes), including result merging and baseline comparison.
This is the fixed step8700/seed42 interim result, not the final five-seed result.

| Metric (100-point scale) | Own SFT | RL step8700 | Paired delta | Log-bootstrap95% CI of delta |
|---|---:|---:|---:|---:|
| Official v1.1 PDMS | 88.8644 | 89.0109 | +0.1465 | [-0.0671,+0.3525] |
| Official v2 one-stage EPDMS | 88.2378 | 88.5025 | +0.2647 | [+0.0615,+0.4527] |

Both means improve slightly. The PDMS interval includes0; this run does not
establish a statistically clear PDMS gain. The EPDMS interval is positive for
this paired single-seed evaluation. Neither result establishes stability across
inference seeds or a broad improvement in all driving criteria.

The gain is primarily progress: the v1 ego-progress component increases1.099
points, while no-at-fault-collision compliance declines0.272points and the v1
TTC component declines0.757points. In v2, progress increases1.231points and TTC
declines0.231points. Component changes are descriptive and not an additive
decomposition of the nonlinear aggregate metric.

PDMS zero-score scenes increase578→595:39 recover from zero,56 become zero.
EPDMS zero-score scenes increase602→624:39 recover,61 become zero. Thus improved
mean scores coexist with regressions that require attention. We have not changed
training, chosen a new recipe or selected a checkpoint based on these navtest
observations. The predeclared one-epoch run continues.

[result.json](result.json) contains paired statistics and confidence intervals;
[components.json](components.json) contains all available component means and
zero-score counts. [completed_evidence/index.json](completed_evidence/index.json)
indexes compressed complete SFT/RL score CSVs, per-scene paired CSVs and official
completion receipts with hashes. Historical launch-time snapshots are retained.

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
