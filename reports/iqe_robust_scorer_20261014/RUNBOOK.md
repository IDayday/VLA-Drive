# Scratch Scorer reward-objective experiment

Working directory: `/mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014`.
Interpreter: `/root/miniconda3/envs/ddp/bin/python`.
Config: `configs/iqe/robust_scorer/mean_risk_v1.yaml`.
Output: `/mnt/project/iqe-robust-scorer-20261014/mean_risk_v1`.
Source banks: `/mnt/project/iqe-specialist-scorer-study-20261014/base70k_v1`.

```bash
cd /mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli --help
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli bind --config configs/iqe/robust_scorer/mean_risk_v1.yaml
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli audit-data --config configs/iqe/robust_scorer/mean_risk_v1.yaml
```

After a live resource check, use one permitted GPU per cell. The environment
selects an actual assigned GPU; `cuda:0` is its process-local ordinal.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli train --config configs/iqe/robust_scorer/mean_risk_v1.yaml --cell R00 --device cuda:0
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli train --config configs/iqe/robust_scorer/mean_risk_v1.yaml --cell R01 --device cuda:0
CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli train --config configs/iqe/robust_scorer/mean_risk_v1.yaml --cell R10 --device cuda:0
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli train --config configs/iqe/robust_scorer/mean_risk_v1.yaml --cell R11 --device cuda:0
```

SIGTERM is handled by the native optimizer-boundary checkpoint. Resume the same
cell, output and source identity; auto resolves the newest complete compatible
receipt and does not silently restart from initialization:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.cli train --config configs/iqe/robust_scorer/mean_risk_v1.yaml --cell R00 --device cuda:0 --resume auto
```

Do not run a second process for an active cell. `RUN.lock` protects each cell.
`--stop-after` is an absolute optimizer-step boundary, not a replacement learning
rate time axis. Finished terminal checkpoints execute no extra optimizer step.
The initialization audit, comparison distribution, dependency identity,
per-step validation predictions and checksummed full states are under
`scorers/<cell>/`. No command above trains Base, changes experts or starts navtest.

After all four complete, the following entry point first locks the stage_val
model choice, then calibrates the separately reported margin, and only then
reads the fixed full dev bank. It also reports read-only D_S fitting diagnostics.
Incomplete cells fail instead of selecting a conveniently early result.

```bash
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.evaluation lock-choice --config configs/iqe/robust_scorer/mean_risk_v1.yaml
CUDA_VISIBLE_DEVICES=1 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.evaluation evaluate --config configs/iqe/robust_scorer/mean_risk_v1.yaml --device cuda:0
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.bundle --config configs/iqe/robust_scorer/mean_risk_v1.yaml
```

The exported research bundle includes the Scorer and private expert states and
the checksummed original Base dependency. Its loader is
`iqe.robust_scorer.bundle.load(path, device)`, which returns the existing
`ResearchSystem` single-VLA wrapper. Default selection is raw argmax; the optional
margin is recorded separately. No GT or metric-cache dependency exists in that
inference wrapper. Real trained-head online/bundle qualification remains an
explicit post-training step; CPU serialization tests do not stand in for it.
