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

After the training process on the assigned device has exited, run the two
commands below in separate processes. They use four real stage_val observations
for the original-batch online/cache and label-independence checks. Timing uses
32 measured image-to-trajectory calls after three warmups for Base, the full
candidate pool, and the pool with the selected trained Scorer. The source
campaign is read-only and the output stays under this experiment's
`qualification/` directory. Do not run them on a device occupied by Base/G2.

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.qualification online --config configs/iqe/robust_scorer/mean_risk_v1.yaml --device cuda:0
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.qualification roundtrip --config configs/iqe/robust_scorer/mean_risk_v1.yaml --device cuda:0
```

The exported research bundle includes the Scorer and private expert states and
the checksummed original Base dependency. Its loader is
`iqe.robust_scorer.bundle.load(path, device)`, which returns the existing
`ResearchSystem` single-VLA wrapper. Default selection is raw argmax; the optional
margin is recorded separately. No GT or metric-cache dependency exists in that
inference wrapper. The two explicit post-training commands above generate the
real trained-head evidence; CPU serialization tests do not stand in for it.

The public loader now restores this bundle version's registered deterministic
backend, including TF32 flags, before constructing the model. This is the same
backend already used by training, cached evaluation and the online reference;
it does not rewrite any checkpoint or reset training seeds. A caller does not
need to invoke the qualification CLI to obtain that execution contract.
The following separate-process check deliberately starts with incompatible
default backend flags, loads through the public API, and compares one real
stage_val image against the existing reference. It does not repeat latency
windows, consume optimizer steps, or alter tolerances.

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python scripts/iqe/verify_robust_scorer_bundle_loader.py --config configs/iqe/robust_scorer/mean_risk_v1.yaml --device cuda:0 --max-samples 1
```

Its output is `qualification/public_loader/REPORT.json`; the actual bounded
launch uses a 900-second timeout on the verified free GPU6 of `training-rl-zt2`.
Inspect `qualification/public_loader/LAUNCH.json` and the process before any
retry. Existing weights and all selected predictions remain unchanged.

The bounded finalizer below waits for all four terminal checkpoints and their
training locks to be released, then for GPU6 to be unoccupied. It only runs the
registered evaluation/export/qualification/report stages, rechecks permanent
host permission before every stage, and never stops or launches training. Its
own `runtime/finalizer/RUN.lock` prevents duplicate controllers. Inspect its
`STATUS.json` before launching another instance.

```bash
ssh training-rl-zt2 'cd /mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014 && /root/miniconda3/envs/ddp/bin/python scripts/iqe/finish_robust_scorer.py --config configs/iqe/robust_scorer/mean_risk_v1.yaml --gpu 6 --expect-host training-rl-zt2 --max-wait-seconds 7200'
```

Actual additional recovery qualification was performed on tsinghua-zt5 GPU1
using 32 real D_S scenes, a disposable R11 head, and eight updates total. Full
training and interrupted/resumed training matched model, Adam, scheduler, RNG,
both sampler streams, next IDs and losses. The 2+2 diagnostic phase split does
not change the production 3000+3000 schedule. Evidence is
`qualification/real_resume_R11/REPORT.json`; those weights are never selected.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.qualification resume-audit --config configs/iqe/robust_scorer/mean_risk_v1.yaml --device cuda:0
/root/miniconda3/envs/ddp/bin/python -m iqe.robust_scorer.report --config configs/iqe/robust_scorer/mean_risk_v1.yaml
```

The full four-cell evaluation is complete. The following read-only diagnostics
reuse checksummed predictions and official banks; they neither fit parameters
nor change the locked model choice. The first reports fitting versus heldout
selection and severe-error costs. The second measures existing close-trajectory
safety comparisons in D_S only, without generating new candidates.

```bash
OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python scripts/iqe/audit_robust_scorer_generalization.py --config configs/iqe/robust_scorer/mean_risk_v1.yaml
OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python scripts/iqe/audit_scorer_boundary_coverage.py --config configs/iqe/robust_scorer/mean_risk_v1.yaml
```

Actual outputs are under `qualification/generalization_audit/` and
`qualification/boundary_coverage/`. The rich bank already contains 1,457
NC/DAC safety-crossing candidate pairs with mean XY trajectory distance at most
0.1 m, across 1,141 D_S scenes and 139 logs. That distance is not physical
clearance. This is evidence about training supervision, not a new heldout
success result or permission to retune against dev. No fifth training cell has
been started under the completed four-cell registration.
