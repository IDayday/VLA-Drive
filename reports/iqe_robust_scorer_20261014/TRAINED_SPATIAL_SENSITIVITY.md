# Trained S11 spatial dependence audit

The already selected original S11 remains unchanged. This read-only diagnostic
uses the first 16 scenes in the fixed D_S manifest and the first 16 stage_val
scenes, chosen without candidate scores. It uses CPU FP32, two threads, no
optimizer, and does not access development/test or select another checkpoint.
The constructor reproduces the entire recorded random initial state hash, then
strictly loads the original selected checkpoint. All 27 spatial state tensors
have changed from initialization; every recorded spatial parameter had finite,
nonzero gradients at the first original training update.

| Population | Intervention | Max score difference [0,1] | Changed choices |
|---|---|---:|---:|
| scorer_fit | shuffle_V | 0.000371337 | 0/16 |
| scorer_fit | zero_V | 0.005123794 | 0/16 |
| stage_val | shuffle_V | 0.000126004 | 0/16 |
| stage_val | zero_V | 0.002233744 | 0/16 |

The modified visual inputs and the spatial reader outputs both change
substantially. The subsequent score changes are small and all choices in this
small diagnostic subset remain the same. This rules out a missing replacement
or a completely untrained/disconnected spatial branch in these executions. It
does **not** establish that visual information is intrinsically unhelpful, nor
prove zero dependence on every scene. The existing full-development aggregate
intervention report and this bounded numerical diagnostic answer different
questions. No new Scorer or recipe was trained based on these observations.

An initial invocation used the original training constructor and was safely
rejected because it attempted to reuse a path-specific resolved-recipe receipt
from another worktree. No old artifact was overwritten. The diagnostic now
constructs without writing to the old campaign and verifies the complete
initial state hash before loading the trained weights.

Reproduction (refuses to overwrite an existing audit):

```bash
cd /mnt/project/DriveVLA-M0-iqe-robust-scorer-20261014
PYTHONPATH=. OMP_NUM_THREADS=2 /root/miniconda3/envs/ddp/bin/python scripts/iqe/audit_trained_spatial_use.py --config configs/iqe/robust_scorer/mean_risk_v1.yaml --scenes-per-role 16
```

Evidence: `/mnt/project/iqe-robust-scorer-20261014/mean_risk_v1/qualification/trained_S11_spatial_sensitivity.json`

SHA256: `8520e8bc316da961288d8f1076f635ff5967df86d30f141a08355f9e979f996b`.
