# Current-source regression

Command, from `/mnt/project/DriveVLA-M0-iqe-closeout-924b003`:

```bash
/root/miniconda3/envs/ddp/bin/python -m pytest tests/iqe -q
```

Result: **153 passed, 2 skipped, 4 warnings, 37.98s**. Full output: [final_regression.log](evidence/final_regression.log). The optional NCCL tests are the two skips; each ran separately and passed on free A800 GPUs. See [nccl_test.log](evidence/nccl_test.log) and [nccl_terminal_test.log](evidence/nccl_terminal_test.log). Warnings concern deprecated dependencies, not numerical failures.

Coverage retained: independent-process selector initialization, warm-start/Router rows, latest compatible receipt selection and corrupt/partial rejection, historical best snapshots, data admission fault injection/content identity, expansion→Scorer-only→expansion, genuine residual_v2, frozen generators and scoring context, DDP unequal-denominator loss/gradient equivalence, exact optimizer-boundary resume. C1/C2/C3 negative and crash cases add behavioral coverage. Execution tests cover epoch/tail identities, prefetch ownership/failure, actual trainer fixture resume, backend transition and resource gate.

The protected Query Base, expert definition, loss, S0 adapter, contracts and IO are byte-identical to 924b003. Hash evidence: `PROTECTED_BASE_COMPATIBILITY.json`. A preflight compatibility receipt explicitly permits only aggregate post-training-code changes, rechecks protected dependencies and keeps the historical contract unchanged. Wrong model/input/source/metric semantics still fail.

Real 22k integration and 8-GPU 21k execution qualification are separately reported; tensor fixtures are not counted as real integration. No tests require full navtest, full training or ACTIVE changes.

Current environment: Python3.10.20, PyTorch2.5.1+cu124/CUDA12.4, A80080GB for real GPU tests. `evidence/TEST_ENVIRONMENT.json` records exact versions. The final added legacy-input test ensures qualify/freeze rechecks source bytes even without reevaluating a model.
