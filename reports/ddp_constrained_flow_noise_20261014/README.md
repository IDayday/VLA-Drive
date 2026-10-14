# DDP constrained Flow-Noise: execution record

Source start: `6edb05d576b40a9718421e86c06d560f184a11c6`.
Branch: `feature/ddp-learned-noise-constrained-grpo-20261014`.
Status: implementation and acceptance in progress; no formal training result yet.

The only formal comparisons are legacy correlated, score-corrected `C_SDE`
and diagonal learned `N_FLOW_NOISE`. Both retain the complete 359-tensor DDP
action model (819,503,620 parameters), G16/K10, training seed 42 and original
single-candidate ODE evaluation seed 2101. The new line additionally trains the
RLinf noise MLP. Fixed SFT weight SHA-256:
`9a26685aa3838a2e1b89ab2d92997664afde4b259fed6683851f42781ad16eb4`.

The old all-BF16 condition cache is excluded. The replacement uses the original
VLAAgent execution: BF16 Qwen, FP32 history/action prefix, canonical processor,
prompt and action-token extraction, before trainable `qwen_proj`. It contains
only token identity and observation features. Qualification on real TRAIN
scenes passed exact cache/direct ODE comparisons at batch 1/2 and candidate
chunk 1/16, including the original inference call. Identity:
`80a2dd4b417a3154cbbf575044daea586d945b54e014eee22269d7bd48d39e0b`.
An earlier qualification attempt failed because the launch environment lacked
the deterministic CuBLAS setting; its evidence is preserved. Full TRAIN cache
generation and missing canonical v1.1 metric-cache production are running.

The first 23 core tests passed on the authorized Tsinghua zt4 host. They cover
upstream outputs/gradients, time/sign conversion, densities, KL and sigma
gradients, score-function arithmetic, component advantages, dual direction,
collector ordering/caps/resumption and geometric costs. This is arithmetic
acceptance, not evidence of driving improvement. Real full-head learning,
distributed/resume qualification, two formal curves and ODE evaluation remain
unfinished.

Commands already executed from this worktree:

```bash
python -m scripts.analysis.constrained_flow_noise.lock_upstream --output /mnt/project/constrained-flow-noise-20261014/upstream --lock configs/flow_rl/upstream_lock.json
python -m pytest tests/constrained_flow_noise/test_core.py -q
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.analysis.constrained_flow_noise.build_condition_cache --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --output /mnt/project/constrained-flow-noise-20261014/condition_cache --priority-tokens /mnt/project/constrained-flow-noise-20261014/cache_priority.json
python -m scripts.analysis.constrained_flow_noise.build_training_metric_cache --population /mnt/project/flow-safe-improvement-20261013/population/manifest.json --cache-root /mnt/project/constrained-flow-noise-20261014/metric_cache_v1 --reuse-root /mnt/project/flow-safe-improvement-20261013/metric_cache_v1 --output /mnt/project/constrained-flow-noise-20261014/metric_cache_build --workers 16
```

Python environment: `/mnt/project/flow-safe-improvement-20261013/venv/bin/python`.
Launches set `CUBLAS_WORKSPACE_CONFIG=:4096:8`, CPU BLAS/OMP threads to one,
and validate the current dispatch policy. Private run records and failure
evidence are under `/mnt/project/constrained-flow-noise-20261014`; they are not
uploaded. No original data, features, weights or private scene images are committed.

Upstream boundaries: [patch inventory](../../starVLA/rl/constrained_flow_noise/vendor/PATCHES.md).
Formal calibration, cost budget, optimizer/resume and outcome sections will be
updated with measured evidence; a target PDMS above 92 is not assumed achieved.
