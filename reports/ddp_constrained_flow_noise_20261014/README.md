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

The first 30 core/runtime tests passed on the authorized Tsinghua zt4 host. They cover
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

The finite calibration is complete: 128 predeclared TRAIN scenes, G16 each,
one C_SDE reference and at most two noise ranges. The initial discrete-std range
`[0.0105409, 0.1]` was rejected because its median requested-XY second difference
was 2.80 times C_SDE (registered limit 2). The only allowed half-range passed
and is locked to `[0.0052704627669472965, 0.05000000000000001]`.

| Initial stochastic policy | PDMS (0–100) | Joint NC/DAC/TTC full pass | Pairwise ADE (m) | Median XY second difference (m) |
| --- | ---: | ---: | ---: | ---: |
| C_SDE | 91.5867 | 96.1426% | 0.1322 | 0.2111 |
| N_FLOW_NOISE original range, rejected | 87.7386 | 91.1621% | 0.3396 | 0.5905 |
| N_FLOW_NOISE selected half-range | 91.5540 | 96.0449% | 0.1797 | 0.3514 |

These are calibration means over actual stochastic candidates, **not trained
ODE deployment results**. More geometric diversity is not yet evidence of
useful driving exploration. No DEV/Navtest data selected the range.
The fixed road/object thresholds are 0.1720945/0.9274561 metres, the registered
positive-margin 5th percentiles among C_SDE candidates fully passing all three
official safety metrics. Common unfiltered cost budgets are
`[0, 0.03271484375, 0.005859375, 0.02867518074, 0.01117817716]`.
The first three are event rates; the latter two are bounded geometric costs.
A zero empirical NC budget does not establish zero true collision probability.

The eight-GPU storage probe confirmed that existing DeepSpeed converts a
separate FP32 noise group to BF16. The implementation therefore retains the
audited uniform BF16 storage and FP32 optimizer masters, and evaluates the
noise module through differentiable FP32 parameter views and FP32 fixed bounds.
Actual full-DDP kernel checks passed: legacy C_SDE chain unchanged; eager versus
CUDA graph chain exactly equal; noise-disabled means exactly equal to original
Euler; sigma-only gradients reached all six MLP tensors and shared DiT; both
sigma-only and mean-only changes altered ratios. Flat versus serial initial
mean-ratio errors were at most `1.02e-5` (acceptance `1e-4`), with no PPO update
performed in that kernel test. A first graph attempt failed on constructing a
constant during capture; fixed FP32 bound tensors are now prepared before capture.
Only configuration constants are cached, never noise outputs or trainable hidden states.
