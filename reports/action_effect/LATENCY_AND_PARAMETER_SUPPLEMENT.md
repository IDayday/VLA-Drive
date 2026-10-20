# Actual online latency, parameters and supplemental CPU operator audit

The actual common-load S0/A1/B1-effect online qualification passed from fixed
deployment source `64b115d2c7498d7b6c3376baf72dda5b6af29fe7`. The supplemental
CPU audit used source `20179f542be2952ed572923777fca076cbe07c9c`. Neither this
audit nor this report modifies the immutable deployment worktree, its trained
models or its original `online/COMPLETE.json`.

Actual evidence:

```text
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2/online/COMPLETE.json
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2/online/WORKER_RESULT.json
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/cpu_operator_cost_cache_v1/COMPLETE.json
/mnt/project/action-effect-repair-20261020-budget-v2/qualification/cpu_operator_cost_cache_v1/AUDIT.json
```

`ONLINE_COST_AND_CACHE_ACTUAL.json` is a byte-identical copy of the supplemental
AUDIT, including checksums, source/bundle pins, operator tables and external
trace locations. Its identity is
`eb4c90c21fe43dfb57b123f57e40faf94443cbe6cdad0714e405d8905728e571`.

## Real image-to-trajectory latency

These are the existing actual GPU results, not CPU-profiler timings. All three
small heads were resident with one shared frozen S0, batch1, original FP32-master
eval and TF32off. Each mode used2 warmups and8 measured repeats on the same real
TRAIN scene `fc90fd5b50ae59fe`, with original scene-noise seed42. Timing includes
the current JSON and three original image opens/decode/crop/resize, processor,
visual/language execution, original Euler10, optional A refiner and delivery of
the final physical trajectory to CPU. Weight loading and diagnostic copies are
outside timing. The OS/image file cache is warm across repeated observations.

| Actual mode | Mean seconds | p50 seconds | p95 seconds | Measured n |
|---|---:|---:|---:|---:|
| Original S0 | 1.329762 | 1.329242 | 1.332660 | 8 |
| S0 + A1 explicit refiner | 1.336315 | 1.335788 | 1.342403 | 8 |
| B1-effect, one internalized generation | 1.366010 | 1.365946 | 1.368230 | 8 |

Observed mean differences are A +6.553ms (+0.493%) and B +36.247ms (+2.726%).
These are descriptive differences in this one-scene bounded run, not a paired
population confidence interval or a throughput guarantee. CPU head-only costs
are not substituted for this real image-to-trajectory latency.

Per actual prediction, hooks verified visual1,language1,planner projection1
and original DiT10 in every mode. A additionally called its refiner1; B called
its velocity adapter10 and refiner0. B never generated an original complete
trajectory and then ran an extra correction. The full Base was loaded once;
there was no second VLM or action head. Original auxiliary prediction networks
and B's training-only effect heads were stripped. Label/future/GT/cache poison
did not change outputs. No real simulator or G1 teacher ran online.

Peak allocated GPU memory was12,860,709,888 bytes for S0/B and12,860,710,400
bytes for A (about11.977GiB or12.861GB decimal). Reserved peak was13,304,332,288
bytes (12.391GiB). These scopes have **all requested small heads resident**;
they are not three separate minimal-memory deployment configurations.

## Actual parameters

| Component | Parameters | Accounting |
|---|---:|---|
| Shared original S0 | 2,951,547,396 | Actual stripped online model |
| Original action head | 819,503,620 | Already included in S0; do not add twice |
| A1 private refiner | 2,711,824 | +0.091878% of S0 |
| B1-effect private velocity adapter | 2,706,948 | +0.091713% of S0 |
| Original qwen projection MLP | 5,508,096 | Already inside original action head |
| Additional VLM / additional action head | 0 / 0 | Verified shared ownership |
| Deployed B effect auxiliary parameters | 0 | Training-only projection/heads removed |

The supplemental audit strictly reloaded the actual A/B private export tensors
on CPU and obtained the same private counts as the original online result.
Neither export loads an external Scorer or a new trained encoder.

## Installed-profiler operator FLOPs: explicit lower bounds

The extra measurement is CPU-only torch2.5.1+cu124,1 thread, FP32, autocastoff,
using the real same-scene frozen memory[1,152,2048], ego[1,1,4], planner
condition[1,8,2048], Base physical trajectory[1,8,3] and noise[1,8,4]. Only the
actual small modules and original small qwen projection were constructed.
No full VLM/DiT, G1 teacher, simulator or training step was created.

| Profiled region | Calls | Operator-covered FLOPs lower bound | GFLOPs lower bound |
|---|---:|---:|---:|
| A1 `CoefficientRefiner`, including torch reconstruction | 1 | 270,568,484 | 0.270568484 |
| B1-effect velocity adapter | 10 | 422,728,360 | 0.422728360 |
| Original shared `qwen_proj` MLP | 1 | 88,080,384 | 0.088080384 |

The original projection is common work of S0/A/B, not a new B-only cost.
The adapter's own context projection executes **inside each of its10 calls**;
it is not counted only once. For B's shape-based cost, the actual fixed initial
noise was replayed as state at10 time values. These are representative actual
shapes, not a claimed recreation of intermediate states from a complete ODE
rollout. Operator FLOPs depend on these fixed graph shapes; no trajectory or
planning-quality conclusion uses those profiling outputs.

PyTorch reports supported `addmm/mm/bmm` and some elementwise operators here.
The records also contain zero-reported `aten::_native_multi_head_attention`,
`aten::scaled_dot_product_attention`, CPU flash-attention and softmax events.
Their zero values mean **unreported work**, not free attention. Some nested
matrix operations are counted while other fused attention arithmetic is not.
Layer normalization, activation, geometry/NumPy screening, image processing,
all original visual/language work and all original full-DiT work are not
completely covered by these numbers. No whole-VLA FLOPs total, percentage of
whole-VLA compute, FLOPs-derived speedup or achieved GPU TFLOPs is claimed.
CPU and GPU fusion/kernel graphs can differ; this is not a GPU kernel profile.

The original canonical head artifact was checksum-verified with one3,278,134,372
byte sequential read. The audit then mmap-loaded its metadata and copied only
the four original qwen-projection tensors (22,032,384 bytes) into the small MLP.
It did not instantiate or copy the819M-parameter original DiT into an audit model.
That bounded checksum IO is additional audit cost and is not hidden as22MB-only
disk access. No head blob was duplicated on disk.

CPU profiler initialization emitted one library GPU-property probe warning
(`gpuGetDeviceCount ... code100`) despite CPU-only activities and hidden CUDA.
`torch.cuda.is_initialized()` remained false; there were0 GPU model executions,
0 VLM forwards,0 simulations and0 optimizer updates. No GPU speed was measured
by this supplemental audit. Traces remain under the external qualification
directory and are pinned in the JSON; they are not committed as model/data.

## Exact online versus frozen-cache supplement

The original online receipt still says `cached_comparison_status=NOT_REQUESTED`.
That original result is preserved; this later independent receipt provides the
additional checks without relabeling or rewriting it.

For the same actual online scene and observation hash:

- Original S0 raw8x4 and physical8x3 equal the original64-context cache **bit for
  bit**, including dtype/shape and signed-zero bytes.
- A1 online physical equals the same final1000-checkpoint validation prediction
  **bit for bit**; maximum absolute error0.
- B1-effect online physical equals its same final1000-checkpoint validation
  prediction **bit for bit**; maximum absolute error0.

The actual `qualify_online_deployment.compare_prediction` helper verifies scene,
observation, original Base and exact private checkpoint/arm identity before
joining A/B. The offline prediction files each contain233 validation scenes,
but **only this one real online scene** was compared. This is not233 online
replays or a full-population cache-equivalence claim. Old Base snapshots or DEV
means were not substituted. No future labels or physical feedback enter the
small-module profiling inputs or the online policy.

## Reproduction and checks

Run from the fixed audit source, with all GPU devices hidden:

```bash
CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.audit_online_cost \
  --online-root /mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2/online \
  --deployment-source /mnt/project/VLA-Drive-effect-online-final-closure-20261020 \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --campaign-binding /mnt/project/action-effect-repair-20261020/CAMPAIGN_BINDING.json \
  --a-predictions /mnt/project/action-effect-repair-20261020-budget-v2/evaluation/A1/seed_42/validation/prediction/PREDICTIONS.json \
  --b-predictions /mnt/project/action-effect-repair-20261020-budget-v2/evaluation/B1-effect/seed_42/validation/prediction/PREDICTIONS.json \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/cpu_operator_cost_cache_v1
```

The command above is the actual completed invocation. Repeating it requires
a fresh output suffix; the audit refuses to overwrite evidence. The qualified
deployment source must remain byte-identical to both export inventories.

```text
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/action_effect/test_online_cost_audit.py
5 passed in 1.22s
python -m tools.action_effect.audit_online_cost --help
exit0
```

The CPU tests verify exact-byte/signed-zero handling, wrong source and GPU-visible
refusal, no invented FLOPs for unsupported attention, and actual small-linear
profiler arithmetic. Scientific method gains, effect calibration and population
safety remain the responsibility of separate completed matched evaluations.
