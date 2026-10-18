# One explicit C1 advantage-scale adaptation

The legacy `within_scene_center_only` default is retained. The explicit new
`scene_center_rollout_std_v1` is restricted to C1 with BC and policy weights one.
Original rewards, four actual rollouts, MC50, original repeated BC/dropout,
optimizer groups, precision, sampler and LR horizon are unchanged.

The new helper runs once after complete official feedback and before MC/current
microbatches. Its error vote precedes statistics collectives. Global FP64 sample
variance scales the original FP32 scene-centered advantages; the numerator is
not re-centered. Zero feedback remains zero. It uses no training RNG.

The numeric and target identities explicitly bind the new method. Rank feedback
receipts bind actual reward/mask/output tensor hashes, rollout, context, old
policy and every rank's common statistics. Native states store the last consumed
common receipt hash. Resume verifies it before restoring RNG; the existing
terminal return remains before the optimizer loop. The native evaluation reader
accepts this explicit contract without relaxing legacy strict loading.

The diagnostic-manifest helper copies the already audited TRAIN manifest:
PROFILE retains all entries and their original order; SMOKE takes an explicit
score-independent prefix of at most 32. Parent hashes are recorded. No images,
targets or context labels are changed, and normal access verification remains.

Executed affected tests (CPU, including real two-process Gloo and tiny original
DiT/Adam arithmetic; no real-image qualification claimed):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_fpo.py tests/st_world/test_flow_training.py \
  tests/st_world/test_flow_diagnostic_manifest.py tests/st_world/test_advantage_scale.py \
  tests/st_world/test_flow_rollout_scale_integration.py tests/st_world/test_flow_checkpoints.py
```

Result: **97 passed**, one installed pynvml deprecation warning, **12.26 seconds**.
The complete command output is retained in `TEST_OUTPUT.txt`.

Independent correctness review and bounded real SMOKE/PROFILE are still required.
This implementation does not establish planning benefit or authorize a formal
run. It does not resume paused feature research or alter an existing checkpoint.
