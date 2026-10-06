# ReCogDrive Stage2 evaluation

`tools.recogdrive_stage2.evaluate` builds a current-only index and exports the
unchanged official agent's single trajectory. `run_evaluation` acquires the
existing host/GPU leases, verifies original `compute_trajectory` parity on a
real scene in every GPU partition, runs the official parallel v1 scorer and
independently replays the same complete bank through the unmodified official
submission evaluator. Every token and all seven metric fields must agree to
1e-8 on the raw score scale. Partial smoke results are explicitly diagnostic.

The primary checkpoint is the completed 200-epoch optimized-label Stage2
`final.ckpt`, not a released IL/RL model and not a best Navtest checkpoint.
The trained action/state modules load strictly from all `agent.*` tensors.
The public fixed Stage1 identity is checked against the training registration.
The checkpoint's fixed, non-trainable `eta_logit=+inf` represents the official
eta=1 boundary and is the only permitted nonfinite parameter.

Precision is explicit: original Stage1 weights/computation are BF16, its full
final hidden sequence is converted to FP32 as during label extraction, and the
Stage2 master weights and inference computations are FP32, with TF32 disabled.
This is not an all-FP32 backbone measurement. The original five-step DDIM,
sampling noise and output clamp/denormalization remain unchanged. A newly
registered scene-bound RNG makes the single-candidate evaluation reproducible
across GPU allocations; it is not claimed equivalent to an unspecified global
RNG ordering. Sampling seed42 and training seed0 are recorded separately.

Inference reads current front RGB, four legal ego history states, navigation,
velocity and acceleration. Future trajectories/images, optimized labels and
metric cache contents never enter the inference model. Official scoring uses
the canonical 12,146 scenes/136 logs, original full-precision metric caches and
all traffic object classes. It does not use ReCogDrive's GRPO scorer settings.

Entrypoints (all paths and allocations are supplied in an immutable plan):

```bash
python -m tools.recogdrive_stage2.evaluate manifest --help
python -m tools.recogdrive_stage2.evaluate export --help
python -m tools.recogdrive_stage2.run_evaluation --plan /path/to/registration.json
```

Resume uses the same plan, checkpoint hash, source SHA, GPU partition and scene
seeds. Existing per-scene content receipts are verified before reuse. Failed
or missing scenes cannot be discarded to produce a complete benchmark.
