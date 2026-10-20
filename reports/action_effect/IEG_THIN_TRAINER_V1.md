# IEG shared-loop training and canonical head-only qualification

Date: 2026-10-20 UTC. Implementation source starts at a8948831e72900cec0ba0b377852edc7f0d287b5,
with common small-module trainer f43eca75 cherry-picked as ef551eef923e00e4080be4c3309874251bf75e2e.
This report does not independently approve its author's IEG mathematics.

The new `tools.action_effect.train_effect_flow` is a thin client of the common
optimizer/scheduler/checkpoint/RNG/sampler implementation. It does not introduce
another optimizer loop, Base model, visual network, simulation backend, or target
schema. B0, B1-trajectory and B1-effect each initialize the same-sized zero-output
private adapter independently from the same canonical frozen S0 action branch.

The actual shared `action_effect_repair_dataset_v1` is joined to frozen context by
scene, source log, role, observation, full context-entry identity and exact Base
physical trajectory. Registration/completion identities and Base binding are
checked. Only fit rows enter backward. Verified local repairs and original-raw
identity keeps form a fixed 1:1 reference pool; oversampling adds references, not
unique observations. The common replacement sampler yields this expected ratio;
actual repair/keep exposure counts are logged per optimizer update. Either empty
class blocks training. No threshold or class balance is changed from DEV results.

Every optimizer batch draws eight original Beta/noise samples per scene before
microbatching. Changing scene microbatch changes compute grouping, not draw
assignment. Repair velocity, keep velocity, physical effect and physical direction
have separate full-batch denominators. Keeps reuse the original raw target exactly,
including the original sin/cos vector. Their already-computed Base velocity is
reused, avoiding redundant 819M action-head passes. Base stays frozen/eval.

The B1-effect teacher bridge strictly requires this campaign's trained frozen
`delta_feedback` G1 teacher, with matching Base hash and physical label schema.
The bridge decodes raw Base/modified trajectories using the original normalizer;
W/A/current-ego context is separately supplied through the deployment whitelist.
Labels remain loss targets. The bridge remains differentiable with respect to
the adapter output, not its own parameters. Undefined sin/cos headings fail.
The velocity-derived endpoint proxy is not a completed ODE trajectory. Its loss
alone cannot demonstrate improved planning: full Euler10 and real NAVSIM remain
required offline measurements.

The training command records original head/config/receipt, context/repair artifacts,
sampler reference identity, seeds, explicit loss weights, adapter config and source.
Resume restores private parameters, full optimizer/scheduler, sampler consumption,
Python/NumPy/Torch RNG and both independent Flow RNG streams through the shared
loop. Frozen dependencies are reloaded strictly. Fixed source/runtime are required.
Offline endpoint validation is explicitly recorded; no simulator runs in backward.

## CPU evidence and limits

`python -m pytest tests/action_effect/test_flow_internalization.py tests/action_effect/test_flow_trainer.py tests/action_effect/test_small_training.py -q`

Actual result: **24 passed in 16.26s**. Complete output is
`IEG_THIN_TRAINER_CPU_OUTPUT.txt`. The new eight cases cover B0/B1 repeat8 and
microbatch gradient/noise agreement, a single Base call per microbatch, fixed
balanced references, invalid role/keep masks, differentiable physical teacher
bridge, G0 role/context joins, CLI boundaries, both Flow RNG streams during
interruption and a terminal checkpoint that adds zero updates.

Tests use the production DiT class at reduced CPU dimensions and explicitly
synthetic analytic physical feedback. They are not pretrained S0 or simulator
integration. Real training, real teacher compatibility and held-out planning
quality remain untested here. No GPU job, simulation or optimizer on real weights
was launched by this agent. No existing Base, 22k binding or active job changed.

## Real independent-process qualification command

The following is prepared for Lead resource admission after canonical context
extraction completes; it was not executed by this agent:

```bash
python -m tools.action_effect.train_effect_flow \
  --verify-head-only \
  --config configs/action_effect/main.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --output /mnt/project/action-effect-repair-20261020/audit/head_roundtrip64_v1 \
  --device cuda:0 --max-samples 1
```

The manifest and canonical head receipt were observed on shared storage. The
verification explicitly reconstructs **only** the original action head from
its serialized full constructed config and strictly loads canonical FP32 master
state. Two independent sequential processes run original Euler10 from the real
cached condition and scene noise, then compare full raw/physical tensors with
each other and cached Base output exactly. Qwen/simulator implementation imports
are rejected in the workers. All stdout/stderr and the successful receipt are
retained. Failure never becomes PASS or relaxes tolerance.

For training, specify `--arm`, `--repair-dataset`, `--max-updates`, `--output`,
the same actual config/context manifest, seed and admitted device. `--resume`
uses the same output and complete identity; `--stop-after` bounds a segment while
retaining the original scheduler horizon. `--effect-checkpoint` is required only
for B1-effect. The real repair artifact is intentionally not guessed in this
report: shared G0 must first complete and provide its immutable path. Until then
the real three-arm execution is WAITING_INPUTS, not successful training.
