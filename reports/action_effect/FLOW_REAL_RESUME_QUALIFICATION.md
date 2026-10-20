# Route B original-head CUDA resume qualification

Status at source delivery: **NOT_RUN on CUDA**. This patch adds a thin
qualification controller, not a new trainer. The actual existing G1 CUDA resume
PASS does not qualify B's separate noise/time streams. Existing B CPU resume
tests also do not establish actual CUDA exactness. Lead must independently
review this source and dispatch the command below before a CUDA PASS is claimed.

The immutable online deployment worktree remains at `64b115d2` and was not
modified. This source uses a separate sparse worktree and branch
`audit/effect-flow-real-resume-20261020`, starting from root `0c511bb9`.

## Scope and actual assets

The new entry `tools.action_effect.qualify_flow_resume` delegates all three
training processes to `tools.action_effect.train_effect_flow`, which uses the
existing `train(...)` function and shared `training.run_training`. It never
substitutes a tiny head or fake data for the actual experiment. It requires:

```text
config: configs/action_effect/main.yaml
artifact_root in that config: /mnt/project/action-effect-repair-20261020
context: /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json
repairs: /mnt/project/action-effect-repair-20261020/g0/population_000064/REPAIR_DATASET.json
Base: original bound formal_S0_seed42_full100k_v2/milestone_100000
new qualification output: /mnt/project/action-effect-repair-20261020-budget-v2/qualification/flow_real_resume16_v1
```

The two actual input JSON manifests were read as a CPU metadata check: both
are COMPLETE and contain64 scenes. Repair metadata reports1 fit repair,
44 fit identity-preservation scenes and19 validation identity-preservation
scenes. It is not a held-out repair-quality experiment. The existing actual
`load_flow_examples` verifies the original native receipts/context joins and
`balance_examples` builds the same88-reference balanced pool over45 unique
fit scenes. Nothing changes these data, roles, targets or sampling order.

The main config, not the shared-budget-v2 config, is required here because its
artifact root resolves the original64-scene ContextReader campaign binding.
Only the qualification output lives under the new budget-v2 root. No existing
main model/checkpoint/run is overwritten. The large original head remains an
immutable reference and is strictly rebuilt by each actual child process.

## Exactly32 newly scheduled updates

The experiment is one B0 arm at seed42, on one admitted CUDA device:

1. Fresh process: continuous16 updates.
2. Fresh process:8 updates, native save at the optimizer boundary.
3. Fresh process: restore that same split run and add8 updates, reaching16.

All retain original1000-update scheduler horizon, warmup50, batch32,
microbatch8, original repeat8, original Beta(1.5,1),s=.999, FP32 frozen S0
head and trainable parameters, canonical deterministic/TF32off execution.
The only qualification config change is checkpoint/save cadence9, so the first
resumed update has an actual saved state in both runs. No short16-step cosine
is created. Both step16 checkpoints are **OPERATOR_PAUSED**, not a fabricated
completed1000-step endpoint.

Every launch reserves its new-update ceiling in `DISPATCH_BUDGET.json` before
the child starts. The total may not exceed32. A failure or timeout does not
refund possibly executed updates and does not silently start another attempt.
`--resume` can reuse complete matching boundaries without training; if a failed
reservation leaves insufficient budget, the tool refuses more work. An explicit
new budget is required instead of repeating until a favorable result appears.
The normal32-update run has exactly three fresh child launches. Each child has
an explicit1200s wall bound, configurable only up to3600s. A timeout preserves
logs and its budget reservation.

This qualification adds0 VLM forwards,0 teacher calls and0 simulator calls.
There is no terminal16-horizon training run. Therefore actual CUDA terminal
zero-extra-step evidence is **NOT_RUN** in this32-update scope; the existing
separate CPU terminal tests remain CPU evidence. No extra terminal budget is
silently added.

## Complete-state comparisons

At update9 and16, use `audit_effect_resume.exact_hash` to compare all bytes/types
of private parameters, Adam groups/moments/step, scheduler, scene sampler,
Flow noise generator, Flow time generator, Python/NumPy/CPU/all-CUDA RNG,
consumed cursor, run identity, pause/lifecycle state and update count. Also
compare the actual recorded loss terms, gradient norm, LR and consumed count
for update9 and16. Wall-clock durations are excluded. No tolerance-based
numerical comparison is substituted.

The controller reconstructs the small actual adapter parameter-name order on
CPU and verifies optimizer parameter IDs against that order. B0 allocates the
same training auxiliary effect heads for capacity fairness but does not use
them in its loss. Their parameters can legitimately have **no Adam state**.
The audit allows this only for the named `adapter.effect_head.*` and
`adapter.effect_projection.*` parameters. Missing active velocity moments,
wrong parameter ordering, moment shape/FP32 precision or Adam step are errors.
All allocated private parameters are still included in the exact comparison.

The next batch proof reconstructs the actual88-reference pool and sampler from
seed143, consumes8 batches and joins the checkpoint8 sampler state. It then
compares actual reference indices/scene IDs/target kinds and the sampler state
after consuming batch9 with the continuous9 and resumed9 checkpoints. Repeated
references are explicit; unique scenes are not inflated. Flow stream states are
byte-compared at9 and16; the CPU audit does not claim to regenerate CUDA noise
with a CPU generator.

Each state is joined to a checksummed sibling completion receipt and explicit
training identity, including actual source, original Base/head receipt, data,
context, sampling hash, adapter config, recipe, seed and runtime. Foreign
checkpoint paths, receipt lifecycle mismatches and copied completion reports
are rejected. Completed qualification reuse revalidates local state pins and
the unchanged dispatch ledger before returning zero-new-update reuse.

## Commands for Lead

After cherry-picking the fixed reviewed source into an execution worktree,
inspect help without any GPU dispatch:

```bash
CUDA_VISIBLE_DEVICES='' \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.qualify_flow_resume --help
```

Run only after Lead admits an actually available GPU. GPU5 on rl-zt2 was the
proposed placement, not a claim that this report checked its current ownership.
Only that admitted device should be visible; the child sees `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.qualify_flow_resume \
  --config configs/action_effect/main.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020/contexts/MANIFEST_000064.json \
  --repair-dataset /mnt/project/action-effect-repair-20261020/g0/population_000064/REPAIR_DATASET.json \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/flow_real_resume16_v1 \
  --arm B0 --seed 42 --device cuda:0 --max-seconds-per-segment 1200
```

For a native interruption, use exactly the same source, config, data, output
and device mapping with `--resume`. The immutable identity refuses a source,
data, seed or resource-contract change. The existing32-update reservation limit
still applies. A failed/partial run is never erased or silently restarted.

GPU/RAM admission: one currently admitted A80080GB, one CPU thread and enough
host memory to strictly reconstruct the original3.3GB action head. Three child
processes are sequential, releasing their resident head before the next starts.
The five private optimizer-boundary checkpoints are approximately160MiB total
from tensor counts, not a measured disk result; reserve at least512MiB for this
qualification output and logs. No new visual cache or duplicate head blob is
created. Actual GPU peak/RAM/time must come from the run, not this estimate.

Expected output evidence:

```text
IDENTITY.json, QUALIFICATION_CONFIG.json, DISPATCH_BUDGET.json
continuous16/{IDENTITY.json,step_000009.*,step_000016.*,RESULT.json}
split8plus8/{IDENTITY.json,step_000008.*,step_000009.*,step_000016.*,RESULT.json}
*_target*.{COMMAND.json,stdout.log,stderr.log}
RESULT.json
COMPLETE.json only when all actual comparisons pass
```

A PASS supports B0 private-velocity training continuation on this actual CUDA
backend and the same frozen original S0 head. It does not establish convergence,
physical-effect utility, planning gain, cross-hardware bitwise equivalence or
every B1-effect-specific auxiliary loss on CUDA.

## Source/CPU checks completed

```text
python -m pytest -q tests/action_effect/test_flow_resume_qualification.py \
  tests/action_effect/test_real_resume_qualification.py tests/action_effect/test_flow_trainer.py
47 passed in 12.27s
python -m tools.action_effect.qualify_flow_resume --help
exit0
```

The new tests include dormant auxiliary state, missing active Adam state,
wrong FP16 moments/step/order, independent Flow RNG drift, global RNG drift,
loss/gradient/LR drift, actual sampler sequence reconstruction, forbidden
scheduler/batch shortcuts, wrong source/runtime/receipt lifecycle, pre-launch
budget reservation and exhausted/negative-ledger refusal. Tiny state fixtures
are clearly CPU unit tests, not claimed CUDA or real-data training evidence.
The actual CUDA experiment remains NOT_RUN until Lead executes it.
