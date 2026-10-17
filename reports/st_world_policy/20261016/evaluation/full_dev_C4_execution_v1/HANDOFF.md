# C4 fixed full-dev execution preparation

Base: `abd51fd0893804e1f04d9a52f9188d5669b33296`. Worktree:
`/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017`.
This version adds only actual C4 training bindings and its already declared
512/1024/4096 evaluation endpoints. It leaves every older preparation, scientific
rule, control output, deployed helper and runtime unchanged.

C4 registration SHA is
`e87165448c893d660b796eaa3029d839fc684ef8bdf5f3e4a18467b094d0f8fa`;
actual runtime identity is
`4e023061b8ac16114cc1f9e3b5e440cb2c3a23d524d562aa1276270567f5fbbb`.
The actual independent training reviews, approval fingerprint, immutable initial
invocation LAUNCH, config and canonical identity agree. The captured status was
RUNNING at 520 updates. This preparation reads only bounded metadata, not the
full train manifest, images, weights, predictions or scores. It does not give a
new training/evaluation approval or claim that a GPU is available.

`TRAINING_BINDINGS.json` pins these actual inputs. `EXECUTION_PLAN.json` inherits
the prior plan and adds only C4 formal execution/status. Other arm status strings
remain historical. The reward, normalization, C0 controls, complete 1696-scene /
16-log dev population, seeds, single trajectory / original ten Euler steps,
FP32 master evaluation, SDPA, component metrics and paired-log bootstrap
(2000 replicates, seed 20261017) are unchanged. Diagnostics 512/1024 do not select
checkpoints or eliminate an arm; 4096 remains the common endpoint.

`INTENTS.json` fixes exactly three C4 jobs. `COMMANDS.json` supplies their export,
score and already declared paired-analysis templates. No checkpoint is guessed:
tags, COMPLETE receipts, native maps and receipt identities remain null until
Lead seals the explicit actual endpoint in its evaluation registration. A
historical status tag is not a native completeness proof. No native directory
or weight file was scanned or opened in this preparation.

C4 adds three previously unwired execution jobs to the existing sixteen; it
adds no scientific arm. Each job reserves 1 GiB, exports all 1696 scenes on one
GPU/world1, scores with two CPU workers, and has a shared-ledger total score
admission cap of 1728 (1696 primary plus at most 32 remaining admissions).
The three C4 caps total 5184 with 5088 primary admissions and 3 GiB output
reservation. Each endpoint retains at most two export invocations and two score
invocations, including failed/unknown admitted attempts; resume does not double
the score cap. Existing C0 outputs are reused. There is no native checkpoint copy
or new tensor/teacher cache.

The complete campaign keeps four shared export slots and two shared CPU score
slots, with two workers per score. Export/resume/score for a given job stay on
its exact pinned route. The same three evaluation hosts remain allowed:
`training-rl-zt2`, `training-vla-zt`, `training-vlawm-tsinghua-zt` (aliases still
checked against current dispatch policy). C4's training host does not extend
this evaluation allowlist. Host latency remains unmeasured by this helper.

The adjacent `C4_HELPER_v5` contains only candidate copies of deployed v4 with a
fixed C4 metadata pin set and C4 arm dispatch in the existing loader. Native
completion, entire population, exact source, routing, environment, GPU UUID,
fresh process/memory/disk checks, locks, budgets, durable admissions, resume and
W's fdf-training/6714-evaluation distinction are inherited. v1-v4 remain intact.
Independent review and Lead exact-byte deployment are required before use.
