# Thin campaign, IEG evaluation and inference-only export

Date: 2026-10-20 UTC. This is implementation evidence, not positive planning
evidence. No GPU, simulation, real training, export of actual model weights or
new NAVTEST was run by this agent while implementing these entries.

`tools.action_effect.run_campaign` dispatches real audit, G0, G1, route-A,
route-B and evaluation entries. `--phase all` explicitly requires
`--execute-ready-stages`. Missing contexts, physically verified repair data or
trained teachers/checkpoints become BLOCKED with nonzero exit. Dry run remains
NOT_RUN. CPU simulation/scoring children hide CUDA and limit BLAS to one thread.
Both routes retain independent run directories; their shared state is immutable
data, not parameters or optimizers. The runner does not schedule hosts, stop
jobs, run NAVTEST, or update ACTIVE.

All model namespaces include the selected population, so the 64-scene cost
pilot cannot silently become the 1024-scene training experiment. Actual reserved
simulation calls in G0 and selected evaluation ledgers share a total invocation
ceiling. Conflicting immutable historical budget registrations are blocked.
Command, stdout and stderr records are versioned per invocation. The existing
common trainers handle optimizer, sampler, RNG, ordinary pause and terminal
resume; the campaign does not duplicate them. A paused training result is
INCONCLUSIVE, never called a completed scientific comparison.

`tools.action_effect.evaluate_effect_flow` strictly reconstructs the original
canonical master action head and this run's private adapter. It uses each
scene's original cached condition and original fixed noise for exactly one
Euler10 generation. It does not load the route-A refiner or trained G1 teacher,
nor generate a complete original trajectory and post-process it. Inference sets
deterministic algorithms, disables TF32 and cuDNN benchmark, and disables
autocast explicitly. Official evaluation calls the shared
`evaluate_refiner.score` in a separate bounded CPU invocation; no scoring
algorithm or feedback extractor is copied.

The current prediction path uses registered TRAIN fit/validation roles. It does
not pretend that the 64-scene qualification cohort is the complete independent
1696-scene DEV population. The separate DEV collector/reader is owned by the
other agent; its actual complete artifact is a prerequisite for that result.

`tools.action_effect.export_models` exports only the private adapter/refiner
state, pure configs and immutable original Base/head references. It strips
optimizer, teacher, labels and training cache requirements. Route A preserves
its explicit residual refiner and geometry check. Route B preserves one full
Euler10 generation. Neither is silently set ACTIVE. Export includes a
transitively enumerated Python source archive and per-file checksums; it never
archives weights, images, dataset files or secrets. These are research bundles
with the original VLA referenced once, not another VLM copy.

The roundtrip command extracts that source archive into a fresh directory and
runs a new process on the same explicitly supplied legal current tensors/noise.
The worker denies simulator and training/feedback module imports and reads no
training checkpoint/cache. Exact outputs are required; no tolerance is relaxed.
This verifies small-head deployment in isolation, not image-to-action latency
or proof of full online VLA integration.

## Current CPU evidence

Seven new CPU behavior tests passed in **3.50 seconds**. They cover bounded dry
run with zero jobs, explicit all-stage opt-in, latest complete checkpoint within
the named run, corrupt/foreign rejection, aggregate reserved-call budgeting,
population namespaces, canonical inference flags, source dependency closure,
strict small A-head reconstruction without training checkpoint reads, and A/B
input whitelists. Synthetic fixtures are labelled and not reported as S0 runs.

All three actual CLI `--help` commands completed successfully. A fresh temporary
source-only directory containing the computed 224-file / 3,125,133-byte closure
successfully imported both deployment routes with the runtime deny-list active;
it executed no model or simulator. Actual trained bundle roundtrips await Lead
resource admission and model artifacts.

Examples use real existing config/context/data locations; checkpoint/output
arguments must identify the selected run, not an untrained placeholder:

```bash
python -m tools.action_effect.run_campaign --config configs/action_effect/main.yaml --phase audit
python -m tools.action_effect.run_campaign --config configs/action_effect/main.yaml --phase g0 --max-samples 64 --max-simulation-budget 32768 --resume
python -m tools.action_effect.run_campaign --config configs/action_effect/main.yaml --phase g1 --arm delta_feedback --max-samples 64 --max-updates 32 --device cuda:0
python -m tools.action_effect.run_campaign --config configs/action_effect/main.yaml --phase all --dry-run --max-samples 1024 --max-updates 1000
python -m tools.action_effect.export_models --help
```

The actual checkpoint is always resolved once within the named arm/run from its
receipt, model state and identity, or supplied explicitly. The baseline and
validation population never change to improve the reported result. No final
claim about G0 repair opportunity, G1 benefit, route-A/B gain, safety or latency
can be inferred from these entrypoint tests.
