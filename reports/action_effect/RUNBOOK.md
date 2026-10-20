# Physical action-effect campaign: executable runbook

> The original preparation snapshot and its dated WAITING rows below are retained
> as history. Current executed evidence is in
> [EVIDENCE_STATUS.md](../effect_repair/EVIDENCE_STATUS.md),
> [EVIDENCE_INDEX.json](../effect_repair/EVIDENCE_INDEX.json) and the five canonical
> G0/G1/G2/G3/final reports. Do not relaunch completed collectors/models from an
> old readiness row or duplicate a live job. No command here is an automatic
> dispatch instruction.

This is a preparation/runbook delivery, not evidence that the following commands
were executed. Lead alone dispatches on currently authorized resources. Preserve
the active Base, other studies and the existing TRAIN/DEV collectors. Check their
actual jobs before any resume; do not start another producer for the same output.
Use an immutable source checkout, since source SHA is part of training identity.

## Fixed identities and observed readiness

Actual configuration: `configs/action_effect/shared_budget_v2.yaml`.
Actual artifact root: `/mnt/project/action-effect-repair-20261020-budget-v2`.
Original S0 binding remains
`/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`, SHA identity
`3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`.
This is original mature DiT S0 at 100000, not Query Base, C0 continuation or a
diagnostic model. The actual configuration retains 1000-update small-head
horizons, 50 warmup updates, batch32/microbatch8 and original FP32 inference.

Read-only file observation at **2026-10-20 02:43:04 UTC**:

| Artifact | Observed state |
| --- | --- |
| `CAMPAIGN_BINDING.json` | Present |
| `contexts/MANIFEST_000512.json` | Present; existence alone is not a qualification PASS |
| `contexts/MANIFEST_001024.json` | WAITING_INPUTS |
| `g0/population_001024/PAIR_DATASET.json` | WAITING_INPUTS |
| `g0/population_001024/REPAIR_DATASET.json` | WAITING_INPUTS |
| `evaluation_contexts/dev1696_seed42_v1/EVAL_REGISTRATION.json` | Present |
| `evaluation_contexts/dev1696_seed42_v1/EVAL_CONTEXT_MANIFEST.json` | Final completion must still be verified |
| `models/population_001024/...` scientific endpoints | WAITING_INPUTS; paths below are future output namespaces |

At that observation, G0 population256 and population512 call ledgers contained
4431 and 5971 reserved native calls, respectively, with zero reserved retries.
These are counters at that instant, not final success counts or projected costs.
No 1024 artifact or endpoint is invented in this runbook. Existing real16
qualifications live under `qualification/`; they are not scientific 1000-update
endpoints and must not be substituted into the main comparisons.

## Budget and independent-stage behavior

Without `--arm`, `--phase all` now expands explicitly:

```text
audit -> g0 -> G1 delta_feedback
      -> A0 -> B0 -> B1-trajectory -> A1 -> B1-effect
      -> individual matched S0/A0/B0/B1-trajectory/A1/B1-effect evaluations
```

`--continue-independent-on-failure` allows A0/B0/B1-trajectory to execute even
if G1 fails. A1 and B1-effect remain WAITING/BLOCKED when their own teacher is
missing. Evaluation is also split by arm, so a missing endpoint does not hide
results from other ready arms. An explicit `--arm B0` does not launch A1 or
B1-effect; only that route and required prerequisites/evaluations are planned.

`--max-updates` remains the **per-arm scheduler horizon**. An expanded `all` or
six-control G1 execution additionally requires explicit `--max-total-updates`.
The default six training arms therefore declare 6000 total scheduled new updates
when each uses 1000. The reservation is persisted before each child starts.
Failures do not silently refund work, extend the budget or reset a model.
`--stop-after N` limits new updates while retaining the supplied horizon.

Model outputs always include population/family/arm/seed, for example:

```text
models/population_001024/g1/delta_feedback/seed_42/
models/population_001024/route_a/A0/seed_42/
models/population_001024/route_a/A1/seed_42/
models/population_001024/route_b/B0/seed_42/
models/population_001024/route_b/B1-trajectory/seed_42/
models/population_001024/route_b/B1-effect/seed_42/
```

The per-dispatch update ledger is `runner/.../UPDATE_BUDGET.json`. Its ceiling
does not grow on resume. Existing model progress requires `--resume`; shared
trainer locks protect the same actual output across different dispatch plans.
Each subprocess still uses the original trainer rather than a copied optimizer
loop. Complete/terminal training resumes add zero optimizer updates.

The simulation ceiling is **32768 aggregate reserved calls**, not that many
per arm. The dispatcher includes all G0 prefix-population ledgers and all
evaluation arms/seeds under this artifact root. Existing reservations and retries
count. If a previous immutable stage ceiling conflicts with reallocation, the
runner blocks rather than editing it. Additional methods cannot silently obtain
another full simulation budget. The direct child tools retain their own bounded
receipts; Lead must coordinate manual parallel jobs within the same approved
total. Reaching a budget may legitimately leave later evaluations BLOCKED.

## Inspect and finish shared inputs

These commands use real CLI interfaces verified with `--help`. The collector
resume command is only for a confirmed stopped/incomplete producer, not an
instruction to duplicate the current active producer.

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase audit

python -m tools.action_effect.collect_base_trajectories \
  --config configs/action_effect/shared_budget_v2.yaml \
  --max-samples 1024 --device cuda:0 --resume

python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase g0 \
  --max-samples 1024 --max-simulation-budget 32768 \
  --max-seconds 3600 --resume
```

The G0 wrapper validates actual feedback/completion and constructs pairs and
repair manifests. It does not treat a marker's presence as verified data. Native
timeout/failure stays incomplete; no failed simulation becomes a valid zero.

The independent DEV1696 registration already exists. Finish/verify it after all
registered collector workers exit; do not use TRAIN context as a DEV substitute:

```bash
python -m tools.action_effect.collect_evaluation_context \
  --config configs/action_effect/shared_budget_v2.yaml --phase finalize \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/evaluation_contexts/dev1696_seed42_v1

python -m tools.action_effect.collect_evaluation_context \
  --config configs/action_effect/shared_budget_v2.yaml --phase verify \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/evaluation_contexts/dev1696_seed42_v1
```

Missing shards yield failure/WAITING, not a partial population renamed COMPLETE.
DEV is inference/evaluation only; it never enters any training sampler.

## Default complete bounded campaign

First inspect the resolved task list without launching any job:

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase all --dry-run \
  --max-samples 1024 --max-updates 1000 --max-total-updates 6000 \
  --max-simulation-budget 32768 --device cuda:0 --seed 42
```

After the needed data and affected-scope qualifications are ready, the following
actually executes the ready stages. Use the allocated CUDA device in an
immutable checkout; this command is not itself a multi-server scheduler.

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase all \
  --execute-ready-stages --continue-independent-on-failure --resume \
  --max-samples 1024 --max-updates 1000 --max-total-updates 6000 \
  --max-simulation-budget 32768 --max-seconds 3600 \
  --device cuda:0 --seed 42
```

Default evaluation here uses the fixed TRAIN validation-log role, not NAVTEST
and not the final DEV population. A missing prerequisite makes overall status
INCONCLUSIVE while preserving successful independent stage receipts.

## Six G1 controls and absolute-score Refiner

The optional fixed G1 matrix is exactly `absolute_score`, `terminal_pdms`,
`absolute_feedback`, `delta_feedback`, `trajectory_only`, `shuffled_context`.
It does not launch a privileged-feature placeholder or load outside weights.

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase g1 --g1-matrix \
  --max-samples 1024 --max-updates 1000 --max-total-updates 6000 \
  --device cuda:0 --seed 42 --resume --continue-independent-on-failure
```

Use `--phase g1 --arm NAME` for one explicitly assigned GPU/arm. Each model has
its own named output and locks. No automatic privileged/GT context is supplied.
ScoreRefiner can be added to unfiltered `all` with `--include-score-refiner`;
without a completed own `absolute_score` teacher, it remains WAITING_INPUTS.
To execute six G1 controls plus six route models in one uninitialized plan,
explicitly declare `--g1-matrix --include-score-refiner --max-total-updates 12000`.
That extra work is opt-in, not hidden in the default 6000-update plan.

## Individual routes and ordinary/terminal resume

Independent head-only experiments need no G1 checkpoint:

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase route-a --arm A0 \
  --max-samples 1024 --max-updates 1000 --device cuda:0 --seed 42 --resume
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase route-b --arm B0 \
  --max-samples 1024 --max-updates 1000 --device cuda:0 --seed 42 --resume
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase route-b --arm B1-trajectory \
  --max-samples 1024 --max-updates 1000 --device cuda:0 --seed 42 --resume
```

Use the same command after ordinary interruption. `--resume` reads only that
population/arm/seed's latest valid checkpoint, including Adam, scheduler, RNG,
sampler and consumed cursor. A MAX_UPDATES/EARLY_STOPPED checkpoint completes
receipts with zero extra training. Do not pass a shorter max-updates value to
make a short run: keep 1000 and add `--stop-after 16` when a bounded diagnostic is
explicitly authorized, under a separate diagnostic identity/output.

A1/B1-effect can resolve the own delta teacher from the exact named model run;
they do not search arbitrary project checkpoints:

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase route-a --arm A1 \
  --max-samples 1024 --max-updates 1000 --device cuda:0 --seed 42 --resume
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase route-b --arm B1-effect \
  --max-samples 1024 --max-updates 1000 --device cuda:0 --seed 42 --resume
```

Absent teachers are WAITING_INPUTS. B1-effect uses verified-pair native-effect
and representation distillation, not an unsupported physical gradient at an
arbitrary noisy endpoint. Its training-only auxiliary heads are stripped from
deployment. B0/B1-trajectory/B1-effect retain independent initialization,
optimizers, checkpoints and the same velocity-adapter capacity.

## Matched fixed DEV evaluation and comparison

Only after model endpoints and the complete 1696/16-log DEV manifest are ready:

```bash
python -m tools.action_effect.run_campaign \
  --config configs/action_effect/shared_budget_v2.yaml --phase evaluation --arm A1 \
  --max-samples 1024 --evaluation-role dev_report --evaluation-max-samples 1696 \
  --evaluation-context-manifest /mnt/project/action-effect-repair-20261020-budget-v2/evaluation_contexts/dev1696_seed42_v1/EVAL_CONTEXT_MANIFEST.json \
  --evaluation-binding /mnt/project/action-effect-repair-20261020-budget-v2/evaluation_contexts/dev1696_seed42_v1/EVAL_REGISTRATION.json \
  --max-simulation-budget 32768 --max-seconds 3600 --device cuda:0 --seed 42 --resume
```

Repeat with explicit `--arm A0`, `B0`, `B1-trajectory`, `B1-effect` or the trained
ScoreRefiner within the same remaining simulation allowance. The S0 result is
re-scored using exactly the same context/population; do not subtract historical
91.388 from these native single-reference results. Saved prediction reuse must
match hashes. Partial official results cannot be called full-population PDMS.

Each result namespace is:
`evaluation/ARM/seed_42/dev_report/population_001696/score/RESULT.json`.
After these actual files and COMPLETE receipts exist, use
`python -m tools.action_effect.compare_methods --help` and explicitly name them
as `--baseline .../S0/.../RESULT.json --method NAME=.../ARM/.../RESULT.json`.
Use 2000 fixed-seed paired source-log bootstrap draws. Comparison checks the
entire matched population/protocol and does no model or threshold selection.

## Export and real-image qualification

Select only a completed checkpoint from the explicit model run, then export
with `tools.action_effect.export_models --phase export --route A|B`. New export
directories must be outside the training context cache. Actual trained endpoint
paths are still WAITING_INPUTS; no guessed step filename is presented as an
existing artifact. `run_campaign.latest_complete(exact_run)` is available for
one-time explicit resolution, and source/receipt/hash checks remain strict.

`tools.action_effect.qualify_online_deployment --help` describes fresh-process
real-image S0/A/B wiring and latency. It requires the new actual small bundles,
one shared strict-loaded S0, at least two warmups/eight measurements, one visual
and one language forward, and one Euler10 generation. It never loads GT,
feedback, simulator or training-context features. See
`ONLINE_DEPLOYMENT_QUALIFICATION_TOOL.md`; no cached-head timing substitutes for
image-to-final-trajectory latency. Neither export nor this campaign updates
ACTIVE, runs NAVTEST, pushes, merges, or changes the active Base.

## Dispatcher regression evidence

Source-only change tested with:

```text
python -m pytest -q tests/action_effect/test_campaign_independent_stages.py tests/action_effect/test_campaign_exports.py
14 passed in 4.38s
python -m tools.action_effect.run_campaign --help
exit 0
```

The negative dispatcher test uses explicitly stubbed child processes: a failed
G1 child is recorded by the actual `Dispatcher.call` boundary, and the A0/B0/
B1-trajectory CLI calls still occur with continuation enabled. Missing teachers
block only their dependent models; no `--effect-checkpoint` is injected into
independent arms. Tests also cover explicit single-arm intent, fixed six G1
controls, missing own absolute teacher, duplicate named progress, reservation
exhaustion, zero-update completed resume and simulation-call accumulation
across prefix populations/other arms/seeds. These are orchestration tests, not
new real training or simulator evidence. No GPU, simulation or training was
dispatched to develop this runbook/change.


## Executed full campaign addendum (2026-10-20, after initial preparation)

Actual code progression is disclosed by each run's immutable IDENTITY. The six
main G1 models execute sourceb7cc81fb with their original `.2` temporal loss.
Later reviewed reporting/controls/ablation source is not hot-written into those
runtime snapshots. Six G1 and16 A/B models now each have1000-step COMPLETE
receipts. The formal fleet is **not** the smaller default6000-update dispatcher
plan above: its pre-registered full plan has22 main models /22000updates.
`FORMAL_PLAN_AMENDMENT_01.json` adds exactly one independent1000-step no-temporal
G1 arm (23000 total ceiling) and3424 native-call allowance for two nonlearned DEV
controls. Qualification updates are separately recorded, not disguised as models.
The original plan remains immutable and no DEV result tunes these recipes.

- TRAIN1024 context and strict pair/repair datasets are COMPLETE:20678pairs,
  983pair scenes,758fit-pair scenes and225heldout-pair scenes. Full role requests
  remain791fit/233validation. Eight validation scenes have no native valid
  interventions; they are not missing model predictions or automatically scored0.
- DEV1696/16log context is fully sealed;2,226,901,212B payload,0new action-head bytes.
- S0/RandomLocal/FixedLateBrake complete current-protocol means are
  91.388363/90.485193/91.243356. Their official RESULT paths are
  `evaluation/ARM/seed_42/dev_report/score/RESULT.json` (the actual fleet path,
  rather than the optional generic dispatcher's additional population component).
- Fitting results use `evaluation/ARM/seed_42/fit/score/RESULT.json`; validation
  uses the corresponding `validation` role. Do not confuse the TRAIN holdout
  population with independent DEV, or fit evidence with generalization.

Existing fully verified data should be consumed, not rebuilt. The new independent
G1 comparison is reporting-only and reads each existing final CPU COMPLETE:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m tools.action_effect.compare_eligible_effect_models \
  --comparison six \
  --evaluations-json reports/action_effect/G1_ORIGINAL_EVALUATION_PATHS.json \
  --step 1000 --bootstrap 2000 --seed 42 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/g1_evaluation/derived_eligible_comparison
```

This explicit versioned report preserves the original233-scene COMPLETE/METRICS,
verifies every native validation pair before admitting the complete225-scene pair
task, and refuses genuinely missing predictions. It does not run forward, simulate
or train. A structural forced-identity extension defines only incremental0 for
8no-pair scenes, never fabricated absolute full-population PDMS. For temporal
ablation, use `--comparison temporal` and a JSON mapping `full`/`no_temporal` to
their two actual original evaluation COMPLETE receipts; the same source admission
and strict actual weight/sampler/backend matching apply.

The optional extra no-temporal run must use the separately reviewed source and
namespace, preserving main loss/data/seed/horizon. It is not another default
teacher or automatic six-arm search. Resume uses exactly its own output+identity:

```bash
python -m tools.action_effect.train_effect_model \
  --config configs/action_effect/shared_budget_v2.yaml \
  --context-manifest /mnt/project/action-effect-repair-20261020-budget-v2/contexts/MANIFEST_001024.json \
  --pair-dataset /mnt/project/action-effect-repair-20261020-budget-v2/g0/population_001024/PAIR_DATASET.json \
  --arm delta_feedback --temporal-loss-weight 0 \
  --device cuda:0 --seed 42 --max-updates 1000 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/g1_ablation/no_temporal/seed_42
```

Add `--resume` only for its already existing stopped run. These commands are
reproducibility/resumption entries, not a request to duplicate the registered
fleet or to consume an occupied GPU. For already terminal1000 checkpoints, the
common resume entry adds zero updates. Do not lower the horizon or reset warmup.
All new actual endpoints/bundles keep explicit Base/checkpoint/data/runtime pins;
no Base is restarted or changed and no NAVTEST or ACTIVE is invoked.
