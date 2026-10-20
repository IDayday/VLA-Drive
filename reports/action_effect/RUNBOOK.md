# Physical action-effect campaign: completed-run and reproduction runbook

The registered real run is complete:23 formal1000-update models (six G1 controls,
one dense-temporal ablation, seven explicit refiners and nine Flow models),31
complete official score stages (19 DEV policies,12 seed42 fit/validation stages),
actual S0/A1/B1-effect image-to-final qualification, and read-only final statistics.
This is an engineering-complete research result with unsupported method gains;
it is not an instruction to relaunch completed training or simulations. Current
science is in [FINAL_COMPARISON.md](FINAL_COMPARISON.md), all rows in
[FINAL_SYSTEM_RESULTS.csv](FINAL_SYSTEM_RESULTS.csv), and per-source pins in
[final_analysis/FINAL_POLICY_STATISTICS.json](final_analysis/FINAL_POLICY_STATISTICS.json).

## Current immutable identities and completion

- Original mature DiT S0 remains source1493deda/milestone100000, binding
  `3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`.
  This is not a Query Base replacement. No Base/normalizer/noise/Euler10/ACTIVE change.
- Actual configuration: `configs/action_effect/shared_budget_v2.yaml`; artifact
  root `/mnt/project/action-effect-repair-20261020-budget-v2` (abbreviated ROOT below).
- `ROOT/contexts/MANIFEST_001024.json`, all20,678 real effect pairs and1024 repair/
  identity records are COMPLETE:fit791/H6,validation233/H1; source logs are disjoint.
- `ROOT/evaluation_contexts/dev1696_seed42_v1/EVAL_CONTEXT_MANIFEST.json` is COMPLETE,
  SHA781072cb6ed2d8eb22d7aac9c287953e82e5602f3e380c5c5cd455f2921ed479,
  real1696/16logs; W/A/condition payload2,226,901,212B, no giant visual cache.
- Every formal model is its fixed1000-update endpoint, not a16-step qualification,
  resumed warmup, best DEV seed, new Base or super-pool.
- Every DEV RESULT has1696 valid scores,0 invalid/unresolved. All12 fixed seed42
  fit/validation results also completed. Final strict native comparison v1 exited0
  for all18 methods plus S0 in595.908296s. The optional new8-thread reader was not
  used for this production comparison:its controller reused the already complete
  serial evidence without signaling/dispatching a new native-validation job.
- Online actual mean/p50/p95: S01.329762/1.329242/1.332660s,
  A11.336315/1.335788/1.342403s, B1-effect1.366010/1.365946/1.368230s,
  one real TRAIN scene,2 warmups/8 repeats. Cost/poison/hooks/cache equality and
  partial operator FLOPs are in [LATENCY_AND_PARAMETER_SUPPLEMENT.md](LATENCY_AND_PARAMETER_SUPPLEMENT.md).

## Separate budgets and actual cost — do not merge with the old32768 dispatcher

The old serial `run_campaign --phase all` example lower in the historical appendix
had a6000-update/32768-call aggregate cap. It did not execute the full manual
parallel registered matrix and is not the full study's evaluation ceiling.

The formal pre-registration separately allocates17 DEV policies×1712 requests
and6 complete TRAIN fit/validation populations×1040=35,344. The registered control
amendment adds2 DEV policies×1712=3424, giving38,768 formal-evaluation request
capacity. The G0 ceiling32,768 is a separate data-collection budget. The one
extra1000-update dense-temporal arm yields23,000 formal optimizer updates.
Every ceiling includes retries as specified by its actual ledger; none is
silently multiplied per arm.

Actual31 official stages contain38,368 scene-policy evaluations:38,180 reserved
new fixed-reference candidate requests,188 lawful reuses,0 retries/invalid. G0 across prefix
collections contains23,735 canonical requests:the final1024 build newly charges
11,757 and lawfully reuses11,978 earlier-prefix requests. The scoped total is
61,915 requests (38,180+23,735), excluding qualifications and unrelated old
studies. One request invokes a fixed-reference/candidate protocol; it is not a
single internal simulator rollout. Do not count reuses again as new requests.
See [NATIVE_EVALUATION_COST_SUMMARY.json](NATIVE_EVALUATION_COST_SUMMARY.json).

The23 formal training segments total4.2354 GPU-hours, excluding context extraction,
simulation, validation, qualifications and online timing. This is measured segment
wall cost for single-GPU runs, not total fleet reservation or complete study cost.
[TRAINING_EXECUTION_MATRIX.csv](TRAINING_EXECUTION_MATRIX.csv) retains every source,
arm/seed, endpoint, runtime, exposure and segment. The explicit16GiB post-training
reservation is [POST_TRAINING_STORAGE_RESERVATION.json](POST_TRAINING_STORAGE_RESERVATION.json);
no checkpoint/image/large cache is committed or automatically deleted.

## Reproduce only the final read-only statistics

The following completed command reads existing fixed results; it does not call
`compare_methods.load`, native feedback, a model, simulator or optimizer. Choose
a fresh output directory to preserve original evidence. It verifies the exact
18-method final comparison and its EXIT0 before deriving statistics.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python \
  reports/action_effect/analysis/summarize_final_results.py \
  --comparison /mnt/project/action-effect-repair-20261020-budget-v2/comparisons/final_dev1696_all_seeds_v1/COMPARISON.json \
  --comparison-exit /mnt/project/action-effect-repair-20261020-budget-v2/resources/FINAL_DEV_COMPARISON_EXIT.json \
  --output-dir reports/action_effect/reproduction_final_analysis_NEW
```

Actual derivation took2.052s,0 new native reads/forwards/updates. Bootstrap uses
only `paired_summary`,2000 source-log draws/seed42. Each policy uses the same
1696/16logs. Training seeds42/43/44 are reported separately and summarized with
sample SD/range;5088 repeated evaluations are never labeled independent scenes.
Nine A1−A0/B1-effect−B0/B1-effect−B1-trajectory same-seed comparisons do not select
a best seed or replace the registered model.

## Training, recovery, fresh Base and reevaluation commands

The command interfaces in the historical appendix remain real, but their old
WAITING rows and old budget are historical. Shared inputs/23 endpoints are now
complete. Ordinary interruption uses the exact same config/population/arm/seed
and `--resume`; a terminal1000 endpoint must not gain another update or restart
warmup. Actual same-backend full-state CUDA continuation is proven for G1/B0 in
its bounded scope, not every B1-effect/DDP/cross-device combination. Examples:

```bash
EFFECT_ROOT=/mnt/project/action-effect-repair-20261020-budget-v2
EFFECT_CFG=/mnt/project/VLA-Drive-action-effect-repair-20261020/configs/action_effect/shared_budget_v2.yaml

# G1: exact original source b7cc81fb, not the current documentation/source HEAD.
cd "$EFFECT_ROOT/runtime/formal_b7cc81fb"
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.train_effect_model \
  --config "$EFFECT_CFG" --context-manifest "$EFFECT_ROOT/contexts/MANIFEST_001024.json" \
  --pair-dataset "$EFFECT_ROOT/g0/population_001024/PAIR_DATASET.json" \
  --arm delta_feedback --seed 42 --max-updates 1000 --device cuda:0 \
  --output "$EFFECT_ROOT/models/population_001024/g1/delta_feedback/seed_42" --resume

# A/B: exact original source c845d6e9; fixed resolved recipe, dataset and own teacher.
cd "$EFFECT_ROOT/runtime/routes_c845d6e9"
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.train_refiner \
  --config "$EFFECT_CFG" --context-manifest "$EFFECT_ROOT/contexts/MANIFEST_001024.json" \
  --repair-dataset "$EFFECT_ROOT/g0/population_001024/REPAIR_DATASET.json" \
  --arm A0 --seed 42 --max-updates 1000 --device cuda:0 \
  --output "$EFFECT_ROOT/models/population_001024/route_a/A0/seed_42" --resume
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.train_refiner \
  --config "$EFFECT_CFG" --context-manifest "$EFFECT_ROOT/contexts/MANIFEST_001024.json" \
  --repair-dataset "$EFFECT_ROOT/g0/population_001024/REPAIR_DATASET.json" \
  --arm A1 --seed 42 --max-updates 1000 --device cuda:0 \
  --effect-checkpoint "$EFFECT_ROOT/models/population_001024/g1/delta_feedback/seed_42/step_001000.pt" \
  --output "$EFFECT_ROOT/models/population_001024/route_a/A1/seed_42" --resume
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.train_effect_flow \
  --config "$EFFECT_CFG" --context-manifest "$EFFECT_ROOT/contexts/MANIFEST_001024.json" \
  --repair-dataset "$EFFECT_ROOT/g0/population_001024/REPAIR_DATASET.json" \
  --arm B0 --seed 42 --max-updates 1000 --device cuda:0 \
  --output "$EFFECT_ROOT/models/population_001024/route_b/B0/seed_42" --resume
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.train_effect_flow \
  --config "$EFFECT_CFG" --context-manifest "$EFFECT_ROOT/contexts/MANIFEST_001024.json" \
  --repair-dataset "$EFFECT_ROOT/g0/population_001024/REPAIR_DATASET.json" \
  --arm B1-effect --seed 42 --max-updates 1000 --device cuda:0 \
  --effect-checkpoint "$EFFECT_ROOT/models/population_001024/g1/delta_feedback/seed_42/step_001000.pt" \
  --output "$EFFECT_ROOT/models/population_001024/route_b/B1-effect/seed_42" --resume
```

These exact-source examples are operator commands, not dispatches performed by this documentation. Do not run them from a new HEAD against old outputs:source identity must reject that mismatch rather than pretending it is exact continuation. For seeds43/44 use the matching explicit seed and output only; ScoreRefiner uses its own absolute_score teacher, not delta. A fresh reproduction requires a new config/artifact namespace with its source identity registered before training; it must not replace this fixed data or existing checkpoints.
Check current authorized resource/real process identity before using them; do not
start another lineage copy. Original data/model identity mismatches reject resume.
A new Base requires a new campaign/config/artifact root and binding, fresh feature/
score/teacher identities and requalified inputs, not editing this fixed run.
Completed native scores and strict comparison must be reused under their original
identities, not recomputed to obtain a more favorable mean. Original online output
and one-scene supplement are immutable; see [FINAL_ONLINE_CLOSURE_RUNBOOK.md](FINAL_ONLINE_CLOSURE_RUNBOOK.md)
and its later actual cost report for export/independent-process commands.

Final full CPU regression is487 passed,0 failed/skipped,one existing pynvml warning,
46.05s after test-environment isolation. The initial485-pass/2-failure environment
run is also preserved; neither is rewritten. See [REGRESSION_AND_RESUME_REPORT.md](REGRESSION_AND_RESUME_REPORT.md).

## Historical preparation appendix — 02:43 UTC and later partial snapshots

Everything below is the original dated preparation/runbook text and earlier
partial evidence. It is retained for audit, not current readiness. Its missing
artifacts now exist,14/16 snapshot is superseded, serial6000/32768 ceiling is not
the full parallel plan, and optional old score commands do not authorize new work.

---

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
