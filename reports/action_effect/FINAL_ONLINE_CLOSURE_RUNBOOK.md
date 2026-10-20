# Final1000 A1 / B1-effect deployment closure

This is a bounded engineering deployment qualification. It does not train,
simulate, select a checkpoint or produce a population-quality result. The new
thin entry is `tools.action_effect.prepare_final_online`; it delegates exports
and online execution to the existing strictly checked implementations. No Base,
private-model, precision, noise, Euler10, deployment guard or official scorer
function changed in this patch. Source start: `7dc6abd6`.

## Actual current readiness

At the CPU-only check on 2026-10-20 03:28 UTC, both of these explicit final
`RESULT.json` files were absent:

```text
/mnt/project/action-effect-repair-20261020-budget-v2/models/population_001024/route_a/A1/seed_42/RESULT.json
/mnt/project/action-effect-repair-20261020-budget-v2/models/population_001024/route_b/B1-effect/seed_42/RESULT.json
```

The actual planner returned `WAITING_INPUTS`, exit2. It created no output
directory, read no large Base/action-head weights, and dispatched no child.
No final checkpoint was CPU-exported, and no GPU qualification was launched.
These facts are a timestamped input check, not a claim that an endpoint remains
missing later. Rerun the planner after the actual training results appear.

The plan fixes the original S0 binding
`3b910a6d4245580d77f868e4e0af5faf2fc7fccc25c561c935f219ab1bf7f099`,
context `MANIFEST_001024.json` SHA256 `a670570d83d7ca54c047d85d2476fd58508609163fca3bd098fd5275e9e1eb80`,
and repair dataset SHA256 `5c3697801b4da2501b26df712a3fa3ae3773ea30d9e05b6e4ba1b85f2896ad2d`.
The scene is the already registered real TRAIN observation `fc90fd5b50ae59fe`.
Its three current images are loaded by the actual current-only dataset.
Current ego remains normalized historical delta-pose [1,4], not velocity.

## Admission and budget

Lead alone admits and dispatches this job after checking current server policy
and actual available memory. Use a frozen execution worktree containing this
patch and the previously reviewed `629fffd8` guard fix; do not hot-edit the
source between prepare and online phases. The plan pins source commit plus
critical module hashes; each small export also pins its complete source
inventory. A source/endpoint/command change requires a fresh preparation root.

Resource reservation for the proposed run:

- CPU export: two sequential CPU-only processes; B strictly reconstructs the
  original 3,278,134,372-byte FP32 action head from its verified receipt. This
  is not Qwen or a second online head. A 16GiB host-RAM admission reserve is
  recommended for this export stage; no peak for this stage is claimed here.
- Online: one already admitted NVIDIA A800 80GB (or separately reviewed matching
  device), with one visible GPU, two CPU threads and the same full-Base loading
  host-memory allowance used for the original S0 replay. Do not evict live jobs
  to satisfy this preparation. The actual full-worker peak RAM/GPU is measured
  by the existing online tool, not inferred here.
- Exactly one full-S0 load in one fresh online worker. Its existing original
  action head is injected by object identity into B. No action-head cache blob
  is opened in the online worker, and no second VLM/head is constructed there.
- Per mode: 2 warmups + 8 measured predictions + 1 label-poison check. Three
  modes total33 online predictions,33 visual/language forwards and330 original
  DiT steps. A calls only its small refiner after original Euler10; B calls its
  velocity adapter within its only Euler10 generation, with no A refiner.
- Zero new optimizer updates and zero simulator calls. Export children have
  1200s wall limits each; the single online child has1800s by default. Timeout
  preserves logs/failure and is not retried automatically.

These are admission ceilings/definitions, not measured latency or resource
results. The old failed `qualification/online_S0_v1` directory is untouched.

## Commands to execute from the frozen reviewed source

The checked original runtime Python exists at `/root/miniconda3/envs/ddp/bin/python`
(Python3.10.20). The source checkout must include the final commit of this patch.
Every command below uses a new output root and keeps the existing failed run.
No command changes training, navtest, ACTIVE or server processes.

First run the cheap metadata-only planner. A missing/paused/midpoint result
returns exit2; a contract mismatch returns exit1. It does not poll or run a job.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.prepare_final_online \
  --config configs/action_effect/shared_budget_v2.yaml \
  --phase plan --seed 42 --scene-id fc90fd5b50ae59fe --device cuda:0 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2 \
  --max-seconds 1800 --export-max-seconds 1200
```

When both final inputs exist, create fresh small exports with CUDA hidden. The
preflight requires exact arm/seed/Base/context/repair identities, the original
FP32 training runtime, the same own G1 teacher, B's `verified_pair_v1`, a genuine
step1000 `MAX_UPDATES` complete result and its sibling immutable receipt. It
does not use `latest`, `best`, a 16-update qualification or a paused750 endpoint.
The existing export loaders then strictly reconstruct the actual model state.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.prepare_final_online \
  --config configs/action_effect/shared_budget_v2.yaml \
  --phase prepare --seed 42 --scene-id fc90fd5b50ae59fe --device cuda:0 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2 \
  --max-seconds 1800 --export-max-seconds 1200
```

After Lead has admitted one actual GPU, expose only that GPU and run the online
phase. This command is the actual dispatch entry; it was not executed here.
Example visible physical GPU0 is a placeholder for the admitted device, not an
assertion of availability. The worker sees it as `cuda:0`.

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/root/miniconda3/envs/ddp/bin/python -m tools.action_effect.prepare_final_online \
  --config configs/action_effect/shared_budget_v2.yaml \
  --phase online --seed 42 --scene-id fc90fd5b50ae59fe --device cuda:0 \
  --output /mnt/project/action-effect-repair-20261020-budget-v2/qualification/online_S0_A1_B1effect_final1000_seed42_v2 \
  --max-seconds 1800 --export-max-seconds 1200
```

`--phase all` combines the preceding CPU exports and single online launch;
exports still hide CUDA in their own processes. `prepare` and `online` remain
separate so the Lead can stage small exports before admitting a GPU.
`--dry-run` never dispatches either stage. Existing output roots are rejected
for prepare, and any existing `online` directory is rejected for online, whether
its prior worker succeeded or failed. Choose a new suffix for a reviewed retry;
do not delete failure evidence. No automatic repeat of the expensive Base load.

The fixed endpoints may be supplied with `--a-run` and `--b-run`, but they must
still pass all same-population final1000 checks. These are explicit directories,
not fallback search roots. Changing either run changes the immutable plan.

## Optional cached-output comparison

Use only actual explicitly supplied predictions that contain the same TRAIN
scene, observation hash, S0 binding and exact final private checkpoint:

```text
--s0-predictions /actual/S0/PREDICTIONS.json
--a-predictions /actual/A1_final1000/PREDICTIONS.json
--b-predictions /actual/B1effect_final1000/PREDICTIONS.json
```

No such optional artifact is assumed to exist. DEV1696 predictions do not
contain this TRAIN scene and must not be supplied. Pass the same options in
prepare and online so they are pinned in the plan. The launcher joins these
after the worker exits; training/context cache, GT or real physical feedback
never enters the online policy. A missing optional comparison is reported
`NOT_REQUESTED`, not an online/cache equality PASS. Output mismatch fails the
receipt without widening numerical tolerance.

## Existing deployment guarantees and resulting evidence

The worker retains original FP32-master eval, disabled autocast, TF32off,
deterministic backend and `scene_noise(scene_id,42)`. It uses the original
`load_student(..., strip=True)` helper; DINO/future/interaction prediction
networks remain stripped. All modes get the same fixed original noise. B's
training-only effect heads/projection are removed by `velocity_export` and
strictly refused if they remain in a deployment bundle.

Hooks require visual1,language1,planner projection1,DiT10 per prediction;
A additionally refiner1; B additionally velocity adapter10. Each timed call
must actually open all three current images. Labels, GT, LiDAR, future actors,
simulator/effect-teacher/feature-cache modules and data remain denied. Only
previously reviewed trusted importlib atomic code-cache writes are exempt.
The poisoned-label check is outside timing and must preserve the complete
outputs exactly; this is not an input to the model.

Expected new outputs, only after real execution:

```text
<output>/PLAN.json
<output>/PREPARATION.json
<output>/exports/{A1,B1-effect}/{BUNDLE.json,PRIVATE_HEAD.pt,SOURCE.zip}
<output>/online/{COMMAND.json,stdout.log,stderr.log}
<output>/online/{DEPENDENCY_AUDIT.json,WORKER_RESULT.json,COMPLETE.json}
```

The final actual receipt must report p50/p95/mean over eight measured calls
per mode, resident/peak memory, parameter counts, hooks, exact label independence
and optional cached comparisons. Repeated one-scene warm file-cache timing
cannot establish full-population throughput or scientific planning improvement.
Those remain separate official matched DEV evaluation results.

## CPU validation performed

```text
python -m pytest -q tests/action_effect/test_final_online_preparation.py \
  tests/action_effect/test_online_deployment.py tests/action_effect/test_campaign_exports.py
37 passed in 8.77s
python -m tools.action_effect.prepare_final_online --help
exit0
```

New negative/behavior checks cover wrong Base/arm/seed/data/context/horizon/
precision/teacher, legacy proxy B, absent or paused final inputs, foreign receipt
or changed private checkpoint, existing shared-trainer terminal-result pin form,
and no child while waiting. A real small CPU adapter export strips all effect
auxiliary parameters, strictly reloads against the exact same head object and
preserves its Euler10 output. Tiny fixtures are explicit unit tests; no S0
weight, real GPU deployment, final1000 checkpoint or scientific gain is claimed
by these tests. Actual online closure remains `WAITING_INPUTS` until executed.
