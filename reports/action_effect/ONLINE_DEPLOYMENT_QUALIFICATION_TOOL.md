# Real current-image S0 / EGR / IEG qualification

Source implementation starts at `95b46c5d3062fae09da9820a9a329376156824ae`
in isolated branch `audit/effect-online-deployment-20261020`. Current status:
**CPU wiring tests PASS; real image/CUDA deployment NOT_RUN by this author**.
The Lead owns GPU allocation and will select actually trained private models.

`tools.action_effect.qualify_online_deployment` starts a fresh Python worker.
The worker uses the actual `load_student` helper to strictly reconstruct the
once-bound original mature S0 from FP32 masters, with original inference FP32,
TF32 disabled, deterministic backend and scene-noise seed 42. It checks the
original protected source contracts and uses `ForesightCurrentDataset`, which
opens current metadata and real images without loading any ego/future labels.
The pinned observation is checked against the actual dataset index/current
file and its three ordered image paths. Current ego is the original normalized
history delta-pose `[dx,dy,sin(dyaw),cos(dyaw)]`, not velocity.

One `encode_current` call supplies W, action tokens and the unmodified original
planner condition. The three execution modes are:

- S0: the original action head's Euler10, then the original raw-to-physical map.
- A: the same original Euler10 followed by the exported private EGR refiner,
  including its existing geometric fallback. No learned or real feedback is
  required. Auxiliary S0 prediction heads remain stripped.
- B: the same original action-head object plus the exported private velocity
  adapter inside one Euler10 generation. There is no initial S0 rollout,
  second action-head object, second VLM, refiner, teacher or postprocessing.

`export_models.load_bundle` now accepts explicit injection of the already
verified original action head and its Base identity. New B exports contain a
tiny original-head contract directly in the bundle manifest, so the online
path never opens the head blob or even metadata under the training context
cache. It strictly rejects an injected different Base and any deployment
adapter containing training-only effect heads. Old standalone head-only loading
remains available. Online integration requires a fresh export made with this
source contract; legacy bundles lacking the embedded metadata are rejected
rather than silently reading a cache.

Forward hooks require, per prediction, visual=1, language=1, original planner
projection=1 and DiT=10. A additionally requires refiner=1; B requires velocity
adapter=10 and refiner=0. Real input file opens must be exactly the original
three images in every timed prediction. The worker refuses simulator, teacher
trainer and context-reader imports, and denies data opens for ego/future labels,
metric caches, feature-context caches, feedback and LiDAR. These checks enforce
the known repository contracts; they are not a general operating-system sandbox.

For every requested mode, replacing future GT, labels, actor truths, metric and
teacher-cache references, LiDAR and the observation token must leave W/A/ego,
planner condition, raw and final physical outputs byte-identical. Noise is
fixed independently from the registered original scene ID, so a poisoned token
cannot change the seed. No true future labels are read to construct the test.

The launcher may separately read explicit completed prediction JSON artifacts
after the deployment worker has exited. It joins by scene and observation hash,
verifies complete inventory, registration, Base and private-checkpoint identity,
then compares the final physical trajectory without relaxed tolerance. Cached
predictions are never passed to the worker or used as policy inputs.

## Timing scope

The default is two warmups plus eight measured predictions per mode, on the
same real scene and fixed noise. Smaller sample counts are rejected. Every
timed iteration includes current JSON and original image decode/crop/resize,
processor/tokenizer, one VLM, Euler10, optional A refiner and delivery of final
physical trajectory to CPU. CUDA is synchronized at both timing boundaries.
Weight loading, diagnostic W/A copies, poison tests and cached-output comparison
are excluded. Memory peaks are captured before poison diagnostics.

Results record mean/p50/p95 and every sample, allocated/reserved CUDA peaks,
CPU maximum RSS, device/dtype/backend, private parameter counts and actual hooks.
One shared S0 and all requested small heads are resident in the worker, so the
memory scope explicitly includes those heads. This is a repeated one-scene,
warm-file latency measurement, not long-run population throughput, physical
cold-storage performance or scientific planning evidence. No speed is predicted
or reported until the actual CUDA run completes.

## Actual commands

The following is a complete original-S0-only command using existing real paths
and the first already locked TRAIN cohort scene. The Lead must expose exactly
one admitted CUDA device before execution. No command below was dispatched by
this author.

```bash
python -m tools.action_effect.qualify_online_deployment --help

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python -m tools.action_effect.qualify_online_deployment \
  --base-binding /mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json \
  --cohort /mnt/project/action-effect-repair-20261020/COHORT.json \
  --scene-id fc90fd5b50ae59fe \
  --modes S0 --warmup 2 --repeats 8 --max-seconds 1800 \
  --device cuda:0 \
  --output /mnt/project/action-effect-repair-20261020/audit/online_S0_v1
```

For all modes, replace `--modes S0` with `--modes S0 A B` and provide
`--a-bundle` and `--b-bundle` pointing to the actual newly exported `BUNDLE.json`
files. The tool does not invent paths, choose a later checkpoint or export an
untrained placeholder. Those two inputs currently await the Lead's endpoint
selection. Export with the existing, verified CLI:

```bash
python -m tools.action_effect.export_models --phase export --route A \
  --checkpoint "$A_CHECKPOINT" \
  --base-binding /mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json \
  --output "$A_EXPORT" --device cpu
python -m tools.action_effect.export_models --phase export --route B \
  --checkpoint "$B_CHECKPOINT" \
  --base-binding /mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json \
  --output "$B_EXPORT" --device cpu
```

The uppercase variables denote required explicit paths to real trained
checkpoints and new export directories selected by the Lead, not assumed
existing files. Add optional `--s0-predictions`, `--a-predictions` and
`--b-predictions` only when the exact model/scene prediction artifacts exist.
`--dry-run` starts no child or GPU work, including with the internal worker flag.
Output directories are immutable and must be new; failed logs are retained.

CPU helper and existing export regression command:

```text
python -m pytest -q tests/action_effect/test_online_deployment.py tests/action_effect/test_campaign_exports.py
16 passed in 4.44s
python -m tools.action_effect.qualify_online_deployment --help
exit 0
```

Tests exercise actual small adapter/refiner modules with a deliberately tiny
synthetic Base fixture. They establish wiring, hooks, zero-module identity,
forbidden-input behavior, strict same-head injection without a head-cache read,
auxiliary stripping and negative cases. They do not replace real image/S0/GPU
evidence. No weights, images or raw caches are committed.
