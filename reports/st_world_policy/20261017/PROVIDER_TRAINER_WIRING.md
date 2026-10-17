# Bounded teacher: optional trainer wiring, 2026-10-17

Engineering implementation only. No teacher, GPU, training, tensor-cache build,
formal approval, or dispatch was performed by this patch. Existing complete
cache configurations and dataset identities retain their original path. This
uses the original ST DiT/Flow trainer; it is not an IQE Query experiment.

## Implemented boundary

- `tools/st_world_policy/teacher_provider.py` prepares an immutable metadata
  dependency, validates a completed real qualification report and its plan,
  exact request population, COMPLETE current source, full raw teacher contract,
  execution identity, and budgets. The metadata-only preview does not load
  teacher weights. Actual construction uses the existing `FrozenDINOExecutor`
  after run ownership and data/PCA checks, on the owning rank's explicit device.
  The dataset loads the small existing PCA statistics; metadata-only does not
  mean no PCA tensor read. Descriptor validation precedes immutable publication,
  including failure evidence and limits, so failed preparation cannot poison a
  new immutable descriptor path.
- `STWorldTrainingDataset(..., teacher_provider=...)` accepts only the concrete
  provider and its new identity, with `teacher_root=None`. It preserves all
  source/split/ego/image pins and existing target transforms. All declared
  future frames are obtained before the original `sample_requests` call.
  A missing computed record raises through the existing all-rank error guard;
  genuinely absent source frames keep the original false masks.
- `provider_pca.py` registers a **new** `st_pca_provider_channel_scope_v1`,
  bound to this provider and the unchanged fitted PCA identity, original fit
  source, checksum and verified DINO channel family. It performs no fit, basis
  clone, tensor write, width choice, or fake `target_cache_identity` mapping.
  The original `st_pca_channel_scope_v1` route remains unchanged.
- `train_world.py` keeps its original optimizer, loss, sampler and checkpoint
  code. Provider state contains identity only; native resume checks it and
  clears transient RAM before restoring the original RNG and consumed cursor.
  Teacher construction and calls use the already implemented Python/NumPy/
  CPU/device RNG isolation. No future-pair or patch generator is passed to it.
  Backend flags are rechecked after student/official-runtime/DeepSpeed setup and
  before every teacher batch. The inspected `apply_official_model_runtime`
  changes module attention implementations/checkpointing, not global SDP flags;
  a later global mutation is rejected instead of silently accepted.
  Each status boundary writes small per-rank `teacher_provider_rankN.json`
  counters (batch calls, computed images, immutable/RAM hits, resident peak,
  failures, zero tensor disk bytes). Rank0 status/final result includes its
  counters. They reset per invocation and are not sampler/checkpoint progress.

## Deliberately restricted selectable mode

The initial route is `SMOKE` only, at most 64 train scenes, one exact qualified
current source, batch compute at most eight, metadata at most 32 MiB, scene
tensor budget at most 128 MiB. `loader_workers=0` is mandatory: the existing
ThreadPoolExecutor is not compatible with owner-thread teacher execution.

Cross-scene RAM reuse is **disabled** in this trainer route: resident budget is
one byte, less than a record. The existing general bounded provider and its
qualification runner are unchanged. Their LRU can change the shape of a miss
batch after partial eviction, changing FP32 targets. The new trainer dependency
has a different provider identity and cannot claim that untested partial-LRU
resume is exact. Fixed immutable source reuse remains enabled. Each scene's
miss grouping is independent of prior accesses; no future target is persisted
to tensor disk. This is a conservative numerical diagnostic, not a throughput
recommendation or production cache approval.

## Real evidence inspected read-only

Lead's immutable execution runtime was
`/mnt/project/VLA-Drive-st-bounded-label-runtime-v1-20261017`, not this worktree.
Actual result:
`/mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_run_v1/RESULT.json`.

64 scenes, 957 distinct images, 192 current immutable hits, 765 future misses,
three physically absent future views. Lead's run completed 3064 image forwards
and 617 calls in 287.672 seconds; peak RSS 5,243,781,120 bytes and peak CUDA
allocated 1,391,321,600 bytes. It returned **NUMERICAL_REVIEW_REQUIRED**, exit 2,
with no recorded operational errors. Original production batches of eight
(including original tails) reproduced all 957 records bitwise. Current batch
one differed: max absolute 1.50345e-4, aggregate L2 relative 2.24958e-6. Provider
per-scene miss groups differed from the old reference by max absolute 8.26716e-5,
L2 relative 1.15660e-6. Masks, global RNG, patch/pair streams and the tested
full-order cold/warm/restored digests agreed. Partial eviction was not tested.

These are raw teacher results, **not** student loss/gradient/update or training
throughput results. Cold/warm case wall times include tensor comparisons and
audits and must not be advertised as training speed. No threshold was widened.

Active formal SP0–SP3 configs use `deterministic=True`; that real teacher64 run
used `False`. The new selectable configuration explicitly uses `False` and a
new SMOKE identity. It does not mutate active flags or qualify a matched formal
temporal comparison. Formal use remains blocked on same-contract numerical
review, real trainer update/resume, memory and throughput evidence.

## Evidence card / implementation basis

This is an engineering adapter, not a new scientific architecture. Source
basis: original campaign contract §§5.3/6, `st_world_targets.py::TeacherCacheReader`,
`st_world_dataset.py::__getitem__`, `train_world.py::collective_call`, native
checkpoint save/restore, `teacher_pca.py::TeacherPCA.load/teacher_channel_family`,
and the unchanged `ondemand_teacher.py::isolated_rng/FrozenDINOExecutor`.
No new external API or storage framework was introduced. Python OrderedDict and
PyTorch `fork_rng` evidence remains in `BOUNDED_FUTURE_PROVIDER_ENGINEERING.md`.

## Tests and remaining limits

`PROVIDER_TRAINER_WIRING_TEST.txt`: 71 passed, one pre-existing pynvml warning,
8.90 seconds. CPU only, CUDA hidden, two BLAS/OMP/PyTorch process threads.
The selection covers provider wiring, dataset, strict config, PCA, existing
provider including Gloo failure propagation, and native training-state tests.
The 12 new cases cover scope/version pins, missing/incomplete qualification,
partial-LRU rejection, strict factory, old dataset identity, masks, cold resume
with batch-sensitive fixture arithmetic, no refit, and explicit configuration.
Fixture tests are not real teacher or student numerical evidence.
After the small runtime-counter addition, `PROVIDER_TRAINER_COUNTER_TEST.txt`
records the same 12 wiring cases passing in 4.18 seconds. These are a focused
recheck, not 12 additional unique tests to add to the 71-case suite.

Not run: real student forward/backward using this provider; original-cache vs
provider loss/gradient/update; new runtime native save/resume; NCCL; real trainer
VRAM/RAM and update wall. Synchronous fetch adds exposed teacher compute and
disables prefetch, so speed may regress. Parent formal data scale is not admitted
by this 64-scene metadata boundary. No request-aware partial target fetching,
fixed padded teacher batch policy, producer service or full future cache was added.

## Selectable configuration and commands

Real generated configuration:
`configs/st_world_policy/ondemand_teacher64_numeric_smoke.json` (same bytes as
`/mnt/project/st-world-policy-20261016/audits/ondemand_provider_wiring_v1/configured/training_config.json`).
It is T2, native384/PCA384, current+future diagnostic only, history off, one GPU,
global batch two, microbatch one, two updates on the original 20000-update
schedule, at most 900 seconds. This reuses existing dimensions for engineering
tests and does not select the formal spatial winner. It is **not launched or
approved**. Existing base, input data, PCA basis and caches remain untouched.

Metadata-only preparation, already executed successfully (no tensor writes):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.teacher_provider \
  --requests /mnt/project/st-world-policy-20261016/audits/history_data/smoke_train64_v1/TEACHER_REQUESTS.json \
  --source-root /mnt/project/st-world-policy-20261016/audits/history_data/train256_current_cost_v1/native384 \
  --teacher-config /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1/resolved_config_build_rank0.json \
  --qualification-result /mnt/project/st-world-policy-20261016/audits/ondemand_teacher64_run_v1/RESULT.json \
  --output /mnt/project/st-world-policy-20261016/audits/ondemand_provider_wiring_v1/PROVIDER.json \
  --training-config /mnt/project/st-world-policy-20261016/audits/ondemand_provider_wiring_v1/SELECTED_NUMERIC_SMOKE_INPUT.json \
  --pca-fit-cache-root /mnt/project/st-world-policy-20261016/audits/history_data/native384_train64_cost_v1 \
  --configured-output /mnt/project/st-world-policy-20261016/audits/ondemand_provider_wiring_v1/configured
```

Generated provider identity:
`ad994b593bd70801041893a028a194a6d1854bf8c70bbdeb22756fdd72843773`.
The ordinary existing registration must pin the generated `training_config.json`,
fixed data list, existing Base, target contract, environment/resource lease, and
**PROVIDER.json** as `teacher_manifest`. Lead and independent reviewers provide
the new registration/approval; none were manufactured here. There is no truthful
ready-to-launch registration path yet. Once supplied, the unchanged CLI is:
`torchrun --nproc_per_node=1 -m tools.st_world_policy.train_world --registration ... --approval ...`.
For the same run, append `--resume` (and `--acknowledge-stop` only after an operator
stop). This retains complete native state and checks the provider identity.

Both `teacher_provider --help` and `train_world --help` were executed. No
unregistered new trainer, tensor cache, background process or GPU task was started.
