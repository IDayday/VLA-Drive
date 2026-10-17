# One indexed-source student smoke: Lead handoff

**LEAD_PREPARATION_ONLY.** No GPU was inspected, acquired or used. No live
resource lease, storage reservation, run registration or trainer was created.
No tensor/weight/image content was read. This is the input handoff for one
discarded-weight, two-update SMOKE, not a second training framework or a repeat
of the previous continuous/split recovery experiment.

## Factory result and namespace separation

The existing provider factory actually completed once under the Lead's separate
resource approval. Its result SHA256 is
`afb0c77ccea1b6e368995ad68efa7140e30e1f3a8fa1ab97e716a04f27c704d4`.
It consumed 7.737 seconds, sampled peak RSS 644,300,800 bytes and
2,994,053,170 rchar bytes, with zero GPU execution or new teacher tensor writes.
It loaded the original PCA basis; no refit occurred.

Verified actual outputs under
`/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1`:

| Artifact | SHA256 |
|---|---|
| `PROVIDER.json` | `bb53252f1aea91ed54fa7b3917314d2d7f6c1c251d1f89f6a079f0f582f3874d` |
| `configured/training_config.json` | `f6b8bd3ad59a763355130ef8bb0ef9228473f424516c62e630def80e87db948d` |
| `configured/PROVIDER_CHANNEL_SCOPE.json` | `dc9c7c5f6be9213a9d755f17d9c12a34a46c4ddf2f10b3c80c952a03bc6ae41e` |
| `configured/PROVIDER_CONFIG_LINEAGE.json` | `e5203bdfcb7eef1b9fb7ffd59b42d524c037e23488daf60830a75509e3d429d1` |

That root has a **32 MiB metadata allowance**, not permission to place model
checkpoints in its `configured/runs`. All factory outputs remain unchanged.

This handoff's `TRAINING_CONFIG.json` derives from the real factory config with
exactly one change: `runtime.output_root` becomes
`/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/runs`.
Its SHA256 is `2ec8df5c996c51ef0b7b40707c0e66179643bcc5cac317ec4d53d31d0d85b13f`.
The new training root needs its own checkpoint/storage admission. The config can
be copied verbatim to that root as `CONFIG.json`; only the artifact path pin in
registration changes. Do not rewrite the completed factory config.

## What is inherited and what remains untested

The actual immutable runtime is
`/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017`,
commit `31c3dca2fbc697b419bf343ce05d10be0a4fe13f`.
Metadata comparison confirms these six files are byte-identical to the prior
real student qualification at `8c1e3f86268ef5ed79f46d69abf33d5c9ca74bdf`:
`train_world.py`, `training_state.py`, `fp32_master_init.py`, `official_runtime.py`,
`teacher_pca.py` and `starVLA/dataloader/st_world_dataset.py`.

Inherited evidence includes the original current/future/ego supervision,
BF16 execution/FA2/TorchAdamW/ZeRO2 with FP32 master/moments, common-S0 strict
load, native checkpoint inventory and functional restoration. Prior independent
continuous versus split runs already exhibited FA2 backward variability before
recovery. The old result is functional recovery evidence, **not full-model
bitwise continuation evidence**. No third comparison is proposed here.

Changed components are the indexed metadata/provider/PCA application-scope
plumbing and the exact selected population. The new 32-scene teacher-composition
qualification and its independent closure cover 294 keys, source hits/misses,
fixed8 layout changes, full-tensor equality, missing masks, sampling and RNG.
The actual factory now verifies the original PCA tensor and indexed scope.
Neither alone demonstrates that this new source completes a student update.

Still pending: **one real two-update student run with the new indexed32 source**,
including actual current/ego record checks, finite current/future/ego losses,
FP32 master update evidence, indexed provider counters and committed native
saves. This does not qualify formal/full-population throughput, multi-GPU
scaling, convergence, planning quality or a scientific winner.

## Exact proposed run

- Run ID: `indexed_provider_single_updates2_v1`.
- Local physical GPU0, world1, microbatch1, global batch2, accumulation2.
  Actual host/UUID/availability must be freshly checked by the Lead.
- Exactly 32 pinned scenes, manifest SHA256
  `7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8`.
- Common Base binding SHA256
  `84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38`.
- T2/PCA384, native18×24 grid, W48/view, original three views, 96 sampled
  positions, current/future/ego supervision; no history/delta change.
- Original 20,000-step cosine timeline, 1,000-step warmup and optimizer recipe;
  `stop_after=2`, `save_every=1`, loader0, seed42, deterministic flag true,
  effective checkpointing false, 900-second trainer wall limit.
- Frozen fixed8 FP32 teacher with unchanged 14,159,232-byte tensor LRU, existing
  current targets and computed missing labels; no teacher tensor disk output.
- One initial invocation, no resume arm, automatic retry or extra update.
  SMOKE weights remain unusable for formal initialization.

## Native checkpoint disk accounting

The two real complete old checkpoints were read only through small
`COMPLETE.json` receipts and file `stat`. Receipt identities, exact inventories
and physical sizes match. Payload SHA256 values are inherited evidence; their
37 GB contents were **not** reread or newly validated.

| Per checkpoint component | Actual bytes |
|---|---:|
| ZeRO2 optimizer/master/moments | 30,632,173,006 |
| Model state | 6,733,729,058 |
| Rank0 RNG | 25,402 |
| All native tensor files | 37,365,927,466 |
| Complete checkpoint directory including receipt | 37,365,928,655 |

Normal fresh two-update execution saves step1 and step2: **74,731,857,310 bytes**
for both complete directories. The unchanged trainer has **no unconditional
step0 save**. A timeout or operator boundary before the first update can save
step0 and terminate; do not invent a successful three-checkpoint observation.
The budget nevertheless conservatively reserves three full-size slots for
step0/1/2 or one interrupted write in addition to two completed saves:
**112,097,785,965 bytes (104.4 GiB)**.

Proposed independent reservation: **120 GiB = 128,849,018,880 bytes**, leaving
16,751,232,915 bytes for partial/native size variation, logs and metadata.
This counts complete model + optimizer + RNG, not merely an exported model.
Retain the old controller's separate **1 TiB free-space floor** unless the Lead
explicitly reviews a different policy. This is not filesystem quota, fallocate
or a live reservation. Do not delete old checkpoints.

At step2 the existing `can_reuse_just_saved_boundary` validates the just-committed
receipt and file sizes, so the following budget pause does not write a duplicate
step2 snapshot. No resume hashing or checkpoint deletion is added. Old observed
save walls were 85.415 and 86.076 seconds; total old allocation wall was
624.807 seconds. These are historical sizing/timing references, not a prediction
or measured speed for the new indexed run.

## Registration input and controller reuse

`REGISTRATION_INPUTS.json` contains all existing artifact pins, actual clean
runtime SHA/fingerprint, finite execution, proposed run ID, checkpoint sizing
and config lineage. It deliberately is **not** `st_world_run_v1` and cannot be
passed to the trainer. A fresh actual environment and resource lease are absent.

Reuse the checks from the existing private Lead controller
`/mnt/project/st-world-policy-20261016/resources/qualify_fixed_provider_training_v1.py`:
clean source, actual host authorization, actual idle target UUID, per-GPU and
run locks, nonexisting new run, exact config/artifact hashes, environment match,
finite lease, native checkpoint budget, launch/PID/start-time receipt, own-group
watchdog and archived result. Preserve the registered trainer unchanged.

The old script cannot simply be executed: it hardcodes old runtime/GPU1,
64-scene dependencies, a 160 GiB shared two-arm budget, and continuous/split
actions. The Lead's finite control wrapper must instead bind this input record,
GPU0, the new training root and 120 GiB one-run budget. No split, request-stop or
resume action is needed. Use the existing `environment_record`,
`source_fingerprint`, `verify_registration` and `check_formal_approval` functions;
do not build another trainer or generate an approval PASS.

After actual Lead registration, the callable single invocation is:

```bash
cd /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.check_formal_approval \
  --registration /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/REGISTRATION.json \
  --repo /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
timeout --signal=TERM --kill-after=180s 1020s \
  /root/miniconda3/envs/ddp/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 \
  -m tools.st_world_policy.train_world \
  --registration /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_training2_v1/REGISTRATION.json
```

These commands were **not executed**. The registration path is a planned new
artifact. The lease must cover initialization, both saves and watchdog grace;
the old expiry is not reusable. Preserve any incomplete output instead of
automatically restarting. Expected diagnostic completion is exit0,
`completed=2`, `new_optimizer_updates=2`, `exposure=4`, epoch0/offset2,
`PAUSED/BUDGET_PAUSED`, two complete native saves and zero provider failures/new
teacher tensor bytes. Preserve all per-component losses and actual effective
denominators; no selected teacher key counts are invented in advance.

## Preparation evidence

This follow-up's actual metadata/stat preparation took 0.251414 seconds,
explicitly read 310,235 bytes of small JSON/source evidence and peaked at
21,131,264 bytes RSS. It opened no metadata index, tensor, image or model and
did not import the trainer. `prepare_inputs.py` records how these artifacts were
derived. Schema/syntax and one-field config-diff checks passed. No current
training result is claimed.
