# Indexed provider → unchanged PCA384 → student SMOKE handoff

Status: **METADATA_CHECKED_NOT_ADMITTED**. No trainer, teacher, GPU, image,
tensor, weight, resource registry or storage admission was run or changed.
The old 64-scene run is untouched. This preparation does not grant formal or
full-population training permission.

The CPU-only audit ran against immutable runtime
`31c3dca2fbc697b419bf343ce05d10be0a4fe13f` at
`/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017`.
The implementation worktree starts at `42854f40d0dab551463d63abae80a1045b64722f`.
Only this report directory is added; no provider, model, trainer or PCA source
is changed.

## Verified dependencies

| Dependency | Actual identity / result |
|---|---|
| Real indexed composition qualification | `/mnt/project/st-world-policy-20261016/audits/indexed_provider_composition_run_v1/RESULT.json`, SHA256 `a4f4c86f0b5c09d18a58d02e8919dd73e659b48a53ed0d2a25c7d68e497b7a2f` |
| Dataset | `/mnt/project/st-world-policy-20261016/audits/indexed_composition_preparation_v1/SELECTED_SCENES.json`, SHA256 `7c78c58c40456649a93d152f9925ea5e1faa7e08bb65e98cd6c4af3b0c52b9b8` |
| New exact request population | 32 train scenes; 294 image keys; 189 existing-current-cache hits; 105 on-demand misses |
| Request index | `d8e82f21aa8fea1d4353e1a442d22e01fc0a2ad6a36f232602dd7b9521161dd3` |
| Current-cache index | `326a3f37344b4526b8595954cb7d0b1dbb6eabdfdc836ea9582d2dc239dc725b` |
| Original PCA-fit index | `390790f1fc6a469ab76fc855f477d2308122888819ce73b1dc9bb069140716ff` |
| Original PCA identity | `268723c8da3151492fd20a81b0f6f541f348a1187222132b055f4e4a528bfd46` |
| Expected provider identity | `90a908a48eba2e3d94e1d07700178dd08c61cb4aa0943459bbf572bd1b17ff8b` |

The actual existing receipt consumer validated all real composition evidence,
including prior fixed8 qualification, full comparisons, source-code pins and
resource/count consistency. Each of the three metadata indices was opened once;
the bound provider validator reused those objects. New dataset rows exactly
match `future_metadata_qualification32_v2/PLAN.json.selected_rows`, including
order, source-log join and every original current/ego checksum. The request
header and student `identity.json` agree on source identity, original index,
partition and train role. No current JSON, ego tensor or image was opened.
Their physical bytes remain subject to the unchanged trainer checks.

The frozen original PCA is still
`/mnt/project/st-world-policy-20261016/audits/history_data/pca_train64_256_384_v1`.
Its original fit cache is `native384_train64_cost_v1`, not the new 32 scenes.
Its original basis supports 256 and 384 dimensions; the retained T2 diagnostic
uses **384**. The original raw1024 channel family, teacher/transform identities,
FP32 transform, exact PCA384 projected contract and native 18×24 grid all match.
This audit verified metadata and recorded tensor checksum; it did **not** verify
the PCA tensor bytes again and does not claim to have loaded them. Existing
`configure_smoke` and dataset loading will perform that strict check. No refit.

## Recipe and input configuration

`INPUT_CONFIG.json` is a factory input, **not** a registered runnable training
configuration. It intentionally retains the old descriptor pins from the source
config until the existing `configure_smoke` replaces them with the newly
qualified descriptor and its PCA application scope. Do not register this input
file or copy the old 64-scene registration.

Relative to the old continuous `CONFIG.json`, exactly two values change:

- `execution.sample_limit`: 64 → 32.
- `runtime.output_root`: new isolated
  `/mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/configured/runs`.

Everything else remains: T2, S-P2, three views, W48/view, 96 sampled positions,
current and future supervision, no history/delta, 20,000-update scheduler,
1,000-step warmup, AdamW recipe, FP32 master/Adam with BF16 execution and ZeRO2,
TF32 off, original attention backend, seed42, world1/micro1/global2, two gradient
accumulation microbatches, `stop_after=2`, save every update and 900-second cap.
This is a fresh disposable two-update diagnostic from the common registered S0,
not continuation of the old diagnostic and not a 2-step cosine schedule.
It cannot initialize formal work.

The teacher remains independently FP32, fixed actual batch8 with counted padding,
three-view native1024 features and an 8-record RAM LRU of 14,159,232 bytes.
The student is not converted to all-FP32. All future targets are teacher labels,
not additional model inputs. First-use immutable shard checksum behavior remains
unchanged.

## Actual CPU evidence

The saved `HANDOFF.json` records:

- elapsed 5.261174787 seconds;
- logical `/proc/self/io` rchar delta 857,254,384 bytes;
- peak RSS 637,583,360 bytes;
- full-current index opens: 1;
- tensor/image/weight payload reads: 0 (Python audit hook rejects their suffixes);
- GPU/trainer execution and resource registration: false.

The process was limited to two CPU threads, CUDA hidden and a 180-second outer
timeout; guards enforce 2 GiB RSS and 2 GiB logical reads. This is metadata
correctness evidence, not training throughput or numerical requalification.
Existing `teacher_provider --help` and `train_world --help` both executed and
returned zero on the qualified runtime. No new CLI or trainer is introduced.

## Deferred preparation command

`PREPARE_COMMAND.sh` contains the actual existing CLI with all pinned paths and
limits. **It was not executed.** Run only after Lead grants preparation IO and
the original PCA-basis read. This command constructs the real descriptor and
PCA scope and writes `configured/training_config.json`; it starts no training.

```bash
bash /mnt/project/VLA-Drive-st-indexed-composition-20261017/reports/st_world_policy/20261017/indexed_smoke_handoff_v1/PREPARE_COMMAND.sh
```

Why deferred: the existing factory scans the current index four times across
`prepare_indexed`, descriptor validation, `configure_smoke` and PCA registration.
Each scan includes 317,326,781 bytes of source metadata plus the 381,902,848-byte
SQLite file, approximately 2.80 GB total for four scans before other reads.
It also reads the existing 1,590,864-byte PCA tensor. That exceeds this task's
2 GiB metadata-only allowance. This is an explicit preparation-cost limitation,
not a numerical failure or a reason to build a new teacher/cache/trainer.
Do not bypass PCA loading or sign a new COMPLETE receipt to avoid it.

## Lead-only registration and one finite invocation

After preparation, the Lead must independently inspect the generated scope and
config and allocate a fresh resource/storage lease. Required registration
artifacts:

- config: new `configured/training_config.json`, never `INPUT_CONFIG.json`;
- data_manifest: exact new 32-scene file and hash in the table;
- teacher_manifest: new `PROVIDER.json`;
- target_contract: unchanged PCA384 target, equal to the source
  `continuous/TARGET_CONTRACT.json`;
- base_binding: existing `/mnt/project/st-world-policy-20261016/bindings/BASE_BINDING.json`,
  SHA256 `84544c2242c0c7ecdcbe4450073b23a3c556cad4d822860d5b8652ecee9f1e38`;
- environment: newly verified actual runtime record;
- resource_lease: **new** host/device/time/IO/disk/RAM lease, not the expired old
  continuous-run lease;
- exact clean runtime SHA/fingerprint and new run ID;
- execution equal to `HANDOFF.json.execution`, diagnostic weights false.

No registration or lease is fabricated here. Once the Lead has created and
verified the actual registration in the new namespace, the existing CLI is:

```bash
cd /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
# CUDA_VISIBLE_DEVICES must come from the fresh actual Lead lease.
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.check_formal_approval \
  --registration /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/REGISTRATION.json \
  --repo /mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017
/root/miniconda3/envs/ddp/bin/torchrun --standalone --nproc_per_node=1 \
  -m tools.st_world_policy.train_world \
  --registration /mnt/project/st-world-policy-20261016/smoke/world_indexed_provider_native2_v1/REGISTRATION.json
```

The registration path above is a planned output, not an existing approval.
Only one two-update demonstration is proposed. No second continuous/resume
pair, broader population, formal approval, cache construction or active-run
change is included. The parent must also validate checkpoint/output capacity
before launch; the teacher's earlier 16 GiB allocated-memory qualification is
not a student-training memory budget.
