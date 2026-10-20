# Independent resource-only cache namespace migration review

2026-10-20 UTC. Reviewed `c17e685247090e56218181ac75aa9545a049c600` plus source-cache guard `ef7683ec`. Decision: **PASS for this actual data-cache migration and bounded resumption of frozen-S0 feature extraction**. This is not a new Base, inference-numeric migration, permission to restart a Base training process or scientific-result approval.

Original and new outputs remain separate:

- Original: `/mnt/project/action-effect-repair-20261020`.
- Resource execution v2: `/mnt/project/action-effect-repair-20261020-budget-v2`.
- New config: `configs/action_effect/shared_budget_v2.yaml`.
- New logical output cap: 8GiB. Shared free-space reservation: 300GiB. Original action head remains an external verified reference, not a duplicate tensor file.

## Source review

The compatibility comparison excludes exactly `artifact_root` and `resources`; every scientific configuration field must remain equal. The original immutable binding must match the supplied original configuration hash. `prepare(new_cfg)` independently checks the current Base and extraction source/package closure, constructs the same seeded observation cohort and retains the model/input/normalizer contract. Before copying any complete context receipt, the tool requires identical Base checkpoint, Base-binding hash, extraction contract and cohort content SHA.

Only source shards with an existing COMPLETE receipt are eligible. Each original receipt and payload checksum is verified. The body is hardlinked without tensor decoding, recomputation or conversion, then its SHA is verified again. New receipts explicitly bind the new execution identity and original scene order/offsets. The original body/receipts are unchanged; inode ctime/link count may change, as disclosed. No header containing a false original execution identity is written.

The original action-head receipt references its existing outside artifact. Its bytes are checked rather than copied. The actual source head's schema, matching Base hash and `strict_original_master=true` were independently checked below. A generic guard for these three provenance fields in the migration helper would improve future unsupported-source rejection, but there is no unresolved provenance gap for this inspected source.

The exact-request native cache is shared via an explicit symlink. An initial review found that a missing source directory could create a dangling link while returning PASS. `ef7683ec` now requires an existing source directory, rejects pre-existing normal directories, wrong or dangling links and verifies the resulting directory resolution. Existing candidate receipts remain authoritative; any future exact-request reuse still follows the original canonical checksum chain.

## Independent CPU tests

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/action_effect/test_resource_migration_independent_review.py
```

Actual result: **13 passed in 0.13s**. Tiny synthetic files and a mocked new-binding constructor isolate migration/checksum/hardlink semantics. These are not actual model/shard performance or scientific observations.

Covered behavior:

- Exact inode/device and payload bytes across the hardlink, preserved original receipt bytes, explicitly changed execution-binding identity and idempotent rerun.
- Changed scientific seed rejected before calling the new-binding constructor.
- Original configuration hash mismatch rejected.
- Changed Base checkpoint, Base-binding hash, extraction contract or cohort rejected.
- Missing or corrupt completed payload, missing head receipt or native cache rejected.
- Existing wrong normal-directory or dangling-link destination rejected.
- No inference, simulation or model update occurs.

No original S0 replay or complete scientific test matrix was rerun.

## Actual metadata admission

Read-only actual-source checks, with no repeated payload/weight hash scan:

- `main.yaml` and `shared_budget_v2.yaml` match exactly after removing only output root and resource budgets.
- Original config digest matches the original immutable campaign binding.
- Twenty original complete context receipts cover 320 scenes. The tool itself must still perform each payload SHA check before linking.
- Actual action-head receipt has the expected schema, the same Base-binding hash and `strict_original_master=true`; referenced artifact size is 3,278,134,372 bytes. No new head artifact was created.
- Original native-cache directory exists.
- Observed shared available space was 347,689,750,528 bytes, above the declared 322,122,547,200-byte floor. This is a momentary observation, not a reservation guaranteeing future availability.
- Reviewer launched no actual migration, model forward, GPU job or simulator call.

Structured evidence: `RESOURCE_NAMESPACE_INDEPENDENT_METADATA.json`.

## Explicit remaining resource boundary

The common `check_space` traversal does not count files beneath the shared directory symlink toward the **new root's** 8GiB logical cap. This does not affect the approved hardlink migration or continued context extraction, which introduce no native scoring calls. Before expanding G0 simulation through that shared store, record/check its separately owned growth and combine it with the new-root budget; the common shared-free-space floor remains enforced. Do not describe the 8GiB new-root cap as already covering all future shared-native writes.

If free space falls below the new declared floor, retain safe partial context shards and block further writes. No deletion, old-config mutation or process termination is approved by this review.

## Actual migration entry

After the Lead integrates this report and confirms the source producer is not writing new source shards:

```bash
cd /mnt/project/VLA-Drive-action-effect-repair-20261020
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.migrate_execution_namespace \
  --old-config configs/action_effect/main.yaml \
  --config configs/action_effect/shared_budget_v2.yaml
```

Subsequent frozen-context collection and DEV registration use the new resource configuration and its new execution namespace. Preserve original inference code, complete cohort, numerical settings and scene order. The migration result is not proof that all1024 TRAIN or all1696 DEV features have been generated; those manifests must retain their normal completeness gates.
