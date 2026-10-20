# One strict build, two immutable validation receipts

Scope: T0 validation CPU/IO scheduling only. No simulator, model, data population,
labels, training objective, GPU execution or scientific threshold changes.

`build_and_validate_datasets(root)` owns a real strict `build(root)` call and
seals both PAIR and REPAIR standard native-validation receipts from that pass.
The builder decodes each native candidate once into a bounded one-scene cache.
Pair construction and label semantics still perform their existing checks; this
does not claim all semantic validation occurs only once. The receipt inventory
rechecks every pinned body SHA and file version without another native decode.
Pair and repair consumers share a root build lock. Valid unchanged inventories
reuse the same immutable receipt. No externally supplied verified result or
skip-validation flag exists.

Dependencies are observed before their semantic use, with SHA and device/inode/
size/mtime/ctime recorded. Sealing requires the same pins and versions, including
registration, population completion, scene receipts, scene bodies, wrappers,
native receipts, native bodies and ADMITTED records. A coherent mutation of
native data and every parent pin after the strict pass is rejected. Population
length mismatch is rejected before writing a compact COMPLETE artifact.

## Actual original 64-scene check

CPU-only on existing immutable G0 artifacts, 1576 `verified_feedback` calls,
1384 effect pairs, 6436 pinned dependency files. Actual scoped evidence is in
`SINGLE_BUILD_VALIDATION_64.json`.

| Operation | Seconds |
|---|---:|
| Strict build plus both receipts | 70.5358 |
| Both receipts, unchanged inventory reuse | 0.8128 |

PAIR stayed byte-identical: SHA256
`d20b65bef61a0d041683ec5b0b65987a51f4b5ef669e1e87955bbcd9f40a45dc`.
REPAIR stayed byte-identical: SHA256
`198d559133160876f7e14271acc69fcbc414018f3326f99e2b0418931483f994`.
The old earlier first validation took 97.044 seconds; this was not a paired
simultaneous benchmark, so that historical timing is descriptive only. The
structural saving is removal of repeated full native population reconstruction.
This first version uses one CPU worker; full1024 latency remains a material cost.

## Tests and limits

55 targeted CPU tests passed in 5.36 seconds: full inventory modification,
deletion and stat-change negatives, concurrent pair/repair consumers, source and
metadata changes during build, coherently re-pinned source mutation, failed build
without a success receipt, population truncation and G0 transaction semantics.
24 affected G1/Flow/refiner parser tests passed in 13.04 seconds. An older small
G1 fixture was updated to admit the required deployment-input boundary and mock
the new strict-build owner rather than its previous internal implementation.

No real score, new GPU update or 256/1024 full build was run by this change.
Existing old receipts and compact/native evidence were preserved. Science and
cross-host numerical conclusions are unchanged.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/miniconda3/envs/ddp/bin/python -m tools.action_effect.build_effect_pairs \
  --config configs/action_effect/shared_budget_v2.yaml --max-samples 1024 \
  --g0-root /mnt/project/action-effect-repair-20261020-budget-v2/g0/population_001024 \
  --resume
```

This command requires a fully completed and immutable 1024-scene G0 population;
it does not simulate. The resulting pair and repair receipts are reusable by
the existing downstream training parsers.
