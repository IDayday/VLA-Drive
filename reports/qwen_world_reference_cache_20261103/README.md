# Frozen IL reference terms reuse

Base: `5e5220c`. Execution-only, off by default. No live source or job was changed.

`starVLA/rl/qwen_world_research_algorithms/frozen_reference.py` is the shared
`ResearchLoss` / `_physical_anchor` hook for S0/S1/S2/S3/S4/S5. Opt in with
`--frozen-reference-cache`. Its protocol is recorded only in
`identity.execution_optimizations`, outside the existing scientific identity.
All scientific recipe, condition SHA, parameter closure, anchor weights,
coordinates and current-memory reconstruction remain unchanged.

Each saved leaf owns detached, original-dtype `mean/std/vel` from the permanently
frozen IL reference. Its ancestry binds the complete saved chain/times/deltas/W3
indices/condition descriptor, same-run behavior version, pinned IL parent,
original recipe and source identities. A tensor seal is checked on reuse and
native pending restoration. Present invalid caches are rejected, never rebuilt.
Reference module eval/frozen flags and tensor mutation versions are checked.
Current-policy values/KV/graphs and all old log probabilities/advantages are
outside this cache.

Fresh windows populate constants after collection finalization, before the
existing native commit. Normal native cpu_tree/device_tree serialization saves
them. Old pending checkpoints are first admitted by the unchanged strict native
checks; only a missing cache may then be built from the restored same-run IL.
No collection/optimizer step or RNG draw is added. One M4 window should require
three W3 reference head calls per scene instead of twelve; M2 six→three. This is
an expected call reduction, not a measured production speedup.

CPU tests: `tests/test_world_frozen_reference_cache.py` cover original/signed-FM/
FPO losses × x/x_kl/v anchors, four-inner value/gradient/Adam/RNG bit equality,
detached pending serialization, input/parent/source corruption rejection and
real tiny-Scope BF16 current-memory reconstruction. The tiny model is not the
production model.

The operator-allocated full-model probe is
`tools/qwen_world_performance/qualify_reference_cache.py`. It loads the actual
full IL70, reuses existing first-collection scored TRAIN leaves, runs four
original and four cached steps, compares every step's gradient/Adam/parameter/
scheduler/RNG hashes, and round-trips cached pending material at inner2.
It does not generate or score trajectories and cannot dispatch itself.

Suggested explicit arguments after GPU admission:

```sh
python -m tools.qwen_world_performance.qualify_reference_cache \
  --config /mnt/project/qwen-world-performance-rl-20261103/configs/S1_NOMINAL_RESCUE.yaml \
  --manifest /mnt/project/qwen-world-performance-rl-20261103/data/NAVTRAIN_SCENES.json \
  --data /mnt/project/qwen-world-rl-optimization-20261030/data/NAVTRAIN \
  --il-checkpoint /mnt/project/qwen-world-rl-optimization-20261030/IL_PARENT \
  --parent-selection /mnt/project/qwen-world-rl-optimization-20261030/PARENT_SELECTION.json \
  --material /mnt/project/qwen-world-performance-rl-20261103/runs/S1_NOMINAL_RESCUE/behavior/collection000001/rank0.pt \
  --run-identity /mnt/project/qwen-world-performance-rl-20261103/runs/S1_NOMINAL_RESCUE/identity.json \
  --maximum-scenes 1 --output OPERATOR_SELECTED_QUALIFICATION_DIRECTORY
```

No GPU probe has been dispatched by the implementation author. Production
admission requires its actual PASS and a typed `verify_migration` receipt bound
to exact old/new identity, old checkpoint inventory, evidence files and their
hashes. Old behavior/fresh-ratio tolerance is not relaxed. The root operator
alone freezes and applies such a migration.
