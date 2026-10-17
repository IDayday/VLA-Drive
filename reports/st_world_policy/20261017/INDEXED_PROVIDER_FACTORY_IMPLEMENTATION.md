# Finite indexed teacher provider / unchanged PCA basis

Implementation parent: `d3ac224902462cfde00b1320dcaf82eec5c20545`.
The handoff audited Lead `1162b955`; this patch touches only
`teacher_provider.py`, `provider_pca.py`, synthetic tests and this evidence.
No dataset, trainer, fixed8 executor, teacher precision, sampling, target math,
existing metadata, live runtime, approval, PLAN or source population changes.

## Selectable path

`teacher_provider.prepare_indexed` and the existing CLI now accept already
completed request/current metadata indices. They do not build an index/cache,
load teacher weights or write teacher tensors. New schema:
`st_indexed_teacher_provider_dependency_v1`.

The descriptor binds each index's original COMPLETE raw SHA, semantic index and
request identities, explicit reader budget, original request path/SHA and the
qualified original current cache identity. It reuses `MetadataIndex`,
`IndexedTeacherCacheReader`, `IndexedBoundedTeacherProvider` and the existing
`FixedBatchDINOExecutor` factory. Every original cache/source/contract/row/shard
check remains in those modules. The source's COMPLETE identity is not replaced
by an index identity. No full request/record JSON parsing or lazy population
materialization is introduced in this branch.

The old eager descriptor and application scope remain supported. Their common
qualification/limit checks were extracted without changing the rules. Both
branches remain **SMOKE only, train only, at most 64 scenes**, matched to the
exact completed qualification request path/SHA and original current source.
`configure_smoke` remains **at most four updates**, exact sample count,
same deterministic execution, `loader_workers=0`. The indexed branch accepts
only the qualified fixed8 execution, including partial-LRU/RNG evidence;
legacy-repeatability and legacy-reference compatibility remain separate facts.

Metadata indexing or reference comparison alone is not real-teacher numerical
qualification. In particular, completing a full current index does not admit
that different cache identity under a qualification bound to current256, and a
new 32-scene metadata manifest cannot borrow another 64-scene numerical result.
Those mismatches fail rather than silently widening the population.

## PCA application scope

New scope: `st_pca_indexed_provider_channel_scope_v1`.
An explicit metadata index of the **actual original fitted cache** is required.
It retains the original fit cache identity, raw identity/COMPLETE dependency
hashes, teacher contract, channel family, PCA receipt/statistics SHA and the new
provider descriptor/identity. The compact header is used only for the verified
channel family; it is never rehashed as an original full cache identity without
its records. `TeacherPCA.load` still verifies and applies the identical frozen
basis. No refit, dimension choice, normalization change or fake
`target_cache_identity` is used.

PCA validation rechecks an already owned indexed provider rather than opening
duplicate request/source SQLite connections. Fit/provider metadata budgets are
checked together against 256 MiB; defaults are 16 MiB per index, hence 32 MiB
provider metadata and an additional temporary 16 MiB fit-index allowance.
These are metadata-reader budgets, **not total process RSS**. Temporary previews
are closed on completion/error and before nested scope preparation. Actual
teacher weights, activations, per-rank tensor LRU and caller batches retain their
separate existing budgets. Readable metadata still receives startup SHA scans;
this change does not claim to remove that IO.

## Commands (prepared, not dispatched)

`--help` is captured in `evidence/indexed_provider_factory_v1/cli_help.log`.
The variables below must name existing, exactly matched, registered artifacts;
no real artifact/config was created during this implementation:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  /root/miniconda3/envs/ddp/bin/python -m tools.st_world_policy.teacher_provider \
  --request-index "$ST_QUALIFIED_REQUEST_INDEX" \
  --source-index "$ST_QUALIFIED_CURRENT_INDEX" \
  --teacher-config "$ST_ORIGINAL_TEACHER_CONFIG" \
  --qualification-result "$ST_FIXED8_QUALIFICATION_RESULT" \
  --output "$ST_NEW_PROVIDER_DESCRIPTOR" \
  --max-metadata-cache-mib 16 --max-cached-bytes "$ST_QUALIFIED_RAM_BYTES" \
  --max-scene-mib 64 --max-batch-images 8 \
  --training-config "$ST_SELECTED_SMOKE_CONFIG" \
  --pca-fit-cache-root "$ST_ORIGINAL_PCA_FIT_CACHE" \
  --pca-fit-cache-index "$ST_ORIGINAL_PCA_FIT_INDEX" \
  --configured-output "$ST_NEW_SMOKE_NAMESPACE"
```

This emits metadata and a selectable config for the **existing** registered
`train_world` entry. It neither registers nor launches training. A raw-target
SMOKE config omits the two PCA-fit options. Eager `--requests/--source-root`
remain supported; mixing eager/indexed flags, ignored indexed options, wrong
hashes/sources/qualification or larger scope fails nonzero before publication.

## Current evidence and remaining boundary

Focused command (CPU affinity 0,1, no visible GPU, 180-second timeout):

```bash
python -m pytest -q --tb=short \
  tests/st_world/test_indexed_provider_factory.py \
  tests/st_world/test_provider_training.py \
  tests/st_world/test_fixed_provider_training.py \
  tests/st_world/test_provider_scope_boundaries.py \
  tests/st_world/test_teacher_metadata_index.py tests/st_world/test_pca.py
```

Actual result: **103 passed, 20.36 seconds**, including 28 new indexed-factory
cases; one existing pynvml deprecation warning. The full actual result is in
`evidence/indexed_provider_factory_v1/focused_tests.log`.
Coverage includes real `python -m` subprocesses on tiny synthetic files,
unchanged old paths, blocked eager reader/JSON/population iteration, exact
online-executor/immutable-source keys, raw tensor/mask equality, unchanged RNG,
cold RAM restore, identical PCA output, unchanged basis bytes, new scope
identity, changed metadata versions, malformed source/index/hash/budget,
qualification failure, wrong population, and original SMOKE limits.

The initial new fixture setup was rejected because an older PCA toy fixture
omitted source provenance. Tests now create a separate fully sourced toy fit
namespace; production provenance validation was not relaxed.

No real teacher/numerical/trainer run was performed here. Synthetic behavior
does not establish real target/optimizer exactness or throughput. The dataset
still eagerly reads source/selection/pin/history metadata and initially hashes
selected ego-label files, as documented in the preceding handoff. Its full
population RAM/IO and runtime preview ownership remain separate integration
work. No PROFILE/formal gate was opened, no multi-terabyte tensor cache was
created, and no current Base/spatial/temporal task was restarted or modified.

Tested implementation SHA256:

- `teacher_provider.py`: `e9bde7c61e1e6f5e01d75195d3888722bfb6f4d0a32641240275fa3a5ff6c512`
- `provider_pca.py`: `210a68078d51f821c8e000fd3a8817400329b1bbb913770dba1da1c424688481`
