# Independent indexed provider/PCA review: changes requested

Reviewed Lead commit `5e0e2ec3ff236b73d3cf2f850f1e6f17dc71967d`, integrating
implementation `6aeba215171a3d4d1f930d033da34d25733caedd`.
Independent reviewer: `/root/st_future_metadata_correctness`.

**CHANGES_REQUESTED** for two reproduced boundary failures. No implementation
files were modified by this review. No real factory qualification is granted.

## F1: old PCA scope can admit an indexed provider

`tools/st_world_policy/provider_pca.py::validate_provider_scope`, legacy branch
at lines 43–69, checks the old scope identity but does not reject an indexed
target provider or indexed descriptor. A valid old scope can be re-signed with
the legitimate indexed descriptor and corresponding provider identity.
`TeacherPCA.load` then succeeds through that old branch, without the required
indexed fit-cache binding.

Reproduction:
`test_independent_provider_boundaries.py::test_legacy_scope_cannot_be_rebound_to_indexed_provider`.
It uses genuine synthetic old/new descriptors and the same frozen synthetic
basis. Expected ValueError is absent. This does not demonstrate a different PCA
basis being applied; it demonstrates a scope/ownership bypass. The old branch
eagerly parses original fit metadata and constructs another provider preview,
whose indexed connections it does not explicitly close, bypassing the new
bounded fit-index scope and owned-provider revalidation.

Required correction: reject indexed providers/descriptors in the legacy scope
validation branch before legacy fit metadata or extra previews are opened.
Keep valid old-provider/old-scope behavior intact. Do not simply accept both
schemas as interchangeable because their channel family happens to match.

## F2: combined preview budget is checked after opening readers

`provider_pca.py::_register_indexed_scope`, lines 72–91, opens its 16 MiB fit
index and both provider indices before checking their total allowance. With
valid request/source budgets of 128 MiB each, the simultaneous declared reader
allowance reaches **285,212,672 bytes (272 MiB)**, then rejects against 256 MiB.
The function eventually closes all readers and does not publish the scope, but
the admission check is too late. This is a declared concurrent cache-budget
violation, **not a claim that the tiny fixture measured >256 MiB actual RSS**.

Reproduction:
`test_independent_provider_boundaries.py::test_pca_preview_never_opens_over_combined_metadata_allowance`.
The test tracks actual constructed/open SQLite readers, their explicit budgets,
and closure. The bound assertion fails at 272 MiB.

The runtime validation path `_validate_indexed_scope`, lines 131–135, has the
same ordering: open fit index, then add its budget to the already-owned provider.
Both paths need metadata-only budget preflight before opening the additional
reader(s). Preserve the post-open exact identity/budget checks as defense in
depth. Do not lower published budgets to make the counterexample disappear.

## Passing behavior and actual tests

Fixed clean detached runtime:
`/mnt/project/VLA-Drive-st-indexed-provider-review-runtime-5e0e2ec3`.
Exact HEAD, clean status and loaded source hashes were asserted.

| Execution | Actual result | Evidence |
|---|---|---|
| New implementation suite, independently run | 28 passed, 14.86 s | `owner_tests_independent.log` |
| Independent boundary tests | 8 passed, 2 failed, 5.48 s | `independent_tests.log` |

No skip/xfail. Existing pynvml warning only. CPU affinity 0,1, two
OMP/MKL/OpenBLAS threads, CUDA hidden, Python 3.10, bytecode disabled.
Only tiny generated metadata/DB/tensor fixtures and a synthetic executor were
used; no actual source DB, raw asset, model/teacher, GPU, PLAN or runtime job.

Passing evidence covers exact qualified request bytes/source identity, fixed8
execution and RNG/partial-LRU/padding gates, <=64 scenes and <=4 SMOKE updates,
old/indexed raw tensor/key/mask equivalence on fixtures, unchanged original PCA
basis and provenance, default owned-provider PCA revalidation without duplicate
request/source connections, executor-failure cleanup, and actual CLI argument
rejection. Metadata headers are not rehashed as fake original full-cache
identities. Legacy and indexed identity scopes remain distinct in their normal
construction paths; F1 concerns the missing validation boundary.

Tested implementation SHA256:

- teacher_provider.py: `e9bde7c61e1e6f5e01d75195d3888722bfb6f4d0a32641240275fa3a5ff6c512`
- provider_pca.py: `210a68078d51f821c8e000fd3a8817400329b1bbb913770dba1da1c424688481`

Run the tests from the fixed runtime, adding its `tests/st_world` directory to
PYTHONPATH for the repository's existing short fixture imports:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  tests/st_world/test_indexed_provider_factory.py -q

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 0,1 /root/miniconda3/envs/ddp/bin/python -m pytest \
  --import-mode=importlib \
  /mnt/project/VLA-Drive-st-future-metadata-correctness-20261017/reports/st_world_policy/20261017/correctness/indexed_provider_factory_v1/test_independent_provider_boundaries.py -q
```

The recorded runs used equivalent pytest harnesses with HEAD/path assertions.
The existing dataset's eager metadata initialization remains acknowledged and
unchanged. Full-population provider use, real-teacher numerical qualification,
training throughput and formal training are outside this review and unapproved.
