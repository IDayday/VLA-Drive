# Independent review fixes, v2

The original independent 6e20c8b6 counterexamples remain unchanged in
`metadata_storage_guard_review/`; owner did not replace their failed logs.
Review findings MG-F01, MG-F02 and MG-F03 were correct and are fixed here.

- **MG-F01:** pending retries now recheck fresh Lead quiescence under the same
  CLAIMS.lock, before state publication and before completion publication.
  Expiry blocks without changing ledger or discarding pending/snapshot files.
  Explicit optional `renew_quiescence_receipt` / `renew_quiescence_sha256` allow
  recovery. The renewed Lead receipt must carry all original bindings and
  `resumes_migration_id` equal to the exact original transition. It must pass
  the same root/controller/process-evidence checks and fresh time interval.
  The original transition ID/pending/snapshot/history remain immutable; the
  completion records the actual fresh receipt path and SHA used. A completed
  transition is read-only on later retry. Replacing the original quiescence
  argument to generate a second pending transition is rejected.
- **MG-F02:** write admission validates every planned parent component against
  `directory_patterns` and the registered depth even when no such directory
  exists yet. The previous existing-directory census was insufficient. An
  unregistered ancestor rejects before mkdir/file IO; existing symlink and
  filesystem protections remain in place.
- **MG-F03:** thin runner now validates source pins and callback capability
  **before** parser/build or storage admission. `source_pins` binds the exact
  Python entry file and every declared callback dependency by path/raw SHA256.
  The entry point must declare `METADATA_STORAGE_CALLBACK_PROTOCOL` equal to
  `st_metadata_storage_check_v1` and a tuple
  `METADATA_STORAGE_CALLBACK_DEPENDENCIES`. Loaded entry file must match the
  registered source cwd. Missing/changed/extra dependency pins reject. Actual
  callback counters remain a postcondition, not the former sole gate.

The owner of index/future preparation adds the capability declarations only
with the real callback integration. Current agreed dependencies:

- index CLI: `starVLA.dataloader.st_world_metadata_index` and
  `starVLA.dataloader.st_world_future_provenance`.
- future CLI: those two plus `tools.st_world_policy.prepare_current_teacher`
  (contains source_population).

The CLI itself is always additionally required. Model weights are not source
pins. A one-second periodic wrapper around bounded parse/hash loops is valid;
write/create/publication checks cannot be throttled. The outer independent
process monitor and timeout are still required for blocked native calls; this
change adds no RPC, process dispatch or model computation.

Actual owner command:

```text
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/ddp/bin/python -m pytest -q \
  tests/st_world/test_metadata_storage_guard.py \
  tests/st_world/test_cache_storage_budget.py \
  tests/st_world/test_review_metadata_guard.py
```

Result: **93 passed**, one existing pynvml deprecation warning. This includes
all three independent counterexamples unchanged, four renewal crash windows,
wrong/active renewed receipts, planned ancestor/depth cases, and five source
capability failures that prove parser/build/output admission did not occur.
Full current output is `REVIEW_FIX_TEST_OUTPUT_v2.txt`.

Status: IMPLEMENTED / OWNER_TESTED. Await independent re-review. No real policy,
ledger, metadata build, image/teacher access, GPU or training task was changed.
